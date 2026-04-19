"""Realtime inference service for robot integration."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from src.annotation.quality_gates import GateStatus, QualityGate
from src.features.extractor import FeatureExtractor
from src.ingestion.ingestor import DataIngestor
from src.skeleton.extractor import SkeletonExtractor

LOGGER = logging.getLogger(__name__)


@dataclass
class TrackState:
    """Current state for one tracked worker."""

    track_id: str
    frame_idx: int
    timestamp_ms: int
    quality_gate: str
    action_class: str
    risk_score: float
    mean_confidence: float
    bbox: list[float]
    feature_vector: list[float]


class RealtimeEngine:
    """Continuously process frames and keep latest state in memory."""

    def __init__(
        self,
        source: str,
        session_id: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        cfg = config or {}
        ingestion_cfg = cfg.get("ingestion", {}) if isinstance(cfg.get("ingestion"), dict) else {}
        skeleton_cfg = cfg.get("skeleton", {}) if isinstance(cfg.get("skeleton"), dict) else {}
        quality_cfg = cfg.get("quality_gates", {}) if isinstance(cfg.get("quality_gates"), dict) else {}
        runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg.get("runtime"), dict) else {}

        self.ingestor = DataIngestor(
            source=source,
            mode="realtime",
            session_id=session_id,
            target_resolution=(
                int(ingestion_cfg.get("target_width", 640)),
                int(ingestion_cfg.get("target_height", 480)),
            ),
            dataset_fps=int(ingestion_cfg.get("dataset_fps", 2)),
            realtime_fps=int(ingestion_cfg.get("realtime_fps", 15)),
        )
        self.extractor = SkeletonExtractor(
            {
                "device": str(runtime_cfg.get("device", "cpu")),
                "yolo_model": skeleton_cfg.get("yolo_model", "models/checkpoints/yolov8s-pose.pt"),
                "yolo_conf_threshold": float(skeleton_cfg.get("yolo_conf_threshold", 0.6)),
                "ensemble_conf_threshold": float(skeleton_cfg.get("ensemble_conf_threshold", 0.7)),
                "mediapipe_model_complexity": int(skeleton_cfg.get("mediapipe_model_complexity", 1)),
                "mediapipe_tasks_model_path": skeleton_cfg.get("mediapipe_tasks_model_path"),
                "mediapipe_tasks_model_url": skeleton_cfg.get("mediapipe_tasks_model_url"),
            }
        )
        self.quality_gate = QualityGate(
            detection_threshold=float(quality_cfg.get("g1_detection_threshold", 0.6)),
            auto_accept_threshold=float(quality_cfg.get("g2_auto_accept_threshold", 0.65)),
            review_threshold=float(quality_cfg.get("g2_review_threshold", 0.35)),
            min_visible_keypoints=int(quality_cfg.get("min_visible_keypoints", 8)),
        )
        self.feature_extractor = FeatureExtractor()

        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_tracks: dict[str, TrackState] = {}
        self._track_history: dict[str, list[dict[str, Any]]] = {}
        self._events: deque[dict[str, Any]] = deque(maxlen=500)
        self._frames_processed = 0
        self._started_at = 0.0

    def start(self) -> None:
        """Start background realtime loop."""
        if self._running:
            return
        self._running = True
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._run_loop, name="realtime-engine", daemon=True)
        self._thread.start()
        LOGGER.info("Realtime engine started.")

    def stop(self) -> None:
        """Stop background loop."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        LOGGER.info("Realtime engine stopped.")

    def snapshot(self) -> dict[str, Any]:
        """Current in-memory system snapshot."""
        with self._lock:
            tracks = [asdict(item) for item in self._latest_tracks.values()]
            return {
                "running": self._running,
                "frames_processed": self._frames_processed,
                "uptime_s": max(0.0, time.time() - self._started_at) if self._started_at else 0.0,
                "track_count": len(tracks),
                "tracks": tracks,
            }

    def recent_events(self, limit: int = 100) -> list[dict[str, Any]]:
        """Recent frame/track events."""
        with self._lock:
            return list(self._events)[-max(1, limit) :]

    def robot_commands(self) -> list[dict[str, Any]]:
        """Simple robot action suggestions from risk + action."""
        commands: list[dict[str, Any]] = []
        with self._lock:
            for track in self._latest_tracks.values():
                command = "monitor"
                if track.risk_score >= 0.85:
                    command = "emergency_stop_zone"
                elif track.action_class in {"fall", "danger_posture"} and track.risk_score >= 0.65:
                    command = "slowdown_and_alert"
                elif track.action_class in {"reach_overhead", "bend"} and track.risk_score >= 0.55:
                    command = "warn_operator"

                commands.append(
                    {
                        "track_id": track.track_id,
                        "command": command,
                        "risk_score": track.risk_score,
                        "action_class": track.action_class,
                        "timestamp_ms": track.timestamp_ms,
                    }
                )
        return commands

    def _run_loop(self) -> None:
        """Main realtime processing loop."""
        try:
            for record in self.ingestor.iter_frames():
                if not self._running:
                    break

                frame = record.frame
                metadata = record.metadata
                self._frames_processed += 1

                detections = self.extractor.extract(frame=frame, metadata=metadata)
                for detection in detections:
                    detection["frame_width"] = int(frame.shape[1])
                    detection["frame_height"] = int(frame.shape[0])

                    gate = self.quality_gate.evaluate(
                        keypoint_payload=detection,
                        frame_width=int(frame.shape[1]),
                        frame_height=int(frame.shape[0]),
                    )
                    detection["quality_gate"] = gate.status.value

                    track_id = str(detection.get("track_id", "worker_000"))
                    history = self._track_history.setdefault(track_id, [])
                    history.append(detection)
                    if len(history) > 30:
                        history.pop(0)

                    feature_result = self.feature_extractor.extract(history, imu_features=None)
                    vector = feature_result["vector"].astype(float).tolist()
                    action_class, risk_score = self._heuristic_action_risk(vector)

                    state = TrackState(
                        track_id=track_id,
                        frame_idx=int(detection.get("frame_idx", -1)),
                        timestamp_ms=int(detection.get("timestamp_ms", 0)),
                        quality_gate=gate.status.value,
                        action_class=action_class,
                        risk_score=risk_score,
                        mean_confidence=float(detection.get("mean_confidence", 0.0)),
                        bbox=[float(v) for v in detection.get("bbox", [0.0, 0.0, 0.0, 0.0])],
                        feature_vector=[float(v) for v in vector],
                    )

                    event = {
                        "track_id": state.track_id,
                        "frame_idx": state.frame_idx,
                        "timestamp_ms": state.timestamp_ms,
                        "action_class": state.action_class,
                        "risk_score": state.risk_score,
                        "quality_gate": state.quality_gate,
                    }

                    with self._lock:
                        self._latest_tracks[track_id] = state
                        self._events.append(event)
        except Exception:
            LOGGER.exception("Realtime engine loop failed.")
            self._running = False

    def _heuristic_action_risk(self, feature_vector: list[float]) -> tuple[str, float]:
        """Fallback action/risk until trained models are plugged in."""
        torso_angle = abs(feature_vector[0])
        velocity = max(0.0, feature_vector[13])
        accel = max(0.0, feature_vector[14])
        ppe = min(1.0, max(0.0, feature_vector[9]))

        if torso_angle > 45 and accel > 80:
            action = "fall"
        elif torso_angle > 35:
            action = "bend"
        elif velocity > 55:
            action = "walk"
        elif torso_angle > 25 and velocity < 10:
            action = "reach_overhead"
        else:
            action = "idle"

        risk_raw = 0.35 * (torso_angle / 60.0) + 0.35 * min(1.0, accel / 120.0) + 0.2 * min(1.0, velocity / 120.0) + 0.1 * (1.0 - ppe)
        risk = float(min(1.0, max(0.0, risk_raw)))
        return action, risk


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config when available."""
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def create_realtime_app(
    source: str,
    session_id: str,
    config_path: str | Path = "configs/pipeline.yaml",
) -> FastAPI:
    """Create FastAPI app and bind realtime engine lifecycle."""
    app = FastAPI(title="Tentellect Realtime Service", version="0.1.0")
    engine = RealtimeEngine(source=source, session_id=session_id, config=load_config(config_path))

    @app.on_event("startup")
    async def _on_startup() -> None:
        engine.start()

    @app.on_event("shutdown")
    async def _on_shutdown() -> None:
        engine.stop()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "running": engine.snapshot()["running"]}

    @app.get("/state")
    async def state() -> dict[str, Any]:
        return engine.snapshot()

    @app.get("/events")
    async def events(limit: int = 100) -> dict[str, Any]:
        return {"events": engine.recent_events(limit=limit)}

    @app.get("/robot/commands")
    async def robot_commands() -> dict[str, Any]:
        return {"commands": engine.robot_commands()}

    return app

