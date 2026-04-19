"""End-to-end Tentellect pipeline entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from src.annotation.annotator import AnnotationRouter, LabelStudioPusher
from src.annotation.quality_gates import GateStatus, QualityGate
from src.annotation.storage import DatasetWriter
from src.features.extractor import FeatureExtractor
from src.ingestion.ingestor import DataIngestor
from src.skeleton.extractor import SkeletonExtractor

LOGGER = logging.getLogger("run_pipeline")


@dataclass
class PipelineSummary:
    """Runtime summary for one pipeline session."""

    session_id: str
    source: str
    mode: str
    frames_processed: int
    skeletons_detected: int
    auto_accept_count: int
    review_count: int
    discard_count: int
    duration_s: float
    skeleton_output: str
    feature_output: str
    feature_meta_output: str
    summary_output: str
    coco_output: str


class PipelineRunner:
    """Coordinate ingestion, extraction, gating, routing, and feature export."""

    def __init__(
        self,
        source: str,
        mode: str,
        session_id: str,
        output_root: Path,
        ingestor: DataIngestor,
        extractor: SkeletonExtractor,
        quality_gate: QualityGate,
        router: AnnotationRouter,
        feature_extractor: FeatureExtractor,
        dataset_writer: DatasetWriter,
    ) -> None:
        """Initialize pipeline runner with all required components."""
        self.source = source
        self.mode = mode
        self.session_id = session_id
        self.output_root = output_root
        self.ingestor = ingestor
        self.extractor = extractor
        self.quality_gate = quality_gate
        self.router = router
        self.feature_extractor = feature_extractor
        self.dataset_writer = dataset_writer

    def run(self, max_frames: int = 0, validate_coco: bool = False) -> PipelineSummary:
        """Run full pipeline on configured source."""
        start_time = time.time()

        skeleton_records: list[dict[str, Any]] = []
        feature_vectors: list[np.ndarray] = []
        feature_meta: list[dict[str, Any]] = []
        track_history: dict[str, list[dict[str, Any]]] = {}

        frames_processed = 0
        auto_accept_count = 0
        review_count = 0
        discard_count = 0

        for frame_record in self.ingestor.iter_frames():
            frame = frame_record.frame
            metadata = frame_record.metadata
            frames_processed += 1

            detections = self.extractor.extract(frame=frame, metadata=metadata)

            for detection in detections:
                detection["frame_width"] = int(frame.shape[1])
                detection["frame_height"] = int(frame.shape[0])

                gate_result = self.quality_gate.evaluate(
                    keypoint_payload=detection,
                    frame_width=int(frame.shape[1]),
                    frame_height=int(frame.shape[0]),
                )

                status = gate_result.status
                detection["quality_gate"] = status.value
                detection["quality_reason"] = gate_result.reason

                if status == GateStatus.AUTO_ACCEPT:
                    auto_accept_count += 1
                elif status == GateStatus.REVIEW:
                    review_count += 1
                else:
                    discard_count += 1

                route_result = self.router.route(
                    skeleton=detection,
                    gate_result=status,
                    frame_image=frame if status == GateStatus.REVIEW else None,
                )

                track_id = str(detection.get("track_id", "worker_000"))
                history = track_history.setdefault(track_id, [])
                history.append(detection)
                if len(history) > 30:
                    history.pop(0)

                feature_result = self.feature_extractor.extract(history, imu_features=None)
                feature_vector = np.asarray(feature_result["vector"], dtype=np.float32)

                feature_vectors.append(feature_vector)
                feature_meta.append(
                    {
                        "frame_idx": int(detection.get("frame_idx", -1)),
                        "timestamp_ms": int(detection.get("timestamp_ms", 0)),
                        "track_id": track_id,
                        "quality_gate": status.value,
                        "route": str(route_result.get("route", "")),
                        "imu_available": bool(feature_result["imu_available"]),
                    }
                )

                skeleton_records.append(detection)

            if max_frames > 0 and frames_processed >= max_frames:
                break

        output_paths = self._write_outputs(
            skeleton_records=skeleton_records,
            feature_vectors=feature_vectors,
            feature_meta=feature_meta,
            validate_coco=validate_coco,
        )

        duration_s = time.time() - start_time

        summary = PipelineSummary(
            session_id=self.session_id,
            source=self.source,
            mode=self.mode,
            frames_processed=frames_processed,
            skeletons_detected=len(skeleton_records),
            auto_accept_count=auto_accept_count,
            review_count=review_count,
            discard_count=discard_count,
            duration_s=duration_s,
            skeleton_output=str(output_paths["skeleton_output"]),
            feature_output=str(output_paths["feature_output"]),
            feature_meta_output=str(output_paths["feature_meta_output"]),
            summary_output=str(output_paths["summary_output"]),
            coco_output=str(output_paths["coco_output"]),
        )

        output_paths["summary_output"].write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
        return summary

    def _write_outputs(
        self,
        skeleton_records: list[dict[str, Any]],
        feature_vectors: list[np.ndarray],
        feature_meta: list[dict[str, Any]],
        validate_coco: bool,
    ) -> dict[str, Path]:
        """Write processed skeleton/features and COCO export artifacts."""
        skeleton_dir = self.output_root / "skeletons"
        feature_dir = self.output_root / "features"
        skeleton_dir.mkdir(parents=True, exist_ok=True)
        feature_dir.mkdir(parents=True, exist_ok=True)

        skeleton_output = skeleton_dir / f"{self.session_id}.json"
        skeleton_output.write_text(
            json.dumps(
                {
                    "session_id": self.session_id,
                    "source": self.source,
                    "mode": self.mode,
                    "records": skeleton_records,
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )

        feature_output = feature_dir / f"{self.session_id}.npy"
        if feature_vectors:
            matrix = np.vstack(feature_vectors).astype(np.float32)
        else:
            matrix = np.zeros((0, 20), dtype=np.float32)
        np.save(feature_output, matrix)

        feature_meta_output = feature_dir / f"{self.session_id}_meta.json"
        feature_meta_output.write_text(json.dumps(feature_meta, indent=2), encoding="utf-8")

        coco_output = self.output_root / "coco_export.json"
        self.dataset_writer.export_coco(output_path=coco_output, validate_with_pycocotools=validate_coco)

        summary_output = self.output_root / f"{self.session_id}_summary.json"

        return {
            "skeleton_output": skeleton_output,
            "feature_output": feature_output,
            "feature_meta_output": feature_meta_output,
            "summary_output": summary_output,
            "coco_output": coco_output,
        }


def infer_mode(source: str, explicit_mode: str) -> str:
    """Infer ingestion mode from source when mode=auto."""
    if explicit_mode != "auto":
        return explicit_mode

    source_path = Path(source)
    if source_path.exists() and source_path.is_dir():
        return "batch"
    if source_path.exists() and source_path.is_file() and source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
        return "batch"
    return "dataset"


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config file."""
    path = Path(config_path)
    if not path.exists():
        return {}

    try:
        import yaml
    except ImportError:
        LOGGER.warning("PyYAML is not installed; proceeding with default config values.")
        return {}

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Tentellect end-to-end pipeline.")
    parser.add_argument("--input", required=True, help="Input video path, image folder, or stream URL")
    parser.add_argument("--mode", choices=["auto", "dataset", "realtime", "batch"], default="auto")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--output-root", default="data/processed")
    parser.add_argument("--db-path", default="data/processed/annotations.db")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--queue-reviews", action="store_true")
    parser.add_argument("--labelstudio-url", default="http://localhost:8080")
    parser.add_argument("--labelstudio-key", default="")
    parser.add_argument("--labelstudio-project-id", type=int, default=0)
    parser.add_argument("--validate-coco", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_pipeline_components(args: argparse.Namespace) -> PipelineRunner:
    """Build configured pipeline components and return runner."""
    config = load_config(args.config)

    mode = infer_mode(args.input, args.mode)
    session_id = args.session_id or f"session_{int(time.time())}"

    ingestion_cfg = config.get("ingestion", {}) if isinstance(config.get("ingestion"), dict) else {}
    skeleton_cfg = config.get("skeleton", {}) if isinstance(config.get("skeleton"), dict) else {}
    quality_cfg = config.get("quality_gates", {}) if isinstance(config.get("quality_gates"), dict) else {}
    runtime_cfg = config.get("runtime", {}) if isinstance(config.get("runtime"), dict) else {}

    ingestor = DataIngestor(
        source=args.input,
        mode=mode,
        session_id=session_id,
        target_resolution=(
            int(ingestion_cfg.get("target_width", 640)),
            int(ingestion_cfg.get("target_height", 480)),
        ),
        dataset_fps=int(ingestion_cfg.get("dataset_fps", 2)),
        realtime_fps=int(ingestion_cfg.get("realtime_fps", 15)),
    )

    skeleton_extractor = SkeletonExtractor(
        {
            "device": str(runtime_cfg.get("device", "cpu")),
            "yolo_model": skeleton_cfg.get("yolo_model", "yolov8s-pose.pt"),
            "yolo_conf_threshold": float(skeleton_cfg.get("yolo_conf_threshold", 0.6)),
            "ensemble_conf_threshold": float(skeleton_cfg.get("ensemble_conf_threshold", 0.7)),
            "mediapipe_model_complexity": int(skeleton_cfg.get("mediapipe_model_complexity", 1)),
            "mediapipe_tasks_model_path": skeleton_cfg.get("mediapipe_tasks_model_path"),
            "mediapipe_tasks_model_url": skeleton_cfg.get(
                "mediapipe_tasks_model_url",
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            ),
        }
    )

    quality_gate = QualityGate(
        detection_threshold=float(quality_cfg.get("g1_detection_threshold", 0.60)),
        auto_accept_threshold=float(quality_cfg.get("g2_auto_accept_threshold", 0.85)),
        review_threshold=float(quality_cfg.get("g2_review_threshold", 0.50)),
        min_visible_keypoints=int(quality_cfg.get("min_visible_keypoints", 8)),
    )

    dataset_writer = DatasetWriter(db_path=args.db_path)

    label_pusher = None
    if args.queue_reviews:
        label_pusher = LabelStudioPusher(
            api_url=args.labelstudio_url,
            api_key=args.labelstudio_key or None,
            project_id=args.labelstudio_project_id if args.labelstudio_project_id > 0 else None,
        )

    router = AnnotationRouter(dataset_writer=dataset_writer, label_studio_pusher=label_pusher)
    feature_extractor = FeatureExtractor()

    return PipelineRunner(
        source=args.input,
        mode=mode,
        session_id=session_id,
        output_root=Path(args.output_root),
        ingestor=ingestor,
        extractor=skeleton_extractor,
        quality_gate=quality_gate,
        router=router,
        feature_extractor=feature_extractor,
        dataset_writer=dataset_writer,
    )


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    setup_logging(args.log_level)

    try:
        runner = build_pipeline_components(args)
        summary = runner.run(max_frames=args.max_frames, validate_coco=args.validate_coco)
        LOGGER.info("Pipeline completed: %s", json.dumps(asdict(summary), indent=2))
        return 0

    except Exception:
        LOGGER.exception("Pipeline execution failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
