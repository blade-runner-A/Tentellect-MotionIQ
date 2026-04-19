"""Skeleton extraction module with YOLO/MediaPipe selection logic."""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - dependency may be unavailable in local tests
    YOLO = None  # type: ignore[assignment]

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - dependency may be unavailable in local tests
    mp = None  # type: ignore[assignment]

_DEFAULT_POSE_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

try:
    from ultralytics.trackers.byte_tracker import BYTETracker
except Exception:  # pragma: no cover - import path may change across ultralytics releases
    BYTETracker = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

MEDIAPIPE_33_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

COCO_TO_MEDIAPIPE_INDEX = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


@dataclass
class SkeletonCandidate:
    """Intermediate candidate representation from pose models."""

    bbox: list[float]
    detection_confidence: float
    keypoints_17: list[dict[str, float | str]]
    keypoints_33: list[dict[str, float | str]]
    world_3d: list[dict[str, float | str]]
    mean_confidence: float


class SkeletonExtractor:
    """Extract worker skeletons using model-selection logic from PRD."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize models and runtime configuration.

        Args:
            config: Runtime configuration dictionary.
        """
        self.config = config
        self.device = str(config.get("device", "cpu"))
        self.yolo_conf_threshold = float(config.get("yolo_conf_threshold", config.get("confidence_threshold", 0.25)))
        self.low_conf_threshold = float(config.get("ensemble_conf_threshold", 0.7))

        self._next_track_id = 1
        self._fallback_track_cache: dict[tuple[int, int], str] = {}

        self.yolo_model = self._init_yolo_model(model_path=config.get("yolo_model"))
        self.mediapipe_pose = None
        self._mediapipe_tasks_landmarker = None
        self._init_mediapipe_backends(
            model_path=config.get("mediapipe_tasks_model_path"),
            model_url=config.get("mediapipe_tasks_model_url", _DEFAULT_POSE_TASK_URL),
            model_complexity=int(config.get("mediapipe_model_complexity", 1)),
        )
        self.bytetrack_params = {"track_thresh": 0.5, "track_buffer": 30, "match_thresh": 0.8}
        self.byte_tracker = self._init_bytetrack()

    def extract(self, frame: np.ndarray, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract skeleton predictions for a single frame.

        Args:
            frame: Frame image as BGR uint8 array.
            metadata: Ingestion metadata dictionary.

        Returns:
            List of skeleton output dictionaries matching pipeline schema.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected BGR frame with shape (H, W, 3).")

        yolo_candidates = self._run_yolo_detection(frame)
        mediapipe_candidates = self._run_mediapipe_detection(frame)

        if len(yolo_candidates) <= 1 and mediapipe_candidates:
            selected = mediapipe_candidates
        else:
            selected = yolo_candidates

        selected_mean = float(np.mean([candidate.mean_confidence for candidate in selected])) if selected else 0.0
        if (
            selected
            and selected_mean < self.low_conf_threshold
            and yolo_candidates
            and mediapipe_candidates
        ):
            selected = self._ensemble_detections(yolo_candidates, mediapipe_candidates)

        track_ids = self._assign_track_ids(selected, frame.shape)

        frame_idx = int(metadata.get("frame_idx", -1))
        timestamp_ms = int(metadata.get("timestamp_ms", -1))
        session_id = str(metadata.get("session_id", "unknown_session"))

        outputs: list[dict[str, Any]] = []
        for idx, candidate in enumerate(selected):
            track_id = track_ids[idx] if idx < len(track_ids) else f"worker_{self._next_track_id:03d}"

            outputs.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp_ms": timestamp_ms,
                    "session_id": session_id,
                    "track_id": track_id,
                    "bbox": [float(v) for v in candidate.bbox],
                    "detection_confidence": float(candidate.detection_confidence),
                    "keypoints_17": candidate.keypoints_17,
                    "keypoints_33": candidate.keypoints_33,
                    "world_3d": candidate.world_3d,
                    "mean_confidence": float(candidate.mean_confidence),
                    "quality_gate": "UNSET",
                    "ppe": {
                        "helmet": 0.0,
                        "vest": 0.0,
                        "gloves": 0.0,
                        "glasses": 0.0,
                    },
                }
            )

        return outputs

    def _init_yolo_model(self, model_path: str | None) -> Any | None:
        """Load YOLO model once for reuse across all frames."""
        if YOLO is None:
            LOGGER.warning("Ultralytics is unavailable; YOLO extraction disabled.")
            return None

        resolved_model = model_path or "yolov8s-pose.pt"

        try:
            if not Path(resolved_model).exists() and resolved_model != "yolov8s-pose.pt":
                LOGGER.warning("YOLO model path does not exist: %s. Falling back to default weights.", resolved_model)
                resolved_model = "yolov8s-pose.pt"
            return YOLO(resolved_model)
        except Exception:
            LOGGER.exception("Failed to initialize YOLO model from %s", resolved_model)
            return None

    def _repo_root(self) -> Path:
        """Project root (parent of ``src/``)."""
        return Path(__file__).resolve().parents[2]

    def _ensure_pose_task_file(self, model_path: Path | None, model_url: str) -> Path | None:
        """Return path to ``.task`` bundle, downloading once if needed."""
        if model_path is not None:
            resolved = Path(model_path)
            if resolved.is_file():
                return resolved
            LOGGER.warning("MediaPipe task model path missing: %s", resolved)
            return None

        default_path = self._repo_root() / "models" / "checkpoints" / "pose_landmarker_lite.task"
        if default_path.is_file():
            return default_path

        try:
            default_path.parent.mkdir(parents=True, exist_ok=True)
            LOGGER.info("Downloading MediaPipe pose landmarker model to %s", default_path)
            urllib.request.urlretrieve(model_url, default_path)  # noqa: S310 — fixed Google CDN URL
        except Exception:
            LOGGER.exception("Failed to download MediaPipe pose task model.")
            return None

        return default_path if default_path.is_file() else None

    def _init_mediapipe_backends(
        self,
        model_path: str | None,
        model_url: str,
        model_complexity: int,
    ) -> None:
        """Initialize legacy ``solutions.pose`` or MediaPipe Tasks PoseLandmarker."""
        if mp is None:
            LOGGER.warning("MediaPipe is unavailable; MediaPipe extraction disabled.")
            return

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            try:
                self.mediapipe_pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=model_complexity,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                LOGGER.info("MediaPipe legacy Pose (solutions) initialized.")
                return
            except Exception:
                LOGGER.exception("Failed to initialize MediaPipe legacy Pose model.")

        try:
            from mediapipe.tasks.python.core import base_options as mp_base_options
            from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
        except Exception:
            LOGGER.exception("MediaPipe Tasks API unavailable; MediaPipe pose disabled.")
            return

        task_file = self._ensure_pose_task_file(
            Path(model_path) if model_path else None,
            model_url,
        )
        if task_file is None:
            LOGGER.warning(
                "MediaPipe Tasks pose model file missing. Install a .task file or use Python 3.11 "
                "with mediapipe<0.10.14 for the legacy solutions API."
            )
            return

        try:
            opts = PoseLandmarkerOptions(
                base_options=mp_base_options.BaseOptions(model_asset_path=str(task_file)),
                running_mode=RunningMode.IMAGE,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
            )
            self._mediapipe_tasks_landmarker = PoseLandmarker.create_from_options(opts)
            LOGGER.info("MediaPipe Tasks PoseLandmarker initialized.")
        except Exception:
            LOGGER.exception("Failed to initialize MediaPipe Tasks PoseLandmarker.")

    def _mediapipe_tasks_detect(self, rgb_frame: np.ndarray) -> Any | None:
        """Run Tasks API pose detection; returns legacy-like object or None."""
        if self._mediapipe_tasks_landmarker is None or mp is None:
            return None

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            return self._mediapipe_tasks_landmarker.detect(mp_image)
        except Exception:
            LOGGER.exception("MediaPipe Tasks inference failed.")
            return None

    @staticmethod
    def _tasks_result_to_legacy_like(tasks_result: Any) -> Any | None:
        """Adapt PoseLandmarkerResult to an object with ``pose_landmarks`` / ``pose_world_landmarks``."""

        class _LandmarkList:
            """Mimic ``pose_landmarks.landmark`` from the legacy solutions API."""

            def __init__(self, items: list[Any]) -> None:
                self.landmark = items

        if not tasks_result.pose_landmarks:
            return None

        class _LegacyLike:
            def __init__(self) -> None:
                self.pose_landmarks = _LandmarkList(list(tasks_result.pose_landmarks[0]))
                if tasks_result.pose_world_landmarks:
                    self.pose_world_landmarks = _LandmarkList(list(tasks_result.pose_world_landmarks[0]))
                else:
                    self.pose_world_landmarks = None

        return _LegacyLike()

    def _init_bytetrack(self) -> Any | None:
        """Initialize ByteTrack using PRD-required thresholds."""
        if BYTETracker is None:
            LOGGER.warning("BYTETracker is unavailable; fallback track IDs will be used.")
            return None

        try:
            return BYTETracker(
                track_thresh=self.bytetrack_params["track_thresh"],
                track_buffer=self.bytetrack_params["track_buffer"],
                match_thresh=self.bytetrack_params["match_thresh"],
            )
        except TypeError:
            LOGGER.warning("BYTETracker signature mismatch; fallback track IDs will be used.")
            return None
        except Exception:
            LOGGER.exception("Failed to initialize BYTETracker; fallback track IDs will be used.")
            return None

    def _run_yolo_detection(self, frame: np.ndarray) -> list[SkeletonCandidate]:
        """Run YOLO pose model and parse detections into candidates."""
        if self.yolo_model is None:
            return []

        try:
            if hasattr(self.yolo_model, "predict"):
                results = self.yolo_model.predict(frame, conf=self.yolo_conf_threshold, device=self.device, verbose=False)
            else:
                results = self.yolo_model(frame)
        except Exception:
            LOGGER.exception("YOLO inference failed.")
            return []

        candidates: list[SkeletonCandidate] = []
        for result in results:
            boxes_obj = getattr(result, "boxes", None)
            keypoints_obj = getattr(result, "keypoints", None)
            if boxes_obj is None:
                continue

            boxes_xyxy = self._to_numpy(getattr(boxes_obj, "xyxy", []))
            boxes_conf = self._to_numpy(getattr(boxes_obj, "conf", []), default=np.zeros((len(boxes_xyxy),), dtype=np.float32))

            kp_xy = None
            kp_conf = None
            if keypoints_obj is not None:
                kp_xy = self._to_numpy(getattr(keypoints_obj, "xy", None))
                kp_conf = self._to_numpy(getattr(keypoints_obj, "conf", None))

            for idx, bbox in enumerate(boxes_xyxy):
                confidence = float(boxes_conf[idx]) if idx < len(boxes_conf) else 0.0

                points_xy = kp_xy[idx] if kp_xy is not None and idx < len(kp_xy) else np.empty((0, 2), dtype=np.float32)
                points_conf = (
                    kp_conf[idx] if kp_conf is not None and idx < len(kp_conf) else np.full((len(points_xy),), confidence)
                )

                keypoints_17 = self._build_coco_keypoints(points_xy, points_conf, fallback_conf=confidence)
                mean_conf = float(np.mean([point["conf"] for point in keypoints_17])) if keypoints_17 else confidence

                candidates.append(
                    SkeletonCandidate(
                        bbox=[float(v) for v in bbox.tolist()],
                        detection_confidence=confidence,
                        keypoints_17=keypoints_17,
                        keypoints_33=[],
                        world_3d=[],
                        mean_confidence=mean_conf,
                    )
                )

        return candidates

    def _run_mediapipe_detection(self, frame: np.ndarray) -> list[SkeletonCandidate]:
        """Run MediaPipe pose extraction for single-person detailed landmarks."""
        if self.mediapipe_pose is None and self._mediapipe_tasks_landmarker is None:
            return []

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.mediapipe_pose is not None:
                result = self.mediapipe_pose.process(rgb_frame)
            else:
                tasks_result = self._mediapipe_tasks_detect(rgb_frame)
                result = self._tasks_result_to_legacy_like(tasks_result) if tasks_result is not None else None
        except Exception:
            LOGGER.exception("MediaPipe inference failed.")
            return []

        if result is None or result.pose_landmarks is None:
            return []

        height, width = frame.shape[:2]
        landmarks = result.pose_landmarks.landmark
        world_landmarks = result.pose_world_landmarks.landmark if result.pose_world_landmarks else []

        keypoints_33: list[dict[str, float | str]] = []
        world_3d: list[dict[str, float | str]] = []

        for idx, landmark in enumerate(landmarks):
            name = MEDIAPIPE_33_NAMES[idx] if idx < len(MEDIAPIPE_33_NAMES) else f"mp_{idx}"
            x_px = float(np.clip(landmark.x * width, 0, width - 1))
            y_px = float(np.clip(landmark.y * height, 0, height - 1))
            confidence = float(getattr(landmark, "visibility", 0.0))
            keypoints_33.append({"name": name, "x": x_px, "y": y_px, "conf": confidence})

            if idx < len(world_landmarks):
                world = world_landmarks[idx]
                world_3d.append(
                    {
                        "name": name,
                        "x": float(world.x),
                        "y": float(world.y),
                        "z": float(world.z),
                    }
                )

        keypoints_17 = self._mediapipe_to_coco(keypoints_33)
        if not keypoints_17:
            return []

        xs = [point["x"] for point in keypoints_17]
        ys = [point["y"] for point in keypoints_17]
        bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
        mean_confidence = float(np.mean([point["conf"] for point in keypoints_17]))

        return [
            SkeletonCandidate(
                bbox=bbox,
                detection_confidence=mean_confidence,
                keypoints_17=keypoints_17,
                keypoints_33=keypoints_33,
                world_3d=world_3d,
                mean_confidence=mean_confidence,
            )
        ]

    def _ensemble_detections(
        self,
        yolo_candidates: list[SkeletonCandidate],
        mediapipe_candidates: list[SkeletonCandidate],
    ) -> list[SkeletonCandidate]:
        """Merge low-confidence detections by choosing stronger candidate per index."""
        if not yolo_candidates:
            return mediapipe_candidates
        if not mediapipe_candidates:
            return yolo_candidates

        merged: list[SkeletonCandidate] = []
        shared = min(len(yolo_candidates), len(mediapipe_candidates))

        for idx in range(shared):
            yolo_item = yolo_candidates[idx]
            mp_item = mediapipe_candidates[idx]

            if abs(yolo_item.mean_confidence - mp_item.mean_confidence) > 0.1:
                merged.append(mp_item if mp_item.mean_confidence >= yolo_item.mean_confidence else yolo_item)
                continue

            merged.append(self._average_candidate(yolo_item, mp_item))

        if len(yolo_candidates) > shared:
            merged.extend(yolo_candidates[shared:])
        if len(mediapipe_candidates) > shared:
            merged.extend(mediapipe_candidates[shared:])

        return merged

    def _average_candidate(self, first: SkeletonCandidate, second: SkeletonCandidate) -> SkeletonCandidate:
        """Average compatible numeric fields across two candidates."""
        averaged_bbox = [float((a + b) / 2.0) for a, b in zip(first.bbox, second.bbox)]

        keypoints_17 = first.keypoints_17 if first.keypoints_17 else second.keypoints_17
        if first.keypoints_17 and second.keypoints_17 and len(first.keypoints_17) == len(second.keypoints_17):
            keypoints_17 = []
            for p1, p2 in zip(first.keypoints_17, second.keypoints_17):
                keypoints_17.append(
                    {
                        "name": str(p1["name"]),
                        "x": float((float(p1["x"]) + float(p2["x"])) / 2.0),
                        "y": float((float(p1["y"]) + float(p2["y"])) / 2.0),
                        "conf": float((float(p1["conf"]) + float(p2["conf"])) / 2.0),
                    }
                )

        return SkeletonCandidate(
            bbox=averaged_bbox,
            detection_confidence=float((first.detection_confidence + second.detection_confidence) / 2.0),
            keypoints_17=keypoints_17,
            keypoints_33=second.keypoints_33 if second.keypoints_33 else first.keypoints_33,
            world_3d=second.world_3d if second.world_3d else first.world_3d,
            mean_confidence=float((first.mean_confidence + second.mean_confidence) / 2.0),
        )

    def _assign_track_ids(self, candidates: list[SkeletonCandidate], frame_shape: tuple[int, ...]) -> list[str]:
        """Assign track IDs, preferring ByteTrack and falling back to spatial caching."""
        if not candidates:
            return []

        if self.byte_tracker is not None:
            try:
                assigned = self._track_with_bytetrack(candidates, frame_shape)
                if len(assigned) == len(candidates):
                    return assigned
            except Exception:
                LOGGER.exception("ByteTrack update failed; using fallback track IDs.")

        return self._track_with_spatial_fallback(candidates)

    def _track_with_bytetrack(self, candidates: list[SkeletonCandidate], frame_shape: tuple[int, ...]) -> list[str]:
        """Update ByteTrack state and map tracks to output IDs."""
        if self.byte_tracker is None:
            return []

        detections = np.array(
            [[candidate.bbox[0], candidate.bbox[1], candidate.bbox[2], candidate.bbox[3], candidate.detection_confidence] for candidate in candidates],
            dtype=np.float32,
        )

        height, width = int(frame_shape[0]), int(frame_shape[1])
        tracks = self.byte_tracker.update(detections, (height, width), (height, width))

        track_ids: list[str] = []
        for idx, _ in enumerate(candidates):
            if idx < len(tracks) and hasattr(tracks[idx], "track_id"):
                track_ids.append(f"worker_{int(tracks[idx].track_id):03d}")
            else:
                track_ids.append("")

        if any(not track_id for track_id in track_ids):
            return self._track_with_spatial_fallback(candidates)

        return track_ids

    def _track_with_spatial_fallback(self, candidates: list[SkeletonCandidate]) -> list[str]:
        """Assign stable IDs by coarse spatial hashing when tracker is unavailable."""
        assigned: list[str] = []

        for candidate in candidates:
            key = self._spatial_key(candidate.bbox)
            existing = self._fallback_track_cache.get(key)
            if existing:
                assigned.append(existing)
                continue

            track_id = f"worker_{self._next_track_id:03d}"
            self._next_track_id += 1
            self._fallback_track_cache[key] = track_id
            assigned.append(track_id)

        return assigned

    def _spatial_key(self, bbox: list[float]) -> tuple[int, int]:
        """Create a coarse spatial key from bbox center coordinates."""
        cx = int((bbox[0] + bbox[2]) * 0.5)
        cy = int((bbox[1] + bbox[3]) * 0.5)
        return cx // 32, cy // 32

    def _build_coco_keypoints(
        self,
        points_xy: np.ndarray,
        points_conf: np.ndarray,
        fallback_conf: float,
    ) -> list[dict[str, float | str]]:
        """Build COCO keypoint list from model xy/conf arrays."""
        if points_xy.size == 0:
            return []

        keypoints: list[dict[str, float | str]] = []
        for idx, name in enumerate(COCO_KEYPOINT_NAMES):
            if idx >= len(points_xy):
                break

            x_val = float(points_xy[idx][0])
            y_val = float(points_xy[idx][1])
            conf_val = float(points_conf[idx]) if idx < len(points_conf) else float(fallback_conf)
            keypoints.append({"name": name, "x": x_val, "y": y_val, "conf": conf_val})

        return keypoints

    def _mediapipe_to_coco(self, keypoints_33: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
        """Map MediaPipe 33 landmarks to COCO 17 keypoints."""
        mapped: list[dict[str, float | str]] = []

        for name in COCO_KEYPOINT_NAMES:
            mp_index = COCO_TO_MEDIAPIPE_INDEX[name]
            if mp_index >= len(keypoints_33):
                continue

            point = keypoints_33[mp_index]
            mapped.append(
                {
                    "name": name,
                    "x": float(point["x"]),
                    "y": float(point["y"]),
                    "conf": float(point["conf"]),
                }
            )

        return mapped

    def _to_numpy(self, value: Any, default: np.ndarray | None = None) -> np.ndarray:
        """Convert model output to numpy array safely."""
        if value is None:
            return default if default is not None else np.array([])

        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            try:
                return np.asarray(value.numpy())
            except Exception:
                pass

        try:
            return np.asarray(value)
        except Exception:
            return default if default is not None else np.array([])
