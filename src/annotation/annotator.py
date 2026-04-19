"""Annotation routing and Label Studio integration helpers."""

from __future__ import annotations

import base64
import logging
from typing import Any

import cv2
import numpy as np

from src.annotation.quality_gates import GateResult, GateStatus
from src.annotation.storage import COCO_KEYPOINT_NAMES, DatasetWriter

LOGGER = logging.getLogger(__name__)


class LabelStudioPusher:
    """Push low-confidence samples to Label Studio review queue."""

    def __init__(
        self,
        api_url: str = "http://localhost:8080",
        api_key: str | None = None,
        project_id: int | None = None,
        client: Any | None = None,
        project_title: str = "Tentellect Review Queue",
    ) -> None:
        """Initialize Label Studio client and optional project metadata."""
        self.api_url = api_url
        self.api_key = api_key
        self.project_id = project_id
        self.project_title = project_title
        self.from_name = "keypoint"
        self.to_name = "image"

        if client is not None:
            self.client = client
        else:
            self.client = self._build_client(api_url=api_url, api_key=api_key)

    def push_review_task(
        self,
        frame_base64: str,
        skeleton: dict[str, Any],
        metadata: dict[str, Any],
    ) -> str:
        """Create a Label Studio review task with keypoint pre-annotations."""
        project = self._ensure_project()

        width, height = self._infer_dimensions(skeleton)
        prediction_result = self._build_prediction_result(skeleton=skeleton, width=width, height=height)

        payload = {
            "data": {
                "image": f"data:image/jpeg;base64,{frame_base64}",
                "meta": metadata,
            },
            "predictions": [
                {
                    "model_version": "tentellect_auto_v1",
                    "result": prediction_result,
                }
            ],
        }

        if hasattr(project, "import_tasks"):
            response = project.import_tasks([payload])
            task_id = self._extract_task_id(response)
            return task_id

        if hasattr(self.client, "create_task") and self.project_id is not None:
            response = self.client.create_task(self.project_id, payload)
            task_id = self._extract_task_id(response)
            return task_id

        raise RuntimeError("Unable to submit task to Label Studio: unsupported client/project interface.")

    def _build_client(self, api_url: str, api_key: str | None) -> Any:
        """Construct Label Studio SDK client lazily."""
        try:
            from label_studio_sdk import Client
        except ImportError as exc:  # pragma: no cover - depends on optional runtime package
            raise RuntimeError(
                "label-studio-sdk is required for review queue integration. Install it via pip."
            ) from exc

        if not api_key:
            raise ValueError("api_key is required when creating Label Studio SDK client.")

        return Client(url=api_url, api_key=api_key)

    def _ensure_project(self) -> Any:
        """Get or create review project with keypoint label config."""
        if self.project_id is not None:
            if hasattr(self.client, "get_project"):
                return self.client.get_project(self.project_id)
            return self.client

        label_config = self._build_label_config()

        if hasattr(self.client, "start_project"):
            project = self.client.start_project(title=self.project_title, label_config=label_config)
        elif hasattr(self.client, "create_project"):
            project = self.client.create_project(title=self.project_title, label_config=label_config)
        else:
            raise RuntimeError("Unable to create Label Studio project with provided client.")

        self.project_id = int(getattr(project, "id", 0)) if getattr(project, "id", None) is not None else None
        return project

    def _build_label_config(self) -> str:
        """Build Label Studio KeyPointLabels XML configuration."""
        labels = "\n".join(f'      <Label value="{name}" />' for name in COCO_KEYPOINT_NAMES)
        return (
            "<View>\n"
            "  <Image name=\"image\" value=\"$image\"/>\n"
            f"  <KeyPointLabels name=\"{self.from_name}\" toName=\"{self.to_name}\">\n"
            f"{labels}\n"
            "  </KeyPointLabels>\n"
            "</View>"
        )

    def _build_prediction_result(self, skeleton: dict[str, Any], width: int, height: int) -> list[dict[str, Any]]:
        """Convert skeleton keypoints into Label Studio prediction result format."""
        keypoints = skeleton.get("keypoints_17", [])
        if not isinstance(keypoints, list):
            return []

        result: list[dict[str, Any]] = []
        for idx, item in enumerate(keypoints):
            if not isinstance(item, dict):
                continue

            label = str(item.get("name", "")).strip()
            if label not in COCO_KEYPOINT_NAMES:
                continue

            x_val = float(item.get("x", 0.0))
            y_val = float(item.get("y", 0.0))
            conf = float(item.get("conf", 0.0))

            x_pct = float(np.clip((x_val / max(width, 1)) * 100.0, 0.0, 100.0))
            y_pct = float(np.clip((y_val / max(height, 1)) * 100.0, 0.0, 100.0))

            result.append(
                {
                    "id": f"auto_{idx}",
                    "type": "keypointlabels",
                    "from_name": self.from_name,
                    "to_name": self.to_name,
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "score": conf,
                    "value": {
                        "x": x_pct,
                        "y": y_pct,
                        "keypointlabels": [label],
                    },
                }
            )

        return result

    def _extract_task_id(self, response: Any) -> str:
        """Extract created task identifier from SDK response payload."""
        if isinstance(response, list) and response:
            first = response[0]
            if isinstance(first, dict):
                for key in ("id", "task", "task_id"):
                    if key in first:
                        return str(first[key])

        if isinstance(response, dict):
            for key in ("id", "task", "task_id"):
                if key in response:
                    return str(response[key])

        return ""

    def _infer_dimensions(self, skeleton: dict[str, Any]) -> tuple[int, int]:
        """Infer image dimensions from skeleton metadata or bbox fallback."""
        width = int(skeleton.get("frame_width", 0))
        height = int(skeleton.get("frame_height", 0))

        bbox = skeleton.get("bbox", [])
        if isinstance(bbox, list) and len(bbox) == 4:
            width = max(width, int(float(bbox[2])) + 1)
            height = max(height, int(float(bbox[3])) + 1)

        if width <= 0:
            width = 640
        if height <= 0:
            height = 480

        return width, height


class AnnotationRouter:
    """Route annotations to dataset storage or Label Studio review queue."""

    def __init__(
        self,
        dataset_writer: DatasetWriter,
        label_studio_pusher: LabelStudioPusher | None = None,
    ) -> None:
        """Initialize router dependencies."""
        self.dataset_writer = dataset_writer
        self.label_studio_pusher = label_studio_pusher

    def route(
        self,
        skeleton: dict[str, Any],
        gate_result: GateResult | GateStatus | str,
        frame_image: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Route one skeleton payload according to quality gate decision."""
        status = self._coerce_status(gate_result)

        if status == GateStatus.DISCARD:
            return {"route": "discard", "status": status.value}

        record_id = self.dataset_writer.write_annotation(
            skeleton=skeleton,
            quality_gate=status.value,
            human_verified=False,
        )

        if status == GateStatus.AUTO_ACCEPT:
            return {
                "route": "dataset",
                "status": status.value,
                "record_id": record_id,
            }

        if self.label_studio_pusher is None:
            LOGGER.warning("Review routing requested but Label Studio pusher is not configured.")
            return {
                "route": "review_unqueued",
                "status": status.value,
                "record_id": record_id,
            }

        if frame_image is None:
            raise ValueError("frame_image is required to queue REVIEW tasks.")

        frame_base64 = self._encode_frame(frame_image)
        task_metadata = {
            "session_id": skeleton.get("session_id", "unknown_session"),
            "frame_idx": int(skeleton.get("frame_idx", -1)),
            "timestamp_ms": int(skeleton.get("timestamp_ms", 0)),
            "track_id": skeleton.get("track_id", "worker_000"),
            "record_id": record_id,
        }

        task_id = self.label_studio_pusher.push_review_task(
            frame_base64=frame_base64,
            skeleton=skeleton,
            metadata=task_metadata,
        )

        return {
            "route": "review_queue",
            "status": status.value,
            "record_id": record_id,
            "task_id": task_id,
        }

    def _coerce_status(self, gate_result: GateResult | GateStatus | str) -> GateStatus:
        """Convert gate result variant to GateStatus enum."""
        if isinstance(gate_result, GateResult):
            return gate_result.status

        if isinstance(gate_result, GateStatus):
            return gate_result

        if isinstance(gate_result, str):
            return GateStatus(gate_result)

        raise TypeError(f"Unsupported gate_result type: {type(gate_result)!r}")

    def _encode_frame(self, frame_image: np.ndarray) -> str:
        """Encode frame image as base64 JPEG string."""
        ok, encoded = cv2.imencode(".jpg", frame_image)
        if not ok:
            raise RuntimeError("Failed to encode review frame as JPEG.")

        return base64.b64encode(encoded.tobytes()).decode("ascii")
