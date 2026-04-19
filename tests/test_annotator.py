"""Tests for annotation router and review queue routing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.annotation.annotator import AnnotationRouter
from src.annotation.quality_gates import GateStatus
from src.annotation.storage import DatasetWriter


class FakeLabelStudioPusher:
    """Fake Label Studio client for routing tests."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def push_review_task(self, frame_base64: str, skeleton: dict, metadata: dict) -> str:
        self.calls.append(
            {
                "frame_base64": frame_base64,
                "skeleton": skeleton,
                "metadata": metadata,
            }
        )
        return "task_123"


def _sample_skeleton() -> dict:
    return {
        "session_id": "session_router",
        "frame_idx": 3,
        "timestamp_ms": 1500,
        "track_id": "worker_001",
        "bbox": [10.0, 20.0, 60.0, 120.0],
        "keypoints_17": [
            {"name": "nose", "x": 30.0, "y": 24.0, "conf": 0.95},
            {"name": "left_shoulder", "x": 22.0, "y": 45.0, "conf": 0.91},
            {"name": "right_shoulder", "x": 38.0, "y": 45.0, "conf": 0.92},
        ],
        "ppe": {"helmet": 0.8, "vest": 0.7, "gloves": 0.2, "glasses": 0.1},
    }


def test_auto_accept_routes_to_dataset_writer(tmp_path: Path) -> None:
    writer = DatasetWriter(db_path=tmp_path / "annotations.db")
    router = AnnotationRouter(dataset_writer=writer)

    result = router.route(_sample_skeleton(), GateStatus.AUTO_ACCEPT)

    assert result["route"] == "dataset"
    assert result["status"] == "AUTO_ACCEPT"

    row = writer.get_annotation(result["record_id"])
    assert row is not None
    assert row["quality_gate"] == "AUTO_ACCEPT"


def test_review_routes_to_label_studio_queue(tmp_path: Path) -> None:
    writer = DatasetWriter(db_path=tmp_path / "annotations.db")
    fake_pusher = FakeLabelStudioPusher()
    router = AnnotationRouter(dataset_writer=writer, label_studio_pusher=fake_pusher)

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    result = router.route(_sample_skeleton(), GateStatus.REVIEW, frame_image=frame)

    assert result["route"] == "review_queue"
    assert result["task_id"] == "task_123"
    assert len(fake_pusher.calls) == 1

    row = writer.get_annotation(result["record_id"])
    assert row is not None
    assert row["quality_gate"] == "REVIEW"


def test_review_raises_when_frame_missing(tmp_path: Path) -> None:
    writer = DatasetWriter(db_path=tmp_path / "annotations.db")
    fake_pusher = FakeLabelStudioPusher()
    router = AnnotationRouter(dataset_writer=writer, label_studio_pusher=fake_pusher)

    with pytest.raises(ValueError):
        router.route(_sample_skeleton(), GateStatus.REVIEW)


def test_discard_skips_storage_write(tmp_path: Path) -> None:
    writer = DatasetWriter(db_path=tmp_path / "annotations.db")
    router = AnnotationRouter(dataset_writer=writer)

    result = router.route(_sample_skeleton(), GateStatus.DISCARD)

    assert result["route"] == "discard"
    assert result["status"] == "DISCARD"

    # No annotation should exist for discarded route.
    row = writer.find_latest_annotation("session_router", 3, "worker_001")
    assert row is None
