"""Tests for dataset storage and COCO export."""

from __future__ import annotations

from pathlib import Path

from src.annotation.storage import DatasetWriter


def _sample_skeleton(confidence: float = 0.9) -> dict:
    keypoints = [
        {"name": "nose", "x": 50.0, "y": 20.0, "conf": confidence},
        {"name": "left_eye", "x": 46.0, "y": 18.0, "conf": confidence},
        {"name": "right_eye", "x": 54.0, "y": 18.0, "conf": confidence},
        {"name": "left_ear", "x": 43.0, "y": 20.0, "conf": confidence},
        {"name": "right_ear", "x": 57.0, "y": 20.0, "conf": confidence},
        {"name": "left_shoulder", "x": 38.0, "y": 40.0, "conf": confidence},
        {"name": "right_shoulder", "x": 62.0, "y": 40.0, "conf": confidence},
        {"name": "left_elbow", "x": 32.0, "y": 58.0, "conf": confidence},
        {"name": "right_elbow", "x": 68.0, "y": 58.0, "conf": confidence},
        {"name": "left_wrist", "x": 28.0, "y": 76.0, "conf": confidence},
        {"name": "right_wrist", "x": 72.0, "y": 76.0, "conf": confidence},
        {"name": "left_hip", "x": 44.0, "y": 75.0, "conf": confidence},
        {"name": "right_hip", "x": 56.0, "y": 75.0, "conf": confidence},
        {"name": "left_knee", "x": 44.0, "y": 95.0, "conf": confidence},
        {"name": "right_knee", "x": 56.0, "y": 95.0, "conf": confidence},
        {"name": "left_ankle", "x": 44.0, "y": 118.0, "conf": confidence},
        {"name": "right_ankle", "x": 56.0, "y": 118.0, "conf": confidence},
    ]

    return {
        "session_id": "session_storage",
        "frame_idx": 12,
        "timestamp_ms": 6000,
        "track_id": "worker_001",
        "bbox": [28.0, 18.0, 72.0, 118.0],
        "keypoints_17": keypoints,
        "ppe": {"helmet": 0.9, "vest": 0.8, "gloves": 0.4, "glasses": 0.2},
    }


def test_write_and_get_annotation(tmp_path: Path) -> None:
    writer = DatasetWriter(db_path=tmp_path / "annotations.db")
    skeleton = _sample_skeleton()

    record_id = writer.write_annotation(skeleton=skeleton, quality_gate="AUTO_ACCEPT")
    row = writer.get_annotation(record_id)

    assert row is not None
    assert row["id"] == record_id
    assert row["quality_gate"] == "AUTO_ACCEPT"
    assert row["human_verified"] is False
    assert row["skeleton"]["track_id"] == "worker_001"


def test_merge_human_review_updates_existing_row(tmp_path: Path) -> None:
    writer = DatasetWriter(db_path=tmp_path / "annotations.db")
    original = _sample_skeleton(confidence=0.55)

    record_id = writer.write_annotation(skeleton=original, quality_gate="REVIEW")

    reviewed = _sample_skeleton(confidence=1.0)
    merged_id = writer.merge_human_review(
        session_id="session_storage",
        frame_idx=12,
        track_id="worker_001",
        reviewed_skeleton=reviewed,
    )

    updated = writer.get_annotation(record_id)

    assert merged_id == record_id
    assert updated is not None
    assert updated["human_verified"] is True
    assert updated["quality_gate"] == "REVIEW"
    assert updated["skeleton"]["keypoints_17"][0]["conf"] == 1.0


def test_export_coco_writes_valid_structure(tmp_path: Path) -> None:
    writer = DatasetWriter(db_path=tmp_path / "annotations.db")
    skeleton = _sample_skeleton()

    writer.write_annotation(skeleton=skeleton, quality_gate="AUTO_ACCEPT")
    output_path = tmp_path / "coco_export.json"

    payload = writer.export_coco(output_path=output_path, validate_with_pycocotools=False)

    assert output_path.exists()
    assert len(payload["images"]) == 1
    assert len(payload["annotations"]) == 1
    assert payload["annotations"][0]["num_keypoints"] == 17
    assert len(payload["annotations"][0]["keypoints"]) == 51
