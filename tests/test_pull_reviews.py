"""Tests for Label Studio review parsing and merge script helpers."""

from __future__ import annotations

from pathlib import Path

from scripts.pull_reviews import merge_completed_reviews, parse_completed_tasks
from src.annotation.storage import DatasetWriter


def _seed_skeleton() -> dict:
    return {
        "session_id": "session_review",
        "frame_idx": 8,
        "timestamp_ms": 4000,
        "track_id": "worker_001",
        "bbox": [10.0, 20.0, 60.0, 120.0],
        "keypoints_17": [
            {"name": "nose", "x": 25.0, "y": 30.0, "conf": 0.4},
            {"name": "left_shoulder", "x": 20.0, "y": 44.0, "conf": 0.5},
            {"name": "right_shoulder", "x": 36.0, "y": 44.0, "conf": 0.5},
        ],
        "ppe": {},
    }


def _sample_completed_task(record_id: str) -> dict:
    return {
        "id": 77,
        "data": {
            "meta": {
                "record_id": record_id,
                "session_id": "session_review",
                "frame_idx": 8,
                "timestamp_ms": 4000,
                "track_id": "worker_001",
            }
        },
        "annotations": [
            {
                "id": 55,
                "was_cancelled": False,
                "result": [
                    {
                        "type": "keypointlabels",
                        "original_width": 640,
                        "original_height": 480,
                        "score": 0.98,
                        "value": {
                            "x": 10.0,
                            "y": 10.0,
                            "keypointlabels": ["nose"],
                        },
                    },
                    {
                        "type": "keypointlabels",
                        "original_width": 640,
                        "original_height": 480,
                        "score": 0.97,
                        "value": {
                            "x": 15.0,
                            "y": 20.0,
                            "keypointlabels": ["left_shoulder"],
                        },
                    },
                    {
                        "type": "keypointlabels",
                        "original_width": 640,
                        "original_height": 480,
                        "score": 0.96,
                        "value": {
                            "x": 25.0,
                            "y": 20.0,
                            "keypointlabels": ["right_shoulder"],
                        },
                    },
                ],
            }
        ],
    }


def test_parse_completed_tasks_extracts_review_payload() -> None:
    parsed = parse_completed_tasks([_sample_completed_task("abc")])

    assert len(parsed) == 1
    assert parsed[0]["session_id"] == "session_review"
    assert parsed[0]["frame_idx"] == 8
    assert parsed[0]["track_id"] == "worker_001"
    assert len(parsed[0]["skeleton"]["keypoints_17"]) == 17


def test_merge_completed_reviews_marks_record_human_verified(tmp_path: Path) -> None:
    writer = DatasetWriter(db_path=tmp_path / "annotations.db")
    record_id = writer.write_annotation(_seed_skeleton(), quality_gate="REVIEW")

    tasks = [_sample_completed_task(record_id)]
    parsed = parse_completed_tasks(tasks)
    merged_count = merge_completed_reviews(writer, parsed)

    updated = writer.get_annotation(record_id)

    assert merged_count == 1
    assert updated is not None
    assert updated["human_verified"] is True
    assert updated["quality_gate"] == "REVIEW"
    assert updated["skeleton"]["keypoints_17"][0]["conf"] > 0.9
