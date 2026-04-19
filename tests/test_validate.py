"""Tests for dataset validation logic."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_datasets import run_validation, validate_coco, validate_sh17


MINIMAL_COCO_PAYLOAD = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "person"}],
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_validate_coco_passes_for_minimal_valid_structure(tmp_path: Path) -> None:
    data_root = tmp_path / "data"

    (data_root / "coco" / "images" / "train2017").mkdir(parents=True)
    (data_root / "coco" / "images" / "val2017").mkdir(parents=True)

    _write_json(data_root / "coco" / "annotations" / "person_keypoints_train2017.json", MINIMAL_COCO_PAYLOAD)
    _write_json(data_root / "coco" / "annotations" / "person_keypoints_val2017.json", MINIMAL_COCO_PAYLOAD)

    result = validate_coco(data_root)

    assert result.passed is True
    assert result.errors == []


def test_validate_coco_fails_when_required_files_missing(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    (data_root / "coco" / "images" / "train2017").mkdir(parents=True)

    result = validate_coco(data_root)

    assert result.passed is False
    assert len(result.errors) >= 1


def test_validate_sh17_and_run_validation(tmp_path: Path) -> None:
    data_root = tmp_path / "data"

    image_dir = data_root / "sh17" / "images"
    label_dir = data_root / "sh17" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    (image_dir / "frame_001.jpg").write_bytes(b"abc")
    (label_dir / "frame_001.txt").write_text("0 0.5 0.5 0.2 0.2", encoding="utf-8")

    sh17_result = validate_sh17(data_root)
    all_results = run_validation(data_root)

    assert sh17_result.passed is True
    assert sh17_result.stats["images"] == 1
    assert sh17_result.stats["labels"] == 1
    assert len(all_results) == 3
