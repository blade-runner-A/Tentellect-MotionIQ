"""Dataset validation checks for Week 1 pipeline initialization."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("validate_datasets")


@dataclass
class DatasetValidationResult:
    """Validation result for a single dataset."""

    dataset: str
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


def setup_logging(level: str) -> None:
    """Configure process-wide logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _safe_json_load(path: Path) -> dict[str, Any] | None:
    """Load JSON file with defensive error handling."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        LOGGER.error("Invalid JSON in %s: %s", path, exc)
        return None


def _count_files(path: Path, patterns: tuple[str, ...] | None = None) -> int:
    """Count files in a directory, optionally filtering by suffix patterns."""
    if not path.exists():
        return 0

    files = [p for p in path.rglob("*") if p.is_file()]
    if not patterns:
        return len(files)

    pattern_set = tuple(pattern.lower() for pattern in patterns)
    return sum(1 for p in files if p.suffix.lower() in pattern_set)


def validate_coco(data_root: Path) -> DatasetValidationResult:
    """Validate COCO keypoints dataset structure and annotation files."""
    result = DatasetValidationResult(dataset="coco", passed=True)

    train_dir = data_root / "coco" / "images" / "train2017"
    val_dir = data_root / "coco" / "images" / "val2017"
    train_ann = data_root / "coco" / "annotations" / "person_keypoints_train2017.json"
    val_ann = data_root / "coco" / "annotations" / "person_keypoints_val2017.json"

    for required_path in (train_dir, val_dir, train_ann, val_ann):
        if not required_path.exists():
            result.errors.append(f"Missing required COCO path: {required_path}")

    if result.errors:
        result.passed = False
        return result

    for annotation_path in (train_ann, val_ann):
        payload = _safe_json_load(annotation_path)
        if payload is None:
            result.errors.append(f"Invalid JSON: {annotation_path}")
            continue

        for required_key in ("images", "annotations", "categories"):
            if required_key not in payload:
                result.errors.append(f"Missing key '{required_key}' in {annotation_path}")

    result.stats["train_images"] = _count_files(train_dir, patterns=(".jpg", ".jpeg", ".png"))
    result.stats["val_images"] = _count_files(val_dir, patterns=(".jpg", ".jpeg", ".png"))

    if result.stats["train_images"] == 0:
        result.warnings.append("COCO train split has zero images.")
    if result.stats["val_images"] == 0:
        result.warnings.append("COCO val split has zero images.")

    if result.errors:
        result.passed = False

    return result


def validate_sh17(data_root: Path) -> DatasetValidationResult:
    """Validate SH17 folder structure and label/image counts."""
    result = DatasetValidationResult(dataset="sh17", passed=True)

    image_dir = data_root / "sh17" / "images"
    label_dir = data_root / "sh17" / "labels"

    if not image_dir.exists():
        result.errors.append(f"Missing SH17 images directory: {image_dir}")
    if not label_dir.exists():
        result.errors.append(f"Missing SH17 labels directory: {label_dir}")

    if result.errors:
        result.passed = False
        return result

    image_count = _count_files(image_dir, patterns=(".jpg", ".jpeg", ".png"))
    label_count = _count_files(label_dir, patterns=(".txt",))

    result.stats["images"] = image_count
    result.stats["labels"] = label_count

    if image_count == 0:
        result.warnings.append(
            "SH17 has zero images. If you only cloned the GitHub repo, download the image pack from "
            "Kaggle (see data/sh17/README.md) and place YOLO images under sh17/images/ and labels under sh17/labels/."
        )
    if label_count == 0:
        result.warnings.append("SH17 has zero label files.")

    if image_count and label_count and abs(image_count - label_count) > max(20, image_count // 10):
        result.warnings.append("SH17 image/label counts differ significantly.")

    if result.errors:
        result.passed = False

    return result


def validate_isafety(data_root: Path, strict: bool = False) -> DatasetValidationResult:
    """Validate iSafetyBench clips and annotation file presence."""
    result = DatasetValidationResult(dataset="isafety", passed=True)

    clip_dir = data_root / "isafety" / "clips"
    ann_path = data_root / "isafety" / "annotations.json"

    if not clip_dir.exists():
        message = f"Missing iSafety clips directory: {clip_dir}"
        if strict:
            result.errors.append(message)
        else:
            result.warnings.append(message)
            result.passed = True
            return result

    video_count = _count_files(clip_dir, patterns=(".mp4", ".avi", ".mkv", ".mov"))
    result.stats["clips"] = video_count

    if video_count == 0:
        result.warnings.append("iSafety clips directory is present but contains no video files.")

    if not ann_path.exists():
        result.warnings.append(f"Missing optional iSafety annotation file: {ann_path}")

    if result.errors:
        result.passed = False

    return result


def run_validation(data_root: Path, strict_isafety: bool = False) -> list[DatasetValidationResult]:
    """Run all dataset validators and return their results."""
    return [
        validate_coco(data_root),
        validate_sh17(data_root),
        validate_isafety(data_root, strict=strict_isafety),
    ]


def summarize_results(results: list[DatasetValidationResult]) -> bool:
    """Log validation summary and return True if no hard failures."""
    has_failure = False

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        LOGGER.info("[%s] %s", status, result.dataset)

        for key, value in result.stats.items():
            LOGGER.info("  stat %s=%d", key, value)

        for warning in result.warnings:
            LOGGER.warning("  warning: %s", warning)

        for error in result.errors:
            LOGGER.error("  error: %s", error)

        if not result.passed:
            has_failure = True

    return not has_failure


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Validate Tentellect datasets.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Path to data directory")
    parser.add_argument("--strict-isafety", action="store_true", help="Treat missing iSafety dataset as an error")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    setup_logging(args.log_level)

    results = run_validation(data_root=args.data_root, strict_isafety=args.strict_isafety)
    all_passed = summarize_results(results)

    if all_passed:
        LOGGER.info("Dataset validation completed successfully.")
        return 0

    LOGGER.error("Dataset validation failed. See errors above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
