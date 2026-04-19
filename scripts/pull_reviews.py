"""Pull completed Label Studio reviews and merge them into SQLite storage."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.annotation.storage import COCO_KEYPOINT_NAMES, DatasetWriter

LOGGER = logging.getLogger("pull_reviews")


def parse_completed_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse completed Label Studio tasks into normalized review records."""
    parsed: list[dict[str, Any]] = []
    for task in tasks:
        record = _parse_single_task(task)
        if record is not None:
            parsed.append(record)
    return parsed


def merge_completed_reviews(dataset_writer: DatasetWriter, reviews: list[dict[str, Any]]) -> int:
    """Merge parsed reviews into dataset storage and return merged count."""
    merged = 0
    for review in reviews:
        record_id = str(review.get("record_id", "")).strip()
        skeleton = dict(review["skeleton"])

        if record_id:
            if dataset_writer.update_human_review_by_id(record_id=record_id, reviewed_skeleton=skeleton):
                merged += 1
                continue

        dataset_writer.merge_human_review(
            session_id=str(review["session_id"]),
            frame_idx=int(review["frame_idx"]),
            track_id=str(review["track_id"]),
            reviewed_skeleton=skeleton,
            timestamp_ms=int(review.get("timestamp_ms", 0)),
        )
        merged += 1

    return merged


def fetch_completed_tasks(api_url: str, api_key: str, project_id: int) -> list[dict[str, Any]]:
    """Fetch completed tasks from Label Studio project."""
    try:
        from label_studio_sdk import Client
    except ImportError as exc:  # pragma: no cover - optional runtime package
        raise RuntimeError("label-studio-sdk is required to fetch reviews from Label Studio.") from exc

    client = Client(url=api_url, api_key=api_key)

    if hasattr(client, "get_project"):
        project = client.get_project(project_id)
        if hasattr(project, "get_tasks"):
            tasks = project.get_tasks()
        else:
            tasks = []
    elif hasattr(client, "get_tasks"):
        tasks = client.get_tasks(project=project_id)
    else:
        raise RuntimeError("Unable to fetch tasks with provided Label Studio client.")

    completed = []
    for task in tasks:
        annotations = task.get("annotations", [])
        if not annotations:
            continue
        if any(not bool(annotation.get("was_cancelled", False)) for annotation in annotations):
            completed.append(task)

    return completed


def load_tasks_from_json(path: str | Path) -> list[dict[str, Any]]:
    """Load exported Label Studio tasks from local JSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError("Task JSON payload must be a list of task objects.")


def _parse_single_task(task: dict[str, Any]) -> dict[str, Any] | None:
    """Parse one Label Studio task into review record format."""
    data = task.get("data", {}) if isinstance(task.get("data"), dict) else {}
    meta = data.get("meta", {}) if isinstance(data.get("meta"), dict) else {}

    annotations = task.get("annotations", [])
    if not isinstance(annotations, list) or not annotations:
        return None

    active_annotations = [item for item in annotations if isinstance(item, dict) and not bool(item.get("was_cancelled", False))]
    if not active_annotations:
        return None

    annotation = active_annotations[-1]
    result_items = annotation.get("result", [])
    keypoints = _parse_keypoint_results(result_items)
    if not keypoints:
        return None

    session_id = str(meta.get("session_id", "unknown_session"))
    frame_idx = int(meta.get("frame_idx", -1))
    track_id = str(meta.get("track_id", "worker_000"))
    timestamp_ms = int(meta.get("timestamp_ms", 0))

    skeleton = {
        "session_id": session_id,
        "frame_idx": frame_idx,
        "timestamp_ms": timestamp_ms,
        "track_id": track_id,
        "bbox": _bbox_from_keypoints(keypoints),
        "keypoints_17": keypoints,
        "ppe": {},
    }

    return {
        "record_id": meta.get("record_id", ""),
        "session_id": session_id,
        "frame_idx": frame_idx,
        "track_id": track_id,
        "timestamp_ms": timestamp_ms,
        "skeleton": skeleton,
    }


def _parse_keypoint_results(result_items: Any) -> list[dict[str, float | str]]:
    """Parse Label Studio keypointlabels results into COCO keypoint list."""
    if not isinstance(result_items, list):
        return []

    keyed: dict[str, dict[str, float]] = {}
    for item in result_items:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "keypointlabels":
            continue

        value = item.get("value", {})
        if not isinstance(value, dict):
            continue

        labels = value.get("keypointlabels", [])
        if not isinstance(labels, list) or not labels:
            continue

        label = str(labels[0])
        if label not in COCO_KEYPOINT_NAMES:
            continue

        original_width = int(item.get("original_width", 640) or 640)
        original_height = int(item.get("original_height", 480) or 480)

        x_pct = float(value.get("x", 0.0))
        y_pct = float(value.get("y", 0.0))

        x_px = (x_pct / 100.0) * original_width
        y_px = (y_pct / 100.0) * original_height
        conf = float(item.get("score", 1.0))

        current = keyed.get(label)
        if current is None or conf >= current["conf"]:
            keyed[label] = {"x": x_px, "y": y_px, "conf": conf}

    output: list[dict[str, float | str]] = []
    for name in COCO_KEYPOINT_NAMES:
        point = keyed.get(name, {"x": 0.0, "y": 0.0, "conf": 0.0})
        output.append({"name": name, "x": float(point["x"]), "y": float(point["y"]), "conf": float(point["conf"])})

    return output


def _bbox_from_keypoints(keypoints: list[dict[str, float | str]]) -> list[float]:
    """Create xyxy bbox from visible keypoints."""
    visible = [point for point in keypoints if float(point.get("conf", 0.0)) > 0.0]
    if not visible:
        return [0.0, 0.0, 1.0, 1.0]

    xs = [float(point["x"]) for point in visible]
    ys = [float(point["y"]) for point in visible]
    return [min(xs), min(ys), max(xs), max(ys)]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Pull completed Label Studio reviews into SQLite.")
    parser.add_argument("--db-path", default="data/processed/annotations.db")
    parser.add_argument("--api-url", default="http://localhost:8080")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--project-id", type=int, default=0)
    parser.add_argument("--tasks-json", default="")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    """Configure script logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    setup_logging(args.log_level)

    if args.tasks_json:
        tasks = load_tasks_from_json(args.tasks_json)
    else:
        api_key = args.api_key or os.getenv("LABEL_STUDIO_API_KEY", "")
        if not api_key:
            LOGGER.error("Label Studio API key is required when --tasks-json is not provided.")
            return 1
        if args.project_id <= 0:
            LOGGER.error("A valid --project-id is required when fetching tasks from Label Studio.")
            return 1

        tasks = fetch_completed_tasks(api_url=args.api_url, api_key=api_key, project_id=args.project_id)

    reviews = parse_completed_tasks(tasks)
    writer = DatasetWriter(db_path=args.db_path)
    merged_count = merge_completed_reviews(writer, reviews)

    LOGGER.info("tasks_seen=%d", len(tasks))
    LOGGER.info("reviews_parsed=%d", len(reviews))
    LOGGER.info("reviews_merged=%d", merged_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
