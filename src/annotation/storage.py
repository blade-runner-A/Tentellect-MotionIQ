"""SQLite-backed storage and COCO export for annotation records."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

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


class DatasetWriter:
    """Manage annotation persistence and export operations."""

    def __init__(self, db_path: str | Path = "data/processed/annotations.db") -> None:
        """Initialize writer and create schema when absent."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def write_annotation(
        self,
        skeleton: dict[str, Any],
        quality_gate: str,
        human_verified: bool = False,
        feature_vector: Sequence[float] | np.ndarray | None = None,
        risk_score: float | None = None,
        action_class: str | None = None,
        record_id: str | None = None,
    ) -> str:
        """Insert a skeleton annotation row and return its record ID."""
        annotation_id = record_id or str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        feature_blob = None
        if feature_vector is not None:
            feature_blob = np.asarray(feature_vector, dtype=np.float32).tobytes()

        payload = {
            "id": annotation_id,
            "session_id": str(skeleton.get("session_id", "unknown_session")),
            "frame_idx": int(skeleton.get("frame_idx", -1)),
            "timestamp_ms": int(skeleton.get("timestamp_ms", 0)),
            "track_id": str(skeleton.get("track_id", "worker_000")),
            "quality_gate": str(quality_gate),
            "human_verified": 1 if human_verified else 0,
            "skeleton_json": json.dumps(skeleton, ensure_ascii=True),
            "ppe_json": json.dumps(skeleton.get("ppe", {}), ensure_ascii=True),
            "feature_vector": feature_blob,
            "risk_score": risk_score,
            "action_class": action_class,
            "created_at": created_at,
        }

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO annotations (
                    id, session_id, frame_idx, timestamp_ms, track_id, quality_gate,
                    human_verified, skeleton_json, ppe_json, feature_vector,
                    risk_score, action_class, created_at
                ) VALUES (
                    :id, :session_id, :frame_idx, :timestamp_ms, :track_id, :quality_gate,
                    :human_verified, :skeleton_json, :ppe_json, :feature_vector,
                    :risk_score, :action_class, :created_at
                )
                """,
                payload,
            )

        return annotation_id

    def merge_human_review(
        self,
        session_id: str,
        frame_idx: int,
        track_id: str,
        reviewed_skeleton: dict[str, Any],
        timestamp_ms: int | None = None,
    ) -> str:
        """Merge reviewed skeleton into existing row or create a new verified row."""
        row = self.find_latest_annotation(session_id=session_id, frame_idx=frame_idx, track_id=track_id)

        if row:
            record_id = str(row["id"])
            self.update_human_review_by_id(record_id=record_id, reviewed_skeleton=reviewed_skeleton)
            return record_id

        reviewed_payload = dict(reviewed_skeleton)
        reviewed_payload.setdefault("session_id", session_id)
        reviewed_payload.setdefault("frame_idx", frame_idx)
        reviewed_payload.setdefault("track_id", track_id)
        reviewed_payload.setdefault("timestamp_ms", int(timestamp_ms or 0))

        return self.write_annotation(
            skeleton=reviewed_payload,
            quality_gate="REVIEW",
            human_verified=True,
        )

    def update_human_review_by_id(self, record_id: str, reviewed_skeleton: dict[str, Any]) -> bool:
        """Update existing annotation row as human-verified."""
        with self._connect() as conn:
            result = conn.execute(
                """
                UPDATE annotations
                SET
                    skeleton_json = :skeleton_json,
                    ppe_json = :ppe_json,
                    human_verified = 1,
                    quality_gate = 'REVIEW'
                WHERE id = :id
                """,
                {
                    "id": record_id,
                    "skeleton_json": json.dumps(reviewed_skeleton, ensure_ascii=True),
                    "ppe_json": json.dumps(reviewed_skeleton.get("ppe", {}), ensure_ascii=True),
                },
            )
            return result.rowcount > 0

    def find_latest_annotation(self, session_id: str, frame_idx: int, track_id: str) -> dict[str, Any] | None:
        """Return latest matching annotation row by session/frame/track."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM annotations
                WHERE session_id = ? AND frame_idx = ? AND track_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (session_id, frame_idx, track_id),
            ).fetchone()

        return self._row_to_dict(row) if row is not None else None

    def get_annotation(self, record_id: str) -> dict[str, Any] | None:
        """Return annotation row by record ID."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM annotations WHERE id = ?", (record_id,)).fetchone()

        return self._row_to_dict(row) if row is not None else None

    def export_coco(
        self,
        output_path: str | Path,
        validate_with_pycocotools: bool = True,
    ) -> dict[str, Any]:
        """Export AUTO_ACCEPT + human-verified records to COCO keypoints JSON."""
        rows = self._fetch_export_rows()

        image_id_map: dict[tuple[str, int], int] = {}
        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        next_annotation_id = 1

        for row in rows:
            skeleton = json.loads(str(row["skeleton_json"]))
            session_id = str(row["session_id"])
            frame_idx = int(row["frame_idx"])
            image_key = (session_id, frame_idx)

            if image_key not in image_id_map:
                image_id = len(image_id_map) + 1
                image_id_map[image_key] = image_id
                width, height = self._infer_image_dimensions(skeleton)
                images.append(
                    {
                        "id": image_id,
                        "file_name": f"{session_id}_{frame_idx:06d}.jpg",
                        "width": width,
                        "height": height,
                    }
                )

            image_id = image_id_map[image_key]
            keypoints_vector, visible_count = self._to_coco_keypoint_vector(skeleton.get("keypoints_17", []))
            bbox_xywh = self._to_coco_bbox_xywh(skeleton)

            annotations.append(
                {
                    "id": next_annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "keypoints": keypoints_vector,
                    "num_keypoints": visible_count,
                    "bbox": bbox_xywh,
                    "area": float(bbox_xywh[2] * bbox_xywh[3]),
                    "iscrowd": 0,
                }
            )
            next_annotation_id += 1

        coco_payload = {
            "info": {
                "description": "Tentellect processed skeleton dataset",
                "version": "1.0",
                "year": datetime.now(timezone.utc).year,
                "date_created": datetime.now(timezone.utc).isoformat(),
            },
            "licenses": [{"id": 1, "name": "internal", "url": ""}],
            "images": images,
            "annotations": annotations,
            "categories": [
                {
                    "id": 1,
                    "name": "person",
                    "supercategory": "person",
                    "keypoints": COCO_KEYPOINT_NAMES,
                    "skeleton": [],
                }
            ],
        }

        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(coco_payload, indent=2, ensure_ascii=True), encoding="utf-8")

        if validate_with_pycocotools:
            self._validate_coco_with_pycocotools(target)

        return coco_payload

    def _initialize_schema(self) -> None:
        """Create SQLite schema expected by the pipeline."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS annotations (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    frame_idx INTEGER NOT NULL,
                    timestamp_ms INTEGER NOT NULL,
                    track_id TEXT NOT NULL,
                    quality_gate TEXT NOT NULL,
                    human_verified INTEGER NOT NULL DEFAULT 0,
                    skeleton_json TEXT NOT NULL,
                    ppe_json TEXT,
                    feature_vector BLOB,
                    risk_score REAL,
                    action_class TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

    def _fetch_export_rows(self) -> list[sqlite3.Row]:
        """Fetch rows eligible for COCO export."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM annotations
                WHERE quality_gate = 'AUTO_ACCEPT' OR human_verified = 1
                ORDER BY created_at ASC
                """
            ).fetchall()

        return rows

    def _to_coco_keypoint_vector(self, keypoints_17: Any) -> tuple[list[float], int]:
        """Convert keypoint payload into COCO [x,y,v]*17 vector."""
        indexed: dict[str, dict[str, float]] = {}
        if isinstance(keypoints_17, list):
            for item in keypoints_17:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                indexed[name] = {
                    "x": float(item.get("x", 0.0)),
                    "y": float(item.get("y", 0.0)),
                    "conf": float(item.get("conf", 0.0)),
                }

        vector: list[float] = []
        visible_count = 0
        for name in COCO_KEYPOINT_NAMES:
            point = indexed.get(name, {"x": 0.0, "y": 0.0, "conf": 0.0})
            conf = float(point["conf"])
            visibility = 2.0 if conf > 0.0 else 0.0
            if visibility > 0.0:
                visible_count += 1

            vector.extend([float(point["x"]), float(point["y"]), visibility])

        return vector, visible_count

    def _to_coco_bbox_xywh(self, skeleton: dict[str, Any]) -> list[float]:
        """Convert bbox to COCO xywh format."""
        bbox = skeleton.get("bbox", [])
        if isinstance(bbox, list) and len(bbox) == 4:
            x1, y1, x2, y2 = [float(value) for value in bbox]
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            return [x1, y1, width, height]

        keypoints = skeleton.get("keypoints_17", [])
        xs: list[float] = []
        ys: list[float] = []
        if isinstance(keypoints, list):
            for item in keypoints:
                if not isinstance(item, dict):
                    continue
                if float(item.get("conf", 0.0)) <= 0.0:
                    continue
                xs.append(float(item.get("x", 0.0)))
                ys.append(float(item.get("y", 0.0)))

        if not xs or not ys:
            return [0.0, 0.0, 1.0, 1.0]

        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        return [min_x, min_y, max(0.0, max_x - min_x), max(0.0, max_y - min_y)]

    def _infer_image_dimensions(self, skeleton: dict[str, Any]) -> tuple[int, int]:
        """Infer image dimensions from skeleton metadata when available."""
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

    def _validate_coco_with_pycocotools(self, coco_path: Path) -> None:
        """Load exported COCO file with pycocotools for structural validation."""
        try:
            from pycocotools.coco import COCO
        except ImportError as exc:  # pragma: no cover - depends on optional runtime package
            raise RuntimeError(
                "pycocotools is required for COCO validation. Install with `pip install pycocotools`."
            ) from exc

        try:
            COCO(str(coco_path))
        except Exception as exc:
            raise RuntimeError(f"COCO export validation failed for {coco_path}: {exc}") from exc

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert sqlite row to dict and parse JSON fields."""
        data = dict(row)
        data["human_verified"] = bool(data.get("human_verified", 0))

        skeleton_json = data.get("skeleton_json")
        if isinstance(skeleton_json, str):
            data["skeleton"] = json.loads(skeleton_json)

        ppe_json = data.get("ppe_json")
        if isinstance(ppe_json, str) and ppe_json:
            data["ppe"] = json.loads(ppe_json)

        return data

    def _connect(self) -> sqlite3.Connection:
        """Open sqlite connection with row factory configured."""
        connection = sqlite3.connect(str(self.db_path))
        connection.row_factory = sqlite3.Row
        return connection
