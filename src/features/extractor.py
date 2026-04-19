"""Feature extraction for worker-level risk and action models."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


class FeatureExtractor:
    """Build fixed 20-dimensional feature vectors from skeleton windows."""

    FEATURE_NAMES = [
        "torso_angle_deg",
        "left_elbow_angle_deg",
        "right_elbow_angle_deg",
        "left_knee_angle_deg",
        "right_knee_angle_deg",
        "shoulder_width_norm",
        "hip_width_norm",
        "zone_id",
        "bbox_area_norm",
        "ppe_compliance_mean",
        "ppe_helmet",
        "ppe_vest",
        "ppe_gloves",
        "velocity_px_per_s",
        "acceleration_px_per_s2",
        "dwell_time_s",
        "torso_pitch_deg",
        "torso_roll_deg",
        "fall_flag",
        "imu_available",
    ]

    def __init__(self, default_frame_width: int = 640, default_frame_height: int = 480) -> None:
        """Initialize feature extractor defaults."""
        self.default_frame_width = max(1, int(default_frame_width))
        self.default_frame_height = max(1, int(default_frame_height))

    def extract(
        self,
        skeleton_window: Sequence[dict[str, Any]],
        imu_features: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract 20-dimensional feature vector from skeleton history.

        Args:
            skeleton_window: Ordered list of skeleton records for one worker.
            imu_features: Optional IMU summary (pitch_deg, roll_deg, fall_flag).

        Returns:
            Dict with float32 vector, feature names, and IMU availability flag.
        """
        if not skeleton_window:
            raise ValueError("skeleton_window must contain at least one skeleton record.")

        ordered = sorted(skeleton_window, key=self._timestamp_key)
        latest = ordered[-1]

        frame_width, frame_height = self._infer_frame_size(latest)
        keypoints = self._keypoint_index(latest.get("keypoints_17", []))

        torso_angle = self._torso_angle_deg(keypoints)
        left_elbow_angle = self._joint_angle_deg(keypoints, "left_shoulder", "left_elbow", "left_wrist")
        right_elbow_angle = self._joint_angle_deg(keypoints, "right_shoulder", "right_elbow", "right_wrist")
        left_knee_angle = self._joint_angle_deg(keypoints, "left_hip", "left_knee", "left_ankle")
        right_knee_angle = self._joint_angle_deg(keypoints, "right_hip", "right_knee", "right_ankle")

        shoulder_width_norm = self._distance_norm(keypoints, "left_shoulder", "right_shoulder", frame_width)
        hip_width_norm = self._distance_norm(keypoints, "left_hip", "right_hip", frame_width)

        center_x, center_y, bbox_area_norm = self._bbox_features(latest, frame_width, frame_height)
        zone_id = self._zone_id(center_x)

        ppe_payload = latest.get("ppe", {}) if isinstance(latest.get("ppe"), dict) else {}
        ppe_helmet = float(ppe_payload.get("helmet", 0.0))
        ppe_vest = float(ppe_payload.get("vest", 0.0))
        ppe_gloves = float(ppe_payload.get("gloves", 0.0))
        ppe_compliance_mean = float(np.mean([ppe_helmet, ppe_vest, ppe_gloves]))

        velocity = self._velocity_px_per_s(ordered)
        acceleration = self._acceleration_px_per_s2(ordered)
        dwell_time = self._dwell_time_s(ordered)

        if imu_features:
            torso_pitch = float(imu_features.get("pitch_deg", imu_features.get("torso_pitch", 0.0)))
            torso_roll = float(imu_features.get("roll_deg", imu_features.get("torso_roll", 0.0)))
            fall_flag = 1.0 if bool(imu_features.get("fall_flag", False)) else 0.0
            imu_available = 1.0
        else:
            torso_pitch = 0.0
            torso_roll = 0.0
            fall_flag = 0.0
            imu_available = 0.0

        vector = np.array(
            [
                torso_angle,
                left_elbow_angle,
                right_elbow_angle,
                left_knee_angle,
                right_knee_angle,
                shoulder_width_norm,
                hip_width_norm,
                zone_id,
                bbox_area_norm,
                ppe_compliance_mean,
                ppe_helmet,
                ppe_vest,
                ppe_gloves,
                velocity,
                acceleration,
                dwell_time,
                torso_pitch,
                torso_roll,
                fall_flag,
                imu_available,
            ],
            dtype=np.float32,
        )

        return {
            "vector": vector,
            "feature_names": list(self.FEATURE_NAMES),
            "imu_available": bool(imu_available),
        }

    def _timestamp_key(self, skeleton: dict[str, Any]) -> int:
        """Sort key for skeleton records."""
        if "timestamp_ms" in skeleton:
            return int(skeleton.get("timestamp_ms", 0))
        frame_idx = int(skeleton.get("frame_idx", 0))
        return frame_idx * 33

    def _infer_frame_size(self, skeleton: dict[str, Any]) -> tuple[int, int]:
        """Infer frame dimensions from skeleton metadata or defaults."""
        width = int(skeleton.get("frame_width", 0))
        height = int(skeleton.get("frame_height", 0))

        bbox = skeleton.get("bbox", [])
        if isinstance(bbox, list) and len(bbox) == 4:
            width = max(width, int(float(bbox[2])) + 1)
            height = max(height, int(float(bbox[3])) + 1)

        if width <= 0:
            width = self.default_frame_width
        if height <= 0:
            height = self.default_frame_height

        return width, height

    def _keypoint_index(self, keypoints_17: Any) -> dict[str, dict[str, float]]:
        """Build name-indexed keypoint map from list payload."""
        index: dict[str, dict[str, float]] = {}
        if not isinstance(keypoints_17, list):
            return index

        for item in keypoints_17:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            index[name] = {
                "x": float(item.get("x", 0.0)),
                "y": float(item.get("y", 0.0)),
                "conf": float(item.get("conf", 0.0)),
            }
        return index

    def _point(self, keypoints: dict[str, dict[str, float]], name: str) -> tuple[float, float] | None:
        """Return keypoint coordinate when available and visible."""
        point = keypoints.get(name)
        if not point:
            return None
        if point["conf"] <= 0.0:
            return None
        return point["x"], point["y"]

    def _torso_angle_deg(self, keypoints: dict[str, dict[str, float]]) -> float:
        """Estimate torso lean angle relative to vertical axis."""
        left_shoulder = self._point(keypoints, "left_shoulder")
        right_shoulder = self._point(keypoints, "right_shoulder")
        left_hip = self._point(keypoints, "left_hip")
        right_hip = self._point(keypoints, "right_hip")

        if not (left_shoulder and right_shoulder and left_hip and right_hip):
            return 0.0

        shoulder_center = ((left_shoulder[0] + right_shoulder[0]) * 0.5, (left_shoulder[1] + right_shoulder[1]) * 0.5)
        hip_center = ((left_hip[0] + right_hip[0]) * 0.5, (left_hip[1] + right_hip[1]) * 0.5)

        dx = shoulder_center[0] - hip_center[0]
        dy = shoulder_center[1] - hip_center[1]

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0

        angle_from_vertical = math.degrees(math.atan2(dx, max(abs(dy), 1e-6)))
        return float(angle_from_vertical)

    def _joint_angle_deg(
        self,
        keypoints: dict[str, dict[str, float]],
        a_name: str,
        b_name: str,
        c_name: str,
    ) -> float:
        """Compute angle ABC in degrees for three keypoints."""
        a = self._point(keypoints, a_name)
        b = self._point(keypoints, b_name)
        c = self._point(keypoints, c_name)
        if not (a and b and c):
            return 0.0

        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])

        norm_ba = math.hypot(ba[0], ba[1])
        norm_bc = math.hypot(bc[0], bc[1])
        if norm_ba <= 1e-6 or norm_bc <= 1e-6:
            return 0.0

        dot = ba[0] * bc[0] + ba[1] * bc[1]
        cosine = np.clip(dot / (norm_ba * norm_bc), -1.0, 1.0)
        return float(math.degrees(math.acos(float(cosine))))

    def _distance_norm(
        self,
        keypoints: dict[str, dict[str, float]],
        left_name: str,
        right_name: str,
        frame_width: int,
    ) -> float:
        """Normalized horizontal distance between two joints."""
        left = self._point(keypoints, left_name)
        right = self._point(keypoints, right_name)
        if not (left and right):
            return 0.0

        return float(math.hypot(left[0] - right[0], left[1] - right[1]) / max(frame_width, 1))

    def _bbox_features(self, skeleton: dict[str, Any], frame_width: int, frame_height: int) -> tuple[float, float, float]:
        """Compute bbox center and normalized area."""
        bbox = skeleton.get("bbox", [])
        if isinstance(bbox, list) and len(bbox) == 4:
            x1, y1, x2, y2 = [float(value) for value in bbox]
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            center_x = (x1 + x2) * 0.5
            center_y = (y1 + y2) * 0.5
            area_norm = (width * height) / max(float(frame_width * frame_height), 1.0)
            return float(center_x), float(center_y), float(area_norm)

        return 0.0, 0.0, 0.0

    def _zone_id(self, center_x: float) -> float:
        """Map worker center position to discrete horizontal zone ID (0,1,2)."""
        width = float(self.default_frame_width)
        if center_x <= width / 3.0:
            return 0.0
        if center_x <= (2.0 * width) / 3.0:
            return 1.0
        return 2.0

    def _velocity_px_per_s(self, ordered: Sequence[dict[str, Any]]) -> float:
        """Compute instantaneous center velocity from last two records."""
        if len(ordered) < 2:
            return 0.0

        prev = ordered[-2]
        curr = ordered[-1]
        prev_center = self._center_from_bbox(prev)
        curr_center = self._center_from_bbox(curr)
        if prev_center is None or curr_center is None:
            return 0.0

        dt = max((self._timestamp_key(curr) - self._timestamp_key(prev)) / 1000.0, 1e-6)
        dist = math.hypot(curr_center[0] - prev_center[0], curr_center[1] - prev_center[1])
        return float(dist / dt)

    def _acceleration_px_per_s2(self, ordered: Sequence[dict[str, Any]]) -> float:
        """Compute finite-difference acceleration from last three centers."""
        if len(ordered) < 3:
            return 0.0

        a = ordered[-3]
        b = ordered[-2]
        c = ordered[-1]

        vab = self._velocity_between(a, b)
        vbc = self._velocity_between(b, c)
        dt = max((self._timestamp_key(c) - self._timestamp_key(b)) / 1000.0, 1e-6)
        return float((vbc - vab) / dt)

    def _dwell_time_s(self, ordered: Sequence[dict[str, Any]]) -> float:
        """Duration covered by current feature window."""
        if len(ordered) < 2:
            return 0.0
        return float((self._timestamp_key(ordered[-1]) - self._timestamp_key(ordered[0])) / 1000.0)

    def _velocity_between(self, left: dict[str, Any], right: dict[str, Any]) -> float:
        """Velocity magnitude between two skeleton records."""
        left_center = self._center_from_bbox(left)
        right_center = self._center_from_bbox(right)
        if left_center is None or right_center is None:
            return 0.0

        dt = max((self._timestamp_key(right) - self._timestamp_key(left)) / 1000.0, 1e-6)
        dist = math.hypot(right_center[0] - left_center[0], right_center[1] - left_center[1])
        return float(dist / dt)

    def _center_from_bbox(self, skeleton: dict[str, Any]) -> tuple[float, float] | None:
        """Return bbox center for one skeleton record."""
        bbox = skeleton.get("bbox", [])
        if not (isinstance(bbox, list) and len(bbox) == 4):
            return None

        x1, y1, x2, y2 = [float(value) for value in bbox]
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5
