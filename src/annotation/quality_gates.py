"""Quality gate evaluation for skeleton annotation decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

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

HEAD_KEYPOINTS = {"nose", "left_eye", "right_eye", "left_ear", "right_ear"}


class GateStatus(str, Enum):
    """Possible quality gate routing outcomes."""

    AUTO_ACCEPT = "AUTO_ACCEPT"
    REVIEW = "REVIEW"
    DISCARD = "DISCARD"


@dataclass(frozen=True)
class GateResult:
    """Output payload for quality gate evaluation."""

    status: GateStatus
    reason: str
    mean_confidence: float
    visible_keypoints: int


class QualityGate:
    """Evaluate detection confidence, pose confidence, and plausibility checks."""

    def __init__(
        self,
        detection_threshold: float = 0.60,
        auto_accept_threshold: float = 0.85,
        review_threshold: float = 0.50,
        min_visible_keypoints: int = 8,
        shoulder_ratio_range: tuple[float, float] = (0.2, 0.8),
        arm_symmetry_tolerance: float = 0.20,
    ) -> None:
        """Initialize quality gate thresholds."""
        self.detection_threshold = detection_threshold
        self.auto_accept_threshold = auto_accept_threshold
        self.review_threshold = review_threshold
        self.min_visible_keypoints = min_visible_keypoints
        self.shoulder_ratio_range = shoulder_ratio_range
        self.arm_symmetry_tolerance = arm_symmetry_tolerance

    def evaluate(
        self,
        keypoint_payload: dict[str, Any] | list[dict[str, Any]],
        frame_width: int,
        frame_height: int,
    ) -> GateResult:
        """Evaluate one frame payload and return quality-gate decision.

        Args:
            keypoint_payload: One skeleton dict or list of skeleton dicts.
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.

        Returns:
            GateResult with status and reason.
        """
        workers = keypoint_payload if isinstance(keypoint_payload, list) else [keypoint_payload]
        if not workers:
            return GateResult(
                status=GateStatus.DISCARD,
                reason="No worker detections in frame payload.",
                mean_confidence=0.0,
                visible_keypoints=0,
            )

        valid_workers = [worker for worker in workers if float(worker.get("detection_confidence", 0.0)) > self.detection_threshold]
        if not valid_workers:
            return GateResult(
                status=GateStatus.DISCARD,
                reason=f"G1 failed: no worker above detection threshold {self.detection_threshold:.2f}.",
                mean_confidence=0.0,
                visible_keypoints=0,
            )

        worker_results: list[GateResult] = []
        for worker in valid_workers:
            keypoints = self._normalize_keypoints(worker.get("keypoints_17", []))
            mean_confidence = self._mean_confidence(keypoints)
            visible_count = self._visible_keypoint_count(keypoints)

            plausibility_passed, plausibility_reason = self._plausibility_check(
                keypoints=keypoints,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            if not plausibility_passed:
                worker_results.append(
                    GateResult(
                        status=GateStatus.REVIEW,
                        reason=f"G3 failed: {plausibility_reason}",
                        mean_confidence=mean_confidence,
                        visible_keypoints=visible_count,
                    )
                )
                continue

            if mean_confidence > self.auto_accept_threshold:
                worker_results.append(
                    GateResult(
                        status=GateStatus.AUTO_ACCEPT,
                        reason=f"G2 passed auto-accept threshold ({mean_confidence:.3f}).",
                        mean_confidence=mean_confidence,
                        visible_keypoints=visible_count,
                    )
                )
                continue

            if mean_confidence >= self.review_threshold:
                worker_results.append(
                    GateResult(
                        status=GateStatus.REVIEW,
                        reason=f"G2 in review band ({mean_confidence:.3f}).",
                        mean_confidence=mean_confidence,
                        visible_keypoints=visible_count,
                    )
                )
                continue

            worker_results.append(
                GateResult(
                    status=GateStatus.DISCARD,
                    reason=f"G2 below review threshold ({mean_confidence:.3f}).",
                    mean_confidence=mean_confidence,
                    visible_keypoints=visible_count,
                )
            )

        return self._aggregate_worker_results(worker_results)

    def _aggregate_worker_results(self, results: list[GateResult]) -> GateResult:
        """Aggregate per-worker results to a single frame-level decision."""
        if not results:
            return GateResult(
                status=GateStatus.DISCARD,
                reason="No valid workers after G1.",
                mean_confidence=0.0,
                visible_keypoints=0,
            )

        if any(result.status == GateStatus.REVIEW for result in results):
            review_result = next(result for result in results if result.status == GateStatus.REVIEW)
            return review_result

        if any(result.status == GateStatus.AUTO_ACCEPT for result in results):
            auto_result = next(result for result in results if result.status == GateStatus.AUTO_ACCEPT)
            return auto_result

        return results[0]

    def _normalize_keypoints(self, keypoints: Any) -> dict[str, dict[str, float]]:
        """Normalize supported keypoint payloads into name-indexed dict format."""
        normalized: dict[str, dict[str, float]] = {}

        if isinstance(keypoints, list) and keypoints and isinstance(keypoints[0], dict):
            for item in keypoints:
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                normalized[name] = {
                    "x": float(item.get("x", 0.0)),
                    "y": float(item.get("y", 0.0)),
                    "conf": float(item.get("conf", 0.0)),
                }
            return normalized

        if isinstance(keypoints, list) and len(keypoints) >= 51 and all(isinstance(value, (int, float)) for value in keypoints[:51]):
            for idx, name in enumerate(COCO_KEYPOINT_NAMES):
                offset = idx * 3
                x_val = float(keypoints[offset])
                y_val = float(keypoints[offset + 1])
                visibility = float(keypoints[offset + 2])
                normalized[name] = {
                    "x": x_val,
                    "y": y_val,
                    "conf": 1.0 if visibility > 0.0 else 0.0,
                }
            return normalized

        if isinstance(keypoints, dict):
            for name, values in keypoints.items():
                if not isinstance(values, dict):
                    continue
                normalized[str(name)] = {
                    "x": float(values.get("x", 0.0)),
                    "y": float(values.get("y", 0.0)),
                    "conf": float(values.get("conf", 0.0)),
                }

        return normalized

    def _mean_confidence(self, keypoints: dict[str, dict[str, float]]) -> float:
        """Compute mean confidence across available keypoints."""
        if not keypoints:
            return 0.0
        values = [point["conf"] for point in keypoints.values()]
        return float(sum(values) / len(values))

    def _visible_keypoint_count(self, keypoints: dict[str, dict[str, float]]) -> int:
        """Count keypoints with confidence above zero."""
        return sum(1 for point in keypoints.values() if point["conf"] > 0.0)

    def _plausibility_check(
        self,
        keypoints: dict[str, dict[str, float]],
        frame_width: int,
        frame_height: int,
    ) -> tuple[bool, str]:
        """Run G3 plausibility checks.

        Checks:
        1) shoulder width / body height ratio in range,
        2) arm symmetry within tolerance,
        3) all visible keypoints inside frame bounds,
        4) minimum visible keypoints,
        5) at least one visible head keypoint.
        """
        if self._visible_keypoint_count(keypoints) < self.min_visible_keypoints:
            return False, f"Visible keypoints below minimum {self.min_visible_keypoints}."

        if not self._all_points_in_bounds(keypoints, frame_width, frame_height):
            return False, "One or more keypoints are outside frame bounds."

        if not self._has_head_signal(keypoints):
            return False, "No visible head keypoints (headless detection)."

        shoulder_ok, shoulder_reason = self._check_shoulder_height_ratio(keypoints)
        if not shoulder_ok:
            return False, shoulder_reason

        arm_ok, arm_reason = self._check_arm_symmetry(keypoints)
        if not arm_ok:
            return False, arm_reason

        return True, "Plausibility checks passed."

    def _all_points_in_bounds(
        self,
        keypoints: dict[str, dict[str, float]],
        frame_width: int,
        frame_height: int,
    ) -> bool:
        """Return True when all visible keypoints are inside frame boundaries."""
        for point in keypoints.values():
            if point["conf"] <= 0.0:
                continue
            x_val = point["x"]
            y_val = point["y"]
            if x_val < 0 or x_val >= frame_width or y_val < 0 or y_val >= frame_height:
                return False
        return True

    def _has_head_signal(self, keypoints: dict[str, dict[str, float]]) -> bool:
        """Require at least one visible head keypoint."""
        for name in HEAD_KEYPOINTS:
            if keypoints.get(name, {}).get("conf", 0.0) > 0.0:
                return True
        return False

    def _check_shoulder_height_ratio(self, keypoints: dict[str, dict[str, float]]) -> tuple[bool, str]:
        """Validate shoulder width to body-height ratio."""
        left_shoulder = keypoints.get("left_shoulder")
        right_shoulder = keypoints.get("right_shoulder")
        if not left_shoulder or not right_shoulder:
            return False, "Missing shoulder keypoints for ratio check."

        if left_shoulder["conf"] <= 0.0 or right_shoulder["conf"] <= 0.0:
            return False, "Shoulder keypoints not visible for ratio check."

        shoulder_width = self._distance(left_shoulder, right_shoulder)

        visible_points = [point for point in keypoints.values() if point["conf"] > 0.0]
        if len(visible_points) < 2:
            return False, "Insufficient visible points for body-height estimation."

        y_values = [point["y"] for point in visible_points]
        body_height = max(y_values) - min(y_values)
        if body_height <= 1e-6:
            return False, "Body-height estimate is near zero."

        ratio = shoulder_width / body_height
        lower, upper = self.shoulder_ratio_range
        if ratio < lower or ratio > upper:
            return False, f"Shoulder/body ratio {ratio:.3f} outside [{lower:.3f}, {upper:.3f}]."

        return True, "Shoulder ratio valid."

    def _check_arm_symmetry(self, keypoints: dict[str, dict[str, float]]) -> tuple[bool, str]:
        """Check left/right arm lengths are within tolerance."""
        left_length = self._arm_length(keypoints, "left_shoulder", "left_elbow", "left_wrist")
        right_length = self._arm_length(keypoints, "right_shoulder", "right_elbow", "right_wrist")

        if left_length is None or right_length is None:
            return False, "Missing arm keypoints for symmetry check."

        denominator = max(left_length, right_length)
        if denominator <= 1e-6:
            return False, "Arm length denominator too small."

        delta = abs(left_length - right_length) / denominator
        if delta > self.arm_symmetry_tolerance:
            return False, f"Arm symmetry delta {delta:.3f} exceeds {self.arm_symmetry_tolerance:.3f}."

        return True, "Arm symmetry valid."

    def _arm_length(
        self,
        keypoints: dict[str, dict[str, float]],
        shoulder_name: str,
        elbow_name: str,
        wrist_name: str,
    ) -> float | None:
        """Compute polyline arm length from shoulder->elbow->wrist."""
        shoulder = keypoints.get(shoulder_name)
        elbow = keypoints.get(elbow_name)
        wrist = keypoints.get(wrist_name)

        if not shoulder or not elbow or not wrist:
            return None

        if shoulder["conf"] <= 0.0 or elbow["conf"] <= 0.0 or wrist["conf"] <= 0.0:
            return None

        return self._distance(shoulder, elbow) + self._distance(elbow, wrist)

    def _distance(self, p1: dict[str, float], p2: dict[str, float]) -> float:
        """Euclidean distance between two keypoint records."""
        return math.hypot(p1["x"] - p2["x"], p1["y"] - p2["y"])
