"""Tests for quality gate edge cases and routing behavior."""

from __future__ import annotations

from src.annotation.quality_gates import GateStatus, QualityGate


def _worker_template(confidence: float = 0.95) -> dict:
    keypoints = [
        {"name": "nose", "x": 50.0, "y": 20.0, "conf": confidence},
        {"name": "left_eye", "x": 45.0, "y": 18.0, "conf": confidence},
        {"name": "right_eye", "x": 55.0, "y": 18.0, "conf": confidence},
        {"name": "left_ear", "x": 42.0, "y": 20.0, "conf": confidence},
        {"name": "right_ear", "x": 58.0, "y": 20.0, "conf": confidence},
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
        "detection_confidence": 0.95,
        "keypoints_17": keypoints,
    }


def test_fully_occluded_worker_routes_to_review() -> None:
    gate = QualityGate()

    worker = _worker_template(confidence=0.0)
    worker["detection_confidence"] = 0.92

    result = gate.evaluate(worker, frame_width=160, frame_height=160)

    assert result.status == GateStatus.REVIEW
    assert "Visible keypoints" in result.reason


def test_multiple_workers_aggregates_to_review_when_any_worker_needs_review() -> None:
    gate = QualityGate()

    auto_worker = _worker_template(confidence=0.95)
    review_worker = _worker_template(confidence=0.70)

    result = gate.evaluate([auto_worker, review_worker], frame_width=160, frame_height=160)

    assert result.status == GateStatus.REVIEW
    assert "G2 in review band" in result.reason


def test_headless_detection_routes_to_review() -> None:
    gate = QualityGate()

    worker = _worker_template(confidence=0.9)
    for point in worker["keypoints_17"]:
        if point["name"] in {"nose", "left_eye", "right_eye", "left_ear", "right_ear"}:
            point["conf"] = 0.0

    result = gate.evaluate(worker, frame_width=160, frame_height=160)

    assert result.status == GateStatus.REVIEW
    assert "headless" in result.reason.lower()


def test_high_confidence_plausible_pose_auto_accepts() -> None:
    gate = QualityGate()

    worker = _worker_template(confidence=0.95)

    result = gate.evaluate(worker, frame_width=160, frame_height=160)

    assert result.status == GateStatus.AUTO_ACCEPT
    assert "auto-accept" in result.reason
