"""Tests for skeleton extractor selection and output schema."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.skeleton.extractor import SkeletonCandidate, SkeletonExtractor


def _make_keypoints(count: int, conf: float) -> list[dict[str, float | str]]:
    return [
        {
            "name": f"kp_{idx}",
            "x": float(10 + idx),
            "y": float(20 + idx),
            "conf": float(conf),
        }
        for idx in range(count)
    ]


def _make_candidate(conf: float, bbox: list[float], include_33: bool = False) -> SkeletonCandidate:
    return SkeletonCandidate(
        bbox=bbox,
        detection_confidence=conf,
        keypoints_17=_make_keypoints(17, conf),
        keypoints_33=_make_keypoints(33, conf) if include_33 else [],
        world_3d=[],
        mean_confidence=conf,
    )


def _make_extractor() -> SkeletonExtractor:
    return SkeletonExtractor({"device": "cpu", "ensemble_conf_threshold": 0.7})


def test_single_person_prefers_mediapipe(monkeypatch: Any) -> None:
    extractor = _make_extractor()

    yolo_candidate = _make_candidate(conf=0.88, bbox=[8.0, 8.0, 40.0, 40.0])
    mp_candidate = _make_candidate(conf=0.92, bbox=[10.0, 12.0, 42.0, 44.0], include_33=True)

    monkeypatch.setattr(extractor, "_run_yolo_detection", lambda _frame: [yolo_candidate])
    monkeypatch.setattr(extractor, "_run_mediapipe_detection", lambda _frame: [mp_candidate])

    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    results = extractor.extract(frame, {"frame_idx": 3, "timestamp_ms": 1500, "session_id": "session_one"})

    assert len(results) == 1
    assert results[0]["frame_idx"] == 3
    assert results[0]["session_id"] == "session_one"
    assert len(results[0]["keypoints_33"]) == 33
    assert results[0]["track_id"].startswith("worker_")


def test_multi_person_prefers_yolo(monkeypatch: Any) -> None:
    extractor = _make_extractor()

    yolo_candidates = [
        _make_candidate(conf=0.9, bbox=[4.0, 4.0, 32.0, 48.0]),
        _make_candidate(conf=0.86, bbox=[40.0, 10.0, 70.0, 60.0]),
    ]

    monkeypatch.setattr(extractor, "_run_yolo_detection", lambda _frame: yolo_candidates)
    monkeypatch.setattr(extractor, "_run_mediapipe_detection", lambda _frame: [_make_candidate(0.95, [5.0, 5.0, 30.0, 50.0], True)])

    frame = np.zeros((96, 128, 3), dtype=np.uint8)
    results = extractor.extract(frame, {"frame_idx": 5, "timestamp_ms": 2500, "session_id": "session_two"})

    assert len(results) == 2
    assert results[0]["bbox"] == [4.0, 4.0, 32.0, 48.0]
    assert results[1]["bbox"] == [40.0, 10.0, 70.0, 60.0]


def test_low_confidence_triggers_ensemble(monkeypatch: Any) -> None:
    extractor = _make_extractor()

    yolo_candidates = [
        _make_candidate(conf=0.52, bbox=[8.0, 8.0, 40.0, 40.0]),
        _make_candidate(conf=0.56, bbox=[42.0, 10.0, 70.0, 44.0]),
    ]
    mp_candidates = [_make_candidate(conf=0.91, bbox=[9.0, 9.0, 41.0, 41.0], include_33=True)]

    called = {"value": False}

    def _fake_ensemble(
        first: list[SkeletonCandidate],
        second: list[SkeletonCandidate],
    ) -> list[SkeletonCandidate]:
        called["value"] = True
        assert first == yolo_candidates
        assert second == mp_candidates
        return second

    monkeypatch.setattr(extractor, "_run_yolo_detection", lambda _frame: yolo_candidates)
    monkeypatch.setattr(extractor, "_run_mediapipe_detection", lambda _frame: mp_candidates)
    monkeypatch.setattr(extractor, "_ensemble_detections", _fake_ensemble)

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    results = extractor.extract(frame, {"frame_idx": 1, "timestamp_ms": 200, "session_id": "session_three"})

    assert called["value"] is True
    assert len(results) == 1
    assert abs(float(results[0]["mean_confidence"]) - 0.91) < 1e-6
