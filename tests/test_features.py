"""Tests for 20-dimensional feature extraction."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from src.features.extractor import FeatureExtractor


def _base_skeleton(frame_idx: int, timestamp_ms: int, x_shift: float = 0.0) -> dict:
    return {
        "session_id": "session_features",
        "frame_idx": frame_idx,
        "timestamp_ms": timestamp_ms,
        "track_id": "worker_001",
        "frame_width": 640,
        "frame_height": 480,
        "bbox": [100.0 + x_shift, 60.0, 180.0 + x_shift, 220.0],
        "keypoints_17": [
            {"name": "nose", "x": 140.0 + x_shift, "y": 72.0, "conf": 0.95},
            {"name": "left_shoulder", "x": 120.0 + x_shift, "y": 100.0, "conf": 0.93},
            {"name": "right_shoulder", "x": 160.0 + x_shift, "y": 100.0, "conf": 0.93},
            {"name": "left_elbow", "x": 112.0 + x_shift, "y": 132.0, "conf": 0.90},
            {"name": "right_elbow", "x": 168.0 + x_shift, "y": 132.0, "conf": 0.90},
            {"name": "left_wrist", "x": 106.0 + x_shift, "y": 162.0, "conf": 0.88},
            {"name": "right_wrist", "x": 174.0 + x_shift, "y": 162.0, "conf": 0.88},
            {"name": "left_hip", "x": 126.0 + x_shift, "y": 160.0, "conf": 0.92},
            {"name": "right_hip", "x": 154.0 + x_shift, "y": 160.0, "conf": 0.92},
            {"name": "left_knee", "x": 126.0 + x_shift, "y": 190.0, "conf": 0.90},
            {"name": "right_knee", "x": 154.0 + x_shift, "y": 190.0, "conf": 0.90},
            {"name": "left_ankle", "x": 126.0 + x_shift, "y": 218.0, "conf": 0.89},
            {"name": "right_ankle", "x": 154.0 + x_shift, "y": 218.0, "conf": 0.89},
        ],
        "ppe": {"helmet": 0.9, "vest": 0.8, "gloves": 0.6, "glasses": 0.2},
    }


def test_feature_vector_has_expected_schema_and_dtype() -> None:
    extractor = FeatureExtractor()
    window = [_base_skeleton(frame_idx=0, timestamp_ms=0)]

    result = extractor.extract(window)

    vector = result["vector"]
    assert vector.shape == (20,)
    assert vector.dtype == np.float32
    assert len(result["feature_names"]) == 20
    assert result["feature_names"][0] == "torso_angle_deg"


def test_temporal_features_capture_motion() -> None:
    extractor = FeatureExtractor()
    window = [
        _base_skeleton(frame_idx=0, timestamp_ms=0, x_shift=0.0),
        _base_skeleton(frame_idx=1, timestamp_ms=100, x_shift=12.0),
        _base_skeleton(frame_idx=2, timestamp_ms=200, x_shift=30.0),
    ]

    result = extractor.extract(window)
    vector = result["vector"]
    name_to_index = {name: idx for idx, name in enumerate(result["feature_names"])}

    assert float(vector[name_to_index["velocity_px_per_s"]]) > 0.0
    assert float(vector[name_to_index["acceleration_px_per_s2"]]) > 0.0
    assert np.isclose(float(vector[name_to_index["dwell_time_s"]]), 0.2, atol=1e-4)


def test_imu_features_are_zeroed_when_unavailable() -> None:
    extractor = FeatureExtractor()
    window = [_base_skeleton(frame_idx=0, timestamp_ms=0)]

    result = extractor.extract(window, imu_features=None)
    vector = result["vector"]
    name_to_index = {name: idx for idx, name in enumerate(result["feature_names"])}

    assert np.isclose(float(vector[name_to_index["torso_pitch_deg"]]), 0.0)
    assert np.isclose(float(vector[name_to_index["torso_roll_deg"]]), 0.0)
    assert np.isclose(float(vector[name_to_index["fall_flag"]]), 0.0)
    assert np.isclose(float(vector[name_to_index["imu_available"]]), 0.0)
    assert result["imu_available"] is False


def test_imu_features_populated_when_available() -> None:
    extractor = FeatureExtractor()
    window = [_base_skeleton(frame_idx=0, timestamp_ms=0)]

    imu_features = {"pitch_deg": 18.5, "roll_deg": -4.0, "fall_flag": True}
    result = extractor.extract(window, imu_features=imu_features)

    vector = result["vector"]
    name_to_index = {name: idx for idx, name in enumerate(result["feature_names"])}

    assert np.isclose(float(vector[name_to_index["torso_pitch_deg"]]), 18.5)
    assert np.isclose(float(vector[name_to_index["torso_roll_deg"]]), -4.0)
    assert np.isclose(float(vector[name_to_index["fall_flag"]]), 1.0)
    assert np.isclose(float(vector[name_to_index["imu_available"]]), 1.0)
    assert result["imu_available"] is True


def test_zone_id_changes_with_horizontal_position() -> None:
    extractor = FeatureExtractor(default_frame_width=640, default_frame_height=480)

    left = _base_skeleton(frame_idx=0, timestamp_ms=0, x_shift=-80.0)
    center = _base_skeleton(frame_idx=1, timestamp_ms=100, x_shift=120.0)
    right = _base_skeleton(frame_idx=2, timestamp_ms=200, x_shift=300.0)

    left_vector = extractor.extract([left])["vector"]
    center_vector = extractor.extract([center])["vector"]
    right_vector = extractor.extract([right])["vector"]

    zone_idx = extractor.FEATURE_NAMES.index("zone_id")

    assert float(left_vector[zone_idx]) == 0.0
    assert float(center_vector[zone_idx]) == 1.0
    assert float(right_vector[zone_idx]) == 2.0
