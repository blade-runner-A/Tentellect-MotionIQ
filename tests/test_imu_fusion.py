"""Tests for IMU fusion and auto-label generation."""

from __future__ import annotations

import numpy as np

from src.imu.fusion import ComplementaryFilter, IMUAutoLabeler


def test_complementary_filter_handles_variable_sample_rates() -> None:
    filt = ComplementaryFilter(alpha=0.98)

    readings = [
        {"timestamp_ms": 0, "ax": 0.0, "ay": 0.0, "az": 1.0},
        {"timestamp_ms": 20, "ax": 0.0, "ay": 0.0, "az": 1.0},
        {"timestamp_ms": 55, "ax": 0.6, "ay": 0.0, "az": 0.8},
        {"timestamp_ms": 90, "ax": 0.6, "ay": 0.0, "az": 0.8},
        {"timestamp_ms": 130, "ax": 0.6, "ay": 0.0, "az": 0.8},
    ]

    orientation = filt.process(readings)

    assert len(orientation) == 5
    assert abs(float(orientation[1]["pitch_deg"])) < 3.0
    assert float(orientation[-1]["pitch_deg"]) > 3.0


def test_interpolate_to_video_timestamps_uses_numpy_interp() -> None:
    labeler = IMUAutoLabeler()

    imu_readings = [
        {"timestamp_ms": 0, "ax": 0.0, "ay": 0.0, "az": 1.0},
        {"timestamp_ms": 100, "ax": 1.0, "ay": 0.0, "az": 1.0},
        {"timestamp_ms": 200, "ax": 2.0, "ay": 0.0, "az": 1.0},
    ]
    video_ts = [0, 50, 100, 150, 200]

    aligned = labeler.interpolate_to_video_timestamps(imu_readings, video_ts)

    assert [entry["timestamp_ms"] for entry in aligned] == video_ts
    assert np.isclose(float(aligned[1]["ax"]), 0.5)
    assert np.isclose(float(aligned[3]["ax"]), 1.5)


def test_synthetic_fall_signature_triggers_fall_event() -> None:
    labeler = IMUAutoLabeler()

    readings = []
    for timestamp_ms in range(0, 2200, 10):
        ax = 0.0
        ay = 0.0
        az = 1.0

        if 500 <= timestamp_ms < 620:
            az = 3.2
        elif 620 <= timestamp_ms < 920:
            az = 0.2

        readings.append(
            {
                "timestamp_ms": timestamp_ms,
                "ax": ax,
                "ay": ay,
                "az": az,
            }
        )

    video_timestamps = list(range(0, 2200, 33))
    labels = labeler.detect_labels(readings, video_timestamps_ms=video_timestamps)

    fall_labels = [label for label in labels if label.label_type == "fall_event"]

    assert len(fall_labels) >= 1
    assert fall_labels[0].start_ms <= 620
    assert fall_labels[0].end_ms >= 860
