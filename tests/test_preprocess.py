"""Tests for ingestion preprocessing utilities."""

from __future__ import annotations

import numpy as np

from src.ingestion.preprocess import normalize_resolution, preprocess_frame


def test_preprocess_frame_preserves_shape_and_dtype() -> None:
    frame = np.random.randint(0, 256, size=(240, 320, 3), dtype=np.uint8)

    processed = preprocess_frame(frame)

    assert processed.shape == frame.shape
    assert processed.dtype == frame.dtype


def test_normalize_resolution_letterboxes_without_distortion() -> None:
    frame = np.full((300, 500, 3), 255, dtype=np.uint8)

    normalized = normalize_resolution(frame, target=(640, 480))

    assert normalized.shape == (480, 640, 3)
    assert int(normalized[0, 0, 0]) == 114
    assert int(normalized[240, 320, 0]) >= 250


def test_preprocess_frame_rejects_non_rgb_like_input() -> None:
    invalid = np.random.randint(0, 256, size=(240, 320), dtype=np.uint8)

    try:
        preprocess_frame(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid frame shape")
