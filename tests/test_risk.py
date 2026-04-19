"""Tests for risk training data helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.training.train_risk import compute_ece, load_feature_matrix


def test_load_feature_matrix_from_npy_21col(tmp_path: Path) -> None:
    data = np.array(
        [
            [*([0.1] * 20), 0],
            [*([0.2] * 20), 1],
            [*([0.3] * 20), 0],
        ],
        dtype=np.float32,
    )
    np.save(tmp_path / "features.npy", data)

    x_matrix, y_vector = load_feature_matrix(tmp_path)

    assert x_matrix.shape == (3, 20)
    assert y_vector.shape == (3,)
    assert y_vector.tolist() == [0, 1, 0]


def test_load_feature_matrix_from_npz_xy(tmp_path: Path) -> None:
    x_matrix = np.array([[0.0] * 20, [1.0] * 20], dtype=np.float32)
    y_vector = np.array([0, 1], dtype=np.int32)
    np.savez(tmp_path / "block.npz", X=x_matrix, y=y_vector)

    x_out, y_out = load_feature_matrix(tmp_path)

    assert x_out.shape == (2, 20)
    assert y_out.tolist() == [0, 1]


def test_compute_ece_distinguishes_well_and_poor_calibration() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=np.int32)
    y_good = np.array([0.05, 0.10, 0.90, 0.95], dtype=np.float32)
    y_bad = np.array([0.90, 0.85, 0.20, 0.15], dtype=np.float32)

    ece_good = compute_ece(y_true, y_good, n_bins=4)
    ece_bad = compute_ece(y_true, y_bad, n_bins=4)

    assert ece_good < ece_bad
    assert ece_good < 0.2
