"""Frame preprocessing utilities for the ingestion layer."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Apply CLAHE on luminance channel to improve uneven lighting scenes.

    Args:
        frame: BGR frame as uint8 numpy array.

    Returns:
        Preprocessed BGR frame with equalized local contrast.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected BGR frame with shape (H, W, 3).")

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)

    merged = cv2.merge([l_eq, a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def normalize_resolution(frame: np.ndarray, target: Tuple[int, int] = (640, 480)) -> np.ndarray:
    """Resize frame with letterboxing to avoid aspect-ratio distortion.

    Args:
        frame: BGR frame as uint8 numpy array.
        target: Output size as (width, height).

    Returns:
        A letterboxed frame with exact target size.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected BGR frame with shape (H, W, 3).")

    target_w, target_h = target
    if target_w <= 0 or target_h <= 0:
        raise ValueError("Target resolution must be positive.")

    height, width = frame.shape[:2]
    scale = min(target_w / width, target_h / height)

    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized

    return canvas
