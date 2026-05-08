"""OpenCV drawing utilities for the Tentellect MotionIQ POC.

Handles skeleton overlay, bounding box rendering, badge text,
and risk colour coding on BGR frames.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# ── Colour palette (BGR) ────────────────────────────────────────────────────
COLOUR_GREEN = (0, 210, 80)
COLOUR_ORANGE = (0, 155, 255)
COLOUR_RED = (0, 50, 230)
COLOUR_YELLOW = (0, 220, 220)
COLOUR_WHITE = (255, 255, 255)
COLOUR_DARK = (20, 20, 20)
COLOUR_TEAL = (200, 200, 0)
COLOUR_PURPLE = (220, 80, 180)

# Action class badge colours
ACTION_COLOURS: dict[str, tuple[int, int, int]] = {
    "idle": (120, 120, 120),
    "walk": (200, 180, 0),
    "bend": (0, 165, 255),
    "fall": (0, 30, 220),
    "reach_overhead": (180, 50, 255),
    "danger_posture": (0, 30, 220),
}

# Quality gate badge colours
GATE_COLOURS: dict[str, tuple[int, int, int]] = {
    "AUTO_ACCEPT": (0, 180, 60),
    "REVIEW": (0, 140, 255),
    "DISCARD": (90, 90, 90),
}

# ── COCO-17 skeleton bone pairs ─────────────────────────────────────────────
COCO_BONES: list[tuple[str, str]] = [
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

# Bone colour segments: upper body vs lower body vs face
UPPER_BODY = {
    "left_shoulder", "right_shoulder", "left_elbow", "left_wrist",
    "right_elbow", "right_wrist", "left_hip", "right_hip",
}


def _risk_colour(risk: float) -> tuple[int, int, int]:
    """Return BGR colour based on risk score."""
    if risk < 0.35:
        return COLOUR_GREEN
    if risk < 0.65:
        return COLOUR_ORANGE
    return COLOUR_RED


def _bone_colour(a: str, b: str) -> tuple[int, int, int]:
    """Return BGR bone colour based on body segment."""
    if a in UPPER_BODY or b in UPPER_BODY:
        return COLOUR_TEAL
    return COLOUR_PURPLE


def draw_skeleton(
    frame: np.ndarray,
    keypoints_17: list[dict[str, Any]],
    risk: float = 0.0,
    alpha: float = 0.85,
) -> np.ndarray:
    """Draw COCO-17 keypoints and bone connections on *frame* in-place.

    Args:
        frame: BGR uint8 image.
        keypoints_17: List of keypoint dicts with keys 'name', 'x', 'y', 'conf'.
        risk: Risk score (0–1), used to colour keypoint dots.
        alpha: Opacity of keypoint circles (blended on a copy).

    Returns:
        The annotated frame (same array, modified in-place).
    """
    kp_map: dict[str, tuple[float, float, float]] = {}
    for kp in keypoints_17:
        name = str(kp.get("name", ""))
        conf = float(kp.get("conf", 0.0))
        if conf > 0.0:
            kp_map[name] = (float(kp["x"]), float(kp["y"]), conf)

    overlay = frame.copy()

    # Draw bones
    for a_name, b_name in COCO_BONES:
        if a_name not in kp_map or b_name not in kp_map:
            continue
        ax, ay, ac = kp_map[a_name]
        bx, by, bc = kp_map[b_name]
        if ac < 0.2 or bc < 0.2:
            continue
        colour = _bone_colour(a_name, b_name)
        cv2.line(
            overlay,
            (int(ax), int(ay)),
            (int(bx), int(by)),
            colour,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    # Draw keypoint dots
    dot_colour = _risk_colour(risk)
    for name, (x, y, conf) in kp_map.items():
        if conf < 0.2:
            continue
        radius = 5 if name == "nose" else 4
        cv2.circle(overlay, (int(x), int(y)), radius, dot_colour, -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, (int(x), int(y)), radius, COLOUR_WHITE, 1, lineType=cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_track_box(
    frame: np.ndarray,
    bbox: list[float],
    track_id: str,
    action: str,
    risk: float,
    gate: str,
) -> np.ndarray:
    """Draw bounding box + info badges for one tracked worker.

    Args:
        frame: BGR uint8 image.
        bbox: [x1, y1, x2, y2] in pixels.
        track_id: Worker track identifier string.
        action: Predicted action class string.
        risk: Risk score 0–1.
        gate: Quality gate status string.

    Returns:
        Annotated frame.
    """
    if len(bbox) != 4:
        return frame

    x1, y1, x2, y2 = [int(v) for v in bbox]
    box_colour = _risk_colour(risk)

    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_colour, 2, lineType=cv2.LINE_AA)

    # ── Top badge: track ID + action ─────────────────────────────────────────
    label = f"{track_id} | {action}"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    badge_y = max(y1 - 6, lh + 8)
    cv2.rectangle(
        frame,
        (x1, badge_y - lh - 6),
        (x1 + lw + 10, badge_y + 2),
        ACTION_COLOURS.get(action, COLOUR_DARK),
        -1,
    )
    cv2.putText(
        frame, label,
        (x1 + 5, badge_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOUR_WHITE, 1, lineType=cv2.LINE_AA,
    )

    # ── Bottom-left badge: risk bar ───────────────────────────────────────────
    bar_x, bar_y = x1, y2 + 4
    bar_w, bar_h = max(x2 - x1, 60), 8
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLOUR_DARK, -1)
    fill_w = int(risk * bar_w)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), box_colour, -1)

    risk_label = f"Risk {risk:.2f}"
    cv2.putText(
        frame, risk_label,
        (bar_x, bar_y + bar_h + 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, box_colour, 1, lineType=cv2.LINE_AA,
    )

    # ── Bottom-right badge: gate status ──────────────────────────────────────
    gate_label = gate.replace("_", " ")
    (gw, gh), _ = cv2.getTextSize(gate_label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    gate_colour = GATE_COLOURS.get(gate, COLOUR_DARK)
    gx = x2 - gw - 10
    gy = bar_y + bar_h + 12
    cv2.putText(
        frame, gate_label,
        (gx, gy),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, gate_colour, 1, lineType=cv2.LINE_AA,
    )

    return frame


def draw_hud(
    frame: np.ndarray,
    fps: float,
    frame_idx: int,
    worker_count: int,
    alert_count: int,
) -> np.ndarray:
    """Draw a heads-up display strip at the top of the frame.

    Args:
        frame: BGR uint8 image.
        fps: Current processing FPS.
        frame_idx: Frame counter.
        worker_count: Number of detected workers.
        alert_count: Number of workers above risk threshold.

    Returns:
        Annotated frame.
    """
    h, w = frame.shape[:2]
    strip_h = 30
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, strip_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Left text
    left = f"MotionIQ POC   FPS: {fps:.1f}   Frame: {frame_idx}"
    cv2.putText(frame, left, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOUR_WHITE, 1, lineType=cv2.LINE_AA)

    # Right text
    right = f"Workers: {worker_count}   Alerts: {alert_count}"
    (rw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    alert_col = COLOUR_RED if alert_count > 0 else COLOUR_GREEN
    cv2.putText(frame, right, (w - rw - 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, alert_col, 1, lineType=cv2.LINE_AA)

    # Divider line
    cv2.line(frame, (0, strip_h), (w, strip_h), (60, 60, 60), 1)
    return frame
