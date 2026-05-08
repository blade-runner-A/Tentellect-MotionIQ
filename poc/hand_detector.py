"""MediaPipe Hands wrapper for egocentric (cap-mounted) camera view.

Detects both hands, extracts a 30-dim feature vector, and draws
annotated landmarks on the frame.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

try:
    import mediapipe as mp
    _MP_HANDS = mp.solutions.hands
    _MP_DRAW = mp.solutions.drawing_utils
    _MP_STYLES = mp.solutions.drawing_styles
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False
    _MP_HANDS = None  # type: ignore
    _MP_DRAW = None  # type: ignore

# Colour palette BGR: left=warm, right=cool
_HAND_COLOURS = {"Left": (60, 140, 255), "Right": (255, 140, 60)}
_CONN_COLOUR = (220, 220, 220)


class HandDetector:
    """Detect and track hands for egocentric view."""

    def __init__(
        self,
        max_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._available = MEDIAPIPE_AVAILABLE
        self._hands = None
        if not self._available:
            LOGGER.warning("MediaPipe unavailable; hand detection disabled.")
            return
        self._hands = _MP_HANDS.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._connections = _MP_HANDS.HAND_CONNECTIONS

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        """Run hand detection and return structured results."""
        h, w = frame_bgr.shape[:2]
        if not self._available or self._hands is None:
            return {"hands": [], "frame_shape": (h, w)}

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        hands: list[dict[str, Any]] = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for lms, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handedness.classification[0].label  # "Left" / "Right"
                conf = float(handedness.classification[0].score)
                points = [
                    {
                        "x": lm.x * w,
                        "y": lm.y * h,
                        "z": float(lm.z),
                    }
                    for lm in lms.landmark
                ]
                hands.append(
                    {
                        "label": label,
                        "confidence": conf,
                        "landmarks": points,
                        "_raw": lms,  # kept for drawing
                    }
                )

        return {"hands": hands, "frame_shape": (h, w)}

    # ── Feature extraction ────────────────────────────────────────────────────

    def extract_features(
        self,
        hands_data: dict[str, Any],
        prev_hands_data: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Build a 30-dim float32 feature vector from hand detections.

        Layout:
            0-9   Left hand  (wrist xy, index-tip xy, thumb-tip xy, openness,
                              pinch, wrist-angle, confidence)
            10-19 Right hand (same layout)
            20    Inter-hand distance
            21    Hands-visible flag (0=none, 0.5=one, 1=both)
            22-23 Left wrist velocity x/y  (vs prev frame)
            24-25 Right wrist velocity x/y
            26-27 Left hand zone (x-zone, y-zone)
            28-29 Right hand zone (x-zone, y-zone)
        """
        features = np.zeros(30, dtype=np.float32)
        hands = hands_data.get("hands", [])
        h, w = hands_data.get("frame_shape", (480, 640))

        left = next((hd for hd in hands if hd["label"] == "Left"), None)
        right = next((hd for hd in hands if hd["label"] == "Right"), None)

        def _fill(hand: dict | None, offset: int) -> None:
            if hand is None:
                return
            lm = hand["landmarks"]
            wrist = lm[0]
            wx, wy = wrist["x"] / max(w, 1), wrist["y"] / max(h, 1)
            features[offset + 0] = wx
            features[offset + 1] = wy
            features[offset + 2] = lm[8]["x"] / max(w, 1)   # index tip
            features[offset + 3] = lm[8]["y"] / max(h, 1)
            features[offset + 4] = lm[4]["x"] / max(w, 1)   # thumb tip
            features[offset + 5] = lm[4]["y"] / max(h, 1)
            tip_ids = [4, 8, 12, 16, 20]
            avg_dist = float(np.mean([
                np.hypot(lm[t]["x"] - lm[0]["x"], lm[t]["y"] - lm[0]["y"])
                for t in tip_ids
            ])) / max(w, 1)
            features[offset + 6] = avg_dist   # openness
            pinch = np.hypot(
                lm[4]["x"] - lm[8]["x"], lm[4]["y"] - lm[8]["y"]
            ) / max(w, 1)
            features[offset + 7] = float(pinch)  # pinch distance
            mid_vec = np.array([lm[9]["x"] - lm[0]["x"], lm[9]["y"] - lm[0]["y"]])
            angle = float(np.degrees(np.arctan2(mid_vec[1], mid_vec[0] + 1e-6)))
            features[offset + 8] = angle / 180.0
            features[offset + 9] = float(hand["confidence"])

        _fill(left, 0)
        _fill(right, 10)

        # Inter-hand
        if left and right:
            lw, rw = left["landmarks"][0], right["landmarks"][0]
            features[20] = float(
                np.hypot(lw["x"] - rw["x"], lw["y"] - rw["y"]) / max(w, 1)
            )
            features[21] = 1.0
        else:
            features[21] = 0.5 if (left or right) else 0.0

        # Velocity vs previous frame
        if prev_hands_data:
            prev_hands = prev_hands_data.get("hands", [])
            prev_left = next((hd for hd in prev_hands if hd["label"] == "Left"), None)
            prev_right = next((hd for hd in prev_hands if hd["label"] == "Right"), None)

            def _vel(cur: dict | None, prev: dict | None, offset: int) -> None:
                if cur and prev:
                    dx = (cur["landmarks"][0]["x"] - prev["landmarks"][0]["x"]) / max(w, 1)
                    dy = (cur["landmarks"][0]["y"] - prev["landmarks"][0]["y"]) / max(h, 1)
                    features[offset] = float(dx)
                    features[offset + 1] = float(dy)

            _vel(left, prev_left, 22)
            _vel(right, prev_right, 24)

        # Zone (divide frame into 3x3 grid)
        def _zone(hand: dict | None, offset: int) -> None:
            if hand:
                wx = hand["landmarks"][0]["x"] / max(w, 1)
                wy = hand["landmarks"][0]["y"] / max(h, 1)
                features[offset] = float(min(int(wx * 3), 2))
                features[offset + 1] = float(min(int(wy * 3), 2))

        _zone(left, 26)
        _zone(right, 28)

        return features

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray, hands_data: dict[str, Any]) -> np.ndarray:
        """Draw hand landmarks and connections on frame in-place."""
        if not self._available or _MP_DRAW is None:
            return frame

        for hand in hands_data.get("hands", []):
            raw = hand.get("_raw")
            if raw is None:
                continue
            label = hand["label"]
            dot_col = _HAND_COLOURS.get(label, (200, 200, 200))

            _MP_DRAW.draw_landmarks(
                frame,
                raw,
                self._connections,
                _MP_DRAW.DrawingSpec(color=dot_col, thickness=2, circle_radius=5),
                _MP_DRAW.DrawingSpec(color=_CONN_COLOUR, thickness=2),
            )

            # Label near wrist
            wrist = hand["landmarks"][0]
            cv2.putText(
                frame,
                f"{label} ✋ {hand['confidence']:.2f}",
                (int(wrist["x"]), max(int(wrist["y"]) - 18, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                dot_col,
                1,
                lineType=cv2.LINE_AA,
            )

        return frame

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
