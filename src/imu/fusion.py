"""IMU fusion and auto-labeling utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class AutoLabel:
    """Auto-generated label from IMU signal analysis."""

    label_type: str
    confidence: float
    start_ms: int
    end_ms: int
    metadata: dict[str, Any]


class ComplementaryFilter:
    """Estimate pitch/roll from accelerometer readings at variable sample rates."""

    def __init__(self, alpha: float = 0.98, reference_dt_s: float = 0.02) -> None:
        """Initialize filter state.

        Args:
            alpha: Baseline smoothing factor in range (0, 1).
            reference_dt_s: Reference timestep in seconds for adaptive alpha.
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in range (0, 1).")
        if reference_dt_s <= 0.0:
            raise ValueError("reference_dt_s must be positive.")

        self.alpha = alpha
        self.reference_dt_s = reference_dt_s
        self.pitch_deg = 0.0
        self.roll_deg = 0.0
        self._last_timestamp_ms: int | None = None

    def reset(self) -> None:
        """Reset internal orientation state."""
        self.pitch_deg = 0.0
        self.roll_deg = 0.0
        self._last_timestamp_ms = None

    def update(self, ax: float, ay: float, az: float, timestamp_ms: int) -> tuple[float, float]:
        """Update filter state and return (pitch_deg, roll_deg)."""
        accel_pitch_deg, accel_roll_deg = self._tilt_from_accel(ax=ax, ay=ay, az=az)

        if self._last_timestamp_ms is None:
            self.pitch_deg = accel_pitch_deg
            self.roll_deg = accel_roll_deg
            self._last_timestamp_ms = timestamp_ms
            return self.pitch_deg, self.roll_deg

        dt_s = max((timestamp_ms - self._last_timestamp_ms) / 1000.0, 0.0)
        alpha_eff = self._adaptive_alpha(dt_s)

        self.pitch_deg = alpha_eff * self.pitch_deg + (1.0 - alpha_eff) * accel_pitch_deg
        self.roll_deg = alpha_eff * self.roll_deg + (1.0 - alpha_eff) * accel_roll_deg
        self._last_timestamp_ms = timestamp_ms

        return self.pitch_deg, self.roll_deg

    def process(self, readings: Sequence[dict[str, Any]]) -> list[dict[str, float | int]]:
        """Process a reading buffer and return orientation sequence."""
        if not readings:
            return []

        output: list[dict[str, float | int]] = []
        for reading in sorted(readings, key=lambda item: int(item["timestamp_ms"])):
            pitch_deg, roll_deg = self.update(
                ax=float(reading["ax"]),
                ay=float(reading["ay"]),
                az=float(reading["az"]),
                timestamp_ms=int(reading["timestamp_ms"]),
            )
            output.append(
                {
                    "timestamp_ms": int(reading["timestamp_ms"]),
                    "pitch_deg": float(pitch_deg),
                    "roll_deg": float(roll_deg),
                }
            )

        return output

    def _adaptive_alpha(self, dt_s: float) -> float:
        """Adjust alpha for variable sampling rates using exponential scaling."""
        if dt_s <= 0.0:
            return self.alpha

        ratio = dt_s / self.reference_dt_s
        return float(np.clip(self.alpha ** ratio, 0.0, 0.999999))

    def _tilt_from_accel(self, ax: float, ay: float, az: float) -> tuple[float, float]:
        """Compute pitch/roll from accelerometer vector."""
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm <= 1e-9:
            return self.pitch_deg, self.roll_deg

        ax_n = ax / norm
        ay_n = ay / norm
        az_n = az / norm

        pitch_rad = math.atan2(ax_n, math.sqrt(ay_n * ay_n + az_n * az_n))
        roll_rad = math.atan2(ay_n, max(1e-9, az_n))

        return math.degrees(pitch_rad), math.degrees(roll_rad)


class IMUAutoLabeler:
    """Generate physics-based labels from IMU readings."""

    def __init__(self, filter_alpha: float = 0.98) -> None:
        """Initialize detector thresholds and orientation estimator."""
        self.orientation_filter = ComplementaryFilter(alpha=filter_alpha)

        self.fall_impact_threshold_g = 2.5
        self.fall_impact_min_duration_ms = 50
        self.fall_post_impact_low_g = 0.5
        self.fall_post_impact_min_duration_ms = 200

        self.impact_threshold_g = 3.0

        self.rapid_sma_threshold = 2.0
        self.static_sma_threshold = 0.3

        self.bend_pitch_threshold_deg = 35.0
        self.bend_min_duration_ms = 3000

    def detect_labels(
        self,
        imu_readings: Sequence[dict[str, Any]],
        video_timestamps_ms: Sequence[int] | None = None,
    ) -> list[AutoLabel]:
        """Detect labels from IMU data, optionally aligned to video timestamps.

        Args:
            imu_readings: Sequence of IMU dictionaries with ax, ay, az, timestamp_ms.
            video_timestamps_ms: Optional target frame timestamps for interpolation.

        Returns:
            Sorted list of AutoLabel entries.
        """
        normalized = self._normalize_readings(imu_readings)
        if not normalized:
            return []

        if video_timestamps_ms:
            working = self.interpolate_to_video_timestamps(normalized, video_timestamps_ms)
        else:
            working = normalized

        timestamps = np.array([int(item["timestamp_ms"]) for item in working], dtype=np.int64)
        ax = np.array([float(item["ax"]) for item in working], dtype=np.float64)
        ay = np.array([float(item["ay"]) for item in working], dtype=np.float64)
        az = np.array([float(item["az"]) for item in working], dtype=np.float64)
        magnitude = np.sqrt(ax * ax + ay * ay + az * az)

        labels: list[AutoLabel] = []
        labels.extend(self._detect_fall_events(timestamps, magnitude))
        labels.extend(self._detect_impact_events(timestamps, magnitude))
        labels.extend(self._detect_rapid_movement_events(timestamps, ax, ay, az))
        labels.extend(self._detect_static_posture_events(timestamps, ax, ay, az))

        orientation = self.orientation_filter.process(working)
        pitch_deg = np.array([float(item["pitch_deg"]) for item in orientation], dtype=np.float64)
        labels.extend(self._detect_bending_events(timestamps, pitch_deg))

        labels.sort(key=lambda item: (item.start_ms, item.end_ms, item.label_type))
        return labels

    def interpolate_to_video_timestamps(
        self,
        imu_readings: Sequence[dict[str, Any]],
        video_timestamps_ms: Sequence[int],
    ) -> list[dict[str, float | int]]:
        """Interpolate IMU values onto video frame timestamps using numpy.interp."""
        if not imu_readings or not video_timestamps_ms:
            return []

        source = self._normalize_readings(imu_readings)
        if not source:
            return []

        source_t = np.array([int(item["timestamp_ms"]) for item in source], dtype=np.float64)
        target_t = np.array(sorted(set(int(ts) for ts in video_timestamps_ms)), dtype=np.float64)

        ax_values = np.array([float(item["ax"]) for item in source], dtype=np.float64)
        ay_values = np.array([float(item["ay"]) for item in source], dtype=np.float64)
        az_values = np.array([float(item["az"]) for item in source], dtype=np.float64)

        ax_interp = np.interp(target_t, source_t, ax_values)
        ay_interp = np.interp(target_t, source_t, ay_values)
        az_interp = np.interp(target_t, source_t, az_values)

        return [
            {
                "timestamp_ms": int(target_t[idx]),
                "ax": float(ax_interp[idx]),
                "ay": float(ay_interp[idx]),
                "az": float(az_interp[idx]),
            }
            for idx in range(len(target_t))
        ]

    def _normalize_readings(self, imu_readings: Sequence[dict[str, Any]]) -> list[dict[str, float | int]]:
        """Normalize and sort IMU payload with required fields."""
        normalized: list[dict[str, float | int]] = []
        for reading in imu_readings:
            if not {"ax", "ay", "az", "timestamp_ms"}.issubset(reading.keys()):
                continue
            normalized.append(
                {
                    "timestamp_ms": int(reading["timestamp_ms"]),
                    "ax": float(reading["ax"]),
                    "ay": float(reading["ay"]),
                    "az": float(reading["az"]),
                }
            )

        normalized.sort(key=lambda item: int(item["timestamp_ms"]))
        return normalized

    def _detect_fall_events(self, timestamps: np.ndarray, magnitude: np.ndarray) -> list[AutoLabel]:
        """Detect fall pattern: high acceleration then near-freefall period."""
        labels: list[AutoLabel] = []
        high_segments = self._segments_from_mask(timestamps, magnitude > self.fall_impact_threshold_g)
        low_segments = self._segments_from_mask(timestamps, magnitude < self.fall_post_impact_low_g)

        for high_start, high_end in high_segments:
            high_duration = timestamps[high_end] - timestamps[high_start]
            if high_duration < self.fall_impact_min_duration_ms:
                continue

            for low_start, low_end in low_segments:
                if low_start <= high_end:
                    continue
                low_duration = timestamps[low_end] - timestamps[low_start]
                if low_duration < self.fall_post_impact_min_duration_ms:
                    continue

                peak_g = float(np.max(magnitude[high_start : high_end + 1]))
                low_min_g = float(np.min(magnitude[low_start : low_end + 1]))
                confidence = float(np.clip(0.7 + 0.1 * (peak_g - self.fall_impact_threshold_g) + 0.1, 0.0, 1.0))

                labels.append(
                    AutoLabel(
                        label_type="fall_event",
                        confidence=confidence,
                        start_ms=int(timestamps[high_start]),
                        end_ms=int(timestamps[low_end]),
                        metadata={
                            "peak_g": peak_g,
                            "post_impact_min_g": low_min_g,
                            "impact_duration_ms": int(high_duration),
                            "low_duration_ms": int(low_duration),
                        },
                    )
                )
                break

        return labels

    def _detect_impact_events(self, timestamps: np.ndarray, magnitude: np.ndarray) -> list[AutoLabel]:
        """Detect impact spikes based on absolute acceleration magnitude."""
        labels: list[AutoLabel] = []
        impact_segments = self._segments_from_mask(timestamps, magnitude > self.impact_threshold_g)

        for start_idx, end_idx in impact_segments:
            peak_g = float(np.max(magnitude[start_idx : end_idx + 1]))
            confidence = float(np.clip(0.75 + 0.08 * (peak_g - self.impact_threshold_g), 0.0, 1.0))
            labels.append(
                AutoLabel(
                    label_type="impact_detected",
                    confidence=confidence,
                    start_ms=int(timestamps[start_idx]),
                    end_ms=int(timestamps[end_idx]),
                    metadata={"peak_g": peak_g},
                )
            )

        return labels

    def _detect_rapid_movement_events(
        self,
        timestamps: np.ndarray,
        ax: np.ndarray,
        ay: np.ndarray,
        az: np.ndarray,
    ) -> list[AutoLabel]:
        """Detect rapid movement windows using 1-second SMA threshold."""
        return self._detect_sma_events(
            timestamps=timestamps,
            ax=ax,
            ay=ay,
            az=az,
            threshold=self.rapid_sma_threshold,
            comparator=lambda value, limit: value > limit,
            label_type="rapid_movement",
            min_duration_ms=1000,
        )

    def _detect_static_posture_events(
        self,
        timestamps: np.ndarray,
        ax: np.ndarray,
        ay: np.ndarray,
        az: np.ndarray,
    ) -> list[AutoLabel]:
        """Detect static posture windows using long-duration low-SMA regions."""
        return self._detect_sma_events(
            timestamps=timestamps,
            ax=ax,
            ay=ay,
            az=az,
            threshold=self.static_sma_threshold,
            comparator=lambda value, limit: value < limit,
            label_type="static_posture",
            min_duration_ms=30000,
        )

    def _detect_bending_events(self, timestamps: np.ndarray, pitch_deg: np.ndarray) -> list[AutoLabel]:
        """Detect sustained forward bending from pitch trajectory."""
        labels: list[AutoLabel] = []
        segments = self._segments_from_mask(timestamps, pitch_deg > self.bend_pitch_threshold_deg)

        for start_idx, end_idx in segments:
            duration = int(timestamps[end_idx] - timestamps[start_idx])
            if duration < self.bend_min_duration_ms:
                continue

            peak_pitch = float(np.max(pitch_deg[start_idx : end_idx + 1]))
            confidence = float(np.clip(0.7 + (peak_pitch - self.bend_pitch_threshold_deg) / 45.0, 0.0, 1.0))
            labels.append(
                AutoLabel(
                    label_type="bending_forward",
                    confidence=confidence,
                    start_ms=int(timestamps[start_idx]),
                    end_ms=int(timestamps[end_idx]),
                    metadata={"peak_pitch_deg": peak_pitch, "duration_ms": duration},
                )
            )

        return labels

    def _detect_sma_events(
        self,
        timestamps: np.ndarray,
        ax: np.ndarray,
        ay: np.ndarray,
        az: np.ndarray,
        threshold: float,
        comparator: Any,
        label_type: str,
        min_duration_ms: int,
    ) -> list[AutoLabel]:
        """Detect contiguous windows satisfying SMA comparator condition."""
        if len(timestamps) < 2:
            return []

        signal = np.abs(ax) + np.abs(ay) + np.abs(az)
        window_ms = 1000
        step_ms = 250

        start_t = int(timestamps[0])
        end_t = int(timestamps[-1])
        if end_t - start_t < window_ms:
            return []

        window_starts = np.arange(start_t, end_t - window_ms + 1, step_ms)
        flags = []

        for w_start in window_starts:
            w_end = w_start + window_ms
            mask = (timestamps >= w_start) & (timestamps <= w_end)
            if not np.any(mask):
                flags.append(False)
                continue
            sma = float(np.mean(signal[mask]))
            flags.append(bool(comparator(sma, threshold)))

        labels: list[AutoLabel] = []
        if not flags:
            return labels

        flags_arr = np.array(flags, dtype=bool)
        segment_idx = self._segments_from_boolean(flags_arr)

        for seg_start_idx, seg_end_idx in segment_idx:
            start_ms = int(window_starts[seg_start_idx])
            end_ms = int(window_starts[seg_end_idx] + window_ms)
            duration = end_ms - start_ms
            if duration < min_duration_ms:
                continue

            win_mask = (timestamps >= start_ms) & (timestamps <= end_ms)
            avg_sma = float(np.mean(signal[win_mask])) if np.any(win_mask) else 0.0
            confidence = float(np.clip(0.6 + abs(avg_sma - threshold) / max(threshold, 1e-6), 0.0, 1.0))

            labels.append(
                AutoLabel(
                    label_type=label_type,
                    confidence=confidence,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    metadata={"avg_sma": avg_sma, "duration_ms": duration},
                )
            )

        return labels

    def _segments_from_mask(self, timestamps: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
        """Return index segments where mask is true."""
        if len(mask) == 0:
            return []

        segments: list[tuple[int, int]] = []
        start_idx: int | None = None

        for idx, flag in enumerate(mask):
            if flag and start_idx is None:
                start_idx = idx
            if not flag and start_idx is not None:
                segments.append((start_idx, idx - 1))
                start_idx = None

        if start_idx is not None:
            segments.append((start_idx, len(mask) - 1))

        return segments

    def _segments_from_boolean(self, flags: np.ndarray) -> list[tuple[int, int]]:
        """Return index ranges for contiguous true values in a boolean array."""
        segments: list[tuple[int, int]] = []
        start_idx: int | None = None

        for idx, flag in enumerate(flags):
            if flag and start_idx is None:
                start_idx = idx
            if not flag and start_idx is not None:
                segments.append((start_idx, idx - 1))
                start_idx = None

        if start_idx is not None:
            segments.append((start_idx, len(flags) - 1))

        return segments
