"""Data ingestion module for video streams and image batches."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Literal, TypedDict

import cv2
import numpy as np

from src.ingestion.preprocess import normalize_resolution, preprocess_frame

try:
    import ffmpeg
except ImportError:  # pragma: no cover - exercised via runtime check
    ffmpeg = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

IngestMode = Literal["dataset", "realtime", "batch"]
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class FrameMetadata(TypedDict):
    """Metadata emitted alongside each ingested frame."""

    frame_idx: int
    timestamp_ms: int
    source_id: str
    session_id: str


@dataclass(frozen=True)
class IngestedFrame:
    """Container for ingested frame data and metadata."""

    frame: np.ndarray
    metadata: FrameMetadata


class DataIngestor:
    """Read source media and yield preprocessed frames with metadata."""

    def __init__(
        self,
        source: str,
        mode: IngestMode,
        session_id: str,
        target_resolution: tuple[int, int] = (640, 480),
        dataset_fps: int = 2,
        realtime_fps: int = 15,
    ) -> None:
        """Initialize ingestor configuration.

        Args:
            source: Local path or stream URL.
            mode: Ingestion mode (dataset/realtime/batch).
            session_id: Session identifier attached to output metadata.
            target_resolution: Output frame size as (width, height).
            dataset_fps: FPS for dataset video extraction.
            realtime_fps: FPS for realtime stream extraction.
        """
        if mode not in {"dataset", "realtime", "batch"}:
            raise ValueError(f"Unsupported ingestion mode: {mode}")

        if target_resolution[0] <= 0 or target_resolution[1] <= 0:
            raise ValueError("Target resolution must contain positive dimensions.")

        if dataset_fps <= 0 or realtime_fps <= 0:
            raise ValueError("FPS values must be positive integers.")

        self.source = source
        self.mode = mode
        self.session_id = session_id
        self.target_resolution = target_resolution
        self.dataset_fps = dataset_fps
        self.realtime_fps = realtime_fps

    def __iter__(self) -> Generator[IngestedFrame, None, None]:
        """Yield ingested frames when class is used as an iterator."""
        yield from self.iter_frames()

    def iter_frames(self) -> Generator[IngestedFrame, None, None]:
        """Yield preprocessed frames with required metadata."""
        if self.mode == "batch":
            yield from self._iter_image_batch()
            return

        yield from self._iter_video_like_source()

    def _iter_image_batch(self) -> Generator[IngestedFrame, None, None]:
        """Load and preprocess images from a local directory (or a single image file)."""
        source_path = Path(self.source)
        if not source_path.exists():
            raise FileNotFoundError(f"Batch source not found: {source_path}")

        if source_path.is_file():
            if source_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                raise FileNotFoundError(f"Batch source is not an image file: {source_path}")
            frame = cv2.imread(str(source_path))
            if frame is None:
                raise FileNotFoundError(f"Unreadable image file: {source_path}")
            yield self._to_record(frame=frame, frame_idx=0, fps=self.dataset_fps)
            return

        if not source_path.is_dir():
            raise FileNotFoundError(f"Batch source directory not found: {source_path}")

        frame_paths = sorted(
            path for path in source_path.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )

        if not frame_paths:
            LOGGER.warning("No image files found in batch source: %s", source_path)
            return

        for frame_idx, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                LOGGER.warning("Skipping unreadable image: %s", frame_path)
                continue

            yield self._to_record(frame=frame, frame_idx=frame_idx, fps=self.dataset_fps)

    def _iter_video_like_source(self) -> Generator[IngestedFrame, None, None]:
        """Decode frames from video-like sources using ffmpeg-python."""
        if ffmpeg is None:
            raise RuntimeError("ffmpeg-python is not installed. Run `pip install ffmpeg-python`.")

        target_fps = self.dataset_fps if self.mode == "dataset" else self.realtime_fps

        try:
            probe = ffmpeg.probe(self.source)
            video_stream = next(
                (stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"),
                None,
            )

            if video_stream is None:
                raise RuntimeError(f"No video stream found in source: {self.source}")

            width = int(video_stream["width"])
            height = int(video_stream["height"])
            if width <= 0 or height <= 0:
                raise RuntimeError(f"Invalid stream dimensions detected for source: {self.source}")

            stream = (
                ffmpeg.input(self.source)
                .filter("fps", fps=target_fps)
                .output("pipe:", format="rawvideo", pix_fmt="bgr24")
            )
            out_bytes, _ = stream.run(capture_stdout=True, capture_stderr=True)

        except FileNotFoundError:
            LOGGER.warning(
                "ffmpeg executable not found; falling back to OpenCV VideoCapture. "
                "Install ffmpeg for faster, more reliable decoding."
            )
            yield from self._iter_video_with_opencv(target_fps=target_fps)
            return
        except ffmpeg.Error as exc:  # type: ignore[attr-defined]
            stderr = exc.stderr.decode("utf-8", errors="ignore") if getattr(exc, "stderr", None) else ""
            raise RuntimeError(f"ffmpeg failed to decode source {self.source}: {stderr}") from exc

        frame_size = width * height * 3
        if frame_size <= 0:
            raise RuntimeError("Computed invalid frame size during ffmpeg decode.")

        if not out_bytes:
            LOGGER.warning("No frame bytes decoded from source: %s", self.source)
            return

        frame_count = len(out_bytes) // frame_size
        for frame_idx in range(frame_count):
            start = frame_idx * frame_size
            end = start + frame_size
            frame_bytes = out_bytes[start:end]
            if len(frame_bytes) != frame_size:
                LOGGER.warning("Skipping partial frame %d from source %s", frame_idx, self.source)
                continue

            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3)).copy()
            yield self._to_record(frame=frame, frame_idx=frame_idx, fps=target_fps)

    def _iter_video_with_opencv(self, target_fps: int) -> Generator[IngestedFrame, None, None]:
        """Best-effort fallback decoder for environments without ffmpeg binary."""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video source: {self.source}")

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if not native_fps or native_fps <= 0:
            native_fps = float(target_fps)

        stride = max(1, int(round(native_fps / float(target_fps))))
        frame_idx = 0
        sampled_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if frame_idx % stride == 0:
                yield self._to_record(frame=frame, frame_idx=sampled_idx, fps=target_fps)
                sampled_idx += 1

            frame_idx += 1

        cap.release()

    def _to_record(self, frame: np.ndarray, frame_idx: int, fps: int) -> IngestedFrame:
        """Preprocess frame and attach standardized metadata."""
        processed = normalize_resolution(preprocess_frame(frame), target=self.target_resolution)

        metadata: FrameMetadata = {
            "frame_idx": frame_idx,
            "timestamp_ms": int((frame_idx / float(fps)) * 1000),
            "source_id": self._build_source_id(),
            "session_id": self.session_id,
        }

        return IngestedFrame(frame=processed, metadata=metadata)

    def _build_source_id(self) -> str:
        """Create a stable source identifier from input source."""
        if self.source.startswith(("http://", "https://", "rtsp://")):
            cleaned = self.source.split("://", maxsplit=1)[-1]
            return cleaned.replace("/", "_")

        path = Path(self.source)
        return path.stem if path.stem else path.name
