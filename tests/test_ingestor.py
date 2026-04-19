"""Tests for ingestion DataIngestor module."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.ingestion.ingestor import DataIngestor


def _write_image(path: Path, value: int) -> None:
    frame = np.full((32, 48, 3), value, dtype=np.uint8)
    ok = cv2.imwrite(str(path), frame)
    if not ok:
        raise RuntimeError(f"Failed to write test image: {path}")


def test_batch_mode_yields_frames_with_metadata(tmp_path: Path) -> None:
    image_dir = tmp_path / "batch_frames"
    image_dir.mkdir(parents=True)

    _write_image(image_dir / "001.jpg", 32)
    _write_image(image_dir / "002.jpg", 96)
    _write_image(image_dir / "003.jpg", 160)

    ingestor = DataIngestor(
        source=str(image_dir),
        mode="batch",
        session_id="session_batch",
        target_resolution=(64, 64),
        dataset_fps=2,
    )

    records = list(ingestor.iter_frames())

    assert len(records) == 3
    assert records[0].frame.shape == (64, 64, 3)
    assert records[0].metadata["frame_idx"] == 0
    assert records[1].metadata["timestamp_ms"] == 500
    assert all(record.metadata["source_id"] == "batch_frames" for record in records)
    assert all(record.metadata["session_id"] == "session_batch" for record in records)


def test_batch_mode_raises_for_missing_directory(tmp_path: Path) -> None:
    missing_dir = tmp_path / "does_not_exist"

    ingestor = DataIngestor(source=str(missing_dir), mode="batch", session_id="session_missing")

    with pytest.raises(FileNotFoundError):
        list(ingestor.iter_frames())


def test_video_mode_decodes_frames_via_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.ingestion.ingestor as ingestor_module

    width = 4
    height = 3
    frame_count = 5

    raw_payload = b"".join(
        np.full((height, width, 3), fill_value=i * 10, dtype=np.uint8).tobytes() for i in range(frame_count)
    )

    class FakeFFmpegStream:
        def filter(self, *_args, **_kwargs):
            return self

        def output(self, *_args, **_kwargs):
            return self

        def run(self, capture_stdout: bool, capture_stderr: bool) -> tuple[bytes, bytes]:
            assert capture_stdout is True
            assert capture_stderr is True
            return raw_payload, b""

    class FakeFFmpegModule:
        class Error(Exception):
            def __init__(self, stderr: bytes = b"") -> None:
                self.stderr = stderr
                super().__init__(stderr.decode("utf-8", errors="ignore"))

        @staticmethod
        def probe(_source: str) -> dict[str, object]:
            return {"streams": [{"codec_type": "video", "width": width, "height": height}]}

        @staticmethod
        def input(_source: str) -> FakeFFmpegStream:
            return FakeFFmpegStream()

    monkeypatch.setattr(ingestor_module, "ffmpeg", FakeFFmpegModule)

    ingestor = DataIngestor(
        source="sample_video.mp4",
        mode="dataset",
        session_id="session_video",
        target_resolution=(4, 3),
        dataset_fps=2,
    )

    records = list(ingestor.iter_frames())

    assert len(records) == frame_count
    assert records[0].metadata["frame_idx"] == 0
    assert records[1].metadata["timestamp_ms"] == 500
    assert records[-1].metadata["frame_idx"] == frame_count - 1
    assert records[0].metadata["source_id"] == "sample_video"


def test_invalid_mode_raises_value_error() -> None:
    with pytest.raises(ValueError):
        DataIngestor(source="dummy", mode="bad_mode", session_id="session_invalid")  # type: ignore[arg-type]
