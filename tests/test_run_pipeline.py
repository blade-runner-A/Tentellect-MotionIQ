"""Tests for run_pipeline helper behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.run_pipeline import PipelineRunner, infer_mode


class _FakeIngestedFrame:
    def __init__(self, frame: np.ndarray, metadata: dict) -> None:
        self.frame = frame
        self.metadata = metadata


class _FakeIngestor:
    def __init__(self) -> None:
        self._records = [
            _FakeIngestedFrame(
                frame=np.zeros((64, 64, 3), dtype=np.uint8),
                metadata={"frame_idx": 0, "timestamp_ms": 0, "source_id": "s", "session_id": "session_test"},
            ),
            _FakeIngestedFrame(
                frame=np.zeros((64, 64, 3), dtype=np.uint8),
                metadata={"frame_idx": 1, "timestamp_ms": 100, "source_id": "s", "session_id": "session_test"},
            ),
        ]

    def iter_frames(self):
        for item in self._records:
            yield item


class _FakeExtractor:
    def extract(self, frame: np.ndarray, metadata: dict):
        return [
            {
                "session_id": metadata["session_id"],
                "frame_idx": metadata["frame_idx"],
                "timestamp_ms": metadata["timestamp_ms"],
                "track_id": "worker_001",
                "bbox": [10.0, 10.0, 30.0, 40.0],
                "detection_confidence": 0.95,
                "keypoints_17": [
                    {"name": "nose", "x": 20.0, "y": 12.0, "conf": 0.9},
                    {"name": "left_shoulder", "x": 16.0, "y": 20.0, "conf": 0.9},
                    {"name": "right_shoulder", "x": 24.0, "y": 20.0, "conf": 0.9},
                    {"name": "left_hip", "x": 17.0, "y": 30.0, "conf": 0.9},
                    {"name": "right_hip", "x": 23.0, "y": 30.0, "conf": 0.9},
                    {"name": "left_elbow", "x": 14.0, "y": 24.0, "conf": 0.9},
                    {"name": "right_elbow", "x": 26.0, "y": 24.0, "conf": 0.9},
                    {"name": "left_wrist", "x": 12.0, "y": 28.0, "conf": 0.9},
                    {"name": "right_wrist", "x": 28.0, "y": 28.0, "conf": 0.9},
                    {"name": "left_knee", "x": 17.0, "y": 36.0, "conf": 0.9},
                    {"name": "right_knee", "x": 23.0, "y": 36.0, "conf": 0.9},
                    {"name": "left_ankle", "x": 17.0, "y": 39.0, "conf": 0.9},
                    {"name": "right_ankle", "x": 23.0, "y": 39.0, "conf": 0.9},
                ],
                "keypoints_33": [],
                "world_3d": [],
                "mean_confidence": 0.9,
                "ppe": {"helmet": 0.9, "vest": 0.8, "gloves": 0.7, "glasses": 0.2},
            }
        ]


class _FakeGateResult:
    def __init__(self, status):
        self.status = status
        self.reason = "ok"


class _FakeQualityGate:
    def evaluate(self, keypoint_payload, frame_width: int, frame_height: int):
        from src.annotation.quality_gates import GateStatus

        return _FakeGateResult(GateStatus.AUTO_ACCEPT)


class _FakeRouter:
    def __init__(self) -> None:
        self.calls = 0

    def route(self, skeleton, gate_result, frame_image=None):
        self.calls += 1
        return {"route": "dataset", "record_id": f"r{self.calls}"}


class _FakeFeatureExtractor:
    def extract(self, skeleton_window, imu_features=None):
        return {"vector": np.ones((20,), dtype=np.float32), "feature_names": [f"f{i}" for i in range(20)], "imu_available": False}


class _FakeDatasetWriter:
    def __init__(self) -> None:
        self.exported = False

    def export_coco(self, output_path, validate_with_pycocotools: bool = False):
        Path(output_path).write_text("{}", encoding="utf-8")
        self.exported = True
        return {}


def test_infer_mode_auto_selects_batch_for_directory(tmp_path: Path) -> None:
    assert infer_mode(str(tmp_path), "auto") == "batch"


def test_pipeline_runner_writes_outputs(tmp_path: Path) -> None:
    runner = PipelineRunner(
        source="input.mp4",
        mode="dataset",
        session_id="session_test",
        output_root=tmp_path,
        ingestor=_FakeIngestor(),
        extractor=_FakeExtractor(),
        quality_gate=_FakeQualityGate(),
        router=_FakeRouter(),
        feature_extractor=_FakeFeatureExtractor(),
        dataset_writer=_FakeDatasetWriter(),
    )

    summary = runner.run(max_frames=2, validate_coco=False)

    assert summary.frames_processed == 2
    assert summary.skeletons_detected == 2
    assert Path(summary.skeleton_output).exists()
    assert Path(summary.feature_output).exists()
    assert Path(summary.feature_meta_output).exists()
    assert Path(summary.summary_output).exists()
