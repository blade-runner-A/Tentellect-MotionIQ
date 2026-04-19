"""Fine-tune YOLOv8 pose model on industrial datasets with MLflow logging."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("train_pose")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for pose fine-tuning."""
    parser = argparse.ArgumentParser(description="Train/fine-tune YOLOv8 pose model for Tentellect.")
    parser.add_argument("--data-config", required=True, help="Path to Ultralytics data YAML")
    parser.add_argument("--model", default="yolov8s-pose.pt", help="Base model checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project-dir", default="models/checkpoints")
    parser.add_argument("--run-name", default="pose_finetune")
    parser.add_argument("--experiment", default="tentellect_pipeline_v1")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    """Configure module logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def maybe_get_mlflow() -> Any | None:
    """Import MLflow lazily if available."""
    try:
        import mlflow

        return mlflow
    except ImportError:
        LOGGER.warning("MLflow is not installed; metrics will only be logged to stdout.")
        return None


def train_pose(args: argparse.Namespace) -> dict[str, float]:
    """Run YOLO pose fine-tune and return metrics dictionary."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required for pose training.") from exc

    project_dir = Path(args.project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    results = model.train(
        data=args.data_config,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=str(project_dir),
        name=args.run_name,
        exist_ok=True,
    )

    metrics = extract_metrics(results)

    if args.export_onnx:
        onnx_path = model.export(format="onnx")
        LOGGER.info("Exported ONNX model: %s", onnx_path)

    return metrics


def extract_metrics(results: Any) -> dict[str, float]:
    """Extract scalar metrics from Ultralytics result object."""
    if results is None:
        return {}

    if hasattr(results, "results_dict") and isinstance(results.results_dict, dict):
        return {str(key): float(value) for key, value in results.results_dict.items() if _is_number(value)}

    metrics: dict[str, float] = {}
    for key in ("fitness", "box_loss", "pose_loss", "kobj_loss"):
        if hasattr(results, key):
            value = getattr(results, key)
            if _is_number(value):
                metrics[key] = float(value)
    return metrics


def _is_number(value: Any) -> bool:
    """Return True when value can be represented as float."""
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    setup_logging(args.log_level)

    mlflow = maybe_get_mlflow()
    if mlflow is not None:
        mlflow.set_experiment(args.experiment)

    run = mlflow.start_run(run_name=args.run_name) if mlflow is not None else None

    try:
        params = {
            "data_config": args.data_config,
            "model": args.model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "patience": args.patience,
            "device": args.device,
            "export_onnx": args.export_onnx,
        }

        if mlflow is not None:
            mlflow.log_params(params)

        metrics = train_pose(args)
        for key, value in metrics.items():
            LOGGER.info("metric %s=%.6f", key, value)

        if mlflow is not None and metrics:
            mlflow.log_metrics(metrics)

        return 0

    except Exception:
        LOGGER.exception("Pose training failed.")
        return 1

    finally:
        if run is not None and mlflow is not None:
            mlflow.end_run()


if __name__ == "__main__":
    raise SystemExit(main())
