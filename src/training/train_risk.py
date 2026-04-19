"""Train calibrated XGBoost risk scorer and persist SHAP explainer."""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("train_risk")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for risk model training."""
    parser = argparse.ArgumentParser(description="Train XGBoost risk model for Tentellect.")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--output-dir", default="models/checkpoints/risk")
    parser.add_argument("--experiment", default="tentellect_pipeline_v1")
    parser.add_argument("--run-name", default="risk_xgboost_train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--skip-shap", action="store_true")
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
        LOGGER.warning("MLflow is not installed; training metadata will be stdout-only.")
        return None


def load_feature_matrix(features_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load feature matrix and labels from .npy/.npz files.

    Supported formats:
    1) .npz with keys X and y
    2) .npy object-dict with keys X and y
    3) .npy array with 21 columns (20 features + 1 label)
    """
    root = Path(features_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Feature directory not found: {root}")

    files = sorted([*root.glob("*.npy"), *root.glob("*.npz")])
    if not files:
        raise FileNotFoundError(f"No .npy or .npz feature files found in {root}")

    x_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []

    for file_path in files:
        if file_path.suffix == ".npz":
            payload = np.load(file_path, allow_pickle=True)
            if {"X", "y"}.issubset(payload.files):
                x_val = np.asarray(payload["X"], dtype=np.float32)
                y_val = np.asarray(payload["y"], dtype=np.int32).reshape(-1)
                if len(x_val) != len(y_val):
                    raise ValueError(f"Row mismatch in {file_path}: X={len(x_val)} y={len(y_val)}")
                x_blocks.append(x_val)
                y_blocks.append(y_val)
            else:
                LOGGER.warning("Skipping %s (missing X/y keys)", file_path)
            continue

        array = np.load(file_path, allow_pickle=True)
        if isinstance(array, np.ndarray) and array.dtype == object and array.shape == ():
            item = array.item()
            if isinstance(item, dict) and {"X", "y"}.issubset(item.keys()):
                x_val = np.asarray(item["X"], dtype=np.float32)
                y_val = np.asarray(item["y"], dtype=np.int32).reshape(-1)
                if len(x_val) != len(y_val):
                    raise ValueError(f"Row mismatch in {file_path}: X={len(x_val)} y={len(y_val)}")
                x_blocks.append(x_val)
                y_blocks.append(y_val)
                continue

        if array.ndim == 2 and array.shape[1] >= 21:
            x_blocks.append(np.asarray(array[:, :20], dtype=np.float32))
            y_blocks.append(np.asarray(array[:, 20], dtype=np.int32).reshape(-1))
            continue

        if array.ndim == 1 and array.shape[0] >= 21:
            x_blocks.append(np.asarray(array[:20], dtype=np.float32).reshape(1, -1))
            y_blocks.append(np.asarray([int(array[20])], dtype=np.int32))
            continue

        LOGGER.warning("Skipping unsupported feature file format: %s", file_path)

    if not x_blocks:
        raise ValueError("No valid feature blocks with labels were loaded.")

    x_matrix = np.concatenate(x_blocks, axis=0)
    y_vector = np.concatenate(y_blocks, axis=0)

    if x_matrix.shape[0] != y_vector.shape[0]:
        raise ValueError("Feature and label row count mismatch after concatenation.")

    if x_matrix.shape[1] != 20:
        raise ValueError(f"Expected 20 feature columns, found {x_matrix.shape[1]}")

    return x_matrix, y_vector


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute expected calibration error for binary probabilities."""
    if len(y_true) == 0:
        return 0.0

    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for idx in range(n_bins):
        lower = bins[idx]
        upper = bins[idx + 1]
        if idx == n_bins - 1:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)

        if not np.any(mask):
            continue

        bin_conf = float(np.mean(y_prob[mask]))
        bin_acc = float(np.mean(y_true[mask]))
        bin_weight = float(np.mean(mask.astype(np.float32)))
        ece += abs(bin_acc - bin_conf) * bin_weight

    return float(ece)


def train_risk_model(
    x_matrix: np.ndarray,
    y_vector: np.ndarray,
    seed: int,
    n_jobs: int,
) -> tuple[Any, Any, dict[str, float]]:
    """Train XGBoost model and fit isotonic calibrator using CV predictions."""
    try:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import log_loss
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "xgboost and scikit-learn are required for risk training. Install via requirements.txt."
        ) from exc

    class_counts = np.bincount(y_vector.astype(np.int64))
    valid_counts = class_counts[class_counts > 0]
    min_class_count = int(valid_counts.min()) if len(valid_counts) else 0
    n_splits = min(5, min_class_count)
    if n_splits < 2:
        raise ValueError("Need at least two samples per class for cross-validation calibration.")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=n_jobs,
        random_state=seed,
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_prob = cross_val_predict(model, x_matrix, y_vector, cv=cv, method="predict_proba")[:, 1]

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(cv_prob, y_vector)

    model.fit(x_matrix, y_vector)

    train_prob = model.predict_proba(x_matrix)[:, 1]
    calibrated_prob = calibrator.transform(train_prob)

    metrics = {
        "train_logloss": float(log_loss(y_vector, train_prob)),
        "train_ece": compute_ece(y_vector, train_prob),
        "train_ece_calibrated": compute_ece(y_vector, calibrated_prob),
    }

    return model, calibrator, metrics


def build_shap_explainer(model: Any, x_matrix: np.ndarray) -> Any:
    """Build SHAP TreeExplainer and run a small warm-up batch."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("shap is required to build risk explainer artifacts.") from exc

    explainer = shap.TreeExplainer(model)
    sample = x_matrix[: min(len(x_matrix), 256)]
    explainer.shap_values(sample)
    return explainer


def save_artifacts(
    model: Any,
    calibrator: Any,
    explainer: Any | None,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Persist model, calibrator, and optional SHAP explainer."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    model_path = target / "risk_xgboost.json"
    model.save_model(str(model_path))

    calibrator_path = target / "risk_isotonic.pkl"
    with calibrator_path.open("wb") as handle:
        pickle.dump(calibrator, handle)

    outputs = {
        "model": model_path,
        "calibrator": calibrator_path,
    }

    if explainer is not None:
        explainer_path = target / "risk_shap_explainer.pkl"
        with explainer_path.open("wb") as handle:
            pickle.dump(explainer, handle)
        outputs["explainer"] = explainer_path

    return outputs


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    setup_logging(args.log_level)

    mlflow = maybe_get_mlflow()
    run = None

    if mlflow is not None:
        mlflow.set_experiment(args.experiment)
        run = mlflow.start_run(run_name=args.run_name)

    try:
        x_matrix, y_vector = load_feature_matrix(args.features_dir)
        LOGGER.info("Loaded dataset: samples=%d features=%d", x_matrix.shape[0], x_matrix.shape[1])

        model, calibrator, metrics = train_risk_model(
            x_matrix=x_matrix,
            y_vector=y_vector,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )

        explainer = None
        if not args.skip_shap:
            explainer = build_shap_explainer(model, x_matrix)

        paths = save_artifacts(model=model, calibrator=calibrator, explainer=explainer, output_dir=args.output_dir)

        for key, value in metrics.items():
            LOGGER.info("metric %s=%.6f", key, value)

        if mlflow is not None:
            mlflow.log_params(
                {
                    "features_dir": args.features_dir,
                    "output_dir": args.output_dir,
                    "seed": args.seed,
                    "n_jobs": args.n_jobs,
                    "skip_shap": args.skip_shap,
                    "samples": int(x_matrix.shape[0]),
                    "features": int(x_matrix.shape[1]),
                }
            )
            mlflow.log_metrics(metrics)
            for name, path in paths.items():
                mlflow.log_artifact(str(path))
                mlflow.set_tag(f"artifact_{name}", str(path))

        return 0

    except Exception:
        LOGGER.exception("Risk training failed.")
        return 1

    finally:
        if run is not None and mlflow is not None:
            mlflow.end_run()


if __name__ == "__main__":
    raise SystemExit(main())
