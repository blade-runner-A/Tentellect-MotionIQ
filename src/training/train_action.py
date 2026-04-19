"""Launch ST-GCN action training via MMAction2 with MLflow tracking."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("train_action")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for action training."""
    parser = argparse.ArgumentParser(description="Train ST-GCN action classifier for Tentellect.")
    parser.add_argument("--mmaction-train-script", default="tools/train.py", help="MMAction2 train script path")
    parser.add_argument("--config", required=True, help="MMAction2 config path")
    parser.add_argument("--work-dir", default="models/checkpoints/action")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--cfg-options", nargs="*", default=[])
    parser.add_argument("--run-name", default="action_stgcn_train")
    parser.add_argument("--experiment", default="tentellect_pipeline_v1")
    parser.add_argument("--dry-run", action="store_true")
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
        LOGGER.warning("MLflow is not installed; command tracking will be stdout-only.")
        return None


def build_train_command(args: argparse.Namespace) -> list[str]:
    """Build MMAction2 command for ST-GCN training."""
    command = [
        sys.executable,
        args.mmaction_train_script,
        args.config,
        "--work-dir",
        args.work_dir,
    ]

    if args.resume_from:
        command.extend(["--resume", args.resume_from])

    if args.cfg_options:
        command.append("--cfg-options")
        command.extend(args.cfg_options)

    return command


def run_training(command: list[str], dry_run: bool = False) -> int:
    """Execute MMAction2 training command."""
    if dry_run:
        LOGGER.info("Dry-run command: %s", " ".join(command))
        return 0

    process = subprocess.run(command, check=False)
    return int(process.returncode)


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    setup_logging(args.log_level)

    Path(args.work_dir).mkdir(parents=True, exist_ok=True)

    command = build_train_command(args)
    mlflow = maybe_get_mlflow()
    run = None

    if mlflow is not None:
        mlflow.set_experiment(args.experiment)
        run = mlflow.start_run(run_name=args.run_name)
        mlflow.log_params(
            {
                "mmaction_train_script": args.mmaction_train_script,
                "config": args.config,
                "work_dir": args.work_dir,
                "resume_from": args.resume_from,
                "dry_run": args.dry_run,
            }
        )
        mlflow.set_tag("train_command", " ".join(command))

    try:
        exit_code = run_training(command=command, dry_run=args.dry_run)
        LOGGER.info("MMAction2 training exit_code=%d", exit_code)

        if mlflow is not None:
            mlflow.log_metric("exit_code", float(exit_code))

        return 0 if exit_code == 0 else 1

    except Exception:
        LOGGER.exception("Action training failed.")
        return 1

    finally:
        if run is not None and mlflow is not None:
            mlflow.end_run()


if __name__ == "__main__":
    raise SystemExit(main())
