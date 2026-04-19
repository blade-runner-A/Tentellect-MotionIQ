"""Run realtime Tentellect service for robot integration."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import uvicorn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.serving.realtime_server import create_realtime_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Tentellect realtime API service.")
    parser.add_argument("--source", required=True, help="Video file, webcam index string, or RTSP URL")
    parser.add_argument("--session-id", default="realtime_session")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    app = create_realtime_app(
        source=args.source,
        session_id=args.session_id,
        config_path=args.config,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

