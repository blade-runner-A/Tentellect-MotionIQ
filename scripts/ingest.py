"""CLI wrapper for DataIngestor module."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ingestion.ingestor import DataIngestor

LOGGER = logging.getLogger("ingest_cli")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Ingest frames and print summary metadata.")
    parser.add_argument("source", help="Source path or URL")
    parser.add_argument("--mode", choices=["dataset", "realtime", "batch"], default="dataset")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit for quick checks")
    parser.add_argument("--sample-count", type=int, default=3, help="Number of metadata samples to print")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    """Configure logging to stdout."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )


def main() -> int:
    """Run ingestion summary command."""
    args = parse_args()
    setup_logging(args.log_level)

    session_id = args.session_id or datetime.now(timezone.utc).strftime("ingest_%Y%m%dT%H%M%SZ")

    ingestor = DataIngestor(source=args.source, mode=args.mode, session_id=session_id)

    frame_count = 0
    samples: list[dict[str, object]] = []

    try:
        for record in ingestor.iter_frames():
            frame_count += 1
            if len(samples) < args.sample_count:
                samples.append(dict(record.metadata))

            if args.max_frames > 0 and frame_count >= args.max_frames:
                break

    except Exception:
        LOGGER.exception("Ingestion failed for source=%s mode=%s", args.source, args.mode)
        return 1

    LOGGER.info("frame_count=%d", frame_count)
    LOGGER.info("sample_metadata=%s", json.dumps(samples, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
