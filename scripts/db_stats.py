"""Quick stats for the local SQLite annotations database."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print stats from annotations.db")
    parser.add_argument("--db", type=Path, default=Path("data/processed/annotations.db"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    conn = sqlite3.connect(str(args.db))
    try:
        cur = conn.cursor()
        total = cur.execute("SELECT COUNT(1) FROM annotations").fetchone()[0]
        auto = cur.execute("SELECT COUNT(1) FROM annotations WHERE quality_gate='AUTO_ACCEPT'").fetchone()[0]
        review = cur.execute("SELECT COUNT(1) FROM annotations WHERE quality_gate='REVIEW'").fetchone()[0]
        discard = cur.execute("SELECT COUNT(1) FROM annotations WHERE quality_gate='DISCARD'").fetchone()[0]
        print(f"db={args.db}")
        print(f"rows_total={total}")
        print(f"rows_auto_accept={auto}")
        print(f"rows_review={review}")
        print(f"rows_discard={discard}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

