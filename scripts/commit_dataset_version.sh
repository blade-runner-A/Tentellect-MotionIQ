#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f "data/processed/annotations.db" ]]; then
  echo "ERROR: annotations database not found at data/processed/annotations.db" >&2
  exit 1
fi

if ! command -v dvc >/dev/null 2>&1; then
  echo "ERROR: dvc command not found. Install DVC before committing dataset versions." >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git command not found." >&2
  exit 1
fi

FRAME_COUNT=$(python - <<'PY'
import sqlite3
conn = sqlite3.connect("data/processed/annotations.db")
count = conn.execute(
    "SELECT COUNT(*) FROM annotations WHERE quality_gate='AUTO_ACCEPT' OR human_verified=1"
).fetchone()[0]
print(count)
conn.close()
PY
)

VERSION="v$(date +%Y%m%d)_${FRAME_COUNT}frames"

dvc add data/processed/
git add data/processed.dvc .gitignore
if git diff --cached --quiet; then
  echo "No dataset changes staged; nothing to commit."
  exit 0
fi

git commit -m "dataset: ${VERSION}"
git tag -a "${VERSION}" -m "Dataset snapshot: ${FRAME_COUNT} frames"
echo "Committed dataset version: ${VERSION}"
