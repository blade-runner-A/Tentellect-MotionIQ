# Tentellect MotionIQ Pipeline

Tentellect MotionIQ is a skeleton-first industrial safety intelligence system.

This repository is the core pipeline and realtime control-plane implementation:

- ingest camera streams or media files
- extract worker skeletons and track identities
- apply quality gates and annotation routing
- compute risk/action features
- expose realtime robot-facing APIs

## Repository Scope

- This repo contains the pipeline/runtime code.
- `docs-site/` is a separate project and is ignored by this repository.
- Operational viewer UI lives in `viewer-ui/` (separate app in this workspace).

## System Naming

- **Platform:** Tentellect MotionIQ
- **Hardware node:** MotionIQ Node
- **Viewer app:** MotionIQ Viewer

See `docs/SYSTEM_IDENTITY.md`.

## Start Here

Read these in order:

1. `docs/START_HERE.md`
2. `docs/PROJECT_STATUS.md`
3. `docs/HARDWARE_CAPTURE_BLUEPRINT.md`

## Current Status

Implemented and working:

- batch pipeline execution (`scripts/run_pipeline.py`)
- realtime API service (`scripts/run_realtime_server.py`)
- robot command endpoint scaffold (`/robot/commands`)
- dataset validation and test suite
- viewer UI app for processed sessions (`viewer-ui/`)

## Quick Start (Local)

```bash
pip install -r requirements.txt
pytest
python scripts/run_pipeline.py --input data/sample_input/people.mp4 --mode auto --session-id smoke_local --max-frames 60
```

Outputs are written to `data/processed/`.

## Realtime Tracking + Robot API

```bash
python scripts/run_realtime_server.py --source data/sample_input/people.mp4 --session-id rt_demo --port 8091
```

Endpoints:

- `GET /health`
- `GET /state`
- `GET /events?limit=100`
- `GET /robot/commands`

Note: current action/risk in realtime uses heuristic fallback until trained models are plugged into serving.

## Viewer UI (Separate App)

```bash
cd viewer-ui
npm install
npm run dev -- --port 3010
```

Open `http://localhost:3010`.

## Hardware Capture (Own Data)

Use `docs/HARDWARE_CAPTURE_BLUEPRINT.md` for:

- edge hardware BOM
- capture/timestamp contract
- robot integration strategy
- V1 vs V2 hardware roadmap

## Build Order (PRD Reference)

1. `scripts/download_datasets.sh`
2. `scripts/validate_datasets.py`
3. `src/ingestion/preprocess.py`
4. `src/ingestion/ingestor.py`
5. `src/skeleton/extractor.py`
6. `src/annotation/quality_gates.py`
7. `src/imu/fusion.py`
8. `src/annotation/annotator.py`
9. `src/annotation/storage.py`
10. `src/features/extractor.py`
11. `src/training/train_pose.py`
12. `src/training/train_action.py`
13. `src/training/train_risk.py`
14. `scripts/run_pipeline.py`
