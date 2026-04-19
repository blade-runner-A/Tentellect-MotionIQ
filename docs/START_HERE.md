# Start Here

This file is the shortest path to getting productive on the pipeline.

## 1) What this project is

This repo is the core implementation for a skeleton-first industrial safety system:

- ingest video/image streams
- extract worker skeletons
- route detections through quality gates
- build feature vectors
- produce risk/action outputs
- expose realtime robot-facing API endpoints

`docs-site/` is intentionally excluded from this repo scope.

## 2) What is already done

- End-to-end batch pipeline works on sample media (`scripts/run_pipeline.py`)
- Realtime service API exists (`scripts/run_realtime_server.py`)
- Viewer UI is in separate app (`viewer-ui/`) to inspect processed sessions
- Core tests pass
- Dataset validation scripts and download scaffolding exist

See `docs/PROJECT_STATUS.md` for detail.

## 3) First run (local smoke)

```bash
pip install -r requirements.txt
pytest
python scripts/run_pipeline.py --input data/sample_input/people.mp4 --mode auto --session-id smoke_local --max-frames 60
```

## 4) Realtime API run

```bash
python scripts/run_realtime_server.py --source data/sample_input/people.mp4 --session-id rt_demo --port 8091
```

Open:

- `http://localhost:8091/health`
- `http://localhost:8091/state`
- `http://localhost:8091/robot/commands`

## 5) Viewer UI run

```bash
cd viewer-ui
npm install
npm run dev -- --port 3010
```

Open `http://localhost:3010`.

## 6) Build your own capture hardware

Start with `docs/HARDWARE_CAPTURE_BLUEPRINT.md`.

