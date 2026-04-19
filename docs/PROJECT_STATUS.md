# Project Status

## Completed

- Pipeline stages implemented:
  - ingestion
  - skeleton extraction
  - quality gates
  - annotation routing
  - storage/export
  - feature extraction
- Training script scaffolds exist:
  - pose/action/risk trainers under `src/training/`
- Realtime service scaffold exists:
  - `src/serving/realtime_server.py`
  - robot command endpoint and event/state endpoints
- Separate visualization app exists:
  - `viewer-ui/` (Next.js + shadcn/ui)
- Validation and tests:
  - pipeline smoke runs pass
  - test suite passing in local environment

## In Progress / Next

- Replace heuristic action/risk in realtime service with trained model inference
- Add model registry and artifact loading policy (ONNX / Torch)
- Wire Label Studio queue end-to-end in active review workflow
- Extend integration tests for realtime service endpoints

## Blockers / Dependencies

- Full SH17 image+label payload still needs Kaggle download placement
- iSafetyBench/OpenMarcie not yet fully populated locally
- ffmpeg binary not installed on Windows machine (OpenCV fallback active)

