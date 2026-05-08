# Project Status

_Last updated: 2026-05-08_

---

## ✅ Completed

### Core Pipeline (production-quality)
- **Ingestion** — video, webcam, RTSP, image-batch (`src/ingestion/`)
- **Skeleton extraction** — YOLO v8-pose + MediaPipe ensemble, ByteTrack identity (`src/skeleton/`)
- **Quality gates** — G1 detection / G2 confidence / G3 plausibility → AUTO_ACCEPT / REVIEW / DISCARD (`src/annotation/quality_gates.py`)
- **Annotation routing** — to SQLite or Label Studio review queue (`src/annotation/annotator.py`)
- **Feature extraction** — 20-dim vector: joint angles, velocity, acceleration, PPE, zone (`src/features/extractor.py`)
- **IMU fusion** — accelerometer/gyroscope integration for fall detection (`src/imu/fusion.py`)
- **Storage / COCO export** — SQLite annotations + COCO JSON export (`src/annotation/storage.py`)
- **Batch pipeline CLI** — full end-to-end run (`scripts/run_pipeline.py`)
- **Realtime FastAPI service** — `/health`, `/state`, `/events`, `/robot/commands` (`scripts/run_realtime_server.py`)
- **Training scaffolds** — pose / action / risk trainers (`src/training/`)
- **Dataset tooling** — download, validate, DVC version control (`scripts/`)
- **Test suite** — 13 test files, all passing

### POC Demo (`poc/`) — ✅ Live
Completed 2026-05-08. A full Streamlit demo running at `http://localhost:8501`.

| File | Purpose |
|---|---|
| `poc/demo.py` | Main Streamlit app — dual mode (egocentric + full-body) |
| `poc/draw_utils.py` | OpenCV skeleton/HUD rendering |
| `poc/hand_detector.py` | MediaPipe Hands wrapper for egocentric view |
| `poc/trainer.py` | Teachable Machine-style KNN action classifier |

**Features:**
- **Egocentric mode** — designed for cap-mounted ESP32-CAM; tracks both hands with MediaPipe (21 landmarks each), extracts 30-dim hand feature vector
- **Full-body mode** — YOLO v8-pose skeleton overlay with ByteTrack identity
- **Teachable Machine training** — record action classes live (60 frames/class), train KNN in-app, get live predictions with confidence scores and per-class risk levels
- **Live dashboard** — FPS, worker count, alert count, action classification, risk score bars, quality gate badges
- **Multi-source input** — webcam, video file upload, ESP32-CAM HTTP/RTSP stream

---

## 🔄 In Progress / Next

- [ ] Replace heuristic action/risk in realtime FastAPI service with trained model inference
- [ ] Add model registry and artifact loading policy (ONNX / Torch)
- [ ] Wire Label Studio queue end-to-end in active review workflow
- [ ] Persist trained KNN model to disk (joblib) and reload on startup
- [ ] Add PPE detection overlay (helmet/vest/gloves) using YOLOv8 custom weights
- [ ] Extend integration tests for realtime service endpoints
- [ ] Build proper viewer UI (replace `viewer-ui/` placeholder)

---

## 🚧 Blockers / Dependencies

- Full SH17 image+label payload needs Kaggle download placement
- iSafetyBench/OpenMarcie not yet fully populated locally
- ffmpeg binary not installed on macOS (OpenCV fallback active — works fine)
- `onnxruntime-gpu` replaced with `onnxruntime` for macOS ARM compatibility
- `viewer-ui/` is an empty placeholder (no frontend source files)

