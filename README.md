# Tentellect MotionIQ Pipeline

> **Skeleton-first industrial safety intelligence for the shop floor.**
> Egocentric (cap-mounted) + full-body pose tracking with live risk scoring and trainable action classification.

---

## What This Repo Is

Tentellect MotionIQ is the core pipeline and realtime control-plane:

- Ingest camera streams or media files (webcam, video, RTSP, ESP32-CAM)
- Extract worker skeletons (YOLO v8-pose + MediaPipe ensemble) or hand landmarks (MediaPipe Hands)
- Apply quality gates and annotation routing
- Compute risk / action features
- Expose realtime robot-facing APIs
- **POC live demo** with Teachable Machine-style in-app training

## Repository Scope

- `src/` — core pipeline modules
- `scripts/` — CLI runners and dataset tooling
- `poc/` — live Streamlit demo (egocentric + training)
- `configs/` — pipeline YAML configuration
- `tests/` — 13 test files
- `docs/` — project docs and blueprints
- `viewer-ui/` — placeholder for future frontend (not yet implemented)

## System Naming

- **Platform:** Tentellect MotionIQ
- **Hardware node:** MotionIQ Node (ESP32-CAM on hard hat)
- **Viewer app:** MotionIQ Viewer

See `docs/SYSTEM_IDENTITY.md`.

---

## 🚀 Quickest Path: Run the POC

```bash
# 1. Create venv and install deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install streamlit pandas

# 2. Launch the live demo
streamlit run poc/demo.py
```

Open **http://localhost:8501** — choose Webcam or upload a video, press **▶ Start**.

### POC Features

| Feature | Details |
|---|---|
| **Egocentric mode** | Designed for cap-mounted ESP32-CAM — tracks both hands with MediaPipe Hands (21 landmarks/hand) |
| **Full-body mode** | YOLO v8-pose skeleton overlay with ByteTrack identity |
| **Live risk scoring** | Heuristic action/risk classifier → visual risk bars, colour-coded overlays |
| **Teachable Machine training** | Record action classes live → train KNN in-app → live predictions with confidence |
| **Multi-source** | Webcam, video file upload, ESP32-CAM HTTP/RTSP stream |
| **Alert dashboard** | Per-worker action, risk, quality gate, confidence metrics |

### ESP32-CAM Integration

Connect the ESP32-CAM to your WiFi network and use its stream URL:
```
http://<ESP32_IP>:81/stream       # MJPEG HTTP stream (default firmware)
rtsp://<ESP32_IP>:8554/stream     # RTSP (if configured)
```
Select **ESP32-CAM (HTTP/RTSP)** in the sidebar and paste the URL.

---

## Batch Pipeline (CLI)

```bash
source .venv/bin/activate
python scripts/run_pipeline.py \
  --input data/sample_input/people.mp4 \
  --mode auto \
  --session-id smoke_local \
  --max-frames 60
```

Outputs are written to `data/processed/`.

---

## Realtime FastAPI Service

```bash
source .venv/bin/activate
python scripts/run_realtime_server.py \
  --source 0 \
  --session-id rt_demo \
  --port 8091
```

Endpoints:

| Endpoint | Description |
|---|---|
| `GET /health` | Service liveness |
| `GET /state` | Current tracked worker states |
| `GET /events?limit=100` | Recent detection events |
| `GET /robot/commands` | Robot action suggestions from risk/action |

> Action/risk uses heuristic fallback until trained models are plugged into serving.

---

## Hardware Capture

Use `docs/HARDWARE_CAPTURE_BLUEPRINT.md` for:

- Edge hardware BOM (ESP32-CAM + hard hat mount)
- Capture / timestamp contract
- Robot integration strategy
- V1 vs V2 hardware roadmap

---

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
15. **`poc/demo.py`** ← POC live demo ✅

---

## Docs

| File | Purpose |
|---|---|
| `docs/START_HERE.md` | Fastest path to running |
| `docs/PROJECT_STATUS.md` | Full completion / in-progress / blockers log |
| `docs/HARDWARE_CAPTURE_BLUEPRINT.md` | ESP32-CAM hardware guide |
| `docs/SYSTEM_IDENTITY.md` | Platform naming conventions |
