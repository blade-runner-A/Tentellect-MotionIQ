# Start Here

_Last updated: 2026-05-08_

This is the fastest path to running the system.

---

## 1) What this project is

Tentellect MotionIQ is a skeleton-first industrial safety system designed for the shop floor:

- Ingest camera streams (webcam, video file, RTSP, ESP32-CAM on a hard hat)
- Extract worker skeletons **or** hand landmarks (egocentric view)
- Route detections through quality gates
- Build feature vectors for risk and action classification
- Expose realtime robot-facing API endpoints
- **In-app Teachable Machine training** — record action classes live, train a KNN classifier, get live predictions

`docs-site/` is intentionally excluded from this repo scope.

---

## 2) Fastest path: run the POC demo

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install streamlit pandas
streamlit run poc/demo.py
```

Open **http://localhost:8501**.

**To use a webcam**: select *Webcam* in the sidebar → press ▶ Start.

**To use an ESP32-CAM**: select *ESP32-CAM (HTTP/RTSP)* → paste URL (e.g. `http://192.168.x.x:81/stream`).

**To train your own actions**:
1. Type an action name (e.g. `Drilling`) and set its risk level
2. Press 🔴 Record — perform the action for ~4 seconds
3. Repeat for more classes
4. Press 🎯 Train Model → live predictions switch to your trained model

---

## 3) POC files

| File | Purpose |
|---|---|
| `poc/demo.py` | Main Streamlit app |
| `poc/draw_utils.py` | OpenCV skeleton / HUD drawing |
| `poc/hand_detector.py` | MediaPipe Hands for egocentric (cap-mounted) view |
| `poc/trainer.py` | Teachable Machine-style KNN classifier |

---

## 4) Batch pipeline (CLI)

```bash
source .venv/bin/activate
python scripts/run_pipeline.py \
  --input data/sample_input/people.mp4 \
  --mode auto \
  --session-id smoke_local \
  --max-frames 60
```

---

## 5) Realtime FastAPI service

```bash
source .venv/bin/activate
python scripts/run_realtime_server.py --source 0 --session-id rt_demo --port 8091
```

- `http://localhost:8091/health`
- `http://localhost:8091/state`
- `http://localhost:8091/robot/commands`

---

## 6) Tests

```bash
source .venv/bin/activate
pytest
```

---

## 7) Build your own capture hardware

Start with `docs/HARDWARE_CAPTURE_BLUEPRINT.md` for the ESP32-CAM BOM and mounting guide.

