"""Tentellect MotionIQ — POC v2  (Egocentric + Trainable)

Egocentric mode: designed for a cap-mounted ESP32-CAM that looks at the
worker's hands / work area.  Hand tracking replaces full-body skeleton.

Trainable mode (Teachable Machine style): record samples per action class,
train a KNN in-app, get live predictions with confidence scores.

Run:
    streamlit run poc/demo.py
"""

from __future__ import annotations

import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st

# ── repo root on path ─────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from poc.draw_utils import draw_hud, draw_skeleton, draw_track_box
from poc.hand_detector import HandDetector
from poc.trainer import ActionTrainer
from src.annotation.quality_gates import QualityGate
from src.features.extractor import FeatureExtractor
from src.skeleton.extractor import SkeletonExtractor

logging.basicConfig(level=logging.WARNING)
LOGGER = logging.getLogger("motioniq_poc")

# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tentellect MotionIQ POC",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background: #0a0c14; }
div[data-testid="metric-container"] {
    background: #141824; border: 1px solid #252a40;
    border-radius: 10px; padding: 8px 14px;
}
div[data-testid="metric-container"] label { color: #7a8499; font-size:.75rem; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-size:1.7rem; font-weight:700; color:#dde3f0;
}
section[data-testid="stSidebar"] { background:#0e1120; }
.block-container { padding-top:.8rem; }
.cls-badge {
    display:inline-block; background:#1e2440; border:1px solid #303860;
    border-radius:8px; padding:4px 10px; margin:3px; font-size:.82rem;
    color:#aabbdd;
}
.cls-badge .cnt { color:#5af; font-weight:700; }
.alert-box { background:#2a0f0f; border-left:4px solid #e05555;
    border-radius:6px; padding:7px 12px; margin:4px 0;
    font-size:.85rem; color:#f5c6c6; }
.safe-box { background:#0d2018; border-left:4px solid #3dcc7a;
    border-radius:6px; padding:7px 12px; margin:4px 0;
    font-size:.85rem; color:#a8eec8; }
.train-result { background:#101c30; border:1px solid #2a4070;
    border-radius:8px; padding:10px 14px; margin:6px 0;
    font-size:.85rem; color:#aac8f5; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS: dict[str, Any] = {
    "running": False,
    "ego_mode": True,
    "source": "0",
    "frame_idx": 0,
    "fps_buf": deque(maxlen=30),
    "track_history": {},
    "prev_hands": None,
    "latest_tracks": {},
    "trainer": ActionTrainer(),
    "recording": False,
    "rec_class": "",
    "rec_target": 60,
    "rec_count": 0,
    "trained": False,
    "train_result": {},
    "yolo_conf": 0.40,
    "risk_thresh": 0.60,
    "max_fps": 15,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# Model cache
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def _load_body_models(yolo_conf: float):
    extractor = SkeletonExtractor({
        "device": "cpu",
        "yolo_model": "yolov8s-pose.pt",
        "yolo_conf_threshold": yolo_conf,
        "ensemble_conf_threshold": 0.60,
        "mediapipe_model_complexity": 1,
    })
    gate = QualityGate(
        detection_threshold=0.30,
        auto_accept_threshold=0.60,
        review_threshold=0.25,
        min_visible_keypoints=5,
    )
    feat = FeatureExtractor()
    return extractor, gate, feat

@st.cache_resource(show_spinner="Loading hand detector…")
def _load_hand_detector():
    return HandDetector(max_hands=2, min_detection_confidence=0.55)

# ─────────────────────────────────────────────────────────────────────────────
# Heuristic fallback (used when no trained model)
# ─────────────────────────────────────────────────────────────────────────────
def _heuristic(fv: list[float]) -> tuple[str, float]:
    if len(fv) < 20:
        return "idle", 0.0
    torso   = abs(fv[0])
    vel     = max(0.0, fv[13])
    accel   = max(0.0, fv[14])
    ppe     = min(1.0, max(0.0, fv[9]))
    if fv[18] > 0.5 or (torso > 45 and accel > 80):
        action = "fall"
    elif torso > 35:
        action = "bend"
    elif vel > 55:
        action = "walk"
    elif torso > 20 and vel < 10:
        action = "reach_overhead"
    else:
        action = "idle"
    risk = min(1.0, 0.35*(torso/60) + 0.35*min(1,(accel/120)) + 0.20*min(1,(vel/120)) + 0.10*(1-ppe))
    return action, float(risk)

def _ego_heuristic(fv: np.ndarray) -> tuple[str, float]:
    """Heuristic for egocentric hand features (30-dim)."""
    if fv[21] < 0.1:
        return "no_hands", 0.0
    left_open  = float(fv[6])   # left hand openness
    right_open = float(fv[16])
    pinch_l    = float(fv[7])
    pinch_r    = float(fv[17])
    inter_dist = float(fv[20])
    vel_lx     = abs(float(fv[22]))
    vel_rx     = abs(float(fv[24]))
    vel        = vel_lx + vel_rx

    avg_open   = (left_open + right_open) / max(fv[21], 0.1)
    avg_pinch  = (pinch_l + pinch_r) / max(fv[21], 0.1)

    if avg_pinch < 0.04:
        action = "gripping"
    elif inter_dist < 0.15 and avg_open < 0.12:
        action = "assembling"
    elif vel > 0.08:
        action = "reaching"
    elif avg_open > 0.25:
        action = "open_hand_idle"
    else:
        action = "working"

    risk = min(1.0, 0.4*(1 - avg_open*3) + 0.3*vel*4 + 0.3*(1 - avg_pinch*8))
    return action, max(0.0, float(risk))

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦺 MotionIQ POC v2")
    st.caption("Industrial Safety Intelligence")
    st.divider()

    # ── Camera source ──────────────────────────────────────────────────────
    st.markdown("### 📷 Camera Source")
    src_type = st.radio("", ["Webcam / USB", "Video file", "ESP32-CAM (HTTP/RTSP)"],
                        label_visibility="collapsed")

    if src_type == "Webcam / USB":
        cam_idx = st.number_input("Camera index", 0, 10, 0, 1)
        sel_src = str(int(cam_idx))
    elif src_type == "Video file":
        up = st.file_uploader("Upload video", type=["mp4","avi","mov","mkv"])
        if up:
            tmp = Path("/tmp") / up.name
            tmp.write_bytes(up.read())
            sel_src = str(tmp)
            st.success(f"Loaded `{up.name}`")
        else:
            sel_src = None
            st.info("Upload a video to start.")
    else:
        rtsp = st.text_input("Stream URL",
            placeholder="http://192.168.x.x:81/stream  or  rtsp://...")
        sel_src = rtsp if rtsp else None
        st.caption("ESP32-CAM default stream: `http://<IP>:81/stream`")

    # ── Mode ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🎥 Detection Mode")
    ego = st.toggle("Egocentric (cap-mounted) mode", value=True)
    st.caption("ON = hand tracking  |  OFF = full-body skeleton")

    # ── Settings ───────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### ⚙️ Settings")
    yolo_conf  = st.slider("YOLO confidence",  0.10, 0.90, 0.40, 0.05)
    risk_thr   = st.slider("Alert threshold",  0.30, 0.95, 0.60, 0.05)
    max_fps    = st.slider("Max FPS",          1,    30,   15,   1)

    # ── Teachable training panel ───────────────────────────────────────────
    st.divider()
    st.markdown("### 🎓 Train Action Classes")
    st.caption("Record 60 frames per class, then press Train.")

    trainer: ActionTrainer = st.session_state.trainer

    new_cls = st.text_input("Action class name",
                            placeholder="e.g. Drilling, Hammering, Idle")
    risk_for_cls = st.slider("Risk level for this class", 0.0, 1.0, 0.3, 0.05)

    col_rec, col_clr = st.columns(2)
    rec_pressed  = col_rec.button("🔴 Record",  use_container_width=True,
                                  disabled=not bool(new_cls))
    clr_pressed  = col_clr.button("🗑 Clear",   use_container_width=True,
                                  disabled=not bool(new_cls))

    if rec_pressed and new_cls:
        trainer.set_risk(new_cls, risk_for_cls)
        st.session_state.rec_class   = new_cls
        st.session_state.rec_target  = 60
        st.session_state.rec_count   = 0
        st.session_state.recording   = True

    if clr_pressed and new_cls:
        trainer.clear_class(new_cls)
        st.session_state.trained = False

    # Sample counts
    counts = trainer.sample_counts()
    if counts:
        badges = "".join(
            f"<span class='cls-badge'>{cls} <span class='cnt'>{n}</span></span>"
            for cls, n in counts.items()
        )
        st.markdown(badges, unsafe_allow_html=True)

    rec_prog = st.empty()
    if st.session_state.recording:
        pct = st.session_state.rec_count / max(st.session_state.rec_target, 1)
        rec_prog.progress(pct, text=f"Recording '{st.session_state.rec_class}' "
                          f"{st.session_state.rec_count}/{st.session_state.rec_target}")

    can_train = trainer.can_train()
    if st.button("🎯 Train Model", use_container_width=True, disabled=not can_train,
                 type="primary"):
        result = trainer.train()
        st.session_state.trained = True
        st.session_state.train_result = result

    if st.session_state.trained and st.session_state.train_result:
        r = st.session_state.train_result
        st.markdown(
            f"<div class='train-result'>✅ Trained on "
            f"<b>{r.get('n_samples',0)}</b> samples • "
            f"<b>{r.get('n_classes',0)}</b> classes • "
            f"Accuracy <b>{r.get('accuracy',0)*100:.0f}%</b></div>",
            unsafe_allow_html=True,
        )

    if st.button("🧹 Clear All Classes", use_container_width=True):
        trainer.clear_all()
        st.session_state.trained = False

    st.divider()
    c1, c2 = st.columns(2)
    start_btn = c1.button("▶ Start", use_container_width=True, type="primary")
    stop_btn  = c2.button("⏹ Stop",  use_container_width=True)

    if start_btn and sel_src is not None:
        st.session_state.running      = True
        st.session_state.ego_mode     = ego
        st.session_state.source       = sel_src
        st.session_state.frame_idx    = 0
        st.session_state.track_history = {}
        st.session_state.prev_hands   = None
        st.session_state.fps_buf      = deque(maxlen=30)
        st.session_state.yolo_conf    = yolo_conf
        st.session_state.risk_thresh  = risk_thr
        st.session_state.max_fps      = max_fps

    if stop_btn:
        st.session_state.running = False

# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────
ego_mode = st.session_state.ego_mode

mode_label = "🧢 Egocentric (Hand-Tracking) Mode" if ego_mode else "🧍 Full-Body Skeleton Mode"
st.markdown(f"## 📡 Live Safety Dashboard — {mode_label}")

m1, m2, m3, m4, m5 = st.columns(5)
fps_ph     = m1.empty()
worker_ph  = m2.empty()
alert_ph   = m3.empty()
action_ph  = m4.empty()
frame_ph   = m5.empty()

vid_col, info_col = st.columns([3, 1])
with vid_col:
    vid_ph = st.empty()
with info_col:
    st.markdown("#### 👷 Active Workers / Hands")
    table_ph  = st.empty()
    st.markdown("---")
    alert_log_ph = st.empty()

# ── Idle landing ─────────────────────────────────────────────────────────────
if not st.session_state.running:
    vid_ph.markdown("""
    <div style="background:#0e1120;border:2px dashed #252a40;border-radius:14px;
    height:440px;display:flex;align-items:center;justify-content:center;
    flex-direction:column;gap:14px;">
      <span style="font-size:3.5rem">🎥</span>
      <span style="color:#4a5070;font-size:1.05rem">
        Choose a source in the sidebar and press <strong style="color:#5080e8">▶ Start</strong>
      </span>
    </div>""", unsafe_allow_html=True)
    for ph in (fps_ph, worker_ph, alert_ph, action_ph, frame_ph):
        ph.metric("–", "–")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Live inference loop
# ─────────────────────────────────────────────────────────────────────────────
source = st.session_state.source

# Load models
hand_det = _load_hand_detector()
if not ego_mode:
    body_ext, body_gate, body_feat = _load_body_models(st.session_state.yolo_conf)

cap_src = int(source) if source.isdigit() else source
cap = cv2.VideoCapture(cap_src)

if not cap.isOpened():
    st.error(f"❌ Cannot open: `{source}` — check camera index or URL.")
    st.session_state.running = False
    st.stop()

frame_delay  = 1.0 / max(1, st.session_state.max_fps)
risk_thresh  = st.session_state.risk_thresh
trainer      = st.session_state.trainer
trained      = st.session_state.trained

try:
    while st.session_state.running:
        t0 = time.perf_counter()

        ok, raw = cap.read()
        if not ok or raw is None:
            if source.isdigit():
                st.warning("⚠️ Camera feed lost.")
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = raw.copy()
        h, w  = frame.shape[:2]
        fidx  = st.session_state.frame_idx
        st.session_state.frame_idx += 1
        meta: dict[str, Any] = {
            "frame_idx": fidx,
            "timestamp_ms": int(fidx * (1000 / max(st.session_state.max_fps, 1))),
            "session_id": "poc_v2",
        }

        # ── Egocentric: hand tracking ────────────────────────────────────────
        if ego_mode:
            prev_h = st.session_state.prev_hands
            hands_data = hand_det.detect(frame)
            feat_vec = hand_det.extract_features(hands_data, prev_h)
            st.session_state.prev_hands = hands_data

            # Prediction
            if trained and trainer.is_trained:
                action, conf, risk = trainer.predict(feat_vec)
            else:
                action, risk = _ego_heuristic(feat_vec)
                conf = 0.0

            # Recording
            if st.session_state.recording:
                st.session_state.rec_count += 1
                trainer.add_sample(st.session_state.rec_class, feat_vec)
                pct = st.session_state.rec_count / max(st.session_state.rec_target, 1)
                rec_prog.progress(min(pct, 1.0),
                    text=f"Recording '{st.session_state.rec_class}' "
                         f"{st.session_state.rec_count}/{st.session_state.rec_target}")
                if st.session_state.rec_count >= st.session_state.rec_target:
                    st.session_state.recording = False
                    rec_prog.success(
                        f"✅ Recorded {st.session_state.rec_target} frames "
                        f"for '{st.session_state.rec_class}'"
                    )

            # Draw
            hand_det.draw(frame, hands_data)

            # Recording overlay
            if st.session_state.recording:
                cv2.circle(frame, (w - 30, 50), 14, (0, 0, 230), -1, cv2.LINE_AA)
                cv2.putText(frame, "REC", (w - 90, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 230), 2, cv2.LINE_AA)

            # Summary track dict for display
            n_hands  = len(hands_data.get("hands", []))
            cur_trks = {
                "hands": {
                    "track_id": f"{n_hands} hand(s)",
                    "action": action,
                    "risk": risk,
                    "gate": "ACTIVE" if n_hands > 0 else "NO DETECT",
                    "mean_confidence": conf,
                    "bbox": [],
                }
            }
            alert_workers = [t for t in cur_trks.values() if t["risk"] >= risk_thresh]

        # ── Full-body mode: skeleton ──────────────────────────────────────────
        else:
            try:
                dets = body_ext.extract(frame=frame, metadata=meta)
            except Exception:
                dets = []

            cur_trks = {}
            for det in dets:
                det["frame_width"]  = w
                det["frame_height"] = h
                gr = body_gate.evaluate(det, w, h)
                track_id = str(det.get("track_id", "worker_000"))
                hist = st.session_state.track_history.setdefault(track_id, [])
                hist.append(det)
                if len(hist) > 30:
                    hist.pop(0)
                try:
                    fv = body_feat.extract(hist).get("vector", np.zeros(20)).tolist()
                except Exception:
                    fv = [0.0] * 20

                if trained and trainer.is_trained:
                    action, conf, risk = trainer.predict(np.array(fv, dtype=np.float32))
                else:
                    action, risk = _heuristic(fv)
                    conf = 0.0

                cur_trks[track_id] = {
                    "track_id": track_id,
                    "bbox": det.get("bbox", []),
                    "keypoints_17": det.get("keypoints_17", []),
                    "action": action,
                    "risk": risk,
                    "gate": gr.status.value,
                    "mean_confidence": conf or float(det.get("mean_confidence", 0)),
                }

                # Recording
                if st.session_state.recording:
                    st.session_state.rec_count += 1
                    trainer.add_sample(st.session_state.rec_class,
                                       np.array(fv, dtype=np.float32))
                    pct = st.session_state.rec_count / max(st.session_state.rec_target, 1)
                    rec_prog.progress(min(pct, 1.0),
                        text=f"Recording '{st.session_state.rec_class}' "
                             f"{st.session_state.rec_count}/{st.session_state.rec_target}")
                    if st.session_state.rec_count >= st.session_state.rec_target:
                        st.session_state.recording = False
                        rec_prog.success(
                            f"✅ Recorded {st.session_state.rec_target} frames "
                            f"for '{st.session_state.rec_class}'"
                        )

                draw_skeleton(frame, det.get("keypoints_17", []), risk=risk)
                draw_track_box(frame, det.get("bbox", []),
                               track_id, action, risk, gr.status.value)

            alert_workers = [t for t in cur_trks.values() if t["risk"] >= risk_thresh]

        st.session_state.latest_tracks = cur_trks

        # ── FPS & metrics ─────────────────────────────────────────────────────
        elapsed = time.perf_counter() - t0
        cur_fps = 1.0 / max(elapsed, 1e-6)
        st.session_state.fps_buf.append(cur_fps)
        avg_fps = float(np.mean(list(st.session_state.fps_buf)))

        n_workers   = len(cur_trks)
        n_alerts    = len(alert_workers)
        top_action  = list(cur_trks.values())[0]["action"] if cur_trks else "–"
        top_risk    = list(cur_trks.values())[0]["risk"]   if cur_trks else 0.0

        # Risk colour on action metric
        risk_col = "🟢" if top_risk < 0.35 else ("🟡" if top_risk < risk_thresh else "🔴")

        fps_ph.metric("FPS",     f"{avg_fps:.1f}")
        worker_ph.metric("Workers", n_workers)
        alert_ph.metric("🚨 Alerts", n_alerts)
        action_ph.metric("Action", f"{risk_col} {top_action}")
        frame_ph.metric("Frame", fidx)

        draw_hud(frame, avg_fps, fidx, n_workers, n_alerts)

        # ── Video frame ───────────────────────────────────────────────────────
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        vid_ph.image(frame_rgb, channels="RGB", width="stretch")  # type: ignore[arg-type]

        # ── Track table ───────────────────────────────────────────────────────
        if cur_trks:
            import pandas as pd
            rows = []
            for t in cur_trks.values():
                rv = t["risk"]
                em = "🟢" if rv < 0.35 else ("🟡" if rv < risk_thresh else "🔴")
                conf_str = (f"{t['mean_confidence']:.0%}"
                            if trained else "(heuristic)")
                rows.append({
                    "ID":     t["track_id"],
                    "Action": t["action"],
                    "Risk":   f"{em} {rv:.2f}",
                    "Gate":   t["gate"],
                    "Conf":   conf_str,
                })
            table_ph.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )
        else:
            table_ph.markdown(
                "<div style='color:#3a4060;padding:12px;font-size:.9rem;'>"
                "Nothing detected in frame</div>",
                unsafe_allow_html=True,
            )

        # ── Alert log ─────────────────────────────────────────────────────────
        if alert_workers:
            html = "".join(
                f"<div class='alert-box'>⚠️ <b>{t['track_id']}</b> — "
                f"{t['action']} | risk {t['risk']:.2f}</div>"
                for t in alert_workers
            )
            alert_log_ph.markdown(html, unsafe_allow_html=True)
        else:
            alert_log_ph.markdown(
                "<div class='safe-box'>✅ All nominal</div>",
                unsafe_allow_html=True,
            )

        # ── Frame rate cap ────────────────────────────────────────────────────
        rem = frame_delay - (time.perf_counter() - t0)
        if rem > 0:
            time.sleep(rem)

finally:
    cap.release()
    st.session_state.running = False
