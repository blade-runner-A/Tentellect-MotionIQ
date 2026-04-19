# Hardware Capture Blueprint (Own Data Collection)

This design is intentionally compatible with the current software stack in this repo.

## Goal

Capture synchronized industrial scene + worker motion data that can be fed directly into:

- `src/ingestion/ingestor.py`
- `src/skeleton/extractor.py`
- `src/features/extractor.py`
- `src/serving/realtime_server.py`

## Recommended Name for the Physical Unit

Use **MotionIQ Node** for the edge capture hardware.

- **System software platform:** **Tentellect MotionIQ**
- **Hardware capture device:** **MotionIQ Node**
- **What it is:** a realtime worker-pose and risk intelligence system for industrial automation and robotics.

## Minimum Viable Hardware (V1)

### Compute

- NVIDIA Jetson Orin NX (preferred) or Orin Nano (budget)
- 16GB RAM recommended
- NVMe SSD 512GB+

### Cameras

- 1x global-shutter RGB camera (1080p, 30 FPS, wide dynamic range)
- Optional second camera for alternate angle

### IMU (optional in V1 but recommended in V2)

- 6-axis IMU module (accelerometer + gyro) for wearable band integration
- BLE or wired microcontroller bridge for timestamped IMU packets

### Time Sync

- NTP sync on boot
- single monotonic clock source for frame and sensor timestamps

### Mechanical

- vibration-isolated mount
- protected enclosure (IP54+ depending environment)
- fixed field-of-view marked on floor for zone mapping repeatability

## Data Format Contract (must follow)

To stay compatible with the existing software:

- Video input saved as `.mp4` or streamed RTSP
- Frame metadata must include:
  - `frame_idx`
  - `timestamp_ms`
  - `source_id`
  - `session_id`
- Skeleton output schema already expected by pipeline:
  - `track_id`, `bbox`, `keypoints_17`, `mean_confidence`, `quality_gate`, `ppe`

## Capture Workflow

1. Start session with deterministic ID:
   - `plantA_line2_shift1_YYYYMMDD_HHMM`
2. Record RGB stream to rotating files (5 min chunks)
3. Log sensor packets with same `timestamp_ms` epoch
4. Write sidecar session manifest:
   - camera intrinsics
   - mount height/angle
   - site/line metadata
5. Run `scripts/run_pipeline.py` on captured chunks
6. Review low-confidence clips in UI/workflow

## Robot Integration Contract

Robots should consume only stable API payloads:

- `/state`: current worker tracks, risk, action
- `/robot/commands`: command suggestions

Robot controller should:

- poll at fixed interval (5-10 Hz)
- apply debounce (2-3 consecutive high-risk observations)
- enforce safety priority over task throughput

## V1 Build Priorities

1. Single camera + Jetson capture and local inference
2. Reliable power and storage logging
3. Stable session/timestamp contract
4. Safe robot command subscription

## V2 Upgrades

- dual-camera fusion
- full IMU wearable synchronization
- edge model acceleration (TensorRT)
- OTA model and config rollout

