# SafeRoadside — fullSystem

A roadside safety system that detects vehicles and pedestrians from a camera feed, predicts their future trajectories, identifies potential collisions, and broadcasts GPS-encoded alerts via BLE.

---

## Overview

The system is a **6-step pipeline** (scripts numbered `0` through `5`):

| Step | Script | Purpose |
|------|--------|---------|
| 0 | `0_createMap.py` | Calibrate pixel ↔ GPS mapping (homography) |
| 1 | `1_conv.py` | Verify calibration (interactive pixel → GPS tool) |
| 2 | `2_guideDraw.py` | Auto-detect vehicle paths via YOLO + DeepSort tracking |
| 3 | `3_drawTrajectories.py` | Manually draw/refine reference trajectories |
| 4 | `4_cleanTrajectories.py` | Downsample and smooth trajectories |
| 5 | `5_realTime.py` | Real-time collision prediction + GPS alerting |

---

## Prerequisites

### Hardware
- NVIDIA GPU with CUDA support
- BLE-capable Bluetooth adapter (for the `shared/sender2.py` alerting component)

### Software / Python packages
```
pip install opencv-python numpy ultralytics deep-sort-realtime mss pyautogui pygame pexpect
```

### Files required
- `tes.png` — aerial/camera image of the road scene (for calibration)
- `road.png` — clean image of the empty road (for trajectory overlay)
- `models/yolo11n.pt` — YOLOv11 nano model weights

---

## Step-by-step usage

### Step 0 — Create the pixel-to-GPS map

```bash
python 0_createMap.py
```

**What it does:**
1. Opens `tes.png` in a window.
2. You click **13 points** on the image, in the **same order** as the 13 GPS coordinates hard-coded in the script.
3. Computes a **homography matrix** (pixel → local metric XY) using RANSAC.

**Input:** `tes.png` + 13 GPS coordinates (edit `coords_str` in the script to match your location).  
**Output:** `map.txt` — reference origin (lat, lon) + 3×3 homography matrix.

> ⚠️ Edit the `coords_str` list in the script with your own GPS coordinates before running.

---

### Step 1 — Verify calibration (optional)

```bash
python 1_conv.py
```

**What it does:**  
Opens `tes.png`. Click anywhere to see the converted GPS coordinate in the terminal.

**Input:** `tes.png` + `map.txt`  
**Output:** Terminal printout of lat/lon for each clicked pixel.

---

### Step 2 — Auto-detect vehicle guide trajectories

```bash
python 2_guideDraw.py
```

**What it does:**
1. Captures your **entire screen** in real time (display a traffic camera feed fullscreen).
2. Runs **YOLO** object detection + **DeepSort** multi-object tracking.
3. Records the center-point pixel path of every detected vehicle.
4. On exit (`q` key), overlays all tracked paths on `road.png`.

**Input:** Live screen capture + `road.png` + `models/yolo11n.pt`  
**Output:** `withguide.png` — road image with colored trajectory dots.

---

### Step 3 — Draw reference trajectories

```bash
python 3_drawTrajectories.py
```

**What it does:**  
Opens a **Pygame** canvas with `withguide.png` as background. Draw clean reference trajectories:
- **Left-click + drag** → freehand drawing
- **Right-click** twice → straight line between two points

Close the window to save.

**Input:** `withguide.png`  
**Output:** `trajetorias.txt` — raw pixel trajectories (`Traj: (x1, y1),(x2, y2),...`).

---

### Step 4 — Clean trajectories

```bash
python 4_cleanTrajectories.py
```

**What it does:**
1. Removes points closer than **20 px** to the previous selected point (downsampling).
2. Applies a **moving-average smoothing** filter (window = 5).

**Input:** `trajetorias.txt`  
**Output:** `trajetoriasClean.txt` — cleaned and smoothed trajectories.

---

### Step 5 — Real-time collision detection & GPS alerting

```bash
python 5_realTime.py
```

**What it does:**
1. Captures the screen in real time and detects/tracks vehicles and pedestrians.
2. **Vehicles**: predicts future position by matching against pre-defined trajectories (sliding-window best match from `trajetoriasClean.txt`).
3. **Pedestrians**: predicts future position using an **Extended Kalman Filter (EKF)** with a constant-acceleration model.
4. **Collision detection**: if any predicted vehicle point is within **30 px** of any predicted pedestrian point → alert.
5. Converts the collision pixel to **GPS coordinates** via the homography matrix.
6. Encodes GPS as **hex bytes** and writes to `shared/data.txt` (consumed by the BLE sender).

**Input:** Live screen capture + `map.txt` + `trajetoriasClean.txt` + `models/yolo11n.pt`  
**Output:**
- On-screen: bounding boxes, predicted trajectories (colored dots), red collision markers.
- Terminal: `[ALERTA] Possível colisão futura em pixel=(x,y) -> lat/lon=(lat, lon)`
- File: hex-encoded GPS appended to `shared/data.txt` (max 5 lines).

---

## Data flow diagram

```
tes.png + GPS coords
        │
        ▼
  [0_createMap.py]
        │
        ▼
     map.txt  (homography + origin lat/lon)
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
  [1_conv.py]                            [5_realTime.py]
  (verification)                               ▲
                                               │
  Screen feed ──► [2_guideDraw.py]             │
                        │                      │
                        ▼                      │
                   withguide.png               │
                        │                      │
                        ▼                      │
                [3_drawTrajectories.py]         │
                        │                      │
                        ▼                      │
                  trajetorias.txt               │
                        │                      │
                        ▼                      │
               [4_cleanTrajectories.py]        │
                        │                      │
                        ▼                      │
                trajetoriasClean.txt ───────────┘
                                               │
                                               ▼
                                         shared/data.txt
                                               │
                                               ▼
                                       [sender2.py] (BLE broadcast)
```

---

## File descriptions

| File | Description |
|------|-------------|
| `map.txt` | Homography matrix + reference lat/lon origin |
| `road.png` | Clean road image (empty, no vehicles) |
| `tes.png` | Aerial/camera image used for calibration |
| `withguide.png` | Road image overlaid with auto-detected vehicle paths |
| `trajetorias.txt` | Raw manually-drawn reference trajectories |
| `trajetoriasClean.txt` | Cleaned + smoothed reference trajectories |
| `models/yolo11n.pt` | YOLOv11 nano weights for object detection |

---

## Key algorithms

### Vehicle trajectory prediction
Matches the observed vehicle path against pre-defined reference trajectories using a **sliding-window distance metric**. If the cumulative point-to-point distance is below a threshold (3000), the continuation of the best-matching reference trajectory is used as the prediction.

### Pedestrian trajectory prediction
Uses an **Extended Kalman Filter (EKF)** with 6 state variables `(x, y, vx, vy, ax, ay)` — position, velocity, and acceleration. The filter is fed with the last 10 observed positions and then extrapolated 5 steps into the future.

### Collision detection
All predicted vehicle points are compared against all predicted pedestrian points. If any pair is closer than **30 pixels**, a collision alert is triggered.

### GPS conversion
Pixel coordinates are converted to GPS via: `pixel → homography → local XY (meters) → lat/lon` using an inverse equirectangular projection centered at the calibration origin.

---

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit the current script |
| `ESC` | Exit (in OpenCV-based scripts) |

