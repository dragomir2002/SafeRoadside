# Demo Recipe — phone → BLE → Windows scanner → WSL bridge → gateway → detector

End-to-end test: real SafeWalk PSMs from your phone in Lisbon get coordinate-translated into the YouTube-intersection video space, fused against detector tracks, and produce warnings.

## One-time setup

### A. Files already in place
- `demo_video.mp4` — 60 s clip from https://www.youtube.com/watch?v=1H0iTzv2jiQ
- `models/yolo11n.pt`
- `gateway.demo.yaml` — loosened association/fusion thresholds
- `gateway_glue.py` — exposes `start_safewalk_http_bridge()`

### B. Wire the bridge into the detector

Edit `5_realTime.py`:

**Line ~16** — extend the import:
```python
from gateway_glue import start_gateway, publish as gw_publish, start_safewalk_injector, start_safewalk_http_bridge
```

**Right after `start_safewalk_injector()` (around line 434)** — add:
```python
start_safewalk_http_bridge()   # listens on 0.0.0.0:8765, anchors to map.txt origin
```

### C. WSL IP for Windows/phone to reach
```
192.168.1.216
```
Re-check after a reboot with `ip addr show eth0 | grep inet`.

### D. Patch the Windows scanner

File: `C:\Users\dlevitch\OneDrive - Cisco\Documents\scan_safewalk.py`

Add at the top:
```python
import requests
WSL_BRIDGE = "http://192.168.1.216:8765/psm"
```

Inside the PSM-decode callback (where lat/lon currently print), add:
```python
try:
    requests.post(WSL_BRIDGE, json={
        "lat":         obs_lat,
        "lon":         obs_lon,
        "track_id":    device_id,
        "speed_mps":   obs_speed,
        "heading_deg": obs_heading,
        "secMark":     obs_secmark,
    }, timeout=0.5)
except Exception as e:
    print(f"[bridge] POST failed: {e}")
```
Rename the variables to match what your script actually exposes.

---

## Run order

### Terminal 1 — WSL — detector + gateway + bridge
```bash
cd /home/dlevitch/Projects/Thesis/SafeRoadside/fullSystem
source /home/dlevitch/Projects/Thesis/SafeCorners-Gateway/.venv/bin/activate
python 5_realTime.py --source demo_video.mp4 --loop
```

Look for these log lines:
- `safecorners gateway started on background thread`
- `safewalk inject watcher started (write to /tmp/safewalk_inject.txt)`
- `safewalk HTTP bridge listening on 0.0.0.0:8765 (video anchor=41.940499, -85.000637)`
- `[INFO] YOLO device: cpu` (or `cuda`)

The OpenCV window should pop up showing the YouTube intersection with bounding boxes + IDs. Console will spit `[GW] pedestrian track N at lat=…` lines for every detected person.

### Terminal 2 — Windows PowerShell — your BLE scanner
```
python "C:\Users\dlevitch\OneDrive - Cisco\Documents\scan_safewalk.py"
```

First PSM that arrives sets the **Lisbon home anchor**. Subsequent PSMs are translated: walk 5 m east in real life → your avatar moves 5 m east in the video. You'll see this in Terminal 1:
```
[BRIDGE] home anchor set: real=(38.7223…, -9.1393…) -> video=(41.940499, -85.000637)
```

### Terminal 3 — WSL — watch fusion / dispatch
```bash
tail -f /home/dlevitch/Projects/Thesis/SafeRoadside/fullSystem/gateway.jsonl | grep -E 'matched|fuse|dispatch'
```

What to look for:
- `"matched": true, "score": 0.x, "dist_m": …` — phone associated with a vehicle near "you"
- `"stage": "fuse", "risk": "imminent"|"probable"` — fusion produced a warning
- `"stage": "dispatch", "action": "emitted"` — warning sent (BLE-out is a no-op on WSL, log-only)

### Stop
Ctrl-C the detector. Ctrl-C the Windows scanner.

---

## Sanity tests (do these BEFORE involving the phone)

### Test 1: bridge alone (curl from anywhere)
```bash
# First POST: sets home anchor
curl -X POST http://192.168.1.216:8765/psm \
  -H 'Content-Type: application/json' \
  -d '{"lat": 38.7223, "lon": -9.1393, "track_id": "TEST"}'
# expect: {"ok":true}

# Second POST: ~5 m east of home → ~5 m east of video anchor
curl -X POST http://192.168.1.216:8765/psm \
  -H 'Content-Type: application/json' \
  -d '{"lat": 38.7223, "lon": -9.13924, "track_id": "TEST"}'
# expect: {"ok":true}

# Bridge status:
curl http://192.168.1.216:8765/
# expect: home=(38.7223…, -9.1393…)  video_anchor=(41.940499, -85.000637)
```

Confirm the second POST shows up in `gateway.jsonl` with lat ~41.940499, lon ~ -85.0006 (a few meters east of the origin).

### Test 2: full pipeline, no phone, no scanner
```bash
cd /home/dlevitch/Projects/Thesis/SafeRoadside/fullSystem
python -c "
import sys, time; sys.path.insert(0, '.')
from gateway_glue import start_gateway, publish, start_safewalk_injector, INJECT_FILE
start_gateway(); start_safewalk_injector(); time.sleep(1)
INJECT_FILE.write_text('41.9405 -85.0006 FAKE\n'); time.sleep(0.5)
for _ in range(3):
    publish('V1', 'car', 41.94055, -85.0006, speed_mps=5.0, heading_deg=180.0)
time.sleep(2); INJECT_FILE.unlink(missing_ok=True)
"
grep -E 'matched|fuse|dispatch' gateway.jsonl | tail -10
```
Expected: at least one `dispatch action=emitted risk=imminent` line.

---

## Troubleshooting

| symptom | likely cause | fix |
|---|---|---|
| Windows can't reach `192.168.1.216:8765` | WSL IP changed | re-run `ip addr show eth0` in WSL, update `WSL_BRIDGE` in scan_safewalk.py |
| `safewalk HTTP bridge listening` line never appears | step B above not done | add `start_safewalk_http_bridge()` after `start_safewalk_injector()` |
| `gateway.jsonl` shows pedestrian observations but no `matched: true` | vehicle and pedestrian too far apart | move video anchor closer to busy traffic — edit `start_safewalk_http_bridge(video_anchor_lat=…, video_anchor_lon=…)` |
| `[GW] pedestrian` lines but no fusion warnings | demo config not loaded | confirm `gateway.demo.yaml` exists in `SafeCorners-Gateway/`; gateway_glue prefers it over `gateway.yaml` |
| `safecorners_gateway not importable` | wrong venv | source `SafeCorners-Gateway/.venv/bin/activate` |
| YOLO crashes on `.to('cuda')` | no GPU in WSL | already guarded — should print `[INFO] YOLO device: cpu` |

---

## What's still pending

- **Visual overlay**: the SafeWalk avatar shows up in `gateway.jsonl` but isn't drawn on the OpenCV window. Need to project (lat,lon) → pixel via inverse homography and draw a distinct marker ("YOU") in `5_realTime.py`'s render loop.
- **Better video anchor**: `map.txt` origin may not be a high-traffic spot. Eyeball the video, pick a lat/lon where cars frequently pass, and override via `start_safewalk_http_bridge(video_anchor_lat=…, video_anchor_lon=…)`.
