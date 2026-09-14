# Scene: BIB_IST

Lisbon intersection (Av. de Berna / P. Londres direction signs visible),
captured with a GoPro from an elevated static mount.

| | |
|---|---|
| Source video | `SafeRoadside/Videos/BIB_IST/GX010090.MP4` (not in git) |
| Resolution | 3840x2160 |
| Frame rate | 29.97 fps |
| Frames | 10 005 (5 min 34 s) |
| Codec | HEVC |
| Camera | static (max ~15 px drift end-to-end, i.e. mount vibration only) |
| Map origin | 38.736407, -9.143549 |

## Coordinate space

Everything in this folder lives in **3840x2160 pixel space**:
`road.png`, `tes.png`, `withguide.png`, `map.txt`, `trajetorias.txt`,
`trajetoriasClean.txt`, and the source video all agree.

This matters. `find_best_trajectory` matches live track points against the
reference trajectories by **absolute position**, so a reference set recorded at
a different resolution silently never matches. Two things enforce the
agreement:

* `2_guideDraw.py --source <video>` records track points in video pixel space
  and warns if the source resolution differs from `road.png`.
* `3_drawTrajectories.py` displays the background fit-to-window but scales every
  saved point back to the full image resolution.

## Files

| File | Produced by | Notes |
|---|---|---|
| `tes.png` | manual | calibration still (identical to `road.png`) |
| `road.png` | manual | clean road frame, background for overlays |
| `map.txt` | step 0 | origin + 3x3 homography |
| `withguide.png` | step 2 | 315 tracked vehicle paths drawn on `road.png` |
| `guide_tracks.txt` | step 2 | raw per-track pixel paths (315 tracks, 74 265 points) |
| `trajetorias.txt` | step 2b | 106 filtered reference trajectories |
| `traj_preview.png` | step 2b | trajectories overlaid (green = start, red = end) |
| `trajetoriasClean.txt` | step 4 | 106 trajectories, 3 566 points, 40 px spacing |
| `models/yolo11n.pt` | copied | real file, not the broken symlink from git |

## Reproducing

Run everything **from this directory** — the scripts resolve `map.txt`,
`road.png` and `models/` relative to the working directory.

Use the gateway venv: it is the only interpreter here with both the CV stack
and `safecorners_gateway` installed.

```bash
cd SafeRoadside/fullSystem/scenes/bib_ist
PY="../../../../SafeCorners-Gateway/.venv/Scripts/python.exe"
VIDEO="../../../Videos/BIB_IST/GX010090.MP4"
```

### Step 2 - detect vehicle guide paths (~55 min)

```bash
$PY ../../2_guideDraw.py --source "$VIDEO" \
    --stride 3 --imgsz 1920 --conf 0.25 --no-display \
    --out withguide.png --dump-tracks guide_tracks.txt
```

`--imgsz 1920` matters on 4K input: at the default 640 the model finds roughly
half as many vehicles (10 vs 23 on frame 0).

### Step 2b - derive reference trajectories (seconds)

```bash
$PY ../../2b_tracksToTrajectories.py \
    --tracks guide_tracks.txt --out trajetorias.txt \
    --preview traj_preview.png --background road.png
```

Filters out parked cars (large point count, no net displacement), detector
flicker, and tracker ID switches. On this scene: 315 raw tracks -> 106 kept
(182 too few points, 102 too short, 72 stationary, 4 not straight enough).

This replaces the manual step 3. To draw by hand instead:
`$PY ../../3_drawTrajectories.py` then skip to step 4.

### Step 4 - downsample + smooth (instant)

```bash
$PY ../../4_cleanTrajectories.py --input trajetorias.txt \
    --output trajetoriasClean.txt --frame-width 3840
```

`--frame-width 3840` sets the downsample spacing to 40 px. That is not
cosmetic: it must equal the detector's `MIN_DISTANCE_THRESHOLD`, which is also
scaled to 40 px at 4K, or the sliding-window match compares point sequences
sampled at different rates.

### Step 5 - real-time detection, prediction, fusion (~33 min)

```bash
$PY ../../5_realTime.py --source "$VIDEO" \
    --no-display --save-video bib_ist_output.mp4
```

Add `--no-gateway` for the pure computer-vision path with no fusion.
Drop `--no-display` for the live preview (downscaled by `--display-width`,
default 1600).

Before a run that should record BLE alerts, clear the outbox — it is capped at
five pending lines and silently discards everything after that:

```bash
> ../../../shared/data.txt
```

## Gotchas hit on this scene

1. **`models` was a broken symlink.** Git on Windows checks out
   `scenes/bib_ist/models` as a 12-byte text file containing `../../models`.
   Replaced with a real copy of `yolo11n.pt`.
2. **`withguide.png` was a placeholder** — pixel-identical to `road.png`.
   Step 2 had never actually been run for this scene.
3. **The SafeWalk anchor was hardcoded to the Michigan demo clip**
   (41.9407, -85.0010). On a Lisbon scene the avatar landed ~7 000 km away and
   fusion could never associate. It now defaults to the scene's `map.txt`
   origin.
4. **Pixel thresholds were tuned at 1080p.** At 4K each one meant half the
   real-world distance. They are now scaled by `frame_width / 1920`
   (collision 30 -> 60 px, trajectory score 3 000 -> 6 000, point spacing
   20 -> 40 px), each overridable from the CLI.
5. **`shared/data.txt` was full of stale Michigan alerts**, so no new alert
   could be written. Step 5 now says so instead of failing silently.

## Known algorithmic limitation

The step-5 collision test compares **every** predicted vehicle point against
**every** predicted pedestrian point with no time alignment. The vehicle
prediction runs up to 50 reference points ahead — a median of ~1 440 px, very
roughly 20 m of road. So a car that will reach a pedestrian's position in five
seconds raises the same alert as one about to hit them, which is why a busy
intersection produces alerts on most frames.

It is a *path-intersection* test, not a time-to-collision test. The
SafeCorners gateway's `fuse` stage is the layer that computes actual TTC and
grades risk (`none` / `low` / `probable` / `imminent`), which is the number to
quote in the evaluation rather than the raw `[ALERTA]` count.
