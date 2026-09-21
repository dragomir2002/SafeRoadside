# Scene: INESC_IST

Lisbon plaza with a curved corner building and a pastelaria on the right,
captured with a GoPro from a **low, static mount** (roughly pedestrian height).

| | |
|---|---|
| Source video | `SafeRoadside/Videos/INESC_IST/GX010088.MP4` (not in git) |
| Resolution | 3840x2160 |
| Frame rate | 29.97 fps |
| Frames | 9 069 (5 min 03 s) |
| Codec | HEVC |
| Camera | static, ~2.2 px drift end to end (steadier than BIB_IST's ~15) |
| Map origin | 38.736741, -9.141845 |
| Cyclists | yes -- the only scene here that has any |

## Coordinate space

Everything in this folder lives in **3840x2160 pixel space**: `road.png`,
`tes.png`, `map.txt`, `guide_tracks.txt`, `trajetorias.txt`,
`trajetoriasClean.txt` and the source video all agree. `tes.png` and `road.png`
are frame 3600, the emptiest frame of the clip (11 vehicles, 2 people).

## Run it

The detector needs two scene-specific switches here, both added 2026-09-21:

```powershell
.\run-bib.ps1 -Scene inesc_ist -- --ground-point bottom --max-range-m 60
```

* `--ground-point bottom` projects the **wheels** of a detection box rather than
  its centre. A box centre floats ~0.75 m above the road, and through a
  ground-plane homography that lands far behind the vehicle: at this camera's
  height, a box centre at image row 1235 reads 40.5 m while its bottom edge
  reads 16.2 m. BIB_IST's elevated mount keeps that error near 10 m and uses the
  centre, which is what its figures were measured with.
* `--max-range-m 60` drops anything further away. Depth resolution collapses
  towards the horizon (image row ~1130 here): 0.12 m per pixel row at row 1300,
  0.64 at row 1240, 2.23 at row 1200. Without the limit a detection one row
  above the horizon is published at a kilometre, and association pairs against
  it. `rsu_kinematics.observation_latlon` implements both.

**Usable depth is about 10-40 m**, which bounds what any warning here can
achieve: lead time = range / closing speed, so 40 m gives 5.7 s against a 7 m/s
vehicle (the local median) but only 2.9 s against a 14 m/s one.

## Files

| File | Produced by | Notes |
|---|---|---|
| `tes.png` | manual | calibration still (frame 3600, identical to `road.png`) |
| `road.png` | manual | clean road frame, background for overlays |
| `map.txt` | step 0 | origin + 3x3 homography |
| `points_clicked.csv` | step 0 | the 13 ground points and their pixels, so a re-fit needs no re-clicking |
| `calib_check.png` | step 0 | each point's coordinate projected back, over a 5 m ground grid |
| `gcp_candidates.png`, `gcp.csv` | manual | candidate points marked on the still, for a future re-survey |
| `withguide.png`, `guide_tracks.txt` | step 2 | 274 vehicle paths, 51 076 points |
| `trajetorias.txt`, `traj_preview.png` | step 2b | 73 reference trajectories (146 rejected: too few points, too short, stationary, not straight) |
| `trajetoriasClean.txt` | step 4 | 73 trajectories, 1 989 points, 40 px spacing |
| `models/yolo11n.pt` | copied | real file, not the broken symlink from git |

`map.rejected.txt` / `calib_check.rejected.png` are a first calibration attempt
kept as a warning: it was fitted against the other scene's coordinates.

## Calibration quality

13 ground points, all kept by RANSAC: **residual RMS 1.00 m, worst
leave-one-out 5.06 m** (the leave-one-out error is what a point's removal costs,
so it, not the residual, is what catches a misclick). Independently checked by
mapping all 51 076 tracked points through the homography and differencing them:
median vehicle speed **6.8 m/s (25 km/h)**, p90 13.3 m/s (48 km/h), which is
ordinary urban traffic and would not come out of a wrong map.

## Reproducing

From this directory, with the gateway venv as the interpreter:

```powershell
$PY = "..\..\..\..\SafeCorners-Gateway\.venv\Scripts\python.exe"
$VIDEO = "..\..\..\Videos\INESC_IST\GX010088.MP4"

# step 0 - calibration: put the ground coordinates in 0_createMap.py's
# coords_str, then click them on tes.png in the same order
& $PY ..\..\0_createMap.py

# step 2 - vehicle guide paths (~37 min)
& $PY ..\..\2_guideDraw.py --source $VIDEO --stride 3 --imgsz 1920 --conf 0.25 `
      --no-display --out withguide.png --dump-tracks guide_tracks.txt

# step 2b - reference trajectories (seconds)
& $PY ..\..\2b_tracksToTrajectories.py --tracks guide_tracks.txt `
      --out trajetorias.txt --preview traj_preview.png --background road.png

# step 4 - downsample to the detector's 40 px spacing
& $PY ..\..\4_cleanTrajectories.py --input trajetorias.txt `
      --output trajetoriasClean.txt --frame-width 3840
```

## Gotchas hit on this scene

* **Calibration points must span depth.** A first fit used points spread across
  the image but all within image rows 1276-1557 (the foreground). It scored well
  on its own points and then extrapolated: image row 1200 mapped to 127 m, row
  1170 landed past its own horizon. Half of all vehicle detections sit above row
  1235, so the far field has to be constrained by real points.
* **Never use building corners as ground points.** Satellite imagery shows a
  roof, which is metres from its base.
* **The clicking window is fitted to the screen.** `0_createMap.py` used to show
  the 4K still at 1:1, so on a smaller display the far half of the scene could
  not be reached at all.
