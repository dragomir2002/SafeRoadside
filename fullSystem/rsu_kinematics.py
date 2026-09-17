"""Kinematics for roadside-unit tracks: pure, importable, tested.

Kept out of 5_realTime.py, which loads YOLO and DeepSort at import time and so
cannot be imported by a test. Three defects lived in the inline version it
replaces (review C, 2026-09-15):

1. float32 geometry. The homography was float32 and, under NumPy 2 (NEP 50),
   the scalars stayed float32 through `Y / R + lat0`, quantising every
   published latitude to a 0.380 m grid and longitude to 0.074 m.
2. Wall-clock speed. Speed was the displacement across ONE video frame
   (1/29.97 s of scene) divided by the WALL time between processed frames
   (~0.17-0.22 s), so every speed was ~19 % of the truth and every TTC ~5x
   too long -- by an amount that varied with host load.
3. Single-frame differencing. Heading was None whenever a frame step was
   <= 5 cm, i.e. for any pedestrian slower than ~1.5 m/s.
"""
from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

EARTH_R_M = 6_371_000.0


class ReplayClock:
    """Scene time for recorded footage, on the monotonic clock's scale.

    Advances by exactly one video frame per frame read, so every duration the
    gateway measures -- TTC, freshness, hysteresis, cooldown -- is in scene
    seconds however fast the host processes frames. Assigning and reading a
    float attribute is atomic in CPython, so the detector thread may set it
    while the gateway thread reads it.
    """

    def __init__(self, fps: float, origin: float | None = None) -> None:
        if not fps or fps <= 0:
            raise ValueError(f"replay clock needs the video frame rate, got {fps!r}")
        self.fps = float(fps)
        self.origin = time.monotonic() if origin is None else float(origin)
        self._t = self.origin

    def set_frame(self, frame_index: int) -> None:
        self._t = self.origin + frame_index / self.fps

    def __call__(self) -> float:
        return self._t


def pixel_to_latlon(u, v, H, lat0_deg, lon0_deg):
    """Image pixel -> (lat, lon) degrees through the ground-plane homography.

    float64 throughout, whatever dtype H arrives in. Returns None when the
    point maps to the horizon (|w| ~ 0). Same local projection as
    5_realTime.xy_to_latlon: X east, Y north, metres from (lat0, lon0).
    """
    x, y, w = np.asarray(H, dtype=np.float64) @ np.array([float(u), float(v), 1.0])
    if abs(w) < 1e-12:
        return None
    X, Y = float(x / w), float(y / w)
    lat0 = math.radians(lat0_deg)
    lat = math.degrees(Y / EARTH_R_M + lat0)
    lon = math.degrees(X / (EARTH_R_M * math.cos(lat0)) + math.radians(lon0_deg))
    return lat, lon


class KinematicsEstimator:
    """Speed (m/s) and compass heading (deg) per track over a scene-time baseline.

    update() returns (None, None) until a track has `min_span_s` of history --
    motion genuinely unknown. At or below `stationary_mps` it returns
    (speed, None): heading is undefined for a still body, and the gateway
    treats that combination as stationary (association.stationary_mps in
    gateway.yaml must equal this threshold).
    """

    def __init__(self, baseline_s: float = 1.0, min_span_s: float = 0.5,
                 stationary_mps: float = 0.5, max_gap_s: float = 1.0) -> None:
        self.baseline_s = baseline_s
        self.min_span_s = min_span_s
        self.stationary_mps = stationary_mps
        self.max_gap_s = max_gap_s
        self._hist: dict = {}

    def update(self, track_id, t: float, lat: float, lon: float):
        h = self._hist.get(track_id)
        if h is None or t - h[-1][0] > self.max_gap_s:
            h = deque()
            self._hist[track_id] = h
        h.append((t, lat, lon))
        # Keep the newest sample that is still at least baseline_s old.
        while len(h) > 1 and t - h[1][0] >= self.baseline_s:
            h.popleft()
        t0, lat0, lon0 = h[0]
        span = t - t0
        if span < self.min_span_s:
            return None, None
        north = math.radians(lat - lat0) * EARTH_R_M
        east = math.radians(lon - lon0) * EARTH_R_M * math.cos(math.radians(lat0))
        speed = math.hypot(east, north) / span
        if speed <= self.stationary_mps:
            return speed, None
        heading = math.degrees(math.atan2(east, north)) % 360.0
        return speed, (0.0 if heading >= 360.0 else heading)

    def prune(self, t: float) -> None:
        """Forget tracks not updated within max_gap_s (DeepSort ids are never reused)."""
        for k in [k for k, h in self._hist.items() if t - h[-1][0] > self.max_gap_s]:
            del self._hist[k]
