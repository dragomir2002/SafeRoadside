"""Kinematics and geometry helpers for roadside-unit tracks."""
from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

EARTH_R_M = 6_371_000.0


class ReplayClock:
    """Scene time for recorded footage: one video frame per frame read."""

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
    """Image pixel to (lat, lon) through the ground-plane homography;
    None when the point maps to the horizon."""
    x, y, w = np.asarray(H, dtype=np.float64) @ np.array([float(u), float(v), 1.0])
    if abs(w) < 1e-12:
        return None
    X, Y = float(x / w), float(y / w)
    lat0 = math.radians(lat0_deg)
    lat = math.degrees(Y / EARTH_R_M + lat0)
    lon = math.degrees(X / (EARTH_R_M * math.cos(lat0)) + math.radians(lon0_deg))
    return lat, lon


def ground_resolution_m(u, v, H, lat0_deg, lon0_deg, dv=1.0):
    """Metres of ground per pixel row at (u, v); None at the horizon."""
    a = pixel_to_latlon(u, v, H, lat0_deg, lon0_deg)
    b = pixel_to_latlon(u, v - dv, H, lat0_deg, lon0_deg)
    if a is None or b is None:
        return None
    lat0 = math.radians(a[0])
    east = math.radians(b[1] - a[1]) * EARTH_R_M * math.cos(lat0)
    north = math.radians(b[0] - a[0]) * EARTH_R_M
    return math.hypot(east, north) / dv


def ground_point_error_m(ltrb, H, lat0_deg, lon0_deg):
    """Metres between the two ground points a box could be placed at."""
    centre = observation_latlon(ltrb, H, lat0_deg, lon0_deg, ground_point="centre")
    bottom = observation_latlon(ltrb, H, lat0_deg, lon0_deg, ground_point="bottom")
    if centre is None or bottom is None:
        return None
    lat0 = math.radians(centre[0])
    east = math.radians(bottom[1] - centre[1]) * EARTH_R_M * math.cos(lat0)
    north = math.radians(bottom[0] - centre[0]) * EARTH_R_M
    return math.hypot(east, north)


# Range in metres of full camera confidence, and of none.
RANGE_FULL_M = 20.0
RANGE_ZERO_M = 60.0
CONF_MAX = 0.95
CONF_MIN = 0.25


def confidence_from_range(range_m, full_m=RANGE_FULL_M, zero_m=RANGE_ZERO_M):
    """Confidence in a detection at that range from the camera, 0 to 1."""
    if range_m is None:
        return CONF_MAX
    if range_m <= full_m:
        return CONF_MAX
    if range_m >= zero_m:
        return CONF_MIN
    frac = (range_m - full_m) / (zero_m - full_m)
    return CONF_MAX - frac * (CONF_MAX - CONF_MIN)


def range_from_m(reference, lat, lon):
    """Ground distance in metres from the camera reference to a point."""
    lat0 = math.radians(reference[0])
    east = math.radians(lon - reference[1]) * EARTH_R_M * math.cos(lat0)
    north = math.radians(lat - reference[0]) * EARTH_R_M
    return math.hypot(east, north)


def camera_reference_latlon(H, lat0_deg, lon0_deg, frame_w, frame_h):
    """Ground point at the bottom centre of the frame: the camera's position."""
    return pixel_to_latlon(frame_w / 2.0, frame_h - 1.0, H, lat0_deg, lon0_deg)


def observation_latlon(ltrb, H, lat0_deg, lon0_deg, *, ground_point="centre",
                       max_range_m=None, reference=None):
    """Ground position of a detection box; None if it is not to be published.
    ground_point is "centre" or "bottom" of the box."""
    x1, y1, x2, y2 = (float(v) for v in ltrb)
    u = (x1 + x2) / 2.0
    v = y2 if ground_point == "bottom" else (y1 + y2) / 2.0
    latlon = pixel_to_latlon(u, v, H, lat0_deg, lon0_deg)
    if latlon is None or max_range_m is None or reference is None:
        return latlon
    lat0 = math.radians(reference[0])
    east = math.radians(latlon[1] - reference[1]) * EARTH_R_M * math.cos(lat0)
    north = math.radians(latlon[0] - reference[0]) * EARTH_R_M
    return None if math.hypot(east, north) > max_range_m else latlon


class KinematicsEstimator:
    """Speed (m/s) and compass heading (deg) of each track.
    No heading at or below stationary_mps; no values before min_span_s."""

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
        """Forget tracks not updated within max_gap_s."""
        for k in [k for k, h in self._hist.items() if t - h[-1][0] > self.max_gap_s]:
            del self._hist[k]
