"""Kinematics for roadside-unit tracks: pure, importable, tested.

Kept out of 5_realTime.py, which loads YOLO and DeepSort at import time and so
cannot be imported by a test. Three defects lived in the inline version it
replaces:

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


def ground_resolution_m(u, v, H, lat0_deg, lon0_deg, dv=1.0):
    """Metres of ground per pixel row at (u, v): the homography's own depth
    resolution there, and therefore the position error of a detection at that
    pixel. Returns None when either row maps to the horizon.

    A ground-plane homography places a pixel exactly; what it cannot do is
    place it *precisely* far away, because the rows crowd together towards the
    vanishing line. At BIB_IST one row is worth centimetres near the camera and
    metres at the far kerb. Expressing that as metres of error puts the RSU on
    the same confidence scale as a phone's GNSS accuracy
    (`ble_in.confidence_from_accuracy`), which is what R8 needs in order to
    weigh the two against each other at all.
    """
    a = pixel_to_latlon(u, v, H, lat0_deg, lon0_deg)
    b = pixel_to_latlon(u, v - dv, H, lat0_deg, lon0_deg)
    if a is None or b is None:
        return None
    lat0 = math.radians(a[0])
    east = math.radians(b[1] - a[1]) * EARTH_R_M * math.cos(lat0)
    north = math.radians(b[0] - a[0]) * EARTH_R_M
    return math.hypot(east, north) / dv


def ground_point_error_m(ltrb, H, lat0_deg, lon0_deg):
    """How far apart the two ground points a detection could be placed at are.

    A box has no single ground truth: its centre and the middle of its bottom
    edge both project onto the road, and they disagree by more the further away
    the object is and the lower the camera sits (`observation_latlon` documents
    why -- a vehicle's centre floats ~0.75 m above the road, so through a
    ground-plane homography it lands beyond the vehicle).

    That disagreement IS the position error, measured rather than modelled: it
    needs no assumption about object height, only the homography and the box.
    In metres, so it feeds the same curve a phone's GNSS accuracy does
    (`ble_in.confidence_from_accuracy`) and both sources land on one scale --
    which is what R8 needs to weigh them against each other.

    Returns None when either point maps to the horizon.
    """
    centre = observation_latlon(ltrb, H, lat0_deg, lon0_deg, ground_point="centre")
    bottom = observation_latlon(ltrb, H, lat0_deg, lon0_deg, ground_point="bottom")
    if centre is None or bottom is None:
        return None
    lat0 = math.radians(centre[0])
    east = math.radians(bottom[1] - centre[1]) * EARTH_R_M * math.cos(lat0)
    north = math.radians(bottom[0] - centre[0]) * EARTH_R_M
    return math.hypot(east, north)


#: Range at which the camera is as good as it gets, and where it stops being
#: worth trusting. Scene-calibrated, exactly as ble_in's ACC_FULL_M/ACC_ZERO_M
#: are hardware-calibrated: on BIB_IST a hard cut at 40 m took the per-second
#: alert from lift 1.00 to 1.17 and at 30 m to 1.42, so the camera's usable
#: range ends somewhere in that band. A low mount ends sooner -- INESC_IST is
#: usable to ~40 m against BIB_IST's ~120 m -- so these belong to the scene.
RANGE_FULL_M = 20.0
RANGE_ZERO_M = 60.0
CONF_MAX = 0.95
CONF_MIN = 0.25


def confidence_from_range(range_m, full_m=RANGE_FULL_M, zero_m=RANGE_ZERO_M):
    """How much to trust a detection that far from the camera, on 0..1.

    The RSU publishes a constant 0.85 today, which is what leaves R8's source
    confidence half unimplemented: a detection at 120 m, where the homography's
    depth resolution has collapsed, is handed to fusion as exactly as
    trustworthy as one at 15 m. Linear between the two knees, flat outside them,
    and on the same 0.95..0.25 scale as `ble_in.confidence_from_accuracy` so the
    camera and a phone can be weighed against each other.

    None (no range available) scores CONF_MAX: absence of a measurement must not
    masquerade as a bad one -- failing open is the lesson of the kinematics gate.
    """
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
    """Ground point at the bottom centre of the frame: the nearest road surface
    the camera can see, and a stand-in for the camera's own position (which the
    homography cannot give, since the camera is not on the ground plane).

    Ranges are measured from here, so "beyond N metres" means N metres from the
    RSU rather than from the arbitrary ground point chosen as the map origin.
    """
    return pixel_to_latlon(frame_w / 2.0, frame_h - 1.0, H, lat0_deg, lon0_deg)


def observation_latlon(ltrb, H, lat0_deg, lon0_deg, *, ground_point="centre",
                       max_range_m=None, reference=None):
    """Where a detection box sits on the ground, or None if it must not be published.

    `ground_point`:
      "centre" -- the box's midpoint, which the earlier figures were
        measured with. A vehicle's centre floats ~0.75 m above
        the road, so through a ground-plane homography it lands *beyond* the
        vehicle; the error grows with range and with how low the camera sits.
      "bottom" -- the middle of the box's bottom edge, i.e. where the wheels
        meet the road, which is the point the homography actually describes.

    `max_range_m` with `reference` (see camera_reference_latlon) drops anything
    further than that. Depth resolution collapses towards the horizon -- at
    INESC_IST one pixel row is worth 2.2 m at image row 1200 -- so a detection
    a row above the horizon publishes at a kilometre instead of being dropped,
    and association would pair a VRU against it. Off by default: with no limit
    nothing is filtered, which is how BIB_IST was measured.
    """
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
