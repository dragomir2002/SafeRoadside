"""RSU kinematics: float64 geometry, scene-time speed, compass heading."""
import math
import random
from pathlib import Path

import numpy as np
import pytest

from rsu_kinematics import KinematicsEstimator, ReplayClock, pixel_to_latlon

SCENE_MAP = Path(__file__).resolve().parents[1] / "scenes" / "bib_ist" / "map.txt"
R = 6_371_000.0
VIDEO_FPS = 29.97002997


def _scene():
    lines = SCENE_MAP.read_text().splitlines()
    lat0, lon0 = map(float, lines[0].split())
    H = np.array([list(map(float, lines[i].split())) for i in (1, 2, 3)], dtype=np.float32)
    return lat0, lon0, H


def test_single_pixel_steps_are_resolved_even_from_a_float32_matrix():
    """Near the calibrated area one pixel is about 4 cm north."""
    lat0, lon0, H = _scene()
    prev = None
    for v in range(1400, 1440):
        lat, _ = pixel_to_latlon(3000, v, H, lat0, lon0)
        if prev is not None:
            step = math.radians(lat - prev) * R
            assert -0.05 < step < -0.03, f"v={v}: north step {step:.4f} m"
        prev = lat


def test_pixel_to_latlon_returns_none_on_the_horizon():
    H = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 0.0]])
    assert pixel_to_latlon(10, 10, H, 38.7, -9.1) is None


def test_replay_clock_advances_exactly_one_frame_per_frame():
    c = ReplayClock(VIDEO_FPS, origin=100.0)
    assert c() == 100.0
    c.set_frame(30)
    assert c() == pytest.approx(100.0 + 30 / VIDEO_FPS)


def test_replay_clock_rejects_a_missing_frame_rate():
    with pytest.raises(ValueError):
        ReplayClock(0.0)


def _drive(est, every_nth, speed_mps, heading_deg, seconds=2.0, lat0=38.7364, lon0=-9.1435):
    out = None
    for i in range(0, int(seconds * VIDEO_FPS), every_nth):
        t = i / VIDEO_FPS
        north = speed_mps * t * math.cos(math.radians(heading_deg))
        east = speed_mps * t * math.sin(math.radians(heading_deg))
        lat = lat0 + math.degrees(north / R)
        lon = lon0 + math.degrees(east / (R * math.cos(math.radians(lat0))))
        out = est.update("T", t, lat, lon)
    return out


@pytest.mark.parametrize("every_nth", [1, 5, 6])
def test_speed_is_in_scene_seconds_whatever_the_processing_rate(every_nth):
    """Processing every frame or every 5th/6th must give the same 6.1 m/s."""
    speed, heading = _drive(KinematicsEstimator(), every_nth, 6.1, 0.0)
    assert speed == pytest.approx(6.1, rel=0.01)
    assert min(heading, 360.0 - heading) == pytest.approx(0.0, abs=0.5)


def test_motion_is_unknown_until_half_a_second_of_history():
    est = KinematicsEstimator()
    assert est.update("T", 0.0, 38.7, -9.1) == (None, None)
    assert est.update("T", 0.3, 38.7, -9.1) == (None, None)


def test_a_still_body_has_a_speed_and_no_heading():
    est = KinematicsEstimator()
    for k in range(40):
        speed, heading = est.update("T", k / VIDEO_FPS, 38.7, -9.1)
    assert speed == 0.0 and heading is None


def test_jitter_on_a_still_body_stays_below_the_stationary_threshold():
    """+-5 cm per frame over a 1 s baseline is at most 0.14 m/s."""
    rng = random.Random(7)
    est = KinematicsEstimator()
    for k in range(60):
        dn, de = rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05)
        speed, heading = est.update("T", k / VIDEO_FPS, 38.7 + math.degrees(dn / R),
                                    -9.1 + math.degrees(de / (R * math.cos(math.radians(38.7)))))
    assert speed < 0.5 and heading is None


def test_a_slow_walker_gets_a_heading():
    """1.3 m/s is above the stationary threshold."""
    speed, heading = _drive(KinematicsEstimator(), 1, 1.3, 0.0)
    assert speed == pytest.approx(1.3, rel=0.01) and heading is not None


@pytest.mark.parametrize("bearing", [45.0, 90.0, 135.0, 225.0, 270.0, 315.0])
def test_heading_is_a_compass_bearing_in_every_quadrant(bearing):
    """Headings in all four quadrants."""
    _, heading = _drive(KinematicsEstimator(), 1, 5.0, bearing)
    assert heading == pytest.approx(bearing, abs=0.5)
    assert 0.0 <= heading < 360.0


def test_history_restarts_after_a_gap():
    est = KinematicsEstimator()
    _drive(est, 1, 5.0, 90.0)
    assert est.update("T", 10.0, 38.7, -9.1) == (None, None)


def test_prune_forgets_tracks_not_seen_recently():
    est = KinematicsEstimator()
    est.update("A", 0.0, 38.7, -9.1)
    est.prune(5.0)
    assert est.update("A", 5.0, 38.7, -9.1) == (None, None)
    assert list(est._hist) == ["A"]


# --- where a detection sits on the ground, and how far it may be ------------

INESC_MAP = Path(__file__).resolve().parents[1] / "scenes" / "inesc_ist" / "map.txt"


def _load_map(path):
    rows = [l.split() for l in path.read_text().splitlines() if l.strip()]
    return np.array([[float(v) for v in r] for r in rows[1:4]]), float(rows[0][0]), float(rows[0][1])


def _range_m(a, b):
    return math.hypot(math.radians(b[1] - a[1]) * R * math.cos(math.radians(a[0])),
                      math.radians(b[0] - a[0]) * R)


def test_the_bottom_of_the_box_is_nearer_than_its_centre():
    """The bottom of a box is on the road; its centre is above it."""
    from rsu_kinematics import observation_latlon, camera_reference_latlon
    H, lat0, lon0 = _load_map(INESC_MAP)
    ref = camera_reference_latlon(H, lat0, lon0, 3840, 2160)
    box = (1800.0, 1200.0, 2040.0, 1280.0)        # a car out at the far road
    centre = observation_latlon(box, H, lat0, lon0, ground_point="centre")
    bottom = observation_latlon(box, H, lat0, lon0, ground_point="bottom")
    assert _range_m(ref, bottom) < _range_m(ref, centre)


def test_the_centre_is_still_the_default():
    from rsu_kinematics import observation_latlon
    H, lat0, lon0 = _load_map(INESC_MAP)
    box = (1800.0, 1400.0, 2040.0, 1480.0)
    assert observation_latlon(box, H, lat0, lon0) == \
           observation_latlon(box, H, lat0, lon0, ground_point="centre")


def test_a_detection_past_the_horizon_is_dropped_by_the_range_guard():
    """A detection just above the horizon would publish a kilometre away."""
    from rsu_kinematics import observation_latlon, camera_reference_latlon
    H, lat0, lon0 = _load_map(INESC_MAP)
    ref = camera_reference_latlon(H, lat0, lon0, 3840, 2160)
    far = (1900.0, 1100.0, 1940.0, 1150.0)
    assert observation_latlon(far, H, lat0, lon0) is not None  # guard off
    assert observation_latlon(far, H, lat0, lon0, max_range_m=60.0, reference=ref) is None


def test_a_normal_detection_survives_the_guard():
    from rsu_kinematics import observation_latlon, camera_reference_latlon
    H, lat0, lon0 = _load_map(INESC_MAP)
    ref = camera_reference_latlon(H, lat0, lon0, 3840, 2160)
    near = (1800.0, 1400.0, 2040.0, 1480.0)
    assert observation_latlon(near, H, lat0, lon0, max_range_m=60.0, reference=ref) is not None
