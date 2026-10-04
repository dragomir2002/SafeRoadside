"""The RSU's confidence follows how well the homography places a detection."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsu_kinematics import ground_resolution_m, pixel_to_latlon  # noqa: E402

SCENE = Path(__file__).resolve().parents[1] / "scenes" / "bib_ist" / "map.txt"


@pytest.fixture(scope="module")
def homography():
    text = SCENE.read_text().splitlines()
    lat0, lon0 = (float(v) for v in text[0].split())
    H = np.array([[float(v) for v in text[i].split()] for i in (1, 2, 3)])
    return H, lat0, lon0


def test_a_pixel_row_is_worth_more_ground_further_away(homography):
    """Depth resolution collapses towards the horizon -- the whole point."""
    H, lat0, lon0 = homography
    near = ground_resolution_m(1920, 2000, H, lat0, lon0)
    far = ground_resolution_m(1920, 1150, H, lat0, lon0)
    assert near is not None and far is not None
    assert far > near


def test_resolution_is_a_positive_distance(homography):
    H, lat0, lon0 = homography
    assert ground_resolution_m(1920, 1800, H, lat0, lon0) > 0.0


def test_a_pixel_on_the_horizon_has_no_resolution(homography):
    """pixel_to_latlon returns None there: no distance to measure."""
    H, lat0, lon0 = homography
    v = next((v for v in range(1200, 0, -1)
              if pixel_to_latlon(1920, v, H, lat0, lon0) is None), None)
    if v is None:
        pytest.skip("this homography has no horizon inside the frame")
    assert ground_resolution_m(1920, v, H, lat0, lon0) is None


def test_the_box_error_grows_with_distance(homography):
    """The two ground points of a box disagree more the further away it is."""
    from rsu_kinematics import ground_point_error_m
    H, lat0, lon0 = homography
    near = ground_point_error_m((1900, 1600, 1940, 2000), H, lat0, lon0)
    far = ground_point_error_m((1900, 1110, 1940, 1150), H, lat0, lon0)
    assert near is not None and far is not None
    assert far > near


def test_confidence_falls_with_range():
    from rsu_kinematics import CONF_MAX, CONF_MIN, confidence_from_range
    assert confidence_from_range(5.0) == CONF_MAX
    assert confidence_from_range(200.0) == CONF_MIN
    mid = confidence_from_range(40.0)
    assert CONF_MIN < mid < CONF_MAX
    assert confidence_from_range(30.0) > confidence_from_range(50.0)


def test_an_unknown_range_is_not_treated_as_a_bad_one():
    """An unknown range is not a poor one."""
    from rsu_kinematics import CONF_MAX, confidence_from_range
    assert confidence_from_range(None) == CONF_MAX


def test_the_knees_are_configurable_per_scene():
    """The range limits are set per scene."""
    from rsu_kinematics import confidence_from_range
    strict = confidence_from_range(25.0, full_m=10.0, zero_m=30.0)
    loose = confidence_from_range(25.0, full_m=40.0, zero_m=120.0)
    assert strict < loose


def test_range_is_measured_from_the_camera_reference():
    from rsu_kinematics import range_from_m
    ref = (38.7362950, -9.1435588)
    assert range_from_m(ref, *ref) == pytest.approx(0.0, abs=1e-6)
    north = range_from_m(ref, ref[0] + 0.001, ref[1])
    assert north == pytest.approx(111.2, rel=0.02)


def test_the_step_is_configurable_and_symmetric(homography):
    """A wider step smooths the answer; it does not change its sign."""
    H, lat0, lon0 = homography
    one = ground_resolution_m(1920, 1600, H, lat0, lon0, dv=1.0)
    four = ground_resolution_m(1920, 1600, H, lat0, lon0, dv=4.0)
    assert math.isclose(one, four, rel_tol=0.35)
