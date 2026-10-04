"""0_createMap.py --points: calibration from a points file."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from rsu_kinematics import pixel_to_latlon

_spec = importlib.util.spec_from_file_location(
    "create_map", Path(__file__).resolve().parents[1] / "0_createMap.py")
create_map = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(create_map)

R = 6371000.0
LAT_REF, LON_REF = 38.7373, -9.1386
# pixel -> local metres; w stays positive over the image
H_TRUE = np.array([[0.02, 0.001, -30.0], [0.0005, -0.05, 60.0], [0.0, 0.0004, 1.0]])
PIXELS = [(x, y) for y in (1100, 1450, 1800) for x in (400, 1300, 2200, 3100)] + [(1900, 2050)]


def _ground(px, py):
    X, Y, w = H_TRUE @ np.array([px, py, 1.0])
    X, Y = X / w, Y / w
    return (LAT_REF + math.degrees(Y / R),
            LON_REF + math.degrees(X / (R * math.cos(math.radians(LAT_REF)))))


LATLONS = [_ground(*p) for p in PIXELS]


def _read_map(path):
    rows = [list(map(float, line.split())) for line in Path(path).read_text().splitlines()]
    return rows[0], np.array(rows[1:4])


def test_an_exact_point_set_fits_and_round_trips_through_map_txt(tmp_path):
    fit = create_map.fit_map(PIXELS, LATLONS)
    assert max(fit["residual_m"]) < 0.01
    assert all(fit["inlier"])
    out = tmp_path / "map.txt"
    create_map.write_map(out, fit["origin"], fit["H"])
    (lat0, lon0), H = _read_map(out)
    for (px, py), (lat, lon) in zip(PIXELS, LATLONS):
        got = pixel_to_latlon(px, py, H, lat0, lon0)
        assert got[0] == pytest.approx(lat, abs=1e-6)
        assert got[1] == pytest.approx(lon, abs=1e-6)


def test_a_gross_misclick_is_flagged_and_does_not_spoil_the_others():
    pixels = list(PIXELS)
    pixels[5] = (pixels[5][0] + 1000, pixels[5][1])     # ~12 m on the ground
    fit = create_map.fit_map(pixels, LATLONS)
    assert fit["inlier"][5] is False
    assert fit["loo_m"][5] > 5.0
    assert max(e for i, e in enumerate(fit["loo_m"]) if i != 5) < 0.1


def test_a_small_misclick_passes_ransac_but_leave_one_out_singles_it_out():
    """A 300 px misclick stays a RANSAC inlier; leave-one-out shows it."""
    pixels = list(PIXELS)
    pixels[5] = (pixels[5][0] + 300, pixels[5][1])
    fit = create_map.fit_map(pixels, LATLONS)
    assert fit["inlier"][5] is True
    others = [e for i, e in enumerate(fit["loo_m"]) if i != 5]
    assert fit["loo_m"][5] > 2 * max(others)


def test_points_file_blank_pixels_are_none(tmp_path):
    p = tmp_path / "gcp.csv"
    p.write_text("name,lat,lon,px,py\n"
                 "kerb A,38.7373,-9.1386,1200,1500\n"
                 "bollard B,38.73741,-9.13852,,\n", encoding="utf-8")
    pts = create_map.load_points(p)
    assert pts[0] == {"name": "kerb A", "lat": 38.7373, "lon": -9.1386, "px": 1200.0, "py": 1500.0}
    assert (pts[1]["px"], pts[1]["py"]) == (None, None)


def test_coordinate_strings_parse_with_their_hemisphere():
    """The coords_str form: west and south negative."""
    assert create_map.parse_latlon("38.736407°N 9.143549°W") == (38.736407, -9.143549)
    assert create_map.parse_latlon("38.736407N 9.143549E") == (38.736407, 9.143549)
    assert create_map.parse_latlon("1.5°S 2.25°W") == (-1.5, -2.25)
