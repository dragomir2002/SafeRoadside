"""Real-world phone coordinates -> what the gateway is told.

On recorded footage the bridge transplants the phone's motion onto the scene:
the first fix becomes "home" and maps to the video anchor, and later fixes are
applied as deltas. In a FIELD trial that is wrong -- the phone is already
inside the calibrated scene, so translating it displaces the avatar by
(first fix - map origin) and the RSU's own detection of the same person can
never be associated with it.
"""
import math
import time

import pytest

from gateway_glue import AnchorTranslator

ANCHOR_LAT, ANCHOR_LON = 38.736407, -9.143549      # BIB_IST map.txt origin
HOME_LAT, HOME_LON = 38.650000, -9.150000          # somewhere else entirely
R_EARTH = 6_371_000.0


def _metres_north(lat_from, lat_to):
    return math.radians(lat_to - lat_from) * R_EARTH


def test_first_fix_maps_to_the_video_anchor():
    t = AnchorTranslator(ANCHOR_LAT, ANCHOR_LON)
    assert t(HOME_LAT, HOME_LON) == (ANCHOR_LAT, ANCHOR_LON)


def test_later_fixes_apply_their_delta_to_the_video_anchor():
    t = AnchorTranslator(ANCHOR_LAT, ANCHOR_LON)
    t(HOME_LAT, HOME_LON)
    ten_m_north = HOME_LAT + math.degrees(10.0 / R_EARTH)
    lat, lon = t(ten_m_north, HOME_LON)
    assert _metres_north(ANCHOR_LAT, lat) == pytest.approx(10.0, abs=0.01)
    assert lon == pytest.approx(ANCHOR_LON, abs=1e-9)


def test_translation_off_returns_the_real_position_unchanged():
    """A field trial: the phone is already in the scene, so do not move it."""
    t = AnchorTranslator(ANCHOR_LAT, ANCHOR_LON, translate=False)
    assert t(HOME_LAT, HOME_LON) == (HOME_LAT, HOME_LON)
    assert t(38.736500, -9.143600) == (38.736500, -9.143600)


@pytest.mark.parametrize("lat,lon,why", [
    (12.34, 56.78, "SafeWalk's pre-fix placeholder"),
    (38.7367524, -9.1437386, "SafeWalk's pre-fix placeholder, BIB_IST test point"),
    (0.0, 0.0, "Null Island"),
    (91.0, 0.0, "latitude out of range"),
    (0.0, 181.0, "longitude out of range"),
])
@pytest.mark.parametrize("translate", [True, False])
def test_junk_fixes_are_dropped_whether_or_not_translation_is_on(lat, lon, why, translate):
    t = AnchorTranslator(ANCHOR_LAT, ANCHOR_LON, translate=translate)
    assert t(lat, lon) is None, why


def test_a_dropped_first_fix_does_not_become_home():
    """The placeholder must not poison the anchor: the next real fix is home."""
    t = AnchorTranslator(ANCHOR_LAT, ANCHOR_LON)
    assert t(12.34, 56.78) is None
    assert t(HOME_LAT, HOME_LON) == (ANCHOR_LAT, ANCHOR_LON)


def test_the_scene_placeholder_does_not_become_home_either():
    """SafeWalk always sends its placeholder first. Taken as home, a real walk
    10 km away would then land 10 km off the scene."""
    t = AnchorTranslator(ANCHOR_LAT, ANCHOR_LON)
    assert t(38.7367524, -9.1437386) is None
    assert t(HOME_LAT, HOME_LON) == (ANCHOR_LAT, ANCHOR_LON)


def test_a_real_fix_a_metre_from_the_scene_placeholder_is_kept():
    """Only the exact placeholder is junk: a simulated or real fix beside it is not."""
    t = AnchorTranslator(ANCHOR_LAT, ANCHOR_LON, translate=False)
    one_m_north = 38.7367524 + math.degrees(1.0 / R_EARTH)
    assert t(one_m_north, -9.1437386) == (one_m_north, -9.1437386)


# --- the flag has to survive the whole POST path, not just the class -------

@pytest.fixture
def bridge_post():
    """Start the real bridge on a free port, capture what reaches the gateway."""
    import json
    import socket
    import urllib.request

    import gateway_glue

    captured = []

    class _Q:
        def put_nowait(self, obs):
            captured.append(obs)

    class _Adapter:
        ingest_q = _Q()

    class _Loop:
        def call_soon_threadsafe(self, fn):
            fn()

    saved = (gateway_glue._started, gateway_glue._adapter_getter, gateway_glue._loop)
    gateway_glue._started = True
    gateway_glue._adapter_getter = lambda: _Adapter()
    gateway_glue._loop = _Loop()

    def run(translate, lat, lon):
        captured.clear()
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        gateway_glue.start_safewalk_http_bridge(
            host="127.0.0.1", port=port,
            video_anchor_lat=ANCHOR_LAT, video_anchor_lon=ANCHOR_LON,
            translate=translate)
        body = json.dumps({"lat": lat, "lon": lon, "track_id": "FIELD-PHONE",
                           "speed_mps": 1.4, "heading_deg": 270.0,
                           "accuracy_m": 5.0, "source": "SafeWalk"}).encode()
        for _ in range(50):                      # the server starts in a thread
            try:
                urllib.request.urlopen(
                    urllib.request.Request(f"http://127.0.0.1:{port}/psm", data=body,
                                           headers={"Content-Type": "application/json"}),
                    timeout=2.0).read()
                break
            except OSError:
                time.sleep(0.05)
        else:
            pytest.fail("bridge never accepted a POST")
        for _ in range(50):
            if captured:
                break
            time.sleep(0.02)
        assert captured, "the POST never reached the gateway"
        return captured[0]

    yield run

    (gateway_glue._started, gateway_glue._adapter_getter, gateway_glue._loop) = saved


def test_posted_coordinates_reach_the_gateway_untouched_with_no_translation(bridge_post):
    obs = bridge_post(translate=False, lat=38.7365000, lon=-9.1436000)
    assert obs.lat == pytest.approx(38.7365000, abs=1e-9)
    assert obs.lon == pytest.approx(-9.1436000, abs=1e-9)


def test_posted_coordinates_are_anchored_to_the_scene_when_translating(bridge_post):
    obs = bridge_post(translate=True, lat=HOME_LAT, lon=HOME_LON)
    assert obs.lat == pytest.approx(ANCHOR_LAT, abs=1e-9)
    assert obs.lon == pytest.approx(ANCHOR_LON, abs=1e-9)
