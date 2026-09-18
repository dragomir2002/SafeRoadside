"""The detector can withhold its own observations, to emulate RSU degradation.

R5 asks what happens when the RSU is occluded. On recorded footage the camera
cannot be covered, so the publish path is filtered instead: the clip still drives
the replay clock, and the gateway sees only what the mode allows.
"""
import pytest

import gateway_glue


def _basic_types(seen):
    """The two publish paths hand over different shapes.

    `publish()` (the detector's own tracks) calls adapter.publish_observation()
    with a plain dict; `publish_peer_observation()` puts a built Observation on
    ingest_q. Both are captured, so normalise.
    """
    return {s["basic_type"] if isinstance(s, dict) else s.basic_type for s in seen}


def _sources(seen):
    return [s.get("source", "RSU") if isinstance(s, dict) else s.source for s in seen]


@pytest.fixture
def captured():
    """Wire gateway_glue to a capturing adapter; restore globals afterwards."""
    seen = []

    class _Q:
        def put_nowait(self, obs):
            seen.append(obs)

    class _Adapter:
        ingest_q = _Q()

        def publish_observation(self, track: dict):
            seen.append(track)

    class _Loop:
        def call_soon_threadsafe(self, fn):
            fn()

    saved = (gateway_glue._started, gateway_glue._adapter_getter,
             gateway_glue._loop, gateway_glue._print_pedestrian_gps)
    gateway_glue._started = True
    gateway_glue._adapter_getter = lambda: _Adapter()
    gateway_glue._loop = _Loop()
    gateway_glue._print_pedestrian_gps = False
    yield seen
    (gateway_glue._started, gateway_glue._adapter_getter,
     gateway_glue._loop, gateway_glue._print_pedestrian_gps) = saved
    gateway_glue.set_rsu_publish_filter("both")


def _publish_one_of_each():
    gateway_glue.publish("V1", "car", 38.7355, -9.1435, speed_mps=13.9, heading_deg=0.0)
    gateway_glue.publish("P1", "person", 38.7364, -9.1435, speed_mps=1.4, heading_deg=270.0)


@pytest.mark.parametrize("mode,expected", [
    ("both", {"vehicle", "pedestrian"}),
    ("vehicles", {"vehicle"}),
    ("vru", {"pedestrian"}),
    ("none", set()),
])
def test_filter_controls_what_reaches_the_gateway(captured, mode, expected):
    gateway_glue.set_rsu_publish_filter(mode)
    _publish_one_of_each()
    assert _basic_types(captured) == expected


def test_an_unknown_filter_mode_is_refused(captured):
    with pytest.raises(ValueError, match="rsu publish filter"):
        gateway_glue.set_rsu_publish_filter("sometimes")


def test_the_filter_does_not_touch_injected_peer_observations(captured):
    """A phone is not the RSU: 'none' must not silence the cooperative path.

    This is the whole point of mode 'none' -- it emulates an absent RSU while
    the scripted or real phone keeps reporting.
    """
    gateway_glue.set_rsu_publish_filter("none")
    gateway_glue.publish_peer_observation(
        lat=38.736407, lon=-9.143549, track_id="PED-1", speed_mps=1.4,
        heading_deg=270.0, accuracy_m=3.0, source="SafeWalk",
        basic_type="pedestrian")
    assert _sources(captured) == ["SafeWalk"]
