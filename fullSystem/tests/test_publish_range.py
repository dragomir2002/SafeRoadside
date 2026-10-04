"""The RSU tells the gateway how far each detection is from the camera."""
import gateway_glue


def _capture(monkeypatch):
    got = []

    class _Adapter:
        def publish_observation(self, obs):
            got.append(obs)

    monkeypatch.setattr(gateway_glue, "_started", True)
    monkeypatch.setattr(gateway_glue, "_adapter_getter", lambda: _Adapter())
    monkeypatch.setattr(gateway_glue, "_rsu_publish", "both")
    return got


def test_publish_carries_the_range_from_the_camera(monkeypatch):
    got = _capture(monkeypatch)
    gateway_glue.publish(7, "car", 38.7, -9.1, t_recv=1.0, range_m=52.5)
    assert got and got[0]["range_m"] == 52.5


def test_publish_without_a_range_sends_none(monkeypatch):
    got = _capture(monkeypatch)
    gateway_glue.publish(7, "car", 38.7, -9.1, t_recv=1.0)
    assert got and got[0]["range_m"] is None
