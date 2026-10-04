"""Scripted actors can be seen by the camera as well as reported by a phone.

A constructed conflict tests the gateway only on the path its actor takes. A
phone actor never exercises what the camera alone would do -- the range limits,
a pedestrian hidden behind a pole, a detection that arrives late -- so a script
sample with "source": "RSU" is published exactly as a detector track is: through
the RSU adapter, with the frame's capture time and its range from the camera.
"""
import gateway_glue


def _capture(monkeypatch, rsu_publish="both"):
    camera, phone = [], []

    class _Adapter:
        def publish_observation(self, obs):
            camera.append(obs)

    monkeypatch.setattr(gateway_glue, "_started", True)
    monkeypatch.setattr(gateway_glue, "_adapter_getter", lambda: _Adapter())
    monkeypatch.setattr(gateway_glue, "_rsu_publish", rsu_publish)
    monkeypatch.setattr(gateway_glue, "publish_peer_observation",
                        lambda **kw: phone.append(kw))
    return camera, phone


SAMPLE = {"t": 12.0, "lat": 38.7364, "lon": -9.1435, "speed_mps": 1.4,
          "heading_deg": 270.0, "track_id": "PED-C1", "basic_type": "pedestrian",
          "accuracy_m": 3.0}


def test_a_camera_actor_is_published_as_a_detector_track(monkeypatch):
    camera, phone = _capture(monkeypatch)
    gateway_glue.publish_script_sample(dict(SAMPLE, source="RSU"), t_recv=5.0,
                                       range_of=lambda lat, lon: 27.5)
    assert not phone
    obs = camera[0]
    assert (obs["track_id"], obs["basic_type"], obs["t_recv"], obs["range_m"]) == \
        ("PED-C1", "pedestrian", 5.0, 27.5)
    assert (obs["speed_mps"], obs["heading_deg"]) == (1.4, 270.0)


def test_a_camera_actor_has_no_gnss_accuracy(monkeypatch):
    """A detector track carries no GNSS accuracy; neither does its stand-in."""
    camera, _ = _capture(monkeypatch)
    gateway_glue.publish_script_sample(dict(SAMPLE, source="RSU"), t_recv=5.0)
    assert camera[0]["accuracy_m"] is None and camera[0]["range_m"] is None


def test_a_camera_cyclist_stays_a_cyclist(monkeypatch):
    camera, _ = _capture(monkeypatch)
    gateway_glue.publish_script_sample(dict(SAMPLE, source="RSU", basic_type="cyclist"),
                                       t_recv=5.0)
    assert camera[0]["basic_type"] == "cyclist"


def test_a_phone_actor_still_takes_the_phone_path(monkeypatch):
    camera, phone = _capture(monkeypatch)
    gateway_glue.publish_script_sample(dict(SAMPLE, source="SafeWalk"), t_recv=5.0)
    assert not camera
    assert (phone[0]["source"], phone[0]["track_id"], phone[0]["accuracy_m"]) == \
        ("SafeWalk", "PED-C1", 3.0)


def test_an_actor_without_a_source_is_a_safewalk_phone(monkeypatch):
    """The script format's default, unchanged."""
    camera, phone = _capture(monkeypatch)
    gateway_glue.publish_script_sample(dict(SAMPLE), t_recv=5.0)
    assert not camera and phone[0]["source"] == "SafeWalk"


def test_a_camera_actor_obeys_the_rsu_publish_filter(monkeypatch):
    """--rsu-publish vehicles withholds camera pedestrians, scripted or not."""
    camera, phone = _capture(monkeypatch, rsu_publish="vehicles")
    gateway_glue.publish_script_sample(dict(SAMPLE, source="RSU"), t_recv=5.0)
    assert not camera and not phone
