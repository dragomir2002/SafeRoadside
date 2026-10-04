"""The detector can dump what it published, in video time."""
import gateway_glue


def test_dump_writes_published_tracks_in_video_time(tmp_path):
    out = tmp_path / "obs.csv"
    gateway_glue.start_observation_dump(out, fps=30.0)
    try:
        gateway_glue.record_observation_sample(
            frame_index=60, source="RSU", track_id="7", basic_type="vehicle",
            lat=38.7364070, lon=-9.1435490, speed_mps=13.9, heading_deg=270.0)
    finally:
        gateway_glue.stop_observation_dump()
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "t_scene,source,track_id,basic_type,lat,lon,speed_mps,heading_deg"
    assert lines[1] == "2.0000,RSU,7,vehicle,38.7364070,-9.1435490,13.9,270.0"


def test_unknown_speed_and_heading_are_written_as_empty_fields(tmp_path):
    """A track without kinematics is still a position the author may use."""
    out = tmp_path / "obs.csv"
    gateway_glue.start_observation_dump(out, fps=30.0)
    try:
        gateway_glue.record_observation_sample(
            frame_index=0, source="RSU", track_id="1", basic_type="pedestrian",
            lat=1.0, lon=2.0, speed_mps=None, heading_deg=None)
    finally:
        gateway_glue.stop_observation_dump()
    assert out.read_text(encoding="utf-8").splitlines()[1] == \
        "0.0000,RSU,1,pedestrian,1.0000000,2.0000000,,"


def test_detector_classes_are_recorded_as_canonical_types(tmp_path):
    """The CSV uses basic_type names, not YOLO class names."""
    out = tmp_path / "obs.csv"
    gateway_glue.start_observation_dump(out, fps=30.0)
    try:
        for cls in ("car", "person", "bicycle", "vehicle"):
            gateway_glue.record_observation_sample(
                frame_index=0, source="RSU", track_id="1", basic_type=cls,
                lat=1.0, lon=2.0, speed_mps=None, heading_deg=None)
    finally:
        gateway_glue.stop_observation_dump()
    types = [line.split(",")[3] for line in
             out.read_text(encoding="utf-8").splitlines()[1:]]
    assert types == ["vehicle", "pedestrian", "cyclist", "vehicle"]


def test_dump_is_a_no_op_when_not_started(tmp_path):
    gateway_glue.stop_observation_dump()          # make sure it is off
    gateway_glue.record_observation_sample(
        frame_index=1, source="RSU", track_id="1", basic_type="vehicle",
        lat=1.0, lon=2.0, speed_mps=None, heading_deg=None)   # must not raise


def test_dump_refuses_a_missing_frame_rate(tmp_path):
    """t_scene is frame/fps, so a zero fps would silently write t_scene=inf."""
    import pytest
    with pytest.raises(ValueError, match="frame rate"):
        gateway_glue.start_observation_dump(tmp_path / "x.csv", fps=0.0)
