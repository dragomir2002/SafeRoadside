"""Scripted actors, replayed in scene time."""
import json

import pytest

from vru_script import VruScript


def _script(tmp_path, samples, name="s.jsonl"):
    p = tmp_path / name
    p.write_text("".join(json.dumps(s) + "\n" for s in samples), encoding="utf-8")
    return p


# --- single actor (the accuracy-plan scenarios) -----------------------------

def test_latest_due_skips_stale_samples_and_never_repeats(tmp_path):
    """At t=1.0 the 0.9 sample is current and 0.1 is history."""
    s = VruScript.load(_script(tmp_path, [
        {"t": 0.1, "lat": 1.0, "lon": 2.0},
        {"t": 0.9, "lat": 1.1, "lon": 2.1},
        {"t": 5.0, "lat": 1.2, "lon": 2.2},
    ]))
    assert s.latest_due(1.0)["lat"] == 1.1
    assert s.latest_due(1.0) is None          # nothing new became due
    assert s.latest_due(5.0)["lat"] == 1.2


def test_latest_due_is_none_before_the_script_starts(tmp_path):
    s = VruScript.load(_script(tmp_path, [{"t": 10.0, "lat": 1.0, "lon": 2.0}]))
    assert s.latest_due(9.999) is None


def test_samples_are_sorted_by_time_even_if_the_file_is_not(tmp_path):
    s = VruScript.load(_script(tmp_path, [
        {"t": 2.0, "lat": 2.0, "lon": 0.0},
        {"t": 1.0, "lat": 1.0, "lon": 0.0},
    ]))
    assert s.latest_due(1.0)["lat"] == 1.0


def test_load_rejects_a_sample_without_coordinates(tmp_path):
    with pytest.raises(ValueError, match="missing 'lat'"):
        VruScript.load(_script(tmp_path, [{"t": 1.0, "lon": 2.0}]))


def test_load_rejects_an_empty_script(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("# only a comment\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no samples"):
        VruScript.load(p)


# --- several actors at once ---

def test_due_returns_the_newest_sample_for_each_actor(tmp_path):
    """A vehicle and a pedestrian in one script, no cross-talk."""
    s = VruScript.load(_script(tmp_path, [
        {"t": 0.5, "lat": 1.0, "lon": 2.0, "track_id": "CAR-1", "basic_type": "vehicle"},
        {"t": 0.6, "lat": 3.0, "lon": 4.0, "track_id": "PED-1", "basic_type": "pedestrian"},
        {"t": 0.9, "lat": 1.1, "lon": 2.1, "track_id": "CAR-1", "basic_type": "vehicle"},
    ]))
    due = s.due(1.0)
    assert {d["track_id"] for d in due} == {"CAR-1", "PED-1"}
    car = next(d for d in due if d["track_id"] == "CAR-1")
    assert car["lat"] == 1.1            # newest for that actor, not both
    assert s.due(1.0) == []             # nothing new became due


def test_due_advances_each_actor_independently(tmp_path):
    s = VruScript.load(_script(tmp_path, [
        {"t": 1.0, "lat": 1.0, "lon": 0.0, "track_id": "CAR-1"},
        {"t": 3.0, "lat": 2.0, "lon": 0.0, "track_id": "CAR-1"},
        {"t": 3.0, "lat": 9.0, "lon": 0.0, "track_id": "PED-1"},
    ]))
    assert [d["track_id"] for d in s.due(1.0)] == ["CAR-1"]
    assert {d["track_id"] for d in s.due(3.0)} == {"CAR-1", "PED-1"}


def test_latest_due_and_due_share_progress_so_nothing_publishes_twice(tmp_path):
    """A sample consumed by either method is spent for both."""
    s = VruScript.load(_script(tmp_path, [{"t": 1.0, "lat": 1.0, "lon": 2.0}]))
    assert s.latest_due(1.0)["lat"] == 1.0
    assert s.due(1.0) == []
