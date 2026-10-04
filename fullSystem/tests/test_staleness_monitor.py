"""A source that goes quiet is reported as stale."""
from gateway_glue import SourceFreshness


def test_a_source_becomes_stale_after_the_silence_window():
    f = SourceFreshness(silence_s=5.0)
    f.saw("RSU", 100.0)
    assert f.stale(104.9) == []
    assert f.stale(105.0) == ["RSU"]


def test_a_refreshed_source_is_not_stale():
    f = SourceFreshness(silence_s=5.0)
    f.saw("RSU", 100.0)
    f.saw("RSU", 104.0)
    assert f.stale(105.0) == []


def test_a_source_never_seen_is_not_reported_stale():
    """A source never seen is not stale."""
    f = SourceFreshness(silence_s=5.0)
    f.saw("RSU", 100.0)
    assert f.stale(200.0) == ["RSU"]        # only the source we actually saw


def test_each_source_is_tracked_separately():
    f = SourceFreshness(silence_s=5.0)
    f.saw("RSU", 100.0)
    f.saw("SafeWalk", 104.0)
    assert f.stale(106.0) == ["RSU"]


def test_stale_relative_needs_no_clock_read():
    """Staleness is judged against the newest record seen, not a clock."""
    f = SourceFreshness(silence_s=5.0)
    f.saw("RSU", 100.0)
    f.saw("SafeWalk", 104.0)
    assert f.stale_relative() == []  # newest is 104.0; RSU is 4 s behind
    f.saw("SafeWalk", 106.0)
    assert f.stale_relative() == ["RSU"]  # RSU is 6 s behind


def test_stale_relative_cannot_see_total_silence():
    """With nothing arriving, nothing is reported stale."""
    f = SourceFreshness(silence_s=5.0)
    f.saw("RSU", 100.0)
    assert f.stale_relative() == []


def test_a_source_that_comes_back_is_reported_as_returned_once():
    """The HUD needs the edge, not the level, or it logs every frame."""
    f = SourceFreshness(silence_s=5.0)
    f.saw("RSU", 100.0)
    assert f.newly_stale(106.0) == ["RSU"]
    assert f.newly_stale(107.0) == []       # already reported
    f.saw("RSU", 108.0)
    assert f.newly_returned() == ["RSU"]
    assert f.newly_returned() == []
