"""ByteTrack (ultralytics model.track) presented through DeepSort's track interface.

The detection loop reads each track through is_confirmed(), time_since_update,
track_id, to_ltrb() and get_det_class(). Keeping that interface means the
tracker is the only thing that changes between the two arms of the A/B.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from bytetrack_adapter import tracks_from_boxes


def _boxes(ids, xyxy, cls):
    return SimpleNamespace(id=None if ids is None else np.array(ids, dtype=float),
                           xyxy=np.array(xyxy, dtype=float),
                           cls=np.array(cls, dtype=float))


def test_boxes_become_tracks_in_original_frame_pixels():
    b = _boxes([7, 12], [[100, 50, 140, 90], [10, 20, 30, 60]], [2, 0])
    tracks = tracks_from_boxes(b, scale=0.5)      # inference frame was half size
    assert [t.track_id for t in tracks] == ["7", "12"]
    assert tracks[0].to_ltrb() == pytest.approx([200, 100, 280, 180])
    assert [t.get_det_class() for t in tracks] == [2, 0]


def test_every_returned_track_passes_the_loops_filter():
    """ByteTrack only reports tracks it has activated and matched this frame."""
    t = tracks_from_boxes(_boxes([3], [[0, 0, 10, 10]], [0]), scale=1.0)[0]
    assert t.is_confirmed() is True
    assert t.time_since_update == 0


def test_a_frame_with_no_tracks_gives_an_empty_list():
    """ultralytics sets boxes.id to None when nothing is tracked."""
    assert tracks_from_boxes(_boxes(None, np.zeros((0, 4)), []), scale=1.0) == []


def test_torch_like_tensors_are_accepted():
    class _T:
        def __init__(self, a):
            self._a = np.asarray(a, dtype=float)

        def cpu(self):
            return self

        def numpy(self):
            return self._a

    b = SimpleNamespace(id=_T([5]), xyxy=_T([[1, 2, 3, 4]]), cls=_T([7]))
    t = tracks_from_boxes(b, scale=1.0)[0]
    assert (t.track_id, t.get_det_class()) == ("5", 7)
    assert t.to_ltrb() == pytest.approx([1, 2, 3, 4])
