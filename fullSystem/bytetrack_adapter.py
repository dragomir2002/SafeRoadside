"""ByteTrack results presented through DeepSort's track interface."""
from pathlib import Path

import numpy as np


def default_tracker_cfg():
    """Absolute path of the ByteTrack config in trackers/."""
    return str(Path(__file__).resolve().parent / "trackers" / "bytetrack_rsu.yaml")


def _arr(x):
    """A torch tensor (ultralytics) or anything array-like -> numpy."""
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x, dtype=float)


class ByteTrackTrack:
    """One tracked box of the current frame."""

    # ByteTrack reports only tracks matched in this frame.
    time_since_update = 0

    def __init__(self, track_id, ltrb, det_class):
        self.track_id = track_id
        self._ltrb = ltrb
        self._cls = det_class

    def is_confirmed(self):
        return True

    def to_ltrb(self):
        return self._ltrb

    def get_det_class(self):
        return self._cls


def tracks_from_boxes(boxes, scale):
    """ultralytics Boxes of a model.track result as a ByteTrackTrack list;
    boxes are divided by scale to return to original-frame pixels."""
    if boxes.id is None:
        return []
    ids, xyxy, cls = _arr(boxes.id), _arr(boxes.xyxy), _arr(boxes.cls)
    return [ByteTrackTrack(str(int(i)), [float(v) / scale for v in box], int(c))
            for i, box, c in zip(ids, xyxy, cls)]
