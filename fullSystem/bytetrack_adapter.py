"""ByteTrack results presented through DeepSort's track interface.

`--tracker bytetrack` replaces YOLO predict + DeepSort update_tracks with one
ultralytics `model.track(...)` call. DeepSort took 80-92 ms of the ~120 ms
frame; ByteTrack does detection and tracking together in ~20 ms, because it
associates on box overlap and detection score alone, with no appearance model.

The detection loop reads each track through is_confirmed(), time_since_update,
track_id, to_ltrb() and get_det_class(). Presenting ByteTrack the same way keeps
the tracker the only difference between the two arms of an A/B.
"""
import numpy as np


def _arr(x):
    """A torch tensor (ultralytics) or anything array-like -> numpy."""
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x, dtype=float)


class ByteTrackTrack:
    """One tracked box of the current frame."""

    # ByteTrack reports only tracks it activated and matched in this frame, so
    # every one is confirmed and fresh -- the loop's filter lets them all pass.
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
    """ultralytics Boxes of a `model.track` result -> ByteTrackTrack list.

    `scale` is the inference resize factor: boxes are divided by it to return
    to original-frame pixels, as the DeepSort path does with its detections.
    Track ids become strings, as deep_sort_realtime's are.
    """
    if boxes.id is None:
        return []
    ids, xyxy, cls = _arr(boxes.id), _arr(boxes.xyxy), _arr(boxes.cls)
    return [ByteTrackTrack(str(int(i)), [float(v) / scale for v in box], int(c))
            for i, box, c in zip(ids, xyxy, cls)]
