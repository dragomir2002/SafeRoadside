"""The RSU's own collision check, per track pair.

5_realTime.py flags a possible collision wherever a vehicle's predicted pixel
point and a VRU's come closer than COLLISION_THRESHOLD, then flattens the points
and loses whose they were. This keeps the owners, so the check can be logged
and scored against the same ground truth as the gateway.
"""
import numpy as np


def conflict_pairs(car_pts, car_owner, vru_pts, vru_owner, threshold):
    """{(vehicle track, VRU track): closest predicted distance in px} for every
    pair whose points come strictly closer than `threshold`."""
    if not len(car_pts) or not len(vru_pts):
        return {}
    c = np.asarray(car_pts, dtype=np.float64).reshape(-1, 2)
    v = np.asarray(vru_pts, dtype=np.float64).reshape(-1, 2)
    d = np.linalg.norm(c[:, None, :] - v[None, :, :], axis=2)
    out = {}
    for ci, vi in zip(*np.nonzero(d < threshold)):
        key = (car_owner[ci], vru_owner[vi])
        dist = float(d[ci, vi])
        if key not in out or dist < out[key]:
            out[key] = dist
    return out
