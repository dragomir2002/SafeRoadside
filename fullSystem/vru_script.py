"""Scripted actors, published in scene time.

Ground truth for the labelled trials of the evaluation plan §2 and the R5 modes of
the implementation plan. The script says
where each actor is at each second of video, so a conflict with a recorded
vehicle -- or between two scripted actors, with the RSU withheld -- exists by
construction instead of being inferred from the system's own output.

Scene time, not wall time: on recorded footage every duration the gateway
measures runs on the replay clock, so a wall-clock avatar (the behaviour of
`start_safewalk_injector`) would drift against the video by the replay
slow-down, about 9x on the reference machine. The two paths must never run
together either -- the same avatar would be ingested twice .

File format: JSONL, one sample per line, `#` comments and blank lines ignored.

    {"t": 12.0, "lat": 38.7364, "lon": -9.1435, "speed_mps": 1.4,
     "heading_deg": 270.0, "track_id": "PED-1", "basic_type": "pedestrian",
     "accuracy_m": 3.0, "source": "SafeWalk"}

`t` is seconds of video; `t`, `lat` and `lon` are required. `track_id` separates
actors, so one file can carry a vehicle and a VRU.
"""
import json
import pathlib

REQUIRED = ("t", "lat", "lon")
DEFAULT_TRACK_ID = "SCRIPT-VRU"


class VruScript:
    def __init__(self, samples):
        self.samples = sorted(samples, key=lambda s: float(s["t"]))
        self._per_actor = {}    # track_id -> index of the next unsent sample

    @classmethod
    def load(cls, path) -> "VruScript":
        samples = []
        for lineno, raw in enumerate(
                pathlib.Path(path).read_text(encoding="utf-8").splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            for key in REQUIRED:
                if key not in rec:
                    raise ValueError(f"{path}:{lineno}: sample missing {key!r}: {rec}")
            samples.append(rec)
        if not samples:
            raise ValueError(f"{path}: no samples")
        return cls(samples)

    def latest_due(self, t_scene: float):
        """The newest sample at or before t_scene not yet returned, else None.

        Single-actor convenience, defined in terms of `due()` so the two share
        progress: whichever is called, a sample is published at most once. An
        independent cursor would let a caller that used both publish the same
        position twice, which is the double-counting trap of the project notes

        Only the newest: publishing every sample that fell due since the last
        frame would put a teleporting VRU into one buffer window and inflate the
        ingest count.
        """
        out = self.due(t_scene)
        return out[-1] if out else None

    def due(self, t_scene: float) -> list:
        """The newest sample now due for EACH actor, at most one per track_id.

        One per actor rather than one overall, so a scripted vehicle and a
        scripted VRU advance independently -- that is what R5 mode M3 needs.
        """
        newest = {}
        for i, sample in enumerate(self.samples):
            if float(sample["t"]) > t_scene:
                break
            tid = sample.get("track_id", DEFAULT_TRACK_ID)
            if i >= self._per_actor.get(tid, 0):
                newest[tid] = (i, sample)
        out = []
        for tid, (i, sample) in newest.items():
            self._per_actor[tid] = i + 1
            out.append(sample)
        return out
