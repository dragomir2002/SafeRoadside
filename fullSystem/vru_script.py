"""Scripted actors, published in scene time. A script is JSONL: one sample
per line with t (seconds of video), lat, lon; track_id names the actor."""
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
        """Newest unreturned sample at or before t_scene, else None."""
        out = self.due(t_scene)
        return out[-1] if out else None

    def due(self, t_scene: float) -> list:
        """The newest due sample of each actor, at most one per track_id."""
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
