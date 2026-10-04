"""Bridge between the synchronous detector loop and the asyncio gateway,
which runs on a background thread with its own event loop."""
from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
_PEDESTRIAN_CLASSES = {"person"}
_CYCLIST_CLASSES = {"bicycle"}

_started = False
_loop = None  # asyncio.AbstractEventLoop set on background thread
# The duration clock; start_gateway(clock=...) replaces it for a replay.
_clock = time.monotonic
_adapter_getter = None  # callable -> RsuAdapter
_print_pedestrian_gps = True  # set by start_gateway(verbose_pedestrians=...)
_safewalk_watcher_started = False
# Which of the detector's own observations reach the gateway; "both" is normal.
_RSU_PUBLISH_MODES = {"both", "vehicles", "vru", "none"}
_rsu_publish = "both"
# CSV of everything published (--dump-observations).
_dump_fh = None
_dump_fps = 0.0
# Platform temp dir: /tmp on Linux, the user's temp dir on Windows.
INJECT_FILE = Path(tempfile.gettempdir()) / "safewalk_inject.txt"

# Latest SafeWalk positions: track_id -> (lat, lon, monotonic time).
_latest_safewalk: dict = {}

# Pipeline counters for the detector's HUD.
_stats: dict = {
    "ingest_rsu": 0, "ingest_phone": 0,
    "assoc_matched": 0, "assoc_unmatched": 0,
    "fused": 0, "fused_coop": 0,
    "dispatched": 0, "suppressed": 0,
    "risk": {"low": 0, "probable": 0, "imminent": 0},
    "last_risk": None, "last_ttc": None, "last_warn_t": 0.0,
    "config": None,
    "safewalk": None,    # latest PSM the phone sent (position/kinematics only)
    "last_fuse": None,   # latest non-none fusion: risk, ttc, per-source conf
    # Per source, the risk of the latest fusion it fed into.
    "source_risk": {},   # source -> {"risk": str, "t": float}
    # PSMs seen but not decoded, in total and by payload length.
    "decode_errors": 0,
    "decode_error_len": {},   # str(payload length) -> count
    # POSTs that decoded but could not become a valid Observation.
    "inject_rejected": 0,
}
_stats_lock = threading.Lock()
# Which sources have gone quiet; see SourceFreshness.
_freshness = None   # SourceFreshness, created below once the class exists


def _observe(rec: dict) -> None:
    """Fold one gateway stage record into the HUD counters."""
    stage = rec.get("stage")
    with _stats_lock:
        if stage == "ingest":
            src = rec.get("source")
            if src == "RSU":
                _stats["ingest_rsu"] += 1
            elif src:
                _stats["ingest_phone"] += 1
            # t_scene on a replay run, wall time live.
            t_rec = rec.get("t_scene", rec.get("t"))
            if src and t_rec is not None and _freshness is not None:
                _freshness.saw(src, t_rec)
        elif stage == "associate":
            _stats["assoc_matched" if rec.get("matched") else "assoc_unmatched"] += 1
        elif stage == "fuse":
            _stats["fused"] += 1
            if len(set(rec.get("sources") or ())) > 1:
                _stats["fused_coop"] += 1
            if rec.get("risk") and rec["risk"] != "none":
                for _src in set(rec.get("sources") or ()):
                    _stats["source_risk"][_src] = {
                        "risk": rec["risk"], "t": time.monotonic(),
                    }
                _stats["last_fuse"] = {
                    "risk": rec.get("risk"),
                    "ttc": rec.get("ttc"),
                    "confidence": rec.get("confidence"),
                    "source_confidence": rec.get("source_confidence") or {},
                    "sources": sorted(set(rec.get("sources") or ())),
                    "t": time.monotonic(),
                }
        elif stage == "ble_scanner":
            # Undecodable PSMs are counted for the HUD.
            if rec.get("action") == "decode_error":
                _stats["decode_errors"] += 1
                n = str(rec.get("bytes"))
                _stats["decode_error_len"][n] = _stats["decode_error_len"].get(n, 0) + 1
        elif stage == "dispatch":
            if rec.get("action") == "emitted":
                _stats["dispatched"] += 1
                r = rec.get("risk")
                if r in _stats["risk"]:
                    _stats["risk"][r] += 1
                _stats["last_risk"] = r
                _stats["last_warn_t"] = time.monotonic()
            else:
                _stats["suppressed"] += 1


def get_gateway_stats() -> dict:
    """Snapshot of the live pipeline counters (safe to call every frame)."""
    with _stats_lock:
        # Copies, so a caller can iterate while the pipeline mutates.
        out = dict(_stats)
        out["risk"] = dict(_stats["risk"])
        out["source_risk"] = {k: dict(v) for k, v in _stats["source_risk"].items()}
        out["decode_error_len"] = dict(_stats["decode_error_len"])
        for k in ("last_fuse", "safewalk"):
            if isinstance(_stats.get(k), dict):
                out[k] = dict(_stats[k])
        # Report the sources that have fallen silent, once per transition.
        if _freshness is not None:
            out["stale_sources"] = _freshness.stale_relative()
            newest = max(_freshness.last_seen.values(), default=None)
            if newest is not None:
                for src in _freshness.newly_stale(newest):
                    log.warning("source %s has gone quiet (no observation for "
                                "%.1f s of pipeline time); continuing on the rest",
                                src, _freshness.silence_s)
            for src in _freshness.newly_returned():
                log.info("source %s is publishing again", src)
        else:
            out["stale_sources"] = []
        return out


def set_verbose_pedestrians(enabled: bool) -> None:
    """Toggle the per-pedestrian '[GW] pedestrian track N at lat=...' line."""
    global _print_pedestrian_gps
    _print_pedestrian_gps = bool(enabled)


def get_latest_safewalk(max_age_s: float = 2.0) -> dict:
    """{track_id: (lat, lon)} of SafeWalk phones seen within max_age_s."""
    now = time.monotonic()
    return {tid: (lat, lon)
            for tid, (lat, lon, ts) in _latest_safewalk.items()
            if now - ts <= max_age_s}


def start_gateway(config_path: Optional[Path] = None,
                  run_id: Optional[str] = None, clock=None) -> bool:
    """Start the gateway on a daemon thread; return True if it came up.
    Later calls do nothing. clock, if given, is the replay clock."""
    global _started, _loop, _adapter_getter, _clock
    if _started:
        return True
    if clock is not None:
        _clock = clock

    try:
        import asyncio
        from safecorners_gateway.config import load_config
        from safecorners_gateway.main import run_gateway, get_rsu_adapter
    except ImportError as e:
        log.warning("safecorners_gateway not importable; detector will run standalone (%s)", e)
        return False

    cfg_path = config_path or _default_config_path()
    if not cfg_path.exists():
        log.warning("gateway config %s missing; detector standalone", cfg_path)
        return False

    cfg = load_config(cfg_path)
    _adapter_getter = get_rsu_adapter
    with _stats_lock:
        _stats["config"] = cfg_path.name

    ready = threading.Event()

    def _runner() -> None:
        global _loop
        loop = asyncio.new_event_loop()
        _loop = loop
        asyncio.set_event_loop(loop)
        # Schedule the gateway, then signal readiness once the adapter exists.
        gateway_task = loop.create_task(
            run_gateway(cfg, run_id=run_id, observer=_observe, clock=clock))

        async def _wait_for_adapter() -> None:
            for _ in range(50):  # up to 5 s
                try:
                    get_rsu_adapter()
                    ready.set()
                    return
                except RuntimeError:
                    await asyncio.sleep(0.1)
            # Not signalled as ready: publish() would return silently.
            log.error("gateway adapter never appeared; %r", gateway_task)

        loop.create_task(_wait_for_adapter())
        try:
            loop.run_forever()
        finally:
            loop.close()

    t = threading.Thread(target=_runner, name="safecorners-gateway", daemon=True)
    t.start()
    if not ready.wait(timeout=10.0):
        log.error("gateway thread did not become ready in 10 s")
        return False

    _started = True
    log.info("safecorners gateway started on background thread")
    return True


def wait_gateway_idle(timeout: float = 60.0) -> bool:
    """Block until the gateway has processed everything published so far."""
    if not _started or _loop is None:
        return False
    try:
        import asyncio
        from safecorners_gateway.main import wait_idle
        ok = asyncio.run_coroutine_threadsafe(wait_idle(), _loop).result(timeout=timeout)
    except Exception as e:
        log.error("waiting for the gateway to drain failed: %r", e)
        return False
    if not ok:
        log.error("gateway queues did not drain")
    return ok


def start_observation_dump(path, fps: float) -> None:
    """Write every published observation to a CSV, in video time."""
    global _dump_fh, _dump_fps
    if _dump_fh is not None:
        return
    if not fps or fps <= 0:
        raise ValueError(f"observation dump needs the video frame rate, got {fps!r}")
    _dump_fps = float(fps)
    _dump_fh = open(path, "w", encoding="utf-8", newline="")
    _dump_fh.write("t_scene,source,track_id,basic_type,lat,lon,speed_mps,heading_deg\n")
    _dump_fh.flush()
    log.info("observation dump -> %s (%.3f fps)", path, _dump_fps)


def stop_observation_dump() -> None:
    global _dump_fh, _dump_fps
    if _dump_fh is not None:
        _dump_fh.close()
    _dump_fh = None
    _dump_fps = 0.0


def record_observation_sample(frame_index: int, source: str, track_id: str,
                              basic_type: str, lat: float, lon: float,
                              speed_mps, heading_deg) -> None:
    """Write one CSV row; a no-op unless start_observation_dump() was called."""
    if _dump_fh is None:
        return
    basic_type = _basic_type_for(basic_type) or basic_type
    t_scene = frame_index / _dump_fps

    def _f(v):
        return "" if v is None else f"{v:.1f}"

    _dump_fh.write(f"{t_scene:.4f},{source},{track_id},{basic_type},"
                   f"{lat:.7f},{lon:.7f},{_f(speed_mps)},{_f(heading_deg)}\n")


def set_rsu_publish_filter(mode: str) -> None:
    """Restrict what the detector itself publishes to the gateway:
    "both", "vehicles", "vru" or "none"."""
    global _rsu_publish
    if mode not in _RSU_PUBLISH_MODES:
        raise ValueError(
            f"rsu publish filter must be one of {sorted(_RSU_PUBLISH_MODES)}, got {mode!r}")
    _rsu_publish = mode
    if mode != "both":
        log.warning("RSU publish filter: %s -- the gateway will NOT see all "
                    "detector observations (R5 trial mode)", mode)


def publish(track_id, obj_class: str, lat: float, lon: float,
            confidence: float = 0.85,
            speed_mps: Optional[float] = None,
            heading_deg: Optional[float] = None,
            t_recv: Optional[float] = None,
            accuracy_m: Optional[float] = None,
            path=None,
            flagged_with=None,
            range_m: Optional[float] = None) -> None:
    """Publish one detector track to the gateway, if it is running."""
    if not _started or _adapter_getter is None:
        return
    basic_type = _basic_type_for(obj_class)
    if basic_type is None:
        return  # class we don't care about (e.g., dog, traffic light)
    if _rsu_publish == "none":
        return
    if _rsu_publish == "vehicles" and basic_type != "vehicle":
        return
    if _rsu_publish == "vru" and basic_type == "vehicle":
        return
    try:
        adapter = _adapter_getter()
    except RuntimeError:
        return  # adapter not yet constructed
    adapter.publish_observation({
        "track_id":    str(track_id),
        "basic_type":  basic_type,
        "lat":         float(lat),
        "lon":         float(lon),
        "speed_mps":   speed_mps,
        "heading_deg": heading_deg,
        "accuracy_m":  accuracy_m,
        "path":        path,
        "flagged_with": list(flagged_with or ()),
        "range_m":     None if range_m is None else float(range_m),
        "confidence":  float(confidence),
        "t_sender":    None,
        "t_recv":      _clock() if t_recv is None else float(t_recv),
    })
    if _print_pedestrian_gps and basic_type == "pedestrian":
        print(f"[GW] pedestrian track {track_id} at lat={lat:.7f}, lon={lon:.7f}")


_SCRIPT_CLASS = {"vehicle": "car", "pedestrian": "person", "cyclist": "bicycle"}


def publish_script_sample(sample: dict, t_recv: float, range_of=None) -> None:
    """Publish one scripted-actor sample: as a camera track if source is "RSU",
    else as a phone report."""
    source = sample.get("source", "SafeWalk")
    lat, lon = float(sample["lat"]), float(sample["lon"])
    track_id = str(sample.get("track_id", "SCRIPT-VRU"))
    basic_type = sample.get("basic_type", "pedestrian")
    if source == "RSU":
        publish(track_id, _SCRIPT_CLASS[basic_type], lat, lon,
                speed_mps=sample.get("speed_mps"),
                heading_deg=sample.get("heading_deg"),
                t_recv=t_recv,
                range_m=range_of(lat, lon) if range_of else None)
        return
    publish_peer_observation(lat=lat, lon=lon, track_id=track_id,
                             speed_mps=sample.get("speed_mps"),
                             heading_deg=sample.get("heading_deg"),
                             accuracy_m=sample.get("accuracy_m", 3.0),
                             source=source, basic_type=basic_type)


_last_you_print = [0.0]  # mutable cell so the closure can update it


def _confidence_from_accuracy(accuracy_m: Optional[float]) -> float:
    """Defer to the BLE-in adapter so both ingest paths share one curve."""
    try:
        from safecorners_gateway.adapters.ble_in import confidence_from_accuracy
        return confidence_from_accuracy(accuracy_m)
    except Exception:
        return 0.6


def _utc_from_sec_mark(sec_mark: int) -> Optional[float]:
    """ms within the minute to absolute UTC, as the BLE-in adapter does it."""
    if not sec_mark:
        return None
    try:
        from safecorners_gateway.adapters.ble_in import _sender_utc_from_sec_mark
        return _sender_utc_from_sec_mark(sec_mark, time.time())
    except Exception:
        return None


def _publish_safewalk(lat: float, lon: float, track_id: str = "FAKE-PHONE",
                       speed_mps: Optional[float] = None,
                       heading_deg: Optional[float] = None,
                       accuracy_m: Optional[float] = None,
                       basic_type_b: int = 1,
                       source: Optional[str] = None,
                       basic_type: Optional[str] = None,
                       t_sender_utc: Optional[float] = None,
                       time_source: Optional[str] = None,
                       radius_of_curve_m: Optional[float] = None,
                       path_confidence: Optional[float] = None) -> None:
    """Inject one peer PSM (inject-file watcher and HTTP bridge).
    source and basic_type, when given, take precedence over basic_type_b."""
    if not _started or _adapter_getter is None:
        return
    try:
        adapter = _adapter_getter()
    except RuntimeError:
        return
    # Bypass the RSU-flavored convenience and push a peer Observation directly.
    try:
        from safecorners_gateway.adapters.ble_in import BASIC_TYPE_TO_INFO
        _src, _btype = BASIC_TYPE_TO_INFO.get(basic_type_b, ("SafeWalk", "unknown"))
    except Exception:
        _src, _btype = "SafeWalk", "pedestrian"
    if source:
        _src = source
    if basic_type:
        # An explicit role takes precedence over basicType.
        _btype = basic_type
    # An unrecognised basicType stays "unknown".

    _tsrc = time_source or "unknown"
    if t_sender_utc is None:
        # No usable sender clock, whatever the sender claimed.
        _tsrc = "unknown"
    import uuid
    from safecorners_gateway.types import Observation
    try:
        obs = Observation(
            source=_src,
            obs_id=uuid.uuid4().hex[:12],
            track_id=track_id,
            basic_type=_btype,
            lat=float(lat),
            lon=float(lon),
            # None stays None.
            speed_mps=None if speed_mps is None else float(speed_mps),
            heading_deg=heading_deg if heading_deg is None else float(heading_deg),
            accuracy_m=accuracy_m,
            # The same accuracy-to-confidence curve as the BLE-in adapter.
            confidence=_confidence_from_accuracy(accuracy_m),
            t_sender=None,
            t_sender_utc=t_sender_utc,
            time_source=_tsrc,
            t_recv=_clock(),
            radius_of_curve_m=radius_of_curve_m,
            path_confidence=path_confidence,
        )
    except ValueError as e:
        # A malformed PSM is rejected and logged.
        log.warning("rejected injected PSM for %s: %s", track_id, e)
        with _stats_lock:
            _stats["inject_rejected"] += 1
        return False
    if _loop is None:
        return False
    def _enqueue() -> None:
        try:
            adapter.ingest_q.put_nowait(obs)
        except Exception:
            pass
    _loop.call_soon_threadsafe(_enqueue)
    published = True
    # Cache for the detector overlay: latest SafeWalk position per track_id.
    _latest_safewalk[track_id] = (float(lat), float(lon), time.monotonic())
    # Snapshot for the HUD of what the phone reports.
    with _stats_lock:
        _stats["safewalk"] = {
            "source": _src,
            "basic_type": _btype,
            "track_id": track_id,
            "lat": float(lat), "lon": float(lon),
            "speed_mps": speed_mps, "heading_deg": heading_deg,
            "accuracy_m": accuracy_m,
            "confidence": _confidence_from_accuracy(accuracy_m),
            "t": time.monotonic(),
        }
    # Throttled "[YOU]" printout, about 2 Hz.
    now = time.monotonic()
    if _print_pedestrian_gps and now - _last_you_print[0] >= 0.5:
        _last_you_print[0] = now
        spd = "?" if speed_mps is None else f"{speed_mps:.1f}"
        hdg = "?" if heading_deg is None else f"{heading_deg:.0f}"
        print(f"[YOU] phone {track_id} @ video=({lat:.7f}, {lon:.7f})  spd={spd}m/s  hdg={hdg}deg")
    return published


class SourceFreshness:
    """When each source last reported, and which have gone quiet."""

    def __init__(self, silence_s: float = 5.0) -> None:
        self.silence_s = float(silence_s)
        self.last_seen: dict = {}
        self._reported: set = set()
        self._returned: list = []

    def saw(self, source: str, t: float) -> None:
        if source in self._reported:
            self._reported.discard(source)
            self._returned.append(source)
        self.last_seen[source] = float(t)

    def stale(self, t: float) -> list:
        return sorted(s for s, last in self.last_seen.items()
                      if t - last >= self.silence_s)

    def stale_relative(self) -> list:
        """Sources stale against the newest record seen; no clock is read."""
        if not self.last_seen:
            return []
        return self.stale(max(self.last_seen.values()))

    def newly_stale(self, t: float) -> list:
        """Sources that crossed the silence threshold since the last call."""
        out = [s for s in self.stale(t) if s not in self._reported]
        self._reported.update(out)
        return out

    def newly_returned(self) -> list:
        """Sources that spoke again after having been reported stale."""
        out, self._returned = sorted(self._returned), []
        return out


_freshness = SourceFreshness()


def publish_peer_observation(**kwargs) -> None:
    """Public entry point for an injected peer observation;
    not subject to the RSU publish filter."""
    _publish_safewalk(**kwargs)


def start_safewalk_injector() -> None:
    """Start a thread that watches INJECT_FILE and publishes its positions;
    each line is "lat lon [track_id]"."""
    global _safewalk_watcher_started
    if _safewalk_watcher_started:
        return
    _safewalk_watcher_started = True

    def _watcher() -> None:
        last_print = 0.0
        while True:
            try:
                if INJECT_FILE.exists():
                    parts = INJECT_FILE.read_text().strip().split()
                    if len(parts) >= 2:
                        lat = float(parts[0]); lon = float(parts[1])
                        tid = parts[2] if len(parts) >= 3 else "FAKE-PHONE"
                        _publish_safewalk(lat, lon, tid)
                        now = time.time()
                        if now - last_print > 5.0:
                            log.info("[GW] injecting SafeWalk @ (%.7f, %.7f) as %s",
                                     lat, lon, tid)
                            last_print = now
            except Exception as e:
                log.debug("safewalk inject watcher error: %s", e)
            time.sleep(0.5)

    threading.Thread(target=_watcher, name="safewalk-injector", daemon=True).start()
    log.info("safewalk inject watcher started (write to %s)", INJECT_FILE)


class AnchorTranslator:
    """Real phone coordinates to the coordinates the gateway is told.
    translate: the first fix maps to the video anchor, later ones by offset."""

    R_EARTH_M = 6_371_000.0

    def __init__(self, video_anchor_lat: float, video_anchor_lon: float,
                 translate: bool = True) -> None:
        self.video_anchor_lat = float(video_anchor_lat)
        self.video_anchor_lon = float(video_anchor_lon)
        self.translate = bool(translate)
        self.home: Optional[tuple[float, float]] = None

    @staticmethod
    def is_junk(lat: float, lon: float) -> bool:
        # SafeWalk's placeholders: (12.34, 56.78) and the BIB_IST test point.
        return bool((abs(lat - 12.34) < 1e-3 and abs(lon - 56.78) < 1e-3)
                    or (abs(lat - 38.7367524) < 5e-7 and abs(lon + 9.1437386) < 5e-7)
                    or (abs(lat) < 1e-6 and abs(lon) < 1e-6)
                    or not (-90.0 <= lat <= 90.0)
                    or not (-180.0 <= lon <= 180.0))

    def __call__(self, real_lat: float, real_lon: float):
        import math

        if self.is_junk(real_lat, real_lon):
            return None
        if not self.translate:
            return real_lat, real_lon
        if self.home is None:
            self.home = (real_lat, real_lon)
            log.info("[BRIDGE] home anchor set: real=(%.7f, %.7f) -> video=(%.7f, %.7f)",
                     real_lat, real_lon, self.video_anchor_lat, self.video_anchor_lon)
            return self.video_anchor_lat, self.video_anchor_lon
        home_lat, home_lon = self.home
        dnorth = math.radians(real_lat - home_lat) * self.R_EARTH_M
        deast = (math.radians(real_lon - home_lon) * self.R_EARTH_M
                 * math.cos(math.radians(home_lat)))
        out_lat = self.video_anchor_lat + math.degrees(dnorth / self.R_EARTH_M)
        out_lon = self.video_anchor_lon + math.degrees(
            deast / (self.R_EARTH_M * math.cos(math.radians(self.video_anchor_lat))))
        return out_lat, out_lon


def start_safewalk_http_bridge(host: str = "0.0.0.0", port: int = 8765,
                               video_anchor_lat: Optional[float] = None,
                               video_anchor_lon: Optional[float] = None,
                               force_heading_deg: Optional[float] = None,
                               force_speed_mps: Optional[float] = None,
                               translate: bool = True) -> None:
    """HTTP bridge: accept PSMs POSTed as JSON to /psm, translate them to the
    scene's coordinates and publish them to the gateway."""
    if video_anchor_lat is None or video_anchor_lon is None:
        # default to map.txt origin
        try:
            map_path = Path(__file__).parent / "map.txt"
            with open(map_path) as f:
                first = f.readline().split()
            video_anchor_lat = float(first[0])
            video_anchor_lon = float(first[1])
            log.info("video anchor defaulted to map.txt origin: (%.6f, %.6f)",
                     video_anchor_lat, video_anchor_lon)
        except Exception as e:
            log.error("could not read map.txt: %s", e)
            return

    _translate = AnchorTranslator(video_anchor_lat, video_anchor_lon,
                                  translate=translate)

    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import json as _json

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *a, **kw):
            pass  # silence default access log

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = _json.loads(self.rfile.read(length))
                real_lat = float(body["lat"])
                real_lon = float(body["lon"])
                tid = body.get("track_id", "PHONE")
                spd = body.get("speed_mps")
                hdg = body.get("heading_deg")
                # GNSS accuracy in metres; it sets the source confidence.
                acc = body.get("accuracy_m")
                # basicType is the fallback when the scanner sends no source.
                btype = int(body.get("basicType", 1))
                src = body.get("source")
                btype_name = body.get("basic_type")
                # Sender clock, or secMark if t_sender_utc is absent.
                t_sender_utc = body.get("t_sender_utc")
                tsrc = body.get("time_source")
                if t_sender_utc is None and body.get("secMark"):
                    t_sender_utc = _utc_from_sec_mark(int(body["secMark"]))
                    tsrc = tsrc or "GPS"
                rcurve = body.get("radius_of_curve_m")
                pconf = body.get("path_confidence")
                # Demo overrides: pin heading and speed.
                if force_heading_deg is not None:
                    hdg = float(force_heading_deg)
                if force_speed_mps is not None:
                    spd = float(force_speed_mps)
                translated = _translate(real_lat, real_lon)
                if translated is None:
                    # Placeholder PSM: answer 200 but do not publish.
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"ok":true,"skipped":"no_gps_lock"}')
                    return
                video_lat, video_lon = translated
                accepted = _publish_safewalk(video_lat, video_lon, tid,
                                  speed_mps=spd, heading_deg=hdg,
                                  accuracy_m=acc, basic_type_b=btype,
                                  source=src, basic_type=btype_name,
                                  t_sender_utc=t_sender_utc, time_source=tsrc,
                                  radius_of_curve_m=rcurve,
                                  path_confidence=pconf)
                if accepted is False:
                    # Decoded, but not a valid Observation: answer 400.
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'{"error":"rejected: invalid observation"}')
                    return
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"ok":true}')
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(f'{{"error":"{e}"}}'.encode())

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            home_str = ("unset" if _translate.home is None
                        else f"({_translate.home[0]:.7f}, {_translate.home[1]:.7f})")
            mode = "anchored to video" if _translate.translate else "real coordinates (no translation)"
            self.wfile.write(f"safewalk bridge OK\nmode={mode}\nhome={home_str}\n"
                             f"video_anchor=({video_anchor_lat}, {video_anchor_lon})\n".encode())

    def _server() -> None:
        # Threading server: the scanner POSTs several times a second.
        srv = ThreadingHTTPServer((host, port), _Handler)
        srv.daemon_threads = True
        if translate:
            log.info("safewalk HTTP bridge listening on %s:%d (video anchor=%.6f, %.6f)",
                     host, port, video_anchor_lat, video_anchor_lon)
        else:
            log.info("safewalk HTTP bridge listening on %s:%d "
                     "(NO TRANSLATION: phone coordinates used as they arrive)",
                     host, port)
        srv.serve_forever()

    threading.Thread(target=_server, name="safewalk-http-bridge", daemon=True).start()


def _basic_type_for(obj_class: str) -> Optional[str]:
    if obj_class in _VEHICLE_CLASSES:
        return "vehicle"
    if obj_class in _PEDESTRIAN_CLASSES:
        return "pedestrian"
    if obj_class in _CYCLIST_CLASSES:
        return "cyclist"
    return None


def _default_config_path() -> Path:
    """Pick the gateway config and log which: SAFECORNERS_GATEWAY_CONFIG, else
    gateway.yaml (gateway.demo.yaml if SAFECORNERS_GATEWAY_PROFILE=demo)."""
    here = Path(__file__).resolve()
    base = here.parents[2] / "SafeCorners-Gateway"

    override = os.environ.get("SAFECORNERS_GATEWAY_CONFIG")
    if override:
        p = Path(override)
        log.info("gateway config: %s (SAFECORNERS_GATEWAY_CONFIG)", p)
        return p

    profile = os.environ.get("SAFECORNERS_GATEWAY_PROFILE", "").lower()
    demo = base / "gateway.demo.yaml"
    if profile == "demo":
        if demo.exists():
            log.warning("gateway config: %s -- DEMO PROFILE, association is "
                        "loosened (radius 25 m, score 0.2, heading/speed "
                        "ignored). Do NOT report evaluation numbers from this.",
                        demo)
            return demo
        log.warning("SAFECORNERS_GATEWAY_PROFILE=demo but %s not found; "
                    "falling back to gateway.yaml", demo)

    prod = base / "gateway.yaml"
    log.info("gateway config: %s (production thresholds)", prod)
    return prod
