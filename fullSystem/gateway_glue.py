"""Bridge between the synchronous detector loop and the asyncio gateway.

The SafeCorners gateway runs as an asyncio app; this detector runs as a
synchronous while-True loop. We start the gateway on a background thread
that owns its own event loop, and expose a sync `publish()` that hands
each detector track to the gateway's RsuAdapter (which is already
threadsafe via loop.call_soon_threadsafe).

Usage from 5_realTime.py:

    from gateway_glue import start_gateway, publish
    start_gateway()                           # once, before the main loop
    ...
    publish(track_id, obj_class, lat, lon)    # once per confirmed track per frame

If the gateway package is not installed, both calls become no-ops so the
detector still runs standalone.
"""
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
_adapter_getter = None  # callable -> RsuAdapter
_print_pedestrian_gps = True  # toggled by start_gateway(verbose_pedestrians=...)
_safewalk_watcher_started = False
# Platform temp dir: /tmp on Linux, the user's temp dir on Windows.
INJECT_FILE = Path(tempfile.gettempdir()) / "safewalk_inject.txt"

# Latest SafeWalk position published, so the detector can render a marker.
# Keyed by track_id -> (video_lat, video_lon, monotonic_ts).
_latest_safewalk: dict = {}

# Live pipeline counters for the detector's on-screen HUD. Fed by the observer
# passed into run_gateway, so the HUD never has to tail gateway.jsonl.
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
    # Per-source: the risk of the most recent fusion this source fed into, so
    # the HUD can colour each subsystem by the alarm IT is contributing to.
    # Neither subsystem emits a risk of its own -- the PSM has no risk field
    # and the RSU publishes raw tracks -- so this is "the risk this source is
    # currently implicated in", not "the risk this source computed".
    "source_risk": {},   # source -> {"risk": str, "t": float}
    # PSM advertisements the gateway's own BLE scanner saw on the inbound
    # manufacturer ID but could not decode, broken down by payload length.
    # A non-zero count against a length the gateway does speak means corrupt
    # frames; against an unknown length it means a dialect it does not.
    "decode_errors": 0,
    "decode_error_len": {},   # str(payload length) -> count
}
_stats_lock = threading.Lock()


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
            # A phone advertising a dialect the gateway cannot parse used to
            # produce no signal anywhere. Surfacing the count on the HUD means
            # "the phone is right there and nothing is arriving" is visible at
            # a glance instead of being inferred.
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
        # Shallow copy plus an explicit copy of every nested container -- a
        # caller iterating a live dict while the pipeline mutates it raises
        # RuntimeError, and this is called every frame.
        out = dict(_stats)
        out["risk"] = dict(_stats["risk"])
        out["source_risk"] = {k: dict(v) for k, v in _stats["source_risk"].items()}
        out["decode_error_len"] = dict(_stats["decode_error_len"])
        return out


def set_verbose_pedestrians(enabled: bool) -> None:
    """Toggle the per-pedestrian '[GW] pedestrian track N at lat=...' line.

    It prints once per pedestrian per frame, which at a busy intersection
    buries everything else in the console. The same information is on the
    HUD and in gateway.jsonl.
    """
    global _print_pedestrian_gps
    _print_pedestrian_gps = bool(enabled)


def get_latest_safewalk(max_age_s: float = 2.0) -> dict:
    """Return {track_id: (lat, lon)} for SafeWalk phantoms seen within max_age_s.

    The detector calls this each frame to draw "YOU" markers on the video.
    Stale entries (no PSM in the last max_age_s seconds) are filtered out.
    """
    now = time.monotonic()
    return {tid: (lat, lon)
            for tid, (lat, lon, ts) in _latest_safewalk.items()
            if now - ts <= max_age_s}


def start_gateway(config_path: Optional[Path] = None,
                  run_id: Optional[str] = None) -> bool:
    """Spin up the gateway on a daemon thread. Returns True on success.

    Safe to call multiple times — subsequent calls are no-ops.
    Returns False (and logs) if the gateway package isn't installed.
    """
    global _started, _loop, _adapter_getter
    if _started:
        return True

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
            run_gateway(cfg, run_id=run_id, observer=_observe))

        async def _wait_for_adapter() -> None:
            for _ in range(50):  # up to 5 s
                try:
                    get_rsu_adapter()
                    ready.set()
                    return
                except RuntimeError:
                    await asyncio.sleep(0.1)
            ready.set()  # give up; publish() will degrade

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


def publish(track_id, obj_class: str, lat: float, lon: float,
            confidence: float = 0.85,
            speed_mps: Optional[float] = None,
            heading_deg: Optional[float] = None) -> None:
    """Publish one detector track to the gateway. No-op if gateway isn't running."""
    if not _started or _adapter_getter is None:
        return
    basic_type = _basic_type_for(obj_class)
    if basic_type is None:
        return  # class we don't care about (e.g., dog, traffic light)
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
        "accuracy_m":  None,
        "confidence":  float(confidence),
        "t_sender":    None,
        "t_recv":      time.monotonic(),
    })
    if _print_pedestrian_gps and basic_type == "pedestrian":
        print(f"[GW] pedestrian track {track_id} at lat={lat:.7f}, lon={lon:.7f}")


_last_you_print = [0.0]  # mutable cell so the closure can update it


def _confidence_from_accuracy(accuracy_m: Optional[float]) -> float:
    """Defer to the BLE-in adapter so both ingest paths share one curve."""
    try:
        from safecorners_gateway.adapters.ble_in import confidence_from_accuracy
        return confidence_from_accuracy(accuracy_m)
    except Exception:
        return 0.6


def _utc_from_sec_mark(sec_mark: int) -> Optional[float]:
    """ms-within-the-minute -> absolute UTC, as the BLE-in adapter does it.

    Only used for older scanners that post a bare `secMark`; a current scanner
    posts `t_sender_utc` directly because it has already decoded the PSM.
    """
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
                       t_sender_utc: Optional[float] = None,
                       time_source: Optional[str] = None,
                       radius_of_curve_m: Optional[float] = None,
                       path_confidence: Optional[float] = None) -> None:
    """Inject one peer PSM (used by the inject-file watcher and HTTP bridge).

    `source` is the producing app as the scanner's decoder determined it from
    the payload dialect, and is authoritative when supplied. Falling back to
    basic_type_b is weaker -- SafeWalk cannot emit a cyclist, so basicType
    alone cannot really distinguish the two apps -- but it keeps older senders
    working. This whole path used to hardcode SafeWalk/pedestrian, so a
    cyclist over the HTTP bridge was silently relabelled a pedestrian and
    SafeBike could never appear as a source here at all.

    t_sender_utc / time_source carry the sender's own clock. The bridge used to
    drop them and hardcode time_source="unknown", which demoted every
    observation on this path to the lowest-confidence time regime of §4.4 --
    even though the phone populates the field and the scanner forwards it. On
    Windows this is the path that actually runs, so R2's primary clock was
    unavailable exactly where it mattered.
    """
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
    # Deliberately NOT coerced to "pedestrian". Reporting an unrecognised
    # basicType as a pedestrian is exactly the role fabrication that was
    # removed from the BLE path: it puts a road user the sender never claimed
    # into the VRU population, and every per-class figure then counts it.
    #
    # The cost is that an "unknown" observation is neither a VRU nor a vehicle
    # to stages/associate.py, so it is dropped from pairing. That is the right
    # trade -- and it is moot with today's senders, since SafeWalk only emits
    # basicType 0/1 and SafeBike only 2. Should a sender ever use 3 (public
    # safety worker) or 4 (animal), decide then what those are, rather than
    # silently having decided already.

    _tsrc = time_source or "unknown"
    if t_sender_utc is None:
        # No usable sender clock, whatever the sender claimed.
        _tsrc = "unknown"
    import uuid
    from safecorners_gateway.types import Observation
    try:
        obs = Observation(
            source=_src,
            obs_id=uuid.uuid4().hex[:8],
            track_id=track_id,
            basic_type=_btype,
            lat=float(lat),
            lon=float(lon),
            speed_mps=0.5 if speed_mps is None else float(speed_mps),
            heading_deg=heading_deg if heading_deg is None else float(heading_deg),
            accuracy_m=accuracy_m,
            # Same GNSS-accuracy -> confidence curve the BLE-in adapter uses, so a
            # phone reaching the gateway over the HTTP bridge is weighted exactly
            # as it would be over the radio. This used to be a flat 0.85 with a
            # made-up 10 m accuracy, which silently discarded the one quality
            # signal the PSM carries.
            confidence=_confidence_from_accuracy(accuracy_m),
            t_sender=None,
            t_sender_utc=t_sender_utc,
            time_source=_tsrc,
            t_recv=time.monotonic(),
            radius_of_curve_m=radius_of_curve_m,
            path_confidence=path_confidence,
        )
    except ValueError as e:
        # A malformed POST must not take down the bridge thread.
        log.warning("rejected injected PSM for %s: %s", track_id, e)
        return
    if _loop is None:
        return
    def _enqueue() -> None:
        try:
            adapter.ingest_q.put_nowait(obs)
        except Exception:
            pass
    _loop.call_soon_threadsafe(_enqueue)
    # Cache for the detector overlay: latest SafeWalk position per track_id.
    _latest_safewalk[track_id] = (float(lat), float(lon), time.monotonic())
    # Fuller snapshot for the HUD: what the phone itself is reporting. Note the
    # inbound PSM carries no risk field -- SafeWalk broadcasts position and
    # kinematics only, and the risk verdict is the gateway's.
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
    # Throttled "[YOU]" printout so the operator can see their own phantom
    # tick alongside the detector's [GW] pedestrian lines. ~2 Hz max.
    now = time.monotonic()
    if _print_pedestrian_gps and now - _last_you_print[0] >= 0.5:
        _last_you_print[0] = now
        spd = "?" if speed_mps is None else f"{speed_mps:.1f}"
        hdg = "?" if heading_deg is None else f"{heading_deg:.0f}"
        print(f"[YOU] phone {track_id} @ video=({lat:.7f}, {lon:.7f})  spd={spd}m/s  hdg={hdg}deg")


def start_safewalk_injector() -> None:
    """Start a background thread that watches INJECT_FILE.

    Each line in the file is "lat lon [track_id]". As long as the file
    exists, we publish a SafeWalk observation at that coord every 0.5s.
    """
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


def start_safewalk_http_bridge(host: str = "0.0.0.0", port: int = 8765,
                               video_anchor_lat: Optional[float] = None,
                               video_anchor_lon: Optional[float] = None,
                               force_heading_deg: Optional[float] = None,
                               force_speed_mps: Optional[float] = None) -> None:
    """HTTP bridge: accept POSTed SafeWalk PSMs from the Windows scanner, translate
    real-world coords to video-anchor coords, and publish to the gateway.

    Endpoint: POST http://<host>:8765/psm
        body: JSON {"lat": 38.6423, "lon": -9.1573, "track_id": "SM-S918",
                    "speed_mps": 0.5, "heading_deg": 90, "secMark": 17338}

    The first POST establishes the "real world home" anchor. Each subsequent
    POST computes (dnorth, deast) in meters from home and applies that delta
    to (video_anchor_lat, video_anchor_lon) before pushing to the gateway.
    Result: walking 5 m east IRL moves the avatar 5 m east in the video frame.

    If no video anchor is given, defaults to the map.txt origin.

    force_heading_deg / force_speed_mps:
        For remote-testing demos: phone GPS heading reflects the real-world
        street, not the video's street geometry. Set force_heading_deg=0 to
        always report "pedestrian walking north across the road" (or whatever
        direction puts you in collision course with traffic in the video).
        Same idea for force_speed_mps if you want a stable demo trajectory.
        In a production on-site deployment, leave both as None to use the
        phone's real values.
    """
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

    home_lat: list = [None]  # mutable cell

    import math
    R_EARTH = 6_371_000.0

    def _translate(real_lat: float, real_lon: float) -> tuple[float, float] | None:
        # Reject obvious placeholder/no-GPS-lock packets. SafeWalk emits
        # (12.34, 56.78) before its first GPS fix; a (0, 0) "Null Island"
        # fix or any wildly-out-of-range coord is also bogus. Returning
        # None makes the bridge drop the PSM instead of poisoning the
        # home anchor or shoving the avatar miles off-frame.
        if (abs(real_lat - 12.34) < 1e-3 and abs(real_lon - 56.78) < 1e-3) \
           or (abs(real_lat) < 1e-6 and abs(real_lon) < 1e-6) \
           or not (-90.0 <= real_lat <= 90.0) \
           or not (-180.0 <= real_lon <= 180.0):
            return None
        if home_lat[0] is None:
            home_lat[0] = real_lat
            home_lat.append(real_lon)  # store lon at index 1
            log.info("[BRIDGE] home anchor set: real=(%.7f, %.7f) -> video=(%.7f, %.7f)",
                     real_lat, real_lon, video_anchor_lat, video_anchor_lon)
            return video_anchor_lat, video_anchor_lon
        # delta in meters from home
        dlat = math.radians(real_lat - home_lat[0])
        dlon = math.radians(real_lon - home_lat[1])
        dnorth = dlat * R_EARTH
        deast = dlon * R_EARTH * math.cos(math.radians(home_lat[0]))
        # apply delta to video anchor
        out_lat = video_anchor_lat + math.degrees(dnorth / R_EARTH)
        out_lon = video_anchor_lon + math.degrees(deast / (R_EARTH * math.cos(math.radians(video_anchor_lat))))
        return out_lat, out_lon

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
                # GNSS accuracy ellipse (semi-major, metres). Drives the
                # source confidence, so a degraded urban-canyon fix is
                # weighted down rather than trusted like an open-sky one.
                acc = body.get("accuracy_m")
                # basicType decides SafeWalk-pedestrian vs SafeBike-cyclist,
                # but only as a fallback: a scanner that decoded the payload
                # knows which app sent it from the wire dialect and says so in
                # `source`, which is authoritative when present.
                btype = int(body.get("basicType", 1))
                src = body.get("source")
                # Sender clock. Previously dropped here, which demoted every
                # bridged observation to time_source="unknown" even when the
                # phone had populated it -- the same decode-then-discard
                # failure as the accuracy field before it. `secMark` is still
                # accepted from older scanners and reconstructed against
                # gateway wall time the way the BLE adapter does it.
                t_sender_utc = body.get("t_sender_utc")
                tsrc = body.get("time_source")
                if t_sender_utc is None and body.get("secMark"):
                    t_sender_utc = _utc_from_sec_mark(int(body["secMark"]))
                    tsrc = tsrc or "GPS"
                rcurve = body.get("radius_of_curve_m")
                pconf = body.get("path_confidence")
                # Demo-mode overrides: pin heading/speed regardless of what
                # the phone reports. Used when the phone is in a different
                # geography than the video and its real heading wouldn't
                # produce a meaningful TTC against the video's traffic.
                if force_heading_deg is not None:
                    hdg = float(force_heading_deg)
                if force_speed_mps is not None:
                    spd = float(force_speed_mps)
                translated = _translate(real_lat, real_lon)
                if translated is None:
                    # Placeholder PSM (no GPS lock yet) -- accept the HTTP
                    # request so the scanner doesn't retry-storm, but skip
                    # publishing to the gateway.
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"ok":true,"skipped":"no_gps_lock"}')
                    return
                video_lat, video_lon = translated
                _publish_safewalk(video_lat, video_lon, tid,
                                  speed_mps=spd, heading_deg=hdg,
                                  accuracy_m=acc, basic_type_b=btype,
                                  source=src,
                                  t_sender_utc=t_sender_utc, time_source=tsrc,
                                  radius_of_curve_m=rcurve,
                                  path_confidence=pconf)
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
            home_str = "unset" if home_lat[0] is None else f"({home_lat[0]:.7f}, {home_lat[1]:.7f})"
            self.wfile.write(f"safewalk bridge OK\nhome={home_str}\nvideo_anchor=({video_anchor_lat}, {video_anchor_lon})\n".encode())

    def _server() -> None:
        # Threading, not plain HTTPServer. BLE re-advertises the same PSM
        # several times a second and the scanner POSTs each sighting, so a
        # single-threaded server serialises them: while one request is being
        # handled the rest sit in the accept queue and blow the scanner's
        # 0.5 s read timeout, which looks like "the bridge is down" when it
        # is really just busy.
        srv = ThreadingHTTPServer((host, port), _Handler)
        srv.daemon_threads = True
        log.info("safewalk HTTP bridge listening on %s:%d (video anchor=%.6f, %.6f)",
                 host, port, video_anchor_lat, video_anchor_lon)
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
    """Pick the gateway config, and say out loud which one was chosen.

    gateway.demo.yaml loosens association so a demo still fires: radius 25 m
    instead of 10, score threshold 0.2 instead of 0.6, and heading/speed
    weights set to 0 so those terms are ignored entirely. That is useful for
    showing the pipeline working, but it disables two of the four association
    terms thesis 4.5 specifies, so numbers produced under it must not be
    reported as evaluation results.

    Selection order:
      1. SAFECORNERS_GATEWAY_CONFIG   explicit path, always wins
      2. gateway.yaml                 production thresholds (default)
      3. gateway.demo.yaml            only via SAFECORNERS_GATEWAY_PROFILE=demo

    This used to silently prefer the demo file whenever it existed, which made
    it very easy to evaluate against loosened thresholds without noticing.
    """
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
