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
INJECT_FILE = Path("/tmp/safewalk_inject.txt")


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

    ready = threading.Event()

    def _runner() -> None:
        global _loop
        loop = asyncio.new_event_loop()
        _loop = loop
        asyncio.set_event_loop(loop)
        # Schedule the gateway, then signal readiness once the adapter exists.
        gateway_task = loop.create_task(run_gateway(cfg, run_id=run_id))

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


def _publish_safewalk(lat: float, lon: float, track_id: str = "FAKE-PHONE") -> None:
    """Inject one fake SafeWalk PSM (used by the inject-file watcher)."""
    if not _started or _adapter_getter is None:
        return
    try:
        adapter = _adapter_getter()
    except RuntimeError:
        return
    # Bypass the RSU-flavored convenience and push a SafeWalk Observation directly.
    import uuid
    from safecorners_gateway.types import Observation
    obs = Observation(
        source="SafeWalk",
        obs_id=uuid.uuid4().hex[:8],
        track_id=track_id,
        basic_type="pedestrian",
        lat=float(lat),
        lon=float(lon),
        speed_mps=0.5,
        heading_deg=None,
        accuracy_m=10.0,
        confidence=0.85,
        t_sender=None,
        t_sender_utc=None,
        time_source="unknown",
        t_recv=time.monotonic(),
    )
    if _loop is None:
        return
    def _enqueue() -> None:
        try:
            adapter.ingest_q.put_nowait(obs)
        except Exception:
            pass
    _loop.call_soon_threadsafe(_enqueue)


def start_safewalk_injector() -> None:
    """Start a background thread that watches /tmp/safewalk_inject.txt.

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
                               video_anchor_lon: Optional[float] = None) -> None:
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

    def _translate(real_lat: float, real_lon: float) -> tuple[float, float]:
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

    from http.server import BaseHTTPRequestHandler, HTTPServer
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
                video_lat, video_lon = _translate(real_lat, real_lon)
                _publish_safewalk(video_lat, video_lon, tid)
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
        srv = HTTPServer((host, port), _Handler)
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
    """Look for gateway.yaml in the SafeCorners-Gateway sibling folder.
    Prefers gateway.demo.yaml if present (loosened thresholds for testing)."""
    here = Path(__file__).resolve()
    base = here.parents[2] / "SafeCorners-Gateway"
    demo = base / "gateway.demo.yaml"
    if demo.exists():
        return demo
    return base / "gateway.yaml"
