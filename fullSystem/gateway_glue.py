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


def _basic_type_for(obj_class: str) -> Optional[str]:
    if obj_class in _VEHICLE_CLASSES:
        return "vehicle"
    if obj_class in _PEDESTRIAN_CLASSES:
        return "pedestrian"
    if obj_class in _CYCLIST_CLASSES:
        return "cyclist"
    return None


def _default_config_path() -> Path:
    """Look for gateway.yaml in the SafeCorners-Gateway sibling folder."""
    here = Path(__file__).resolve()
    return here.parents[2] / "SafeCorners-Gateway" / "gateway.yaml"
