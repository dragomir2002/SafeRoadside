"""Inject a fake SafeWalk PSM into a running gateway.

Usage: python inject_safewalk.py <lat> <lon> [--rate HZ] [--track-id ID]

Drops you at <lat,lon> as a "pedestrian" SafeWalk source and re-publishes
every 1/rate seconds until you Ctrl-C. The gateway's RsuAdapter is
already threadsafe — this script just imports it and pushes Observations
straight into ingest_q, exactly like the BleScanner adapter would.

Run this in a SECOND terminal while 5_realTime.py is running in the FIRST.
Both processes share nothing — but actually... they do need to share a
Python process, because the gateway runs inside the detector's process.

So this script CAN'T be a separate process.

Workaround: this script just bumps a file `/tmp/safewalk_inject.txt`
with "lat lon", and a tiny watcher thread inside the detector reads it
and publishes. See gateway_glue.start_safewalk_injector() — call once
near start_gateway().
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

INJECT_FILE = Path("/tmp/safewalk_inject.txt")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("lat", type=float)
    p.add_argument("lon", type=float)
    p.add_argument("--rate", type=float, default=2.0,
                   help="how many times per second to refresh the inject file (default 2)")
    p.add_argument("--track-id", default="FAKE-PHONE",
                   help="phone identifier (default FAKE-PHONE)")
    p.add_argument("--clear", action="store_true",
                   help="just delete the inject file and exit")
    args = p.parse_args()

    if args.clear:
        INJECT_FILE.unlink(missing_ok=True)
        print(f"removed {INJECT_FILE}")
        return 0

    print(f"injecting SafeWalk @ ({args.lat}, {args.lon}) as track_id={args.track_id}")
    print(f"writing to {INJECT_FILE} at {args.rate} Hz; Ctrl-C to stop")
    period = 1.0 / args.rate
    try:
        while True:
            INJECT_FILE.write_text(f"{args.lat} {args.lon} {args.track_id}\n")
            time.sleep(period)
    except KeyboardInterrupt:
        INJECT_FILE.unlink(missing_ok=True)
        print(f"\nstopped; cleared {INJECT_FILE}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
