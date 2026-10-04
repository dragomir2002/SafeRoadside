"""Step 2b: turn the vehicle paths of step 2 into reference trajectories."""
import argparse
import math
import os
import random
import re
import sys


def parse_args():
    p = argparse.ArgumentParser(
        description="SafeRoadside step 2b - derive reference trajectories from tracks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tracks", default="guide_tracks.txt",
                   help="Raw per-track paths from 2_guideDraw.py --dump-tracks")
    p.add_argument("--out", default="trajetorias.txt", help="Output trajectories file")
    p.add_argument("--preview", default=None,
                   help="Optional PNG overlaying kept trajectories on --background")
    p.add_argument("--background", default="road.png",
                   help="Background image used for --preview")
    p.add_argument("--min-points", type=int, default=12,
                   help="Reject tracks with fewer points than this")
    p.add_argument("--min-displacement", type=float, default=400.0,
                   help="Reject tracks whose start->end distance (px) is below this")
    p.add_argument("--min-arc", type=float, default=500.0,
                   help="Reject tracks whose total path length (px) is below this")
    p.add_argument("--min-straightness", type=float, default=0.35,
                   help="Reject tracks with displacement/arc below this (0..1)")
    p.add_argument("--max-jump", type=float, default=400.0,
                   help="Split a track wherever consecutive points jump further "
                        "than this (px) - usually an ID switch")
    p.add_argument("--dedupe-dist", type=float, default=0.0,
                   help="If > 0, drop a trajectory whose mean distance to an "
                        "already-kept one is below this (px)")
    p.add_argument("--max-out", type=int, default=0,
                   help="Keep at most this many trajectories (0 = no limit); "
                        "longest paths are preferred")
    return p.parse_args()


def load_tracks(path):
    """Read 'Traj: (x, y), (x, y), ...' lines into lists of int tuples."""
    tracks = []
    with open(path, "r") as f:
        for line in f:
            pts = re.findall(r"\((-?\d+),\s*(-?\d+)\)", line)
            if pts:
                tracks.append([(int(x), int(y)) for x, y in pts])
    return tracks


def dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def arc_length(pts):
    return sum(dist(pts[i - 1], pts[i]) for i in range(1, len(pts)))


def split_on_jumps(pts, max_jump):
    """Cut a path wherever it teleports - almost always a tracker ID switch."""
    segments = []
    current = [pts[0]]
    for i in range(1, len(pts)):
        if dist(pts[i - 1], pts[i]) > max_jump:
            segments.append(current)
            current = [pts[i]]
        else:
            current.append(pts[i])
    segments.append(current)
    return segments


def mean_min_distance(a, b):
    """Average distance from each point of a to the nearest point of b."""
    total = 0.0
    for p in a:
        total += min(dist(p, q) for q in b)
    return total / len(a)


def main():
    args = parse_args()

    if not os.path.exists(args.tracks):
        print(f"[ERRO] tracks file not found: {args.tracks}")
        sys.exit(1)

    raw = load_tracks(args.tracks)
    print(f"[INFO] loaded {len(raw)} raw tracks from {args.tracks}")

    stats = {"too_few_points": 0, "too_short_arc": 0,
             "too_little_displacement": 0, "not_straight_enough": 0}
    kept = []

    for pts in raw:
        if len(pts) < 2:
            stats["too_few_points"] += 1
            continue
        for seg in split_on_jumps(pts, args.max_jump):
            if len(seg) < args.min_points:
                stats["too_few_points"] += 1
                continue
            arc = arc_length(seg)
            if arc < args.min_arc:
                stats["too_short_arc"] += 1
                continue
            disp = dist(seg[0], seg[-1])
            if disp < args.min_displacement:
                stats["too_little_displacement"] += 1
                continue
            if disp / arc < args.min_straightness:
                stats["not_straight_enough"] += 1
                continue
            kept.append(seg)

    print(f"[INFO] {len(kept)} candidate trajectories after filtering")
    for k, v in stats.items():
        print(f"         rejected {v:5d}  ({k})")

    # Longest first: they carry the most predictive context downstream.
    kept.sort(key=arc_length, reverse=True)

    if args.dedupe_dist > 0:
        unique = []
        for seg in kept:
            if all(mean_min_distance(seg, u) > args.dedupe_dist for u in unique):
                unique.append(seg)
        print(f"[INFO] {len(unique)} trajectories after dedupe "
              f"(threshold {args.dedupe_dist:.0f} px)")
        kept = unique

    if args.max_out and len(kept) > args.max_out:
        kept = kept[:args.max_out]
        print(f"[INFO] truncated to --max-out {args.max_out}")

    if not kept:
        print("[ERRO] no trajectories survived filtering - loosen the thresholds")
        sys.exit(1)

    with open(args.out, "w") as f:
        for seg in kept:
            f.write("Traj: " + ", ".join(f"({x}, {y})" for x, y in seg) + "\n")

    lengths = [arc_length(s) for s in kept]
    print(f"[OK] {len(kept)} trajectories -> {args.out}")
    print(f"     points per trajectory: min {min(len(s) for s in kept)}, "
          f"max {max(len(s) for s in kept)}")
    print(f"     arc length px: min {min(lengths):.0f}, "
          f"max {max(lengths):.0f}, mean {sum(lengths) / len(lengths):.0f}")

    if args.preview:
        try:
            import cv2
        except ImportError:
            print("[AVISO] opencv not available - skipping preview")
            return
        if not os.path.exists(args.background):
            print(f"[AVISO] background not found: {args.background} - skipping preview")
            return
        img = cv2.imread(args.background)
        for seg in kept:
            color = (random.randint(60, 255), random.randint(60, 255),
                     random.randint(60, 255))
            for i in range(1, len(seg)):
                cv2.line(img, seg[i - 1], seg[i], color, 4)
            cv2.circle(img, seg[0], 12, (0, 255, 0), -1)
            cv2.circle(img, seg[-1], 12, (0, 0, 255), -1)
        cv2.imwrite(args.preview, img)
        print(f"[OK] preview -> {args.preview}  (green = start, red = end)")


if __name__ == "__main__":
    main()
