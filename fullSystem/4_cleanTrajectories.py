"""Step 4: downsample and smooth the reference trajectories."""
import argparse
import math
import os
import re
import sys


def parse_args():
    p = argparse.ArgumentParser(
        description="SafeRoadside step 4 - clean reference trajectories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", default="trajetorias.txt", help="Raw trajectories")
    p.add_argument("--output", default="trajetoriasClean.txt", help="Cleaned output")
    p.add_argument("--min-distance", type=float, default=20.0,
                   help="Drop points closer than this (px) to the last kept point")
    p.add_argument("--window", type=int, default=5,
                   help="Moving-average window for smoothing (points)")
    p.add_argument("--min-points", type=int, default=4,
                   help="Discard a trajectory left with fewer points than this. "
                        "find_best_trajectory needs >4 to be usable")
    p.add_argument("--frame-width", type=int, default=0,
                   help="If set, scale --min-distance by frame-width/reference-width")
    p.add_argument("--reference-width", type=int, default=1920,
                   help="Resolution --min-distance was tuned at")
    return p.parse_args()


def calcular_distancia(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def suavizar_trajetoria(pontos, janela=5):
    """Moving-average smoothing over a window of `janela` points."""
    suavizados = []
    for i in range(len(pontos)):
        start = max(i - janela // 2, 0)
        end = min(i + janela // 2 + 1, len(pontos))
        j = pontos[start:end]
        suavizados.append((int(sum(p[0] for p in j) / len(j)),
                           int(sum(p[1] for p in j) / len(j))))
    return suavizados


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"[ERRO] input not found: {args.input}")
        sys.exit(1)

    min_dist = args.min_distance
    if args.frame_width:
        k = args.frame_width / args.reference_width
        min_dist *= k
        print(f"[INFO] frame width {args.frame_width} -> scale x{k:.2f}, "
              f"min-distance {args.min_distance:.0f} -> {min_dist:.0f} px")

    kept, dropped = 0, 0
    total_in, total_out = 0, 0

    with open(args.input, "r") as infile, open(args.output, "w") as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue

            points = [(int(x), int(y))
                      for x, y in re.findall(r"\((-?\d+),\s*(-?\d+)\)", line)]
            if not points:
                continue
            total_in += len(points)

            # Keep a point once it is far enough from the last kept one.
            selected = []
            last = None
            for point in points:
                if last is None or calcular_distancia(last, point) > min_dist:
                    selected.append(point)
                    last = point

            if len(selected) < args.min_points:
                dropped += 1
                continue

            suavizados = suavizar_trajetoria(selected, janela=args.window)
            total_out += len(suavizados)
            kept += 1
            outfile.write("Traj: " + ", ".join(f"({x}, {y})" for x, y in suavizados) + "\n")

    print(f"[OK] '{args.output}' written")
    print(f"     trajectories: {kept} kept, {dropped} dropped "
          f"(fewer than {args.min_points} points after downsampling)")
    print(f"     points: {total_in} in -> {total_out} out")
    if kept == 0:
        print("[ERRO] no trajectories survived - lower --min-distance")
        sys.exit(1)


if __name__ == "__main__":
    main()
