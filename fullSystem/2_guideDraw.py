"""
Step 2 - Auto-detect vehicle guide trajectories.

Runs YOLO + DeepSort over an input source and records the centre-point pixel
path of every tracked vehicle. On exit the paths are overlaid on road.png and
saved as withguide.png (input for step 3).

Sources:
    --source path/to/video.mp4   a video file  (coords land in video pixel space)
    --source monitor:N           capture monitor N
    --source region:X,Y,W,H      capture a screen region
    (omit)                       interactive monitor picker

IMPORTANT: the recorded points must share a coordinate space with road.png and
map.txt. When calibration was done on frames of a video, pass that video with
--source rather than capturing it through a (differently-sized) screen.
"""
import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import sys
import random
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="SafeRoadside step 2 - auto-detect vehicle guide paths",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help=(
            "Input source. Options:\n"
            "  path/to/video.mp4   - use a video file\n"
            "  monitor:N           - capture monitor N (1=primary, ...)\n"
            "  region:X,Y,W,H      - capture a screen region\n"
            "  (omit)              - interactive monitor picker"
        )
    )
    parser.add_argument("--stride", type=int, default=1,
                        help="Process every Nth frame (video only). Default 1.")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N processed frames (0 = no limit).")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="YOLO inference size. Larger finds smaller/distant "
                             "vehicles on 4K input. Default 1280.")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="YOLO confidence threshold. Default 0.25.")
    parser.add_argument("--no-display", action="store_true",
                        help="Run headless (no preview window) - much faster.")
    parser.add_argument("--out", type=str, default="withguide.png",
                        help="Output overlay image. Default withguide.png.")
    parser.add_argument("--dump-tracks", type=str, default=None,
                        help="Also write raw per-track pixel paths to this file.")
    return parser.parse_args()


class VideoSource:
    """Reads frames from a video file."""

    def __init__(self, path, stride=1):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            print(f"[ERRO] Cannot open video: {path}")
            sys.exit(1)
        self.stride = max(1, stride)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.idx = 0
        print(f"[INFO] Source: video '{path}' ({self.w}x{self.h}, "
              f"{self.total} frames @ {self.fps:.2f} fps, stride={self.stride})")

    def grab(self):
        # Read (and discard) stride-1 frames so tracking sees even spacing.
        for _ in range(self.stride - 1):
            if not self.cap.grab():
                return None
            self.idx += 1
        ret, frame = self.cap.read()
        if not ret:
            return None
        self.idx += 1
        return frame

    def progress(self):
        return self.idx, self.total

    def release(self):
        self.cap.release()


class ScreenSource:
    """Captures a specific monitor or screen region via mss."""

    def __init__(self, region):
        self.sct = mss()
        self.region = region
        print(f"[INFO] Source: screen {region['width']}x{region['height']}")

    def grab(self):
        shot = self.sct.grab(self.region)
        return cv2.cvtColor(np.array(shot, dtype=np.uint8), cv2.COLOR_BGRA2BGR)

    def progress(self):
        return 0, 0

    def release(self):
        pass


def pick_monitor_interactive():
    sct = mss()
    monitors = sct.monitors
    print("\n============================================")
    print("  Available monitors:")
    print("============================================")
    for i, m in enumerate(monitors):
        label = "ALL (virtual)" if i == 0 else f"Monitor {i}"
        print(f"  {i}) {label}  -  {m['width']}x{m['height']}  at ({m['left']}, {m['top']})")
    print("============================================")
    while True:
        try:
            choice = int(input(f"\nSelect monitor [1-{len(monitors) - 1}] (or 0 for all): "))
            if 0 <= choice < len(monitors):
                return monitors[choice]
            print(f"  Invalid. Choose 0 to {len(monitors) - 1}.")
        except (ValueError, EOFError):
            print(f"  Invalid input. Choose 0 to {len(monitors) - 1}.")


def create_source(args):
    s = args.source
    if s is None:
        return ScreenSource(pick_monitor_interactive())
    if s.startswith("monitor:"):
        sct = mss()
        return ScreenSource(sct.monitors[int(s.split(":")[1])])
    if s.startswith("region:"):
        x, y, w, h = map(int, s.split(":")[1].split(","))
        return ScreenSource({"left": x, "top": y, "width": w, "height": h})
    return VideoSource(s, stride=args.stride)


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] YOLO device: {device}")
    model = YOLO("models/yolo11n.pt").to(device)
    tracker = DeepSort(max_age=1000, n_init=1, nn_budget=50,
                       embedder_gpu=(device == "cuda"))

    if not os.path.exists("road.png"):
        print("road.png not found!")
        sys.exit(1)
    image = cv2.imread("road.png")
    print(f"[INFO] road.png is {image.shape[1]}x{image.shape[0]}")

    source = create_source(args)

    # Warn loudly if the source and road.png disagree - points would be
    # recorded in a coordinate space that does not match the calibration.
    if isinstance(source, VideoSource) and (source.w, source.h) != (image.shape[1], image.shape[0]):
        print(f"[AVISO] source is {source.w}x{source.h} but road.png is "
              f"{image.shape[1]}x{image.shape[0]} - trajectories will NOT align "
              f"with map.txt. Re-extract road.png from this video.")

    point_history = {}
    track_colors = {}
    vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']

    def get_track_color(track_id):
        if track_id not in track_colors:
            track_colors[track_id] = (random.randint(50, 255),
                                      random.randint(50, 255),
                                      random.randint(50, 255))
        return track_colors[track_id]

    processed = 0
    try:
        while True:
            frame = source.grab()
            if frame is None:
                print("[INFO] Source exhausted.")
                break

            results = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                    verbose=False)
            detections = []
            for r in results[0].boxes:
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1],
                                   float(r.conf[0]), int(r.cls[0])))

            tracks = tracker.update_tracks(detections, frame=frame)
            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                obj = model.names[int(track.get_det_class())]
                if obj in vehicle_classes:
                    point_history.setdefault(track.track_id, []).append(center)
                if not args.no_display:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{obj} #{track.track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            processed += 1
            if processed % 25 == 0:
                idx, total = source.progress()
                pct = f" ({100.0 * idx / total:.1f}%)" if total else ""
                print(f"[INFO] frame {idx}/{total}{pct} - "
                      f"{len(point_history)} vehicle tracks so far", flush=True)

            if not args.no_display:
                disp = frame
                if disp.shape[1] > 1600:
                    scale = 1600.0 / disp.shape[1]
                    disp = cv2.resize(disp, None, fx=scale, fy=scale)
                cv2.imshow("AutoDraw - Tracking", disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Stopped by user.")
                    break

            if args.max_frames and processed >= args.max_frames:
                print(f"[INFO] Reached --max-frames {args.max_frames}.")
                break
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        source.release()
        cv2.destroyAllWindows()

    # Overlay every tracked path on the road image.
    for track_id, points in point_history.items():
        color = get_track_color(track_id)
        for point in points:
            cv2.circle(image, point, 3, color, -1)
    cv2.imwrite(args.out, image)

    total_pts = sum(len(p) for p in point_history.values())
    print(f"[OK] {len(point_history)} vehicle tracks, {total_pts} points "
          f"-> {args.out}")

    if args.dump_tracks:
        with open(args.dump_tracks, "w") as f:
            for track_id, points in point_history.items():
                pts = ", ".join(f"({x}, {y})" for x, y in points)
                f.write(f"Traj: {pts}\n")
        print(f"[OK] raw tracks -> {args.dump_tracks}")


if __name__ == "__main__":
    main()
