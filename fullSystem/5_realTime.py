import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque
import random
import logging
import sys
import math
import struct
import os
import argparse
import time

from gateway_glue import (start_gateway, publish as gw_publish,
                          start_safewalk_injector, start_safewalk_http_bridge,
                          get_latest_safewalk, get_gateway_stats,
                          set_verbose_pedestrians, wait_gateway_idle)
from rsu_kinematics import KinematicsEstimator, ReplayClock, pixel_to_latlon

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DE LOGGING
# -----------------------------------------------------------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger("ultralytics").setLevel(logging.WARNING)


# -----------------------------------------------------------------------------
# 0) LER O ARQUIVO map.txt
# -----------------------------------------------------------------------------
def load_map_data(map_file="map.txt"):
    """
    Lê lat0_deg, lon0_deg e a matriz de homografia de um arquivo map.txt
    no formato:
       lat0 lon0
       h11 h12 h13
       h21 h22 h23
       h31 h32 h33
    """
    try:
        with open(map_file, "r") as f:
            lines = f.readlines()

        # Primeira linha: lat0_deg, lon0_deg
        lat0_deg, lon0_deg = map(float, lines[0].split())

        # Próximas 3 linhas: matriz de homografia
        h_rows = []
        for i in range(1, 4):
            row_vals = list(map(float, lines[i].split()))
            h_rows.append(row_vals)

        # float64: in float32 the lat/lon arithmetic quantised every published
        # position to a 0.38 m grid (see rsu_kinematics.py).
        homography_mat = np.array(h_rows, dtype=np.float64)

        if homography_mat.shape != (3, 3):
            raise ValueError("A matriz de homografia não possui dimensões 3x3.")

        return lat0_deg, lon0_deg, homography_mat

    except Exception as e:
        print(f"Erro ao ler {map_file}: {e}")
        sys.exit(1)


# -----------------------------------------------------------------------------
# FUNÇÃO P/ CONVERTER (X, Y) EM LAT/LON
# -----------------------------------------------------------------------------
def xy_to_latlon(X, Y, lat0_deg, lon0_deg):
    """
    Inverso da projeção local.
    Dado (X, Y) em metros no sistema local,
    devolve (lat, lon) em graus decimais.
    """
    R = 6371000.0  # raio aproximado da Terra em metros
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)

    lat = Y / R + lat0
    lon = X / (R * math.cos(lat0)) + lon0

    # Converter p/ graus
    lat_deg = math.degrees(lat)
    lon_deg = math.degrees(lon)
    return lat_deg, lon_deg


def latlon_to_xy(lat_deg, lon_deg, lat0_deg, lon0_deg):
    """Inverso de xy_to_latlon: (lat, lon) em graus -> (X, Y) em metros."""
    R = 6371000.0
    lat0 = math.radians(lat0_deg)
    Y = math.radians(lat_deg - lat0_deg) * R
    X = math.radians(lon_deg - lon0_deg) * R * math.cos(lat0)
    return X, Y


# -----------------------------------------------------------------------------
# 1) CARREGAR TRAJETÓRIAS PREDEFINIDAS
# -----------------------------------------------------------------------------
def load_trajectories(file_path):
    """Carrega e retorna uma lista de trajetórias pré-definidas."""
    trajectories = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("Traj: "):
            points_str = line[len("Traj: "):].split('), (')
            points = []
            for p in points_str:
                p = p.replace('(', '').replace(')', '')
                x, y = map(int, p.split(', '))
                points.append((x, y))
            trajectories.append(np.array(points, dtype=np.int32))

    return trajectories


# -----------------------------------------------------------------------------
# 2) FUNÇÃO PARA ENCONTRAR A MELHOR TRAJETÓRIA (VEÍCULOS)
# -----------------------------------------------------------------------------
def find_best_trajectory(current_traj, predefined_trajs, max_points=50, score_threshold=3000):
    """Encontra a melhor continuação de trajetória com base em uma lista de trajetórias pré-definidas."""
    current_length = len(current_traj)
    if current_length < 4:
        return []

    current_arr = np.array(current_traj, dtype=np.float32)
    best_score = float('inf')
    best_match = []

    for traj in predefined_trajs:
        # Só consideramos trajetórias pré-definidas maiores do que a atual
        if len(traj) <= current_length:
            continue

        # Criamos uma janela deslizante para comparar com a current_traj
        for i in range(len(traj) - current_length):
            reference_window = traj[i : i + current_length]
            # Soma das distâncias entre pontos equivalentes
            diff = current_arr - reference_window
            score = np.sum(np.linalg.norm(diff, axis=1))

            if score < best_score:
                best_score = score
                # Pega próximos pontos para prever
                start_pred = i + current_length
                end_pred = start_pred + max_points
                best_match = traj[start_pred:end_pred]

    # Retorna somente se o score encontrado for significativo
    if best_score < score_threshold:
        return best_match
    else:
        return []


# -----------------------------------------------------------------------------
# 3) FILTRO DE KALMAN ESTENDIDO (EKF) PARA PEDESTRES
# -----------------------------------------------------------------------------
def ekf(previous_points, prediction_range=5):
    """
    Retorna uma lista de pontos previstos a partir dos 'previous_points'
    usando um EKF simplificado.
    """
    if len(previous_points) < 3:
        return []

    dt = 1.0
    Q = np.eye(6) * 0.1
    R = np.eye(2) * 0.5

    # Estado inicial (x, y, vx, vy, ax, ay)
    x = np.array([previous_points[0][0], previous_points[0][1], 0, 0, 0, 0], dtype=np.float32)
    P = np.eye(6, dtype=np.float32)

    # F_jacobian
    F_jacobian = np.array([
        [1, 0, dt, 0, 0.5 * dt**2, 0],
        [0, 1, 0, dt, 0, 0.5 * dt**2],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ], dtype=np.float32)

    # Matriz de observação
    H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0]
    ], dtype=np.float32)

    # Alimenta o EKF com os pontos anteriores
    for z in previous_points:
        z = np.array(z, dtype=np.float32)
        # Predição
        x_pred = F_jacobian @ x
        P_pred = F_jacobian @ P @ F_jacobian.T + Q

        # Inovação
        y = z - (H @ x_pred)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        # Atualização
        x = x_pred + K @ y
        I_KH = np.eye(len(K), dtype=np.float32) - (K @ H)
        P = I_KH @ P_pred

    # Gera previsões futuras
    predictions = []
    for _ in range(prediction_range):
        x_pred = F_jacobian @ x
        predictions.append(x_pred[:2].astype(np.int32))
        x = x_pred

    return predictions


# -----------------------------------------------------------------------------
# 4) CONFIGURAÇÃO DE YOLO E DEEPSORT
# -----------------------------------------------------------------------------
import torch as _torch
_DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
print(f"[INFO] YOLO device: {_DEVICE}")
model = YOLO("models/yolo11n.pt").to(_DEVICE) # TODO é possivel alterar ete valor para outros modelos do YOLO
tracker = DeepSort( # TODO ver se ja outra versoes mais fortes do yolo
    max_age=1000,
    n_init=5,
    nn_budget=10,
    embedder_gpu=True
)

# TODO da para mudar o YOLO para outras coisas tipo modelos treinaddos so em carros

# -----------------------------------------------------------------------------
# 5) CARREGAR MAPA (lat/lon + HOMOGRAFIA) E TRAJETÓRIAS
# -----------------------------------------------------------------------------
# Populated in main() from --map / --trajectories. They are module-level
# because the detection loop and the helpers below all read them as globals.
lat0_deg = lon0_deg = None
homography_mat = homography_mat_inv = None
predefined_trajectories = []


def load_scene(map_file="map.txt", traj_file="trajetoriasClean.txt"):
    """Load the homography + reference trajectories for the active scene."""
    global lat0_deg, lon0_deg, homography_mat, homography_mat_inv
    global predefined_trajectories

    lat0_deg, lon0_deg, homography_mat = load_map_data(map_file)
    homography_mat_inv = np.linalg.inv(homography_mat)  # for SafeWalk overlay
    print(f"[INFO] map '{map_file}': origin ({lat0_deg:.6f}, {lon0_deg:.6f})")

    if os.path.exists(traj_file):
        predefined_trajectories = load_trajectories(traj_file)
        print(f"[INFO] {len(predefined_trajectories)} reference trajectories "
              f"from '{traj_file}'")
    else:
        predefined_trajectories = []
        print(f"[AVISO] '{traj_file}' not found - vehicle trajectory prediction "
              f"is DISABLED (pedestrian EKF still runs)")


# -----------------------------------------------------------------------------
# 6) VARIÁVEIS GLOBAIS
# -----------------------------------------------------------------------------
color_map = {}
point_history = {}
missing_track_counter = {}

# Pixel thresholds, expressed at the 1080p reference resolution they were
# tuned on. calibrate_thresholds() rescales them once the true frame width is
# known, so the same value means the same real-world distance at any
# resolution. Without this a 4K scene halves every effective threshold.
REFERENCE_WIDTH = 1920.0
MIN_DISTANCE_THRESHOLD = 20
COLLISION_THRESHOLD = 30
TRAJ_SCORE_THRESHOLD = 3000

# Multiplier for box thickness / font size / marker radii, so annotations stay
# legible on a 4K frame instead of rendering hairline-thin.
DRAW_SCALE = 1.0


def calibrate_thresholds(frame_width, args):
    """Scale the 1080p-tuned pixel thresholds to the actual frame width."""
    global MIN_DISTANCE_THRESHOLD, COLLISION_THRESHOLD, TRAJ_SCORE_THRESHOLD
    global DRAW_SCALE

    k = frame_width / REFERENCE_WIDTH
    DRAW_SCALE = max(1.0, k)
    MIN_DISTANCE_THRESHOLD = (args.min_point_distance
                              if args.min_point_distance is not None else 20 * k)
    COLLISION_THRESHOLD = (args.collision_threshold
                           if args.collision_threshold is not None else 30 * k)
    TRAJ_SCORE_THRESHOLD = (args.traj_score_threshold
                            if args.traj_score_threshold is not None else 3000 * k)
    print(f"[INFO] frame width {frame_width}px -> threshold scale x{k:.2f}: "
          f"min_point_dist={MIN_DISTANCE_THRESHOLD:.0f} "
          f"collision={COLLISION_THRESHOLD:.0f} "
          f"traj_score={TRAJ_SCORE_THRESHOLD:.0f}")

vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
pedestrian_class = 'person'
# SafeBike's counterpart on the RSU side. YOLO labels the bicycle, not the
# rider, so a cyclist usually also raises a 'person' track at the same spot;
# both are VRUs and both are predicted with the EKF below.
cyclist_classes = ['bicycle']
vru_classes = [pedestrian_class] + cyclist_classes


# -----------------------------------------------------------------------------
# 6.1) INPUT SOURCE — video file, specific monitor, or screen region
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="SafeRoadside — Real-time collision detection",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help=(
            "Input source. Options:\n"
            "  path/to/video.mp4   — use a video file\n"
            "  monitor:N           — capture monitor N (1=primary, 2=secondary, ...)\n"
            "  region:X,Y,W,H     — capture a screen region\n"
            "  (omit)              — interactive monitor picker"
        )
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="Loop video file when it ends (only for --source video)"
    )
    # --- scene / resolution -------------------------------------------------
    parser.add_argument(
        "--map", type=str, default="map.txt",
        help="Homography + origin file. Default map.txt"
    )
    parser.add_argument(
        "--trajectories", type=str, default="trajetoriasClean.txt",
        help="Cleaned reference trajectories. Default trajetoriasClean.txt"
    )
    parser.add_argument(
        "--display-width", type=int, default=1600,
        help="Downscale the preview window to this width (0 = native).\n"
             "A 4K frame does not fit on screen; this only affects display."
    )
    parser.add_argument(
        "--save-video", type=str, default=None,
        help="Write the annotated output to this .mp4"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run headless (no preview window)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Silence the per-frame console chatter (per-pedestrian GPS lines, "
             "library INFO logs). The on-screen HUD shows the same state, and "
             "gateway.jsonl still records everything."
    )
    parser.add_argument(
        "--no-hud", action="store_true",
        help="Do not draw the status panel on the video"
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Stop after N frames (0 = run to the end). Useful for a quick check."
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Tag every gateway record with this run id (eval/*.py --run-id ID)"
    )
    parser.add_argument(
        "--no-translate", action="store_true",
        help="FIELD TRIALS: use the phone's real coordinates instead of "
             "transplanting its motion onto this scene's origin. Required when "
             "the phone is physically inside the calibrated scene, otherwise the "
             "avatar is displaced by (first fix - map origin) and cannot "
             "associate with the RSU's own detection of the same person."
    )
    parser.add_argument(
        "--no-gateway", action="store_true",
        help="Skip the SafeCorners fusion gateway / SafeWalk bridge and run\n"
             "the pure computer-vision pipeline only"
    )
    # Pixel thresholds below were tuned on 1080p. They are multiplied by
    # (frame_width / 1920) at runtime so a 4K scene behaves the same in
    # METRES as a 1080p one. Pass a value to pin it instead.
    parser.add_argument(
        "--collision-threshold", type=float, default=None,
        help="Pixel distance for a predicted collision (default: 30 @1080p, scaled)"
    )
    parser.add_argument(
        "--traj-score-threshold", type=float, default=None,
        help="Trajectory-match score cutoff (default: 3000 @1080p, scaled)"
    )
    parser.add_argument(
        "--min-point-distance", type=float, default=None,
        help="Min pixel gap between stored track points (default: 20 @1080p, scaled)"
    )
    return parser.parse_args()


class VideoSource:
    """Reads frames from a video file."""
    def __init__(self, path, loop=False):
        self.path = path
        self.loop = loop
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            print(f"[ERRO] Cannot open video: {path}")
            sys.exit(1)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        # Frames successfully read, across --loop restarts: the replay clock's
        # frame index must never go backwards.
        self.frames_read = 0
        print(f"[INFO] Source: video '{path}' ({w}x{h}, {self.fps:.3f} fps)")

    def grab(self):
        ret, frame = self.cap.read()
        if not ret:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            if not ret:
                return None
        self.frames_read += 1
        return frame

    def progress(self):
        """(current frame, total frames) so the HUD can show position."""
        return (int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    def release(self):
        self.cap.release()


class ScreenSource:
    """Captures a specific monitor or screen region via mss."""
    def __init__(self, region):
        self.sct = mss()
        self.region = region

    def grab(self):
        screenshot = self.sct.grab(self.region)
        frame = np.array(screenshot, dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def release(self):
        pass


def pick_monitor_interactive():
    """Lists all monitors and lets the user pick one."""
    sct = mss()
    monitors = sct.monitors  # [0] = all monitors combined, [1..N] = individual
    print("\n============================================")
    print("  Available monitors:")
    print("============================================")
    for i, m in enumerate(monitors):
        label = "ALL (virtual)" if i == 0 else f"Monitor {i}"
        print(f"  {i}) {label}  —  {m['width']}x{m['height']}  at  ({m['left']}, {m['top']})")
    print("============================================")

    while True:
        try:
            choice = int(input(f"\nSelect monitor [1-{len(monitors)-1}] (or 0 for all): "))
            if 0 <= choice < len(monitors):
                selected = monitors[choice]
                label = "ALL (virtual)" if choice == 0 else f"Monitor {choice}"
                print(f"[INFO] Source: {label} ({selected['width']}x{selected['height']})")
                return selected
            else:
                print(f"  Invalid. Choose 0 to {len(monitors)-1}.")
        except (ValueError, EOFError):
            print(f"  Invalid input. Choose 0 to {len(monitors)-1}.")


def create_source(args):
    """Factory: builds the right source from CLI arguments."""
    source_str = args.source

    # No argument → interactive monitor picker
    if source_str is None:
        region = pick_monitor_interactive()
        return ScreenSource(region)

    # monitor:N
    if source_str.startswith("monitor:"):
        idx = int(source_str.split(":")[1])
        sct = mss()
        if idx < 0 or idx >= len(sct.monitors):
            avail = len(sct.monitors) - 1
            print(f"[ERRO] Monitor {idx} not found. Available: 0..{avail}")
            for i, m in enumerate(sct.monitors):
                print(f"  {i}) {m['width']}x{m['height']} at ({m['left']},{m['top']})")
            sys.exit(1)
        m = sct.monitors[idx]
        print(f"[INFO] Source: monitor {idx} ({m['width']}x{m['height']})")
        return ScreenSource(m)

    # region:X,Y,W,H
    if source_str.startswith("region:"):
        parts = source_str.split(":")[1].split(",")
        x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        region = {"top": y, "left": x, "width": w, "height": h}
        print(f"[INFO] Source: screen region {w}x{h} at ({x},{y})")
        return ScreenSource(region)

    # Otherwise → video file
    return VideoSource(source_str, loop=args.loop)




# -----------------------------------------------------------------------------
# 7) FUNÇÕES DE APOIO
# -----------------------------------------------------------------------------

def gps2hex(lat: float, lon: float) -> str:
    """
    Converte coordenadas GPS (latitude, longitude) para um inteiro 32-bit representado em hexadecimal.
    O input deve ser um double (float) e o output é o int32 respetivo em hexadecimal.
    
    Utiliza um fator de escala de 1e7 (10.000.000) para preservar 7 casas decimais.
    Retorna uma string formatada como "0xNN 0xNN ..." (4 bytes para cada coordenada).
    """
    scale = 10_000_000  # fator de escala para preservar casas decimais
    # Converter o float para int32 (com arredondamento)
    lat_int = int(round(lat * scale))
    lon_int = int(round(lon * scale))
    
    # Empacotar os inteiros em 4 bytes cada (big-endian)
    lat_bytes = struct.pack('>i', lat_int)
    lon_bytes = struct.pack('>i', lon_int)
    
    # Converter os bytes para uma string hexadecimal
    lat_hex = ' '.join(f'0x{b:02X}' for b in lat_bytes)
    lon_hex = ' '.join(f'0x{b:02X}' for b in lon_bytes)
    
    return f"{lat_hex} {lon_hex}"

HDR = 1.35   # section headers are this much bigger than detail lines


def _source_colour(stats, source, now_m, fallback, window_s=3.0):
    """Colour a subsystem block by the risk of the fusion it just fed.

    Neither subsystem emits a risk of its own -- the PSM has no risk field and
    the RSU publishes raw tracks -- so this shows the alarm the source is
    currently implicated in. Falls back to its liveness colour when it has not
    contributed recently.
    """
    sr = (stats.get("source_risk") or {}).get(source)
    if sr and (now_m - sr.get("t", 0.0)) < window_s:
        return _RISK_COLOR.get(sr.get("risk"), fallback), sr.get("risk")
    return fallback, None


def _fmt(v, unit=""):
    """Format an optional numeric field; PSM sentinels arrive as None."""
    return "?" if v is None else f"{v:.2f}{unit}"


_RISK_COLOR = {
    "imminent": (0, 0, 255),      # red
    "probable": (0, 140, 255),    # orange
    "low":      (0, 220, 220),    # yellow
    None:       (180, 180, 180),  # grey - nothing dispatched yet
}


def draw_hud(frame, lines, scale=1.0):
    """Draw a translucent status panel in the top-left corner.

    Each entry of `lines` is (text, colour) or (text, colour, size_mult).
    size_mult scales that line's font, so section headers can be bigger than
    their detail lines. Kept deliberately compact: the console is unusable for
    watching a run because the gateway logs several lines per frame, so this
    panel is the thing you actually watch.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    base_fs = 0.55 * scale
    pad = int(10 * scale)

    # Measure first: line heights vary now, so the panel has to be sized from
    # the actual text rather than a fixed per-line constant.
    rows = []
    total_h = pad
    max_w = 0
    for entry in lines:
        text, colour = entry[0], entry[1]
        mult = entry[2] if len(entry) > 2 else 1.0
        fs = base_fs * mult
        th = max(1, int(round(1.2 * scale * mult)))
        (tw, _), _ = cv2.getTextSize(text or " ", font, fs, th)
        lh = int(26 * scale * mult)
        rows.append((text, colour, fs, th, lh))
        total_h += lh
        max_w = max(max_w, tw)

    w = max_w + 2 * pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + w, pad + total_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (pad, pad), (pad + w, pad + total_h),
                  (90, 90, 90), max(1, int(scale)))

    y = pad
    for text, colour, fs, th, lh in rows:
        y += lh
        if text:
            cv2.putText(frame, text, (pad * 2, y - int(6 * scale)),
                        font, fs, colour, th, cv2.LINE_AA)


def get_random_color():
    """Gera uma cor aleatória em formato BGR."""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def is_point_far_enough(new_point, last_point, threshold=None):
    """Verifica se a distância entre dois pontos é maior que 'threshold'.

    threshold defaults to the *current* MIN_DISTANCE_THRESHOLD. It must be
    read at call time, not bound as a default argument, because
    calibrate_thresholds() rewrites it after the first frame is known.
    """
    if threshold is None:
        threshold = MIN_DISTANCE_THRESHOLD
    return np.linalg.norm(np.array(new_point) - np.array(last_point)) > threshold


# -----------------------------------------------------------------------------
# 8) LOOP PRINCIPAL
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    source = create_source(args)

    # FPS tracking
    fps = 0.0
    frame_count = 0
    fps_start = time.time()

    # YOLO inference size (smaller = faster). Original frame kept for display.
    INFER_WIDTH = 1920

    writer = None          # lazily created once the frame size is known
    calibrated = False     # thresholds rescaled on the first frame

    if args.quiet:
        # The gateway and its libraries log several lines per frame; at a busy
        # intersection that makes the console useless for watching a run.
        set_verbose_pedestrians(False)
        logging.getLogger("safecorners_gateway").setLevel(logging.WARNING)
        logging.getLogger("deep_sort_realtime").setLevel(logging.WARNING)
        logging.getLogger("gateway_glue").setLevel(logging.INFO)

    load_scene(args.map, args.trajectories)

    print("[INFO] Starting detection loop... (press 'q' on the window to quit)")

    # Time base for everything the gateway measures. Recorded footage replays
    # slower than real time, so its clock is the video's own: one frame is
    # 1/fps seconds of scene. Live capture runs in real time, so wall time is
    # scene time. See rsu_kinematics.ReplayClock.
    if isinstance(source, VideoSource):
        clock = ReplayClock(source.fps)
        print(f"[INFO] replay clock: scene time = frame index / {source.fps:.3f} fps")
    else:
        clock = time.monotonic
    kin = KinematicsEstimator()
    t_frame = clock()
    replay_clock = None

    if args.no_gateway:
        print("[INFO] gateway disabled (--no-gateway) - CV pipeline only")
    else:
        # Only a replay clock is injected: the gateway labels a run "replay"
        # exactly when it is given a clock, and live capture runs on its own
        # monotonic clock (passing time.monotonic here mislabelled live runs).
        replay_clock = clock if isinstance(clock, ReplayClock) else None
        if not start_gateway(run_id=args.run_id, clock=replay_clock):
            print("[ERRO] gateway failed to start; refusing to run blind "
                  "(use --no-gateway for the CV pipeline only)")
            sys.exit(1)
        start_safewalk_injector()
        # Anchor the SafeWalk avatar on THIS scene's origin. It used to be
        # pinned to the Michigan demo clip (41.9407, -85.0010), which put the
        # avatar ~7000 km away from any non-demo scene and killed fusion.
        start_safewalk_http_bridge(video_anchor_lat=lat0_deg,
                                   video_anchor_lon=lon0_deg,
                                   translate=not args.no_translate)
        if args.no_translate:
            print("[INFO] --no-translate: phone coordinates are used as they "
                  "arrive (field-trial mode, no video anchoring)")

    try:
        while True:
            t0 = time.time()

            # 1) Captura do frame
            frame = source.grab()
            if frame is None:
                print("[INFO] End of video / no more frames.")
                break
            if isinstance(clock, ReplayClock):
                clock.set_frame(source.frames_read - 1)
            t_frame = clock()   # ONE capture time for every track of this frame

            orig_h, orig_w = frame.shape[:2]

            # 1.5) Rescale the 1080p-tuned pixel thresholds to this resolution
            if not calibrated:
                calibrate_thresholds(orig_w, args)
                calibrated = True
                if args.save_video:
                    writer = cv2.VideoWriter(
                        args.save_video,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        30.0, (orig_w, orig_h))
                    print(f"[INFO] writing annotated video -> {args.save_video}")

            # 2) Resize for YOLO inference if frame is too large
            if orig_w > INFER_WIDTH:
                scale = INFER_WIDTH / orig_w
                infer_frame = cv2.resize(frame, (INFER_WIDTH, int(orig_h * scale)))
            else:
                scale = 1.0
                infer_frame = frame

            # 3) Detecção com YOLO (on smaller frame)
            # imgsz must be explicit: without it ultralytics letterboxes to its
            # 640 px default, discarding the 1920 resize above (on BIB_IST's
            # frame 0 that is 10 vehicles detected instead of 23).
            results = model.predict(infer_frame, imgsz=INFER_WIDTH, verbose=False)
            yolo_boxes = results[0].boxes

            # 4) Converter predições YOLO para DeepSort (scale back to original)
            detections = []
            for box in yolo_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                if scale != 1.0:
                    x1 = int(x1 / scale)
                    y1 = int(y1 / scale)
                    x2 = int(x2 / scale)
                    y2 = int(y2 / scale)
                detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, class_id))

            # 4) Atualização do tracker
            tracks = tracker.update_tracks(detections, frame=frame)
            current_track_ids = set()

            # Listas para armazenar predições futuras
            future_car_points = []
            future_vru_points = []   # pedestrians AND cyclists
            n_vehicles = n_vrus = 0  # this frame, for the HUD
            n_collisions = 0

            # 5) Processar cada track
            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                track_id = track.track_id
                current_track_ids.add(track_id)
                missing_track_counter[track_id] = 0

                # Bounding box + centro
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Histórico de pontos
                if track_id not in point_history:
                    point_history[track_id] = deque(maxlen=20)

                # Adiciona ponto se for distante o suficiente
                if not point_history[track_id] or is_point_far_enough(center, point_history[track_id][-1]):
                    point_history[track_id].append(center)

                obj_class = model.names[int(track.get_det_class())]

                # Atribui cor única
                color = color_map.setdefault(track_id, get_random_color())

                # Publish this track to the SafeCorners gateway. Geometry and
                # velocity live in rsu_kinematics: float64, and a scene-time
                # baseline instead of one wall-clock frame step.
                if not args.no_gateway:
                    try:
                        # The unrounded box centre: truncating to int and
                        # halving threw away up to a pixel, which is metres
                        # at the ranges that set the 4-8 s band.
                        _ll = pixel_to_latlon((ltrb[0] + ltrb[2]) / 2.0,
                                              (ltrb[1] + ltrb[3]) / 2.0,
                                              homography_mat, lat0_deg, lon0_deg)
                        if _ll is not None:
                            _spd, _hdg = kin.update(track_id, t_frame, _ll[0], _ll[1])
                            gw_publish(track_id, obj_class, _ll[0], _ll[1],
                                       speed_mps=_spd, heading_deg=_hdg, t_recv=t_frame)
                    except Exception as _e:
                        # Never let publishing break the detector -- but never
                        # let it fail silently either.
                        if not getattr(main, "_publish_warned", False):
                            print(f"[AVISO] gateway publish failed: {_e!r} "
                                  f"(further failures not printed)")
                            main._publish_warned = True

                # Desenhar bounding box e label
                _t = max(2, int(2 * DRAW_SCALE))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, _t)
                cv2.putText(
                    frame,
                    f"{obj_class} #{track_id}",
                    (x1, y1 - int(10 * DRAW_SCALE)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * DRAW_SCALE,
                    color,
                    _t
                )

                # 6) Gera predições futuras
                if obj_class in vehicle_classes:
                    n_vehicles += 1
                    past_points = list(point_history[track_id])[-20:]
                    best_traj = find_best_trajectory(
                        past_points, predefined_trajectories, max_points=50,
                        score_threshold=TRAJ_SCORE_THRESHOLD)
                    for pt in best_traj:
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), max(5, int(5 * DRAW_SCALE)), color, -1)
                    future_car_points.extend(best_traj)

                elif obj_class in vru_classes:
                    n_vrus += 1
                    past_points = list(point_history[track_id])[-10:]
                    pred_points = ekf(past_points, prediction_range=5)
                    for pt in pred_points:
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), max(5, int(5 * DRAW_SCALE)), color, -1)
                    future_vru_points.extend(pred_points)

            kin.prune(t_frame)   # forget tracks unseen for more than max_gap_s

            # Lock-step replay: let the gateway finish this frame before the
            # replay clock moves on, so its decisions do not depend on how far
            # the detector got ahead (see safecorners_gateway.main.wait_idle).
            if replay_clock is not None and not wait_gateway_idle():
                print("[ERRO] gateway did not drain; stopping rather than "
                      "producing timing-dependent results")
                break

            # 7) DETECTAR POSSÍVEIS COLISÕES (EM PIXEL) E CONVERTER P/ GPS
            for car_pt in future_car_points:
                for person_pt in future_vru_points:
                    dist = np.linalg.norm(np.array(car_pt) - np.array(person_pt))
                    if dist < COLLISION_THRESHOLD:
                        Px = int((car_pt[0] + person_pt[0]) / 2)
                        Py = int((car_pt[1] + person_pt[1]) / 2)

                        n_collisions += 1
                        cv2.circle(frame, (Px, Py), max(20, int(20 * DRAW_SCALE)), (0, 0, 255), -1)

                        pt = np.array([[Px], [Py], [1]], dtype=np.float64)
                        XYW = homography_mat @ pt
                        X, Y, W = XYW[0, 0], XYW[1, 0], XYW[2, 0]

                        if abs(W) < 1e-12:
                            print("W muito próximo de zero; resultado instável para conversão.")
                        else:
                            X /= W
                            Y /= W
                            lat_deg, lon_deg = xy_to_latlon(X, Y, lat0_deg, lon0_deg)

                            if not args.quiet:
                                # One line per conflicting point pair per frame:
                                # unreadable at a busy intersection. The HUD
                                # shows the live count, gateway.jsonl keeps all.
                                print(f"[ALERTA] Possível colisão futura em pixel=({Px},{Py}) "
                                      f"-> lat/lon=({lat_deg:.6f}, {lon_deg:.6f})")

                            # shared/data.txt is the BLE sender's outbox. It is
                            # capped at 5 pending alerts; a stale full file from
                            # a previous run silently swallows every new alert,
                            # so say so rather than failing quietly.
                            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared", "data.txt")

                            try:
                                if os.path.exists(file_path):
                                    with open(file_path, "r", encoding="utf-8") as file:
                                        lines = [l for l in file.readlines() if l.strip()]
                                else:
                                    lines = []

                                if len(lines) < 5:
                                    with open(file_path, "a", encoding="utf-8") as file:
                                        file.write(f"{gps2hex(lat_deg, lon_deg)}\n")
                                elif not getattr(main, "_outbox_warned", False):
                                    print(f"[AVISO] {file_path} already holds 5 "
                                          f"pending alerts - new alerts are NOT "
                                          f"being written. Clear it to re-enable.")
                                    main._outbox_warned = True
                            except OSError as e:
                                print(f"[AVISO] could not update {file_path}: {e}")

            # 7.5) SafeWalk avatar overlay -- draw "YOU" markers for any
            # phone that has published a PSM in the last ~2 s. Maps lat/lon
            # back through the inverse homography to a pixel coordinate.
            try:
                for sw_tid, (sw_lat, sw_lon) in get_latest_safewalk().items():
                    sX, sY = latlon_to_xy(sw_lat, sw_lon, lat0_deg, lon0_deg)
                    sxy = homography_mat_inv @ np.array([sX, sY, 1.0], dtype=np.float64)
                    if abs(sxy[2]) > 1e-9:
                        spx = int(sxy[0] / sxy[2])
                        spy = int(sxy[1] / sxy[2])
                        if 0 <= spx < frame.shape[1] and 0 <= spy < frame.shape[0]:
                            cv2.circle(frame, (spx, spy), max(18, int(18 * DRAW_SCALE)), (0, 255, 255), max(3, int(3 * DRAW_SCALE)))
                            cv2.circle(frame, (spx, spy), max(4, int(4 * DRAW_SCALE)), (0, 255, 255), -1)
                            cv2.putText(frame, f"YOU ({sw_tid})",
                                        (spx + int(22 * DRAW_SCALE), spy - int(8 * DRAW_SCALE)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * DRAW_SCALE,
                                        (0, 255, 255), max(2, int(2 * DRAW_SCALE)), cv2.LINE_AA)
            except Exception as _e:
                pass  # never let overlay errors crash the detector

            # 7.6) HUD -- the console prints several gateway lines per frame,
            # so this panel is what you actually watch a run on.
            if not args.no_hud:
                st = get_gateway_stats()
                phones = get_latest_safewalk()
                now_m = time.monotonic()

                idx, total = source.progress() if hasattr(source, "progress") else (frame_count, 0)
                pos = f"{idx}/{total}" if total else f"{frame_count}"
                cfg_name = st["config"] or "standalone"
                cfg_col = (0, 165, 255) if "demo" in cfg_name else (200, 200, 200)

                WHITE, DIM = (255, 255, 255), (165, 165, 165)
                lines = [
                    (f"SafeCorners  {pos}   {fps:.1f} fps   [{cfg_name}]", WHITE),
                    ("", DIM),
                ]

                # --- what SafeWalk is reporting -----------------------------
                sw = st.get("safewalk")
                fz = (st.get("last_fuse") or {}).get("source_confidence") or {}
                if sw and (now_m - sw["t"]) < 5.0:
                    age_s = now_m - sw["t"]
                    live = age_s < 2.0
                    scol = (0, 255, 0) if live else (0, 200, 255)
                    # Confidence as the fusion actually used it (freshness
                    # weighted) if this source fed the last decision; otherwise
                    # the raw GNSS-accuracy confidence the PSM earns on its own.
                    # Which peer app this is -- SafeWalk (pedestrian) or
                    # SafeBike (cyclist). Driven by the PSM's basicType, not
                    # assumed, so a cyclist shows up as SafeBike.
                    sw_src = sw.get("source") or "SafeWalk"
                    swc = fz.get(sw_src)
                    own = sw.get("confidence")
                    sw_col, sw_risk = _source_colour(st, sw_src, now_m, scol)
                    lines += [
                        (f"{sw_src.upper()}  {sw['track_id']}"
                         + f"  ({sw.get('basic_type', 'pedestrian')})"
                         + (f"   conf {swc:.2f}" if swc is not None
                            else (f"   conf {own:.2f}" if own is not None else ""))
                         + (f"   [{sw_risk.upper()}]" if sw_risk else ""),
                         sw_col, HDR),
                        (f"  says : {sw['lat']:.6f}, {sw['lon']:.6f}", scol),
                        (f"         spd {_fmt(sw['speed_mps'], 'm/s')}"
                         f"  hdg {_fmt(sw['heading_deg'], 'deg')}"
                         f"   {age_s:.1f}s ago", scol),
                        (f"  GNSS acc {_fmt(sw.get('accuracy_m'), 'm')}"
                         + (f"  -> own conf {own:.2f}" if own is not None else ""), scol),
                        ("  (PSM carries no risk -- position only)", DIM),
                    ]
                else:
                    lines += [("SAFEWALK / SAFEBIKE  absent -- RSU only", DIM, HDR)]
                lines.append(("", DIM))

                # --- what the RSU is reporting ------------------------------
                rsuc = fz.get("RSU")
                rsu_col, rsu_risk = _source_colour(st, "RSU", now_m, WHITE)
                lines += [
                    ("RSU (camera)"
                     + (f"   conf {rsuc:.2f}" if rsuc is not None else "")
                     + (f"   [{rsu_risk.upper()}]" if rsu_risk else ""),
                     rsu_col, HDR),
                    (f"  sees : {n_vehicles} vehicles  {n_vrus} VRUs"
                     f"  conflicts {n_collisions}", WHITE),
                    (f"  gw   : {st['assoc_matched']} assoc  {st['fused']} fused"
                     f"  ({st['fused_coop']} coop)", WHITE),
                    ("", DIM),
                ]

                # --- the fused verdict --------------------------------------
                lf = st.get("last_fuse")
                risk = st["last_risk"]
                wage = now_m - st["last_warn_t"] if st["last_warn_t"] else None
                if lf and risk and wage is not None and wage < 3.0:
                    rcol = _RISK_COLOR.get(risk, WHITE)
                    ttc = lf.get("ttc")
                    conf = lf.get("confidence")
                    lines += [
                        (f"GATEWAY SAYS : {risk.upper()}"
                         + (f"   conf {conf:.2f}" if conf is not None else ""),
                         rcol, HDR),
                        (f"  TTC {ttc:.1f}s" if isinstance(ttc, (int, float))
                         else "  TTC --", rcol),
                        (f"  from {' + '.join(lf.get('sources') or ['?'])}"
                         f"   ({wage:.1f}s ago)", rcol),
                    ]
                else:
                    lines += [("GATEWAY SAYS : no active warning",
                               _RISK_COLOR[None], HDR)]
                lines.append((f"  totals: {st['dispatched']} sent   "
                              f"L{st['risk']['low']} P{st['risk']['probable']} "
                              f"I{st['risk']['imminent']}", DIM))

                draw_hud(frame, lines, scale=DRAW_SCALE)

            # 8) Exibição
            if writer is not None:
                writer.write(frame)

            if not args.no_display:
                disp = frame
                if args.display_width and orig_w > args.display_width:
                    s = args.display_width / orig_w
                    disp = cv2.resize(frame, (args.display_width, int(orig_h * s)))
                cv2.imshow("Tracking Inteligente", disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                print(f"[INFO] reached --max-frames {args.max_frames}")
                break
            if frame_count % 50 == 0:
                elapsed = time.time() - fps_start
                fps = frame_count / elapsed if elapsed > 0 else 0.0
                print(f"[INFO] {frame_count} frames processed "
                      f"({fps:.1f} fps average)", flush=True)

    finally:
        source.release()
        if writer is not None:
            writer.release()
            print(f"[OK] annotated video saved -> {args.save_video}")
        cv2.destroyAllWindows()


# -----------------------------------------------------------------------------
# PONTO DE ENTRADA
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()

    # The gateway owns two daemon threads that block forever by design: an
    # asyncio loop in run_forever() and the SafeWalk bridge in serve_forever().
    # Neither unwinds at interpreter shutdown, and killing them mid-teardown
    # made the process report exit 255 even on a fully successful run -- which
    # breaks any script or CI step that checks the exit code.
    #
    # Nothing is lost by exiting hard here: main() has already released the
    # video writer, and the gateway's JSONL log is opened line-buffered
    # (buffering=1), so every record is on disk as it is written. Flush our own
    # streams first, then go.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
