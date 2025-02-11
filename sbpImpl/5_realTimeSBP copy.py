import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque
import random
import pyautogui
import logging
import sys
import re

# ======================================================================
# Setup Logging
# ======================================================================
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# ======================================================================
# Leitura das Trajetórias de Referência
# ======================================================================
def load_reference_trajectories(file_path="trajetoriasClean.txt"):
    """
    Lê um arquivo contendo trajetórias de referência no formato:
       Traj: (100,200), (110,210), (120,220)
    Retorna uma lista de trajetórias, cada qual sendo uma lista de (x, y).
    """
    trajectories = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Traj:"):
                    coords_str = line.split("Traj:")[1].strip()
                    matches = re.findall(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", coords_str)
                    if matches:
                        traj = [(int(x), int(y)) for x, y in matches]
                        trajectories.append(traj)
    except FileNotFoundError:
        print(f"Could not find file: {file_path}. Retornando lista vazia.")
    return trajectories

# ======================================================================
# Função para encontrar a trajetória completa mais semelhante
# ======================================================================
def predizer_trajetoria_mais_semelhante(current_points, reference_trajectories):
    """
    Dado os pontos recentes de uma trajetória (current_points), encontra a trajetória
    de referência mais parecida e retorna toda a sua sequência de pontos a partir do
    momento identificado como mais próximo.
    """
    if not current_points or not reference_trajectories:
        return []

    best_score = float("inf")
    best_full_trajectory = []
    N = len(current_points)

    for ref_traj in reference_trajectories:
        if len(ref_traj) < N:
            continue  # Trajetória muito curta

        # Deslizar uma janela pela trajetória de referência
        for i in range(len(ref_traj) - N):
            candidate_segment = ref_traj[i:i+N]

            # Calcular a distância entre os segmentos
            dist_sum = sum(
                np.hypot(x1 - x2, y1 - y2)
                for (x1, y1), (x2, y2) in zip(current_points, candidate_segment)
            )

            # Se esta trajetória for mais semelhante, armazenamos
            if dist_sum < best_score:
                best_score = dist_sum
                best_full_trajectory = ref_traj[i:]  # Pegamos a trajetória a partir desse ponto

    return best_full_trajectory

# ======================================================================
# Função Principal: Captura de Tela + YOLO + DeepSORT + Previsão
# ======================================================================
def main():
    # 1) Carrega o modelo YOLO e o DeepSORT
    model = YOLO("models/yolo11n.pt")
    tracker = DeepSort(max_age=45, n_init=4, nn_budget=100)

    # 2) Carrega as trajetórias de referência
    reference_trajectories = load_reference_trajectories("trajetoriasClean.txt")
    print(f"DEBUG: Loaded {len(reference_trajectories)} reference trajectories")

    # 3) Define a região de captura de tela
    screen_width, screen_height = pyautogui.size()
    screen_region = {"top": 0, "left": 0, "width": screen_width, "height": screen_height}

    # 4) Estruturas de dados de tracking
    color_map = {}
    point_history = {}
    frame_num = 0

    def get_random_color():
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    with mss() as sct:
        while True:
            # Captura de tela
            screenshot = sct.grab(screen_region)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # YOLO Inference
            results = model.predict(frame)
            detections = []

            for r in results[0].boxes:
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                confidence = float(r.conf[0])
                class_id = int(r.cls[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, class_id))

            # Atualiza o tracker
            tracks = tracker.update_tracks(detections, frame=frame)
            current_track_ids = set()

            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                track_id = track.track_id
                current_track_ids.add(track_id)

                # Pega bounding box do track
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                track_centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Cor do track
                if track_id not in color_map:
                    color_map[track_id] = get_random_color()

                # Histórico de pontos do track
                if track_id not in point_history:
                    point_history[track_id] = deque()

                # Adiciona novo ponto à trajetória do track
                if frame_num % 10 == 0:
                    point_history[track_id].append(track_centroid)

                # Se temos pontos suficientes, fazemos a previsão
                if len(point_history[track_id]) >= 3:
                    recent_points = list(point_history[track_id])
                    best_trajectory = predizer_trajetoria_mais_semelhante(
                        current_points=recent_points,
                        reference_trajectories=reference_trajectories
                    )

                    if best_trajectory:
                        # **Ajuste de Offset** -> Faz a melhor trajetória casar com o ponto real
                        offset_x = track_centroid[0] - best_trajectory[0][0]
                        offset_y = track_centroid[1] - best_trajectory[0][1]

                        best_trajectory_shifted = [
                            (px + offset_x, py + offset_y) for (px, py) in best_trajectory
                        ]

                        # Desenha a trajetória prevista
                        for i in range(len(best_trajectory_shifted) - 1):
                            p1 = best_trajectory_shifted[i]
                            p2 = best_trajectory_shifted[i + 1]
                            cv2.line(frame, p1, p2, color_map[track_id], 2)

            frame_num += 1

            # Mostra
            cv2.imshow("Real-time Tracking + Trajectory Prediction", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
