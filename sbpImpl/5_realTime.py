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
# 1) CARREGAR TRAJETÓRIAS PREDEFINIDAS
# - Otimização: uso de list comprehension e conversão direta dos pontos para arrays, 
#   evitando parsing desnecessário em cada iteração.
# -----------------------------------------------------------------------------
def load_trajectories(file_path):
    """Carrega e retorna uma lista de trajetórias pré-definidas."""
    trajectories = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("Traj: "):
            # Remove o prefixo "Traj: " e espaços
            points_str = line[len("Traj: "):].split('), (')
            # Converte cada string em um tuple (x, y)
            points = []
            for p in points_str:
                p = p.replace('(', '').replace(')', '')
                x, y = map(int, p.split(', '))
                points.append((x, y))
            trajectories.append(np.array(points, dtype=np.int32))

    return trajectories


# -----------------------------------------------------------------------------
# 2) FUNÇÃO PARA ENCONTRAR A MELHOR TRAJETÓRIA
# - Otimizações:
#   - Converter as listas para arrays numpy uma única vez, para facilitar
#     operações vetorizadas.
#   - Usar métodos do NumPy para cálculo de normas/diferenças e reduzir loop.
#   - Retornar rapidamente se não há pontos suficientes.
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
            reference_window = traj[i:i + current_length]
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
# 3) FILTRO DE KALMAN ESTENDIDO (EKF) PARA PEDIDO DE TRAJETÓRIAS DE PEDESTRES
# - Otimizações:
#   - Verificações de tamanho para evitar loop desnecessário.
#   - Evitar conversões repetidas.
# -----------------------------------------------------------------------------
def ekf(previous_points, prediction_range=5):
    """
    Retorna uma lista de pontos previstos a partir dos 'previous_points'
    usando um EKF simplificado.
    """
    if len(previous_points) < 3:
        return []

    # Intervalo de tempo
    dt = 1.0
    # Matriz de covariância do processo
    Q = np.eye(6) * 0.1
    # Matriz de covariância da medida
    R = np.eye(2) * 0.5

    # Estado inicial (x, y, vx, vy, ax, ay)
    x = np.array([previous_points[0][0], previous_points[0][1], 0, 0, 0, 0], dtype=np.float32)
    P = np.eye(6, dtype=np.float32)

    # Matriz de transição de estado (Jacobiano)
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

    # Alimente o EKF com os pontos anteriores
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

    # Gera as previsões futuras
    predictions = []
    for _ in range(prediction_range):
        x_pred = F_jacobian @ x
        predictions.append(x_pred[:2].astype(np.int32))
        x = x_pred

    return predictions


# -----------------------------------------------------------------------------
# 4) CONFIGURAÇÃO DE YOLO E DEEPSORT
# - Otimização: garantia de que o modelo seja carregado apenas uma vez.
# -----------------------------------------------------------------------------
model = YOLO("models/yolo11n.pt").to('cuda')
tracker = DeepSort(
    max_age=1000,
    n_init=5,
    nn_budget=10,
    embedder_gpu=True
)

# Recupera tamanho da tela para captura
screen_width, screen_height = pyautogui.size()
screen_region = {"top": 0, "left": 0, "width": screen_width, "height": screen_height}

# Carregar as trajetórias pré-definidas em memória (apenas uma vez)
predefined_trajectories = load_trajectories("trajetoriasClean.txt")


# -----------------------------------------------------------------------------
# 5) VARIÁVEIS GLOBAIS/AUXILIARES
# - Otimizações:
#   - color_map e point_history continuam como dicionários, mas se considerou 
#     remover missing_track_counter caso não seja utilizado para nada relevante.
# -----------------------------------------------------------------------------
color_map = {}
point_history = {}
missing_track_counter = {}

# Ajustes de thresholds
MIN_DISTANCE_THRESHOLD = 20
COLLISION_THRESHOLD = 30

# Classes específicas
vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
pedestrian_class = 'person'


# -----------------------------------------------------------------------------
# 6) FUNÇÕES DE APOIO
# -----------------------------------------------------------------------------
def get_random_color():
    """Gera uma cor aleatória em formato BGR."""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def is_point_far_enough(new_point, last_point, threshold=MIN_DISTANCE_THRESHOLD):
    """
    Verifica se a distância entre dois pontos é maior que um 'threshold'
    para evitar gravação de pontos muito próximos.
    """
    return np.linalg.norm(np.array(new_point) - np.array(last_point)) > threshold


# -----------------------------------------------------------------------------
# 7) LOOP PRINCIPAL
# - Otimizações:
#   - Usar conversões locais antes do loop (ex. frame para RGB/BGR, etc.).
#   - Evitar conversões repetidas no loop (ex. logs de debug somente quando necessário).
#   - Processar apenas resultados[0].boxes ao invés de results completos.
# -----------------------------------------------------------------------------
def main():
    with mss() as sct:
        while True:
            # Captura de tela
            screenshot = sct.grab(screen_region)
            frame = np.array(screenshot, dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Detecção com YOLO
            results = model.predict(frame, verbose=False)  # verbose=False para suprimir logs
            yolo_boxes = results[0].boxes

            # Convertendo predições do YOLO para o formato esperado pelo DeepSort
            detections = []
            for box in yolo_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, class_id))

            # Atualização do tracker
            tracks = tracker.update_tracks(detections, frame=frame)
            current_track_ids = set()

            # Listas para armazenar predições de carros e pessoas
            car_predictions = []
            person_predictions = []

            for track in tracks:
                # Desconsidera track não confirmada ou atualizada
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                track_id = track.track_id
                current_track_ids.add(track_id)
                missing_track_counter[track_id] = 0

                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Cria histórico do track se não existir
                if track_id not in point_history:
                    point_history[track_id] = deque(maxlen=10)

                # Só adiciona ponto se for distante o suficiente do último
                if not point_history[track_id] or is_point_far_enough(center, point_history[track_id][-1]):
                    point_history[track_id].append(center)

                # Objeto detectado
                obj_class = model.names[int(track.get_det_class())]

                # Atribui cor única ao track
                color = color_map.setdefault(track_id, get_random_color())

                # Desenha bounding box e label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{obj_class} #{track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

                # Gera predições
                if obj_class in vehicle_classes:
                    # Usa método de matching com trajetórias pré-definidas
                    best_traj = find_best_trajectory(list(point_history[track_id])[-20:], predefined_trajectories)
                    car_predictions.extend(best_traj)
                    # Desenha predições
                    for pt in best_traj:
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, color, -1)

                elif obj_class == pedestrian_class:
                    # Usa EKF para gerar previsões
                    pred_points = ekf(list(point_history[track_id])[-10:], 5)
                    person_predictions.extend(pred_points)
                    # Desenha predições
                    for pt in pred_points:
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, color, -1)

            # DETECÇÃO DE POSSÍVEIS COLISÕES
            # Otimização: evitar loops aninhados muito grandes. No geral, 
            # como as predições não costumam ser muitas, é aceitável.
            for car_point in car_predictions:
                for person_point in person_predictions:
                    dist = np.linalg.norm(np.array(car_point) - np.array(person_point))
                    if dist < COLLISION_THRESHOLD:
                        collision_x = int((car_point[0] + person_point[0]) / 2)
                        collision_y = int((car_point[1] + person_point[1]) / 2)
                        cv2.circle(frame, (collision_x, collision_y), 20, (0, 0, 255), -1)

            # Exibição
            cv2.imshow("Tracking Inteligente", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


# -----------------------------------------------------------------------------
# PONTO DE ENTRADA
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
