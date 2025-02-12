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

# Configuração de logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Carregar trajetórias predefinidas
def load_trajectories(file_path):
    trajectories = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("Traj: "):
                points_str = line[len("Traj: "):].strip().split('), (')
                points = []
                for p in points_str:
                    x, y = p.replace('(', '').replace(')', '').split(', ')
                    points.append((int(x), int(y)))
                trajectories.append(points)
    return trajectories

predefined_trajectories = load_trajectories("trajetoriasClean.txt")

# Função para encontrar a melhor trajetória correspondente
def find_best_trajectory(current_traj, predefined_trajs, max_points=50):
    if len(current_traj) < 1:
        return []
    
    best_score = float('inf')
    best_match = []
    
    for traj in predefined_trajs:
        if len(traj) <= len(current_traj):
            continue
            
        for i in range(len(traj) - len(current_traj)):
            reference_window = traj[i:i+len(current_traj)]
            score = 0
            
            for j in range(len(current_traj)):
                score += np.linalg.norm(np.array(current_traj[j]) - np.array(reference_window[j]))
                
            if score < best_score:
                best_score = score
                best_match = traj[i+len(current_traj):i+len(current_traj)+max_points]
    
    return best_match if best_score < 3000 else []

# Função EKF (mantida para peões)
def ekf(previous_points, prediction_range):
    dt = 1.0
    Q = np.eye(6) * 0.1
    R = np.eye(2) * 0.5
    x = np.array([previous_points[0][0], previous_points[0][1], 0, 0, 0, 0])
    P = np.eye(6)
    F_jacobian = np.array([
        [1, 0, dt, 0, 0.5 * dt**2, 0],
        [0, 1, 0, dt, 0, 0.5 * dt**2],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])

    for z in previous_points:
        x_pred = np.array([
            x[0] + x[2] * dt + 0.5 * x[4] * dt**2,
            x[1] + x[3] * dt + 0.5 * x[5] * dt**2,
            x[2] + x[4] * dt,
            x[3] + x[5] * dt,
            x[4],
            x[5]
        ])
        P = F_jacobian @ P @ F_jacobian.T + Q
        y = z - H @ x_pred
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x_pred + K @ y
        P = (np.eye(len(K)) - K @ H) @ P

    predictions = []
    for _ in range(prediction_range):
        x_pred = np.array([
            x[0] + x[2] * dt + 0.5 * x[4] * dt**2,
            x[1] + x[3] * dt + 0.5 * x[5] * dt**2,
            x[2] + x[4] * dt,
            x[3] + x[5] * dt,
            x[4],
            x[5]
        ])
        predictions.append(x_pred[:2])
        x = x_pred

    return predictions

# Configurações iniciais
model = YOLO("models/yolo11n.pt")
tracker = DeepSort(max_age=1000, n_init=1, nn_budget=50)
screen_width, screen_height = pyautogui.size()
screen_region = {"top": 0, "left": 0, "width": screen_width, "height": screen_height}

color_map = {}
point_history = {}
missing_track_counter = {}
min_distance_threshold = 20
vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']  # Ajustar conforme as classes do modelo

def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def is_point_far_enough(new_point, last_point):
    return np.linalg.norm(np.array(new_point) - np.array(last_point)) > min_distance_threshold

def centroid(box):
    x, y, w, h = box
    return (x + w // 2, y + h // 2)

# Loop principal
with mss() as sct:
    frame_num = 0
    while True:
        screenshot = sct.grab(screen_region)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        results = model.predict(frame)
        detections = []
        for r in results[0].boxes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            confidence = float(r.conf[0])
            class_id = int(r.cls[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, class_id))

        tracks = tracker.update_tracks(detections, frame=frame)
        current_track_ids = set()
        detection_centroids = {(x, y, w, h): (centroid((x, y, w, h)), class_id) for (x, y, w, h), _, class_id in detections}

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            track_id = track.track_id
            current_track_ids.add(track_id)
            missing_track_counter[track_id] = 0
            
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            # Determinar classe do objeto
            track_centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            min_distance = float('inf')
            best_class_id = -1
            for det_bbox, (det_centroid, class_id) in detection_centroids.items():
                distance = np.linalg.norm(np.array(track_centroid) - np.array(det_centroid))
                if distance < min_distance:
                    min_distance = distance
                    best_class_id = class_id
            
            obj = "unknown"
            if best_class_id != -1 and min_distance < 50:
                obj = model.names[best_class_id]

            # Atualizar histórico de pontos
            center = (int(track_centroid[0]), int(track_centroid[1]))
            if track_id not in point_history:
                point_history[track_id] = deque(maxlen=30)
            
            if len(point_history[track_id]) == 0 or is_point_far_enough(center, point_history[track_id][-1]):
                point_history[track_id].append(center)

            # Gerar previsões
            predictions = []
            if obj.lower() in vehicle_classes:
                # last 10 items
                current_traj = list(point_history[track_id])[-10:]
                predictions = find_best_trajectory(current_traj, predefined_trajectories)

            # Desenhar resultados
            color = color_map.setdefault(track_id, get_random_color())
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{obj} #{track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Desenhar trajetória e previsões
            #for i in range(1, len(point_history[track_id])):
                #cv2.line(frame, point_history[track_id][i-1], point_history[track_id][i], color, 2)
            
            for pred in predictions:
                if len(pred) >= 2:
                    cv2.circle(frame, (int(pred[0]), int(pred[1])), 5, color, -1)

        # Limpar tracks antigos
        for track_id in list(point_history.keys()):
            if track_id not in current_track_ids:
                missing_track_counter[track_id] = missing_track_counter.get(track_id, 0) + 1
                if missing_track_counter[track_id] > 30:
                    del point_history[track_id]
                    del missing_track_counter[track_id]
                    if track_id in color_map:
                        del color_map[track_id]

        cv2.imshow("Tracking Inteligente", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()