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

# Carregar trajetórias predefinidas para veículos
def load_trajectories(file_path):
    trajectories = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("Traj: "):
                points_str = line[len("Traj: "):].strip().split('), (')
                points = [(int(p.split(', ')[0].replace('(', '')), int(p.split(', ')[1].replace(')', ''))) for p in points_str]
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
            score = sum(np.linalg.norm(np.array(current_traj[j]) - np.array(reference_window[j])) for j in range(len(current_traj)))

            if score < best_score:
                best_score = score
                best_match = traj[i+len(current_traj):i+len(current_traj)+max_points]
    
    return best_match if best_score < 3000 else []

# EKF para prever a trajetória dos pedestres
def ekf(previous_points, prediction_range):
    
    if (len(previous_points) < 3): 
        return []
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
        x_pred = np.dot(F_jacobian, x)
        P = np.dot(np.dot(F_jacobian, P), F_jacobian.T) + Q
        y = z - np.dot(H, x_pred)
        S = np.dot(H, np.dot(P, H.T)) + R
        K = np.dot(P, np.dot(H.T, np.linalg.inv(S)))
        x = x_pred + np.dot(K, y)
        P = (np.eye(len(K)) - np.dot(K, H)) @ P

    predictions = []
    for _ in range(prediction_range):
        x_pred = np.dot(F_jacobian, x)
        predictions.append(x_pred[:2])
        x = x_pred

    return predictions

# Configuração de YOLO e DeepSORT
model = YOLO("models/yolo11n.pt")
tracker = DeepSort(max_age=1000, n_init=1, nn_budget=50)

screen_width, screen_height = pyautogui.size()
screen_region = {"top": 0, "left": 0, "width": screen_width, "height": screen_height}

color_map = {}
point_history = {}
missing_track_counter = {}
min_distance_threshold = 20
vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
pedestrian_class = 'person'
collision_threshold = 30

def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def is_point_far_enough(new_point, last_point):
    return np.linalg.norm(np.array(new_point) - np.array(last_point)) > min_distance_threshold

with mss() as sct:
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
        car_predictions = []
        person_predictions = []

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            track_id = track.track_id
            current_track_ids.add(track_id)
            missing_track_counter[track_id] = 0

            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            if track_id not in point_history:
                point_history[track_id] = deque(maxlen=10)

            if len(point_history[track_id]) == 0 or is_point_far_enough(center, point_history[track_id][-1]):
                point_history[track_id].append(center)

            obj = model.names[int(track.get_det_class())]
            color = color_map.setdefault(track_id, get_random_color())

            # Desenhar bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{obj} #{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            predictions = []
            if obj in vehicle_classes:
                predictions = find_best_trajectory(list(point_history[track_id])[-10:], predefined_trajectories)
                car_predictions.extend(predictions)
            elif obj == pedestrian_class:
                predictions = ekf(list(point_history[track_id])[-10:], 5)
                person_predictions.extend(predictions)

            for pred in predictions:
                cv2.circle(frame, (int(pred[0]), int(pred[1])), 5, color, -1)

        #  DETECT COLLISION 
        for car_point in car_predictions:
            for person_point in person_predictions:
                if np.linalg.norm(np.array(car_point) - np.array(person_point)) < collision_threshold:
                    cv2.circle(frame, (int((car_point[0] + person_point[0]) / 2), 
                                       int((car_point[1] + person_point[1]) / 2)), 
                                       20, (0, 0, 255), -1)

        cv2.imshow("Tracking Inteligente", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
