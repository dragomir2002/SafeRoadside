import os
import glob
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def process_videos(folder_path, model_path="models/yolo11n.pt", output_file="sbp.txt"):
    # Inicia modelo YOLO
    model = YOLO(model_path)
    
    # Inicia rastreador DeepSORT
    tracker = DeepSort(max_age=45, n_init=4, nn_budget=100)

    # Procurar todos os vídeos em folder_path
    exts = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    video_files = []
    for ext in exts:
        video_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not video_files:
        print(f"Não foram encontrados vídeos em {folder_path}")
        return

    # Dicionário para guardar centros:
    # centers[video_name][track_id] = lista de (x_center, y_center)
    from collections import defaultdict
    centers = {}

    for video_path in video_files:
        video_name = os.path.basename(video_path)
        centers[video_name] = defaultdict(list)

        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detecção com YOLO (ajuste conf conforme necessário)
            results = model.predict(frame, conf=0.3, device="cpu")

            # Prepara detections para DeepSORT
            # Formato: ([x, y, w, h], conf, class_id)
            detections = []
            if len(results) > 0 and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))

            tracks = tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                # Ignora tracks não confirmados ou perdidos
                if not track.is_confirmed() or track.time_since_update > 0:
                    continue

                track_id = track.track_id
                # Retorna [left, top, right, bottom]
                l, t, r, b = track.to_ltrb()
                cx = int((l + r) / 2)
                cy = int((t + b) / 2)

                # Identifica a classe do track a partir da detecção mais próxima
                # (caso queira só "car", verifique se model.names[cls_id] == "car")
                closest_cls_id = None
                closest_dist = float("inf")
                for (bbox, confidence, c_id) in detections:
                    x_det, y_det, w_det, h_det = bbox
                    det_cx = x_det + w_det/2
                    det_cy = y_det + h_det/2
                    dist = ((cx - det_cx)**2 + (cy - det_cy)**2)**0.5
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_cls_id = c_id

                # Verifica se a classe é "car"
                if closest_cls_id is not None:
                    class_name = model.names[closest_cls_id].lower()
                    if class_name == "car":
                        # Guarda o centro
                        centers[video_name][track_id].append((cx, cy))

        cap.release()

    # Escreve os resultados em sbp.txt no formato:
    # <video>-<track_id>: (p1x, p1y), (p2x, p2y), ...
    with open(output_file, "w") as f:
        for video_name, tracks_dict in centers.items():
            for t_id, points in tracks_dict.items():
                # Monta string com todos os (x, y)
                points_str = ", ".join([f"({px},{py})" for (px, py) in points])
                f.write(f"{video_name}-{t_id}: {points_str}\n")

    print(f"Processamento concluído! Arquivo salvo em '{output_file}'.")

if __name__ == "__main__":
    folder = "vids"
    process_videos(folder)
