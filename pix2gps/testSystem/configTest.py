#!/usr/bin/env python3
# configTest.py

import cv2
import numpy as np

# Lista de 20 coordenadas lat/lon (exemplo). Ajusta para as tuas.
coords_str = [
    (38.877292, -7.168618),
    (38.877279, -7.168395),
    (38.877147, -7.168243),
    (38.876991, -7.168212),
    (38.876858, -7.168313),
    (38.876822, -7.168479),
    (38.876884, -7.168702),
    (38.876990, -7.168804),
    (38.877129, -7.168841),
    (38.877244, -7.168729),
    (38.877183, -7.168597),
    (38.877141, -7.168419),
    (38.876986, -7.168402),
    (38.876958, -7.168563),
    (38.877080, -7.168655),
    (38.877110, -7.168545),
    (38.877033, -7.168494),
    (38.876988, -7.168690),
    (38.876833, -7.168594),
    (38.877073, -7.168301)
]

clicked_points = []
idx = 0

def mouse_callback(event, x, y, flags, param):
    global clicked_points, idx
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"({coords_str[idx]}) -> Pixel=({x}, {y})")
        idx += 1

def main():
    global idx

    image_path = r"C:\Users\35196\OneDrive\Ambiente de Trabalho\tese\Tese_code\SafeRoadside\pix2gps\testSystem\tes.png"  # Ajusta aqui
    img = cv2.imread(image_path)
    if img is None:
        print("Erro ao carregar imagem.")
        return

    # Prepara janela
    cv2.namedWindow("Clique nos 20 pontos pela ordem da lista coords_str")
    cv2.setMouseCallback("Clique nos 20 pontos pela ordem da lista coords_str", mouse_callback)

    print("===== INSTRUÇÕES =====")
    print("1) Terás 20 coordenadas GPS predefinidas no array coords_str.")
    print("2) Precisarás de clicar na imagem 20 vezes, respeitando a mesma ORDEM.")
    print("3) A cada clique, será registado (lat/lon, px, py).")
    print("Pressiona ESC para sair a qualquer momento.")
    print("======================")

    while True:
        disp = img.copy()
        cv2.imshow("Clique nos 20 pontos pela ordem da lista coords_str", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        # Se já clicaste em todos os pontos, acaba
        if idx >= len(coords_str):
            print("Todos os 20 pontos foram clicados!")
            break

    cv2.destroyAllWindows()

    # Se nem todos foram clicados, confirma
    if len(clicked_points) < len(coords_str):
        print(f"Foram clicados apenas {len(clicked_points)} pontos. A sair.")
        return

    # Guardar no ficheiro gpspix.txt
    # Formato: "latlon_string" px py
    # ou podes guardar JSON, etc. Aqui fica simples em texto.
    with open("gpspix.txt", "w") as f:
        for i, (px, py) in enumerate(clicked_points):
            f.write(f"\"{coords_str[i]}\",{px},{py}\n")

    print("Ficheiro gpspix.txt guardado com sucesso.")

if __name__ == "__main__":
    main()
