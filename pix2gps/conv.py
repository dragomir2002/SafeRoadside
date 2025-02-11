#!/usr/bin/env python3
# conv.py

import cv2
import numpy as np
import math

clicked_point = None

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"Pixel clicado = ({x}, {y})")

def xy_to_latlon(X, Y, lat0_deg, lon0_deg):
    """
    Inverso da projeção local.
    Dado (X, Y) em metros no sistema local,
    devolve (lat, lon) em graus decimais.
    """
    R = 6371000.0
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)

    # Fazemos as contas inversas
    lat = Y / R + lat0
    lon = X / (R * math.cos(lat0)) + lon0

    # Convert para graus
    lat_deg = math.degrees(lat)
    lon_deg = math.degrees(lon)
    return lat_deg, lon_deg

def main():
    global clicked_point

    image_path = r"C:\Users\35196\OneDrive\Ambiente de Trabalho\tese\Tese_code\SafeRoadside\pix2gps\tes.png"
    img = cv2.imread(image_path)
    if img is None:
        print("Erro ao carregar imagem.")
        return

    # Ler a info de map.txt
    try:
        with open("map.txt", "r") as f:
            lines = f.readlines()
        # Exemplo:
        # linha 1: lat0_deg, lon0_deg
        # linha 2: a11, a12, a13
        # linha 3: a21, a22, a23
        lat0_deg, lon0_deg = map(float, lines[0].split())
        a11, a12, a13 = map(float, lines[1].split())
        a21, a22, a23 = map(float, lines[2].split())
        affine_mat = np.array([[a11, a12, a13],
                               [a21, a22, a23]], dtype=np.float32)
    except Exception as e:
        print("Erro a ler map.txt:", e)
        return

    if affine_mat.shape != (2, 3):
        print("Matriz afim em map.txt não tem dimensões 2x3.")
        return

    # Preparar janela
    cv2.namedWindow("Clica no ponto para ver lat/lon")
    cv2.setMouseCallback("Clica no ponto para ver lat/lon", mouse_callback)

    print("Instruções: clica em qualquer ponto para converter de pixel -> lat/lon.\n"
          "ESC para sair.")

    while True:
        disp = img.copy()
        cv2.imshow("Clica no ponto para ver lat/lon", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

        if clicked_point is not None:
            px, py = clicked_point

            # Transformar pixel -> (X, Y) local
            # [X, Y]^T = A * [px, py, 1]^T
            X = affine_mat[0, 0]*px + affine_mat[0, 1]*py + affine_mat[0, 2]
            Y = affine_mat[1, 0]*px + affine_mat[1, 1]*py + affine_mat[1, 2]

            # Converter (X, Y) local para (lat, lon)
            lat_deg, lon_deg = xy_to_latlon(X, Y, lat0_deg, lon0_deg)

            print(f"\nPixel=({px}, {py})\n"
                  f"-> local (X, Y) = ({X:.3f}, {Y:.3f}) pix dist\n"
                  f"-> lat/lon = ({lat_deg:.6f}°, {lon_deg:.6f}°)\n")

            # Reset para poder clicar noutro ponto
            clicked_point = None

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
