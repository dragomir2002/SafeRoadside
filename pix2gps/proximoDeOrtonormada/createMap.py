#!/usr/bin/env python3
# createMap.py

import cv2
import numpy as np
import math

# --- 1. Lista de 13 coordenadas lat/long em formato string (exemplo dado) ---
coords_str = [
    "38.570285°N 7.914937°W",
    "38.570302°N 7.914937°W",
    "38.570320°N 7.914937°W",
    "38.570338°N 7.914937°W",
    "38.570283°N 7.914914°W",
    "38.570283°N 7.914890°W",
    "38.570283°N 7.914867°W",
    "38.570266°N 7.914937°W",
    "38.570248°N 7.914937°W",
    "38.570229°N 7.914937°W",
    "38.570284°N 7.914960°W",
    "38.570283°N 7.914983°W",
    "38.570284°N 7.915006°W"
]

coords_str = [
    "38.877713°N 7.172653°W",
    "38.877889°N 7.173129°W",
    "38.877577°N 7.173119°W",
    "38.877489°N 7.173021°W",
    "38.877611°N 7.172225°W",
    "38.877772°N 7.172280°W",
    "38.877906°N 7.172505°W",
    "38.877818°N 7.172889°W",
    "38.877573°N 7.172813°W",
    "38.877597°N 7.172440°W",
    "38.877797°N 7.172490°W",
    "38.877727°N 7.172831°W",
    "38.877663°N 7.172543°W"
]


# Variáveis globais para clique
clicked_points = []

def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Pixel clicado = ({x}, {y})")

def parse_latlon(coord_str):
    """
    Recebe algo como '38.570285°N 7.914937°W'
    Retorna (lat_dec, lon_dec) em graus decimais,
    sendo que West é negativo e South é negativo.
    """
    # Exemplo de parsing simples (atenção a possíveis espaços e símbolos):
    coord_str = coord_str.strip().upper().replace('°', '')
    # Agora temos algo como "38.570285N 7.914937W"
    parts = coord_str.split()
    # Vamos dividir a latitude e a longitude
    lat_str = parts[0]  # ex: "38.570285N"
    lon_str = parts[1]  # ex: "7.914937W"

    # Funcão auxiliar para cada parte
    def parse_part(s):
        # s ex: '38.570285N'
        # Verifica se tem 'N' ou 'S'
        if 'N' in s:
            val = float(s.replace('N', ''))
        elif 'S' in s:
            val = -float(s.replace('S', ''))
        else:
            val = float(s)  # fallback
        return val

    def parse_part_lon(s):
        # s ex: '7.914937W'
        # Verifica se tem 'E' ou 'W'
        if 'E' in s:
            val = float(s.replace('E', ''))
        elif 'W' in s:
            val = -float(s.replace('W', ''))
        else:
            val = float(s)  # fallback
        return val

    lat_dec = parse_part(lat_str)
    lon_dec = parse_part_lon(lon_str)

    return lat_dec, lon_dec

def latlon_to_xy(lat_deg, lon_deg, lat0_deg, lon0_deg):
    """
    Converte (lat, lon) em graus decimais para
    coordenadas locais (x, y) em metros, usando
    aproximação planar simples em torno de (lat0, lon0).
    """
    R = 6371000.0  # Raio médio da Terra em metros (aprox)
    # Converte para radianos
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)

    # Equações simples de projeção local
    x = R * (lon - lon0) * math.cos(lat0)
    y = R * (lat - lat0)
    return x, y

def main():
    # --- 2. Carregar a imagem ---
    image_path = r"C:\Users\35196\OneDrive\Ambiente de Trabalho\tese\Tese_code\SafeRoadside\pix2gps\tes.png"
    img = cv2.imread(image_path)
    if img is None:
        print("Erro ao carregar a imagem.")
        return

    # --- 3. Converter as 13 coords de lat/lon para (x, y) local ---
    # Escolhemos como referência (lat0, lon0) a primeira coordenada
    # (poderias escolher um "centro médio", mas aqui fica simples).
    lat0_deg, lon0_deg = parse_latlon(coords_str[0])
    real_points = []
    for c in coords_str:
        lat_deg, lon_deg = parse_latlon(c)
        X, Y = latlon_to_xy(lat_deg, lon_deg, lat0_deg, lon0_deg)
        real_points.append([X, Y])

    real_points = np.array(real_points, dtype=np.float32)  # shape (13, 2)

    # --- 4. Pedir ao utilizador que clique 13 pontos na mesma ordem ---
    cv2.namedWindow("Selecione 13 pontos (mesma ordem das coords)")
    cv2.setMouseCallback("Selecione 13 pontos (mesma ordem das coords)", mouse_callback)

    print("Instrucao: clique 13 vezes nos locais exatos que correspondem \n"
          "à ordem das 13 lat/lon no array coords_str.\n"
          "Pressione ESC a qualquer momento para sair.")

    while True:
        disp = img.copy()
        cv2.imshow("Selecione 13 pontos (mesma ordem das coords)", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if len(clicked_points) == 13:
            break

    cv2.destroyAllWindows()

    if len(clicked_points) != 13:
        print("Nao foram selecionados exatamente 13 pontos. A sair.")
        return

    pixel_points = np.array(clicked_points, dtype=np.float32)  # shape (13, 2)

    # --- 5. Calcular a transformacao afim que mapeia pixel -> (X, Y) local ---
    affine_mat, inliers = cv2.estimateAffine2D(pixel_points, real_points)
    # Se quiseres maior robustez a outliers, podes usar estimateAffine2D com RANSAC.

    if affine_mat is None:
        print("Erro ao calcular a transformacao afim.")
        return

    print("Matriz afim encontrada:")
    print(affine_mat)
    # affine_mat é 2x3

    # --- 6. Guardar no ficheiro (por ex. map.txt) ---
    # Precisamos também de guardar lat0_deg e lon0_deg, pois será necessário
    # no passo inverso (XY -> lat/lon).
    # Vamos guardar:
    # linha 1: lat0_deg, lon0_deg
    # linha 2: a11, a12, a13
    # linha 3: a21, a22, a23

    with open("map.txt", "w") as f:
        f.write(f"{lat0_deg} {lon0_deg}\n")
        for row in affine_mat:
            f.write(" ".join(map(str, row)) + "\n")

    print("Ficheiro map.txt gerado com sucesso.")

if __name__ == "__main__":
    main()
