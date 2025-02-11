#!/usr/bin/env python3
# createMap.py

import cv2
import numpy as np
import math

# Lista de 13 coordenadas lat/long em formato string.
# Exemplo (modifica ou adapta às tuas coords):
coords_str = [
    "41.940527°N 85.000850°W",
    "41.940494°N 85.000678°W",
    "41.940510°N 85.000575°W",
    "41.940689°N 85.000545°W",
    "41.940761°N 85.000587°W",
    "41.940798°N 85.000782°W",
    "41.940743°N 85.000902°W",
    "41.940586°N 85.001006°W",
    "41.940600°N 85.000822°W",
    "41.940637°N 85.000672°W",
    "41.940386°N 85.000483°W",
    "41.940969°N 85.001014°W",
    "41.940740°N 85.000380°W"
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
    Recebe algo como "38.877713°N 7.172653°W"
    Retorna (lat_dec, lon_dec) em graus decimais,
    em que W e S são negativos.
    """
    coord_str = coord_str.strip().upper().replace('°', '')
    # Ex: "38.877713N 7.172653W"
    parts = coord_str.split()
    lat_str = parts[0]  # ex: "38.877713N"
    lon_str = parts[1]  # ex: "7.172653W"

    def parse_part_lat(s):
        if 'N' in s:
            return float(s.replace('N', ''))
        elif 'S' in s:
            return -float(s.replace('S', ''))
        return float(s)  # fallback

    def parse_part_lon(s):
        if 'E' in s:
            return float(s.replace('E', ''))
        elif 'W' in s:
            return -float(s.replace('W', ''))
        return float(s)  # fallback

    lat_dec = parse_part_lat(lat_str)
    lon_dec = parse_part_lon(lon_str)
    return lat_dec, lon_dec

def latlon_to_xy(lat_deg, lon_deg, lat0_deg, lon0_deg):
    """
    Converte (lat, lon) em graus decimais para
    coordenadas locais (X, Y) em metros, usando
    uma projeção planar simples em torno de (lat0, lon0).
    """
    R = 6371000.0  # Raio médio da Terra em metros (aprox)
    # Converte para radianos
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)

    # Projeção local
    X = R * (lon - lon0) * math.cos(lat0)
    Y = R * (lat - lat0)
    return X, Y

def main():
    # 1. Carregar a imagem
    image_path = r"C:\Users\35196\OneDrive\Ambiente de Trabalho\tese\Tese_code\SafeRoadside\pix2gps\tes.png"
    img = cv2.imread(image_path)
    if img is None:
        print("Erro ao carregar a imagem.")
        return

    # 2. Converter as 13 coords de lat/lon para (X, Y) local
    #    Usamos como referência (lat0, lon0) a primeira coordenada
    lat0_deg, lon0_deg = parse_latlon(coords_str[0])
    real_points = []
    for c in coords_str:
        lat_deg, lon_deg = parse_latlon(c)
        X, Y = latlon_to_xy(lat_deg, lon_deg, lat0_deg, lon0_deg)
        real_points.append([X, Y])
    real_points = np.array(real_points, dtype=np.float32)  # shape (13, 2)

    # 3. Pedir ao utilizador que clique 13 pontos na mesma ordem
    cv2.namedWindow("Selecione 13 pontos (mesma ordem das coords)")
    cv2.setMouseCallback("Selecione 13 pontos (mesma ordem das coords)", mouse_callback)

    print("Instrução: clique 13 vezes na imagem, na mesma ordem das 13 lat/lon.\n"
          "Pressione ESC para sair a qualquer momento.")

    while True:
        disp = img.copy()
        cv2.imshow("Selecione 13 pontos (mesma ordem das coords)", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if len(clicked_points) == len(coords_str):
            break

    cv2.destroyAllWindows()

    if len(clicked_points) != len(coords_str):
        print(f"Não foram selecionados exatamente {len(coords_str)} pontos. A sair.")
        return

    pixel_points = np.array(clicked_points, dtype=np.float32)  # shape (13, 2)

    # 4. Calcular a Homografia que mapeia pixel -> (X, Y) local
    #    Usamos RANSAC para maior robustez.
    homography_mat, status = cv2.findHomography(pixel_points, real_points, cv2.RANSAC, 5.0)

    if homography_mat is None:
        print("Erro ao calcular a homografia.")
        return

    print("Homografia encontrada (3x3):")
    print(homography_mat)

    # 5. Guardar no ficheiro map.txt:
    #    Linha 1: lat0_deg, lon0_deg
    #    Linhas 2..4: cada linha com 3 valores da matriz
    with open("map.txt", "w") as f:
        f.write(f"{lat0_deg} {lon0_deg}\n")
        for row in homography_mat:
            f.write(" ".join(map(str, row)) + "\n")

    print("Ficheiro map.txt gerado com sucesso.")

if __name__ == "__main__":
    main()
