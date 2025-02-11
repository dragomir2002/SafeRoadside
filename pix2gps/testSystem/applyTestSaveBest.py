#!/usr/bin/env python3

import math
import numpy as np
import itertools
import cv2

def parse_line(line):
    """
    Faz o parsing de uma linha do tipo:
       "(38.877292, -7.168618)",335,63

    Retorna (lat, lon, px, py) como floats.
    """
    line = line.strip()
    # Dividir em 2 partes a partir de ')",'
    parts = line.split(')",')
    if len(parts) != 2:
        raise ValueError(f"Linha com formato inesperado: {line}")

    latlon_str = parts[0]  # ex: '(38.877292, -7.168618)'
    latlon_str = latlon_str.strip().lstrip('"').rstrip('"')  # remove aspas
    latlon_str = latlon_str.strip('()')  # remove parênteses

    # Agora latlon_str deve ser algo como: '38.877292, -7.168618'
    lat_str, lon_str = latlon_str.split(',', 1)
    lat = float(lat_str.strip())
    lon = float(lon_str.strip())

    px_py_str = parts[1].strip()  # ex: '335,63'
    px_str, py_str = px_py_str.split(',', 1)
    px = float(px_str.strip())
    py = float(py_str.strip())

    return lat, lon, px, py

def latlon_to_xy(lat_deg, lon_deg, lat0_deg, lon0_deg):
    """
    Projeção local simples (lat/lon -> (X,Y) em metros),
    tomando (lat0, lon0) como referência (0,0).
    """
    R = 6371000.0
    lat  = math.radians(lat_deg)
    lon  = math.radians(lon_deg)
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)

    X = R * (lon - lon0) * math.cos(lat0)
    Y = R * (lat - lat0)
    return X, Y

def main():
    # 1) Ler gpspix.txt
    #    Formato de cada linha: "(lat, lon)",px,py
    try:
        with open("gpspix.txt", "r") as f:
            lines = f.readlines()
    except:
        print("Erro ao ler gpspix.txt.")
        return

    lat_list = []
    lon_list = []
    pix_list = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        lat, lon, px, py = parse_line(line)
        lat_list.append(lat)
        lon_list.append(lon)
        pix_list.append((px, py))

    n_points = len(pix_list)
    print(f"Carregados {n_points} pontos do ficheiro gpspix.txt.")

    if n_points < 20:
        print("Atenção: esperávamos ter pelo menos 20 pontos.")
        return

    # 2) Converter lat/lon para (X, Y) local
    #    Usamos o 1º ponto como origem (lat0, lon0).
    lat0, lon0 = lat_list[0], lon_list[0]

    real_points = []
    for la, lo in zip(lat_list, lon_list):
        X, Y = latlon_to_xy(la, lo, lat0, lon0)
        real_points.append([X, Y])

    real_points = np.array(real_points, dtype=np.float32)  # shape (n_points, 2)
    pixel_points = np.array(pix_list,   dtype=np.float32)  # shape (n_points, 2)

    # 3) Gerar combinações de n_points pontos, escolhendo 13 para treino e 7 para teste.
    all_indices = range(n_points)
    comb_indices = itertools.combinations(all_indices, 13)

    # Para não sobrecarregar a CPU, vamos limitar a 10k combinações.
    # (Se quiseres avaliar todas, remove este limite, mas pode demorar muito.)
    max_combinations = 10000
    count = 0

    all_errors = []

    # Variáveis para guardar a melhor homografia
    best_error = float("inf")
    best_H = None

    for combo in comb_indices:
        combo = list(combo)
        test_indices = [i for i in all_indices if i not in combo]

        # Arrays para treino (13 pontos)
        train_pixels = pixel_points[combo]  # shape (13, 2)
        train_real   = real_points[combo]   # shape (13, 2)

        # Calcular homografia (pixel->real)
        H, status = cv2.findHomography(train_pixels, train_real, cv2.RANSAC, 5.0)
        if H is None:
            # Pode falhar se os pontos forem degenerados
            continue

        # Avaliar erro nos pontos de teste (7 pontos)
        test_pixels = pixel_points[test_indices]  # shape (7, 2)
        test_real   = real_points[test_indices]   # shape (7, 2)

        # Converter test_pixels -> real usando H
        ones = np.ones((test_pixels.shape[0], 1), dtype=np.float32)
        test_pixels_hom = np.hstack([test_pixels, ones])  # shape (7,3)

        # (X, Y, W) = H * (px, py, 1)
        pred_XYW = (H @ test_pixels_hom.T).T  # shape(7,3)
        W = pred_XYW[:, 2]
        valid_mask = np.abs(W) > 1e-12

        # Normalizar
        pred_X = np.zeros_like(W)
        pred_Y = np.zeros_like(W)

        pred_X[valid_mask] = pred_XYW[valid_mask, 0] / W[valid_mask]
        pred_Y[valid_mask] = pred_XYW[valid_mask, 1] / W[valid_mask]

        pred_points = np.vstack([pred_X, pred_Y]).T  # shape (7,2)

        # Distância Euclideana entre pred_points e test_real
        diffs = pred_points - test_real
        dist = np.sqrt(diffs[:,0]**2 + diffs[:,1]**2)  # array (7,)
        mean_err = np.mean(dist)
        all_errors.append(mean_err)

        # Verifica se esta é a melhor homografia até agora
        if mean_err < best_error:
            best_error = mean_err
            best_H = H.copy()

        count += 1
        if count >= max_combinations:
            break

    if len(all_errors) == 0 or best_H is None:
        print("Não foi calculada nenhuma homografia válida.")
        return

    # Estatísticas globais das combinações
    avg_error = np.mean(all_errors)
    std_error = np.std(all_errors)
    min_error = np.min(all_errors)
    max_error = np.max(all_errors)

    print(f"\nNúmero de combinações avaliadas = {count}")
    print(f"Erro médio (todas as combos) = {avg_error:.3f} m")
    print(f"Desvio padrão                = {std_error:.3f} m")
    print(f"Erro mínimo (em todas combos)= {min_error:.3f} m")
    print(f"Erro máximo (em todas combos)= {max_error:.3f} m")
    print(f"\nMelhor homografia obtida: erro médio = {best_error:.3f} m")

    # 4) Guardar a melhor H no ficheiro map.txt
    #    Formato:
    #      lat0 lon0
    #      (3 linhas da matriz H, cada com 3 valores)
    with open("map.txt", "w") as f:
        f.write(f"{lat0} {lon0}\n")
        for row in best_H:
            f.write(" ".join(str(v) for v in row) + "\n")

    print("\nMelhor matriz homografia guardada em map.txt!")
    print("Estrutura do ficheiro:")
    print("  linha 1: lat0 lon0")
    print("  linhas 2..4: cada linha com 3 valores de H")

if __name__ == "__main__":
    main()
