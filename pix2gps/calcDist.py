#!/usr/bin/env python3
# checkDistance.py

import math

def haversine(lat1, lon1, lat2, lon2):
    """
    Calcula a distância entre dois pontos geográficos (latitude e longitude)
    usando a fórmula de Haversine.
    
    Parâmetros:
        lat1, lon1 - Latitude e longitude do primeiro ponto (graus decimais)
        lat2, lon2 - Latitude e longitude do segundo ponto (graus decimais)
    
    Retorna:
        Distância entre os dois pontos em metros.
    """
    R = 6371000  # Raio médio da Terra em metros

    # Converter graus para radianos
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Diferenças de latitude e longitude
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Fórmula de Haversine
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c  # Distância em metros

def parse_coord(coord_str):
    """
    Converte uma coordenada do formato "38.877721°N 7.172348°W"
    para (latitude, longitude) em graus decimais.

    - 'N' e 'E' são positivos
    - 'S' e 'W' são negativos
    """
    coord_str = coord_str.strip().replace("°", "").upper()
    parts = coord_str.split()

    lat_str = parts[0]  # Exemplo: "38.877721N"
    lon_str = parts[1]  # Exemplo: "7.172348W"

    def parse_part(s):
        if 'N' in s:
            return float(s.replace('N', ''))
        elif 'S' in s:
            return -float(s.replace('S', ''))
        elif 'E' in s:
            return float(s.replace('E', ''))
        elif 'W' in s:
            return -float(s.replace('W', ''))
        else:
            return float(s)  # Fallback

    lat = parse_part(lat_str)
    lon = parse_part(lon_str)

    return lat, lon

def main():
    # Coordenadas fornecidas
    true_coord = "38.877577°N 7.172632°W"
    pred_coord = (38.877579, -7.172634)  # Já está em formato correto
    
    # Converter "true" para formato decimal
    true_lat, true_lon = parse_coord(true_coord)

    # Extrair "pred"
    pred_lat, pred_lon = pred_coord

    # Calcular a distância
    distance = haversine(true_lat, true_lon, pred_lat, pred_lon)

    # Mostrar resultado
    print(f"True:  ({true_lat:.6f}°, {true_lon:.6f}°)")
    print(f"Pred:  ({pred_lat:.6f}°, {pred_lon:.6f}°)")
    print(f"Distância: {distance:.3f} metros")

if __name__ == "__main__":
    main()
