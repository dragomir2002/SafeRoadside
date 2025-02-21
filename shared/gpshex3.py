import struct

def gps2hex(lat: float, lon: float) -> str:
    """
    Converte coordenadas GPS (latitude, longitude) para um inteiro 32-bit representado em hexadecimal.
    O input deve ser um double (float) e o output é o int32 respetivo em hexadecimal.
    
    Utiliza um fator de escala de 1e7 (10.000.000) para preservar 7 casas decimais (igual ao dicionario de mensagens).
    Retorna uma string formatada como "0xNN 0xNN ..." (4 bytes para cada coordenada).
    """
    scale = 10_000_000  # fator de escala para preservar casas decimais
    # Converter o float para int32 (com arredondamento)
    lat_int = int(round(lat * scale))
    lon_int = int(round(lon * scale))
    
    # Empacotar os inteiros em 4 bytes cada (big-endian)
    lat_bytes = struct.pack('>i', lat_int)
    lon_bytes = struct.pack('>i', lon_int)
    
    # Converter os bytes para uma string hexadecimal
    lat_hex = ' '.join(f'0x{b:02X}' for b in lat_bytes)
    lon_hex = ' '.join(f'0x{b:02X}' for b in lon_bytes)
    
    return f"{lat_hex} {lon_hex}"

def hex2gps(hex_str: str) -> str:
    """
    Converte uma string hexadecimal no formato "0xNN 0xNN ..." (representando dois inteiros 32-bit)
    para coordenadas GPS (latitude, longitude).
    
    Recupera o valor int, divide pelo fator de escala (1e7) para retornar o double original.
    Retorna uma string formatada como "lat: ... lon: ..."
    """
    scale = 10_000_000
    # Separar os valores hexadecimais e converter para bytes
    hex_values = hex_str.split()
    # Cada coordenada ocupa 4 bytes
    lat_bytes = bytes(int(h, 16) for h in hex_values[:4])
    lon_bytes = bytes(int(h, 16) for h in hex_values[4:])
    
    # Desempacotar os 4 bytes em int32 (big-endian)
    lat_int = struct.unpack('>i', lat_bytes)[0]
    lon_int = struct.unpack('>i', lon_bytes)[0]
    
    # Converter o inteiro de volta para float usando o fator de escala
    lat = lat_int / scale
    lon = lon_int / scale
    
    return f"lat: {lat}, lon: {lon}"

# Exemplo de uso (my loc)
lat = 38.707827
lon = -9.298649
hx_str = gps2hex(lat, lon)
print("Hexadecimal:", hx_str)
gps_str = hex2gps(hx_str)
print("Coordenadas:", gps_str)
