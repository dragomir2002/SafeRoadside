import struct

def gps2hex(lat, lon):
    """
    Converte coordenadas GPS (latitude, longitude) para o formato hexadecimal IEEE 754 (double, 64 bits).
    Retorna uma string formatada como "0xNN 0xNN ..."
    """
    # Converter para double (IEEE 754, 64 bits, big-endian)
    lat_bytes = struct.pack('>d', lat)
    lon_bytes = struct.pack('>d', lon)
    
    # Formatar como string hexadecimal
    lat_hex = ' '.join(f'0x{b:02X}' for b in lat_bytes)
    lon_hex = ' '.join(f'0x{b:02X}' for b in lon_bytes)
    
    return f"{lat_hex} {lon_hex}"

def hex2gps(hex_str):
    """
    Converte uma string hexadecimal no formato "0xNN 0xNN ..." para coordenadas GPS (latitude, longitude).
    Retorna uma string formatada como "lat: ... lon: ..."
    """
    # Separar os valores hexadecimais e converter para bytes
    hex_values = hex_str.split()
    lat_bytes = bytes(int(h, 16) for h in hex_values[:8])
    lon_bytes = bytes(int(h, 16) for h in hex_values[8:])
    
    # Converter bytes para double (IEEE 754, 64 bits, big-endian)
    lat = struct.unpack('>d', lat_bytes)[0]
    lon = struct.unpack('>d', lon_bytes)[0]
    
    return f"lat: {lat}, lon: {lon}"


# exemplo de use case #
#lat = 33.787668
#lon = -117.852876
#hx_str = gps2hex(lat,lon)
#print(hx_str)
#gps_str = hex2gps(hx_str)
#print(gps_str)