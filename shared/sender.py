import pexpect
import time
import os

# Inicia o bluetoothctl como um processo interativo
bluetoothctl = pexpect.spawn("sudo bluetoothctl", encoding="utf-8", timeout=5)

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
    
    return f"{lat_hex}{lon_hex}"


def send_command(command):
    """Envia um comando para o bluetoothctl e aguarda a resposta."""
    bluetoothctl.sendline(command)
    bluetoothctl.expect("#|\\[bluetooth\\]")

# Configuração inicial do advertising
print("Configurando bluetoothctl...")
send_command("power on")
send_command("menu advertise")
send_command("interval 60 100")
send_command("manufacturer 0xFF 0x00 0x00 0x00")  # Começa com valor fixo
send_command("name SafeRSU")
send_command("back")
send_command("advertise on")
send_command("menu advertise")
print("Configuração inicial concluída. Monitorando o arquivo data.txt...")

# Inicializa o valor atual
current_value = "0x00 0x00 0x00"

# Loop para monitorar o arquivo data.txt e continuar enviando o último valor
while True:
    if os.path.exists("data.txt"):
        with open("data.txt", "r") as f:
            data = f.read().strip().replace("\n", " ")  # Remove espaços e quebras de linha
        
        if data and data != current_value:  # Se houver um novo valor, atualiza
            print(f"Novo valor detectado! Enviando: 0xFF {data}")
            current_value = data  # Atualiza o valor atual
            send_command(f"manufacturer 0xFF {current_value}")
            with open("data.txt", "w") as f:  # Limpa o arquivo após a leitura
                f.write("")
        
        else:
            # Continua enviando o último valor atualizado
            print(f"Reenviando último valor: 0xFF {current_value}")
            send_command(f"manufacturer 0xFF {current_value}")

    else:
        print("Arquivo data.txt não encontrado!")

    time.sleep(5)  # Aguarda 5 segundos antes de verificar novamente
