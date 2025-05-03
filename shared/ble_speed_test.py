import pexpect
import time
import os
import threading
from datetime import datetime

# Inicia o bluetoothctl como um processo interativo
bluetoothctl = pexpect.spawn("sudo bluetoothctl", encoding="utf-8", timeout=5)

def send_command(command):
    """Envia um comando para o bluetoothctl e aguarda o prompt."""
    bluetoothctl.sendline(command)
    bluetoothctl.expect("#|\\[bluetooth\\]")

def get_time_hex():
    """Obtém minuto, segundo e milissegundo atual e codifica em hexa."""
    now = datetime.now()
    minute = now.minute
    second = now.second
    millisecond = int(now.microsecond / 1000)

    ms_high = (millisecond >> 8) & 0xFF
    ms_low = millisecond & 0xFF

    # Retorna string hex formatada para uso no comando
    return f"0x{minute:02X} 0x{second:02X} 0x{ms_high:02X} 0x{ms_low:02X}"

def advertise_loop(interval=1.0):
    """
    Envia dados de tempo em hexadecimal em loop no intervalo definido.
    """
    while True:
        hex_payload = get_time_hex()
        print(f"[INFO] Enviando timestamp: {hex_payload}")
        send_command(f"manufacturer 0xFF {hex_payload}")
        time.sleep(interval)

def main():
    # Configuração inicial do bluetooth
    print("Configurando bluetoothctl...")
    send_command("power on")
    send_command("menu advertise")
    send_command("interval 60 100")
    send_command("manufacturer 0xFF 0x00 0x00 0x00 0x00")
    send_command("name GPS")
    send_command("back")
    send_command("advertise on")
    send_command("menu advertise")
    print("Configuração concluída. Iniciando envio de timestamps BLE...")

    try:
        advertise_loop(interval=0.1)  # envia a cada 100ms
    except KeyboardInterrupt:
        print("Encerrando...")
        send_command("advertise off")
        bluetoothctl.sendline("exit")
        bluetoothctl.close()

if __name__ == "__main__":
    main()
