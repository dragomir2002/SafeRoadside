import pexpect
import time
import os
import threading

# Inicia o bluetoothctl como um processo interativo
bluetoothctl = pexpect.spawn("sudo bluetoothctl", encoding="utf-8", timeout=5)

def send_command(command):
    """Envia um comando para o bluetoothctl e aguarda o prompt."""
    bluetoothctl.sendline(command)
    bluetoothctl.expect("#|\\[bluetooth\\]")

def watch_file(filename):
    """
    Thread que observa o arquivo e envia a primeira linha
    sempre que o arquivo tiver conteúdo.
    """
    while True:
        # Verifica se o arquivo existe
        if os.path.exists(filename):
            with open(filename, "r") as f:
                lines = f.readlines()
            
            # Se houver linhas no arquivo
            if lines:
                first_line = lines[0].strip()

                # Remove a primeira linha do arquivo
                with open(filename, "w") as f:
                    f.writelines(lines[1:])

                # Se a linha não estiver vazia, envia
                if first_line:
                    print(f"[Thread] Enviando novos dados: 0xFF {first_line}")
                    send_command(f"manufacturer 0xFF {first_line}")
                # Se for vazia, envia null
            else: 
                    print(f"[Thread] Ficheiro vazio: 0xFF 0x00 0x00 0x00")
                    send_command(f"manufacturer 0xFF 0x00 0x00 0x00")       
        # Aguarda um pouco antes de verificar de novo
        time.sleep(0.5)

def main():
    # Configuração inicial do bluetooth
    print("Configurando bluetoothctl...")
    send_command("power on")
    send_command("menu advertise")
    send_command("interval 60 100")
    send_command("manufacturer 0xFF 0x00 0x00 0x00")
    send_command("name GPS")
    send_command("back")
    send_command("advertise on")
    send_command("menu advertise")
    print("Configuração inicial concluída. Iniciando monitoramento...")

    # Cria e inicia a thread de monitoramento
    watch_thread = threading.Thread(target=watch_file, args=("data.txt",), daemon=True)
    watch_thread.start()

    # Mantém a thread principal ativa indefinidamente
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Encerrando...")

if __name__ == "__main__":
    main()
