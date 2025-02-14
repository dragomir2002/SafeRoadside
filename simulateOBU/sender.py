import pexpect
import time
import os

# Inicia o bluetoothctl como um processo interativo
bluetoothctl = pexpect.spawn("sudo bluetoothctl", encoding="utf-8", timeout=5)

def send_command(command):
    """Envia um comando para o bluetoothctl e aguarda a resposta."""
    bluetoothctl.sendline(command)
    bluetoothctl.expect("#|\\[bluetooth\\]")

# Configuração inicial do advertising
print("Configurando bluetoothctl...")
send_command("power on")
send_command("menu advertise")
send_command("interval 60 100")
send_command("manufacturer 0xFF 0x00 0x00 0x00")
send_command("name ServicoZe")
send_command("back")
send_command("advertise on")
send_command("menu advertise")
print("Configuração inicial concluída. Monitorando o arquivo data.txt...")

# Loop para monitorar o arquivo data.txt
while True:
    if os.path.exists("data.txt"):
        with open("data.txt", "r") as f:
            data = f.read().strip().replace("\n", " ")  # Remove espaços e quebras de linha
        if data:
            print(f"Enviando novos dados: 0xFF {data}")
            send_command(f"manufacturer 0xFF {data}")
            with open("data.txt", "w") as f:  # Limpa o arquivo
                f.write("")
        else:
            send_command("manufacturer 0xFF 0x00 0x00 0x00")
    else:
        print("Arquivo data.txt não encontrado!")
    time.sleep(5)  # Aguarda 5 segundos antes de verificar novamente