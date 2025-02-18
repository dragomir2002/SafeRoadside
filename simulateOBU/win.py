import bluetooth
import bluetooth._bluetooth as bt
import struct
import sys

dev_id = 0  # ID do dispositivo Bluetooth (geralmente 0)
sock = bt.hci_open_dev(dev_id)

# Pacote de anúncio BLE
hci_cmd = struct.pack("<BB", 0x08, 0x00)  # Comando HCI
bt.hci_send_cmd(sock, 0x08, 0x0008, hci_cmd)
print("Anunciando BLE...")
