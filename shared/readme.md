# Start bluetooth inside VM
lsusb
sudo systemctl start bluetooth
sudo systemctl enable bluetooth
sudo systemctl restart bluetooth

# time shift measurement commands
adb kill-server
adb start-server
adb devices

# missing 
quando enviamos há uma chance de mandar restart 