$advertiser = New-Object Windows.Devices.Bluetooth.Advertisement.BluetoothLEAdvertisementPublisher
$advertisement = $advertiser.Advertisement
$advertisement.LocalName = "MeuBLE"

$advertiser.Start()
Write-Output "Anúncio BLE iniciado..."
Start-Sleep -Seconds 60  # Anuncia por 60 segundos
$advertiser.Stop()
Write-Output "Anúncio BLE parado."
