@echo off
rem Stop the app: whatever is listening on port 7860, and any start.bat window left waiting behind it.
powershell -NoProfile -Command "$app = Get-NetTCPConnection -State Listen -LocalPort 7860 -ErrorAction SilentlyContinue; if ($app) { Stop-Process -Id $app.OwningProcess -Force; Write-Host 'The app has been stopped.' } else { Write-Host 'The app was not running.' }; Get-CimInstance Win32_Process -Filter \"Name='cmd.exe'\" | Where-Object { $_.CommandLine -match 'start\.bat' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
pause
