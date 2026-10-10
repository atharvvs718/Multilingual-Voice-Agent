@echo off
rem Start the database and the app, and open the app in Microsoft Edge.
rem Edge is opened by name, whatever the default browser is: Brave's privacy shield silences the microphone.
rem Double-click this file, or run it from cmd.
cd /d "%~dp0"
if not exist logs mkdir logs

echo Starting the database...
docker compose up -d
if errorlevel 1 (
    echo.
    echo Could not start the database. Open Docker Desktop, wait until it is running, then run this again.
    echo The app will still start, but it cannot use your notes without the database.
    echo.
)

rem In the background: wait until the app answers on port 7860 (up to 4 minutes), then open it in Edge.
start "" /b powershell -NoProfile -Command "$edge = Join-Path ${env:ProgramFiles(x86)} 'Microsoft\Edge\Application\msedge.exe'; if (-not (Test-Path $edge)) { $edge = 'msedge' }; for ($i = 0; $i -lt 120; $i++) { try { (New-Object Net.Sockets.TcpClient('127.0.0.1', 7860)).Close(); Start-Process $edge 'http://localhost:7860'; break } catch { Start-Sleep -Seconds 2 } }"

echo.
echo Starting the app. Edge opens by itself when the page is ready (a few seconds).
echo.
echo Keep this window open while you use the app. To stop the app, press Ctrl+C here or close the window.
echo What the app is doing is written to logs\app.log.
rem Output goes to the log file, not this window: a click inside a console window can freeze a program that writes to it.
".venv\Scripts\python.exe" -u server.py >> logs\app.log 2>&1
echo.
echo The app has stopped. The last lines of logs\app.log say why:
powershell -NoProfile -Command "Get-Content logs\app.log -Tail 8"
pause
