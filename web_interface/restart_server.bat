@echo off
echo Restarting Flask server...
timeout /t 2 /nobreak > nul
taskkill /F /PID 18552 > nul 2>&1
cd /d C:\Users\AnonymousJ\multistrategytrading
start "" "C:\Users\AnonymousJ\multistrategytrading\venv\Scripts\python.exe" "C:\Users\AnonymousJ\multistrategytrading\web_interface\app.py"
exit
