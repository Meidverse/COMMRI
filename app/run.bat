@echo off
cd /d "%~dp0.."
echo Starting NeuroScan AI... the browser will open in a few seconds at http://127.0.0.1:5000
start "" cmd /c "timeout /t 10 >nul & start http://127.0.0.1:5000"
python app\app.py
pause
