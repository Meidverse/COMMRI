@echo off
cd /d "%~dp0.."
echo Starting NeuroScan AI...  open http://127.0.0.1:5000 in your browser
start "" http://127.0.0.1:5000
python app\app.py
pause
