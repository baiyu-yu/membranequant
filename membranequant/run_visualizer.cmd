@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Starting MembraneQuant Visualization Web UI on http://127.0.0.1:7861 ...
python "%~dp0visualizer_app.py"
if errorlevel 1 pause
