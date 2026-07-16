@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Starting MembraneQuant Web UI on http://127.0.0.1:7860 ...
echo (If this fails, try from parent folder: cd .. ^&^& python -m membranequant --webui)
python "%~dp0main.py" --webui --port 7860 %*
if errorlevel 1 pause
