@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" web_panel.py
    pause
    exit /b
)
py -3.12 --version >nul 2>&1
if %errorlevel%==0 (
    py -3.12 web_panel.py
) else (
    python web_panel.py
)
pause
