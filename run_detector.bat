@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo First run setup_windows.bat to install the detector.
  pause
  exit /b 1
)
.venv\Scripts\python.exe earprint_desktop.py
if errorlevel 1 pause
