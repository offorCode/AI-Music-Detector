@echo off
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -c "import torch, torchaudio, numpy, librosa, soundfile, typer, pydub, huggingface_hub, rich, tqdm, yt_dlp, transformers, safetensors, scipy, imageio_ffmpeg" >nul 2>nul
  if not errorlevel 1 goto already_ready
)
where python >nul 2>nul
if errorlevel 1 (
  echo Python 3.11 was not found. Install Python 3.11 and enable Add Python to PATH.
  pause
  exit /b 1
)
python -c "import struct, sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) and struct.calcsize('P')*8 == 64 else 1)"
if errorlevel 1 (
  echo This app currently requires 64-bit Python 3.11.
  pause
  exit /b 1
)
if not exist .venv\Scripts\python.exe python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu --timeout 120 --retries 2
if errorlevel 1 goto failed
python -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo.
echo Setup complete. Run run_detector.bat to start the app.
pause
exit /b 0
:already_ready
echo Setup is already complete. You do not need to install it again.
echo Run run_detector.bat to start the app.
pause
exit /b 0
:failed
echo.
echo Setup failed. Check the error above and internet connection.
pause
exit /b 1
