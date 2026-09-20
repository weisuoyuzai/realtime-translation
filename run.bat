@echo off
rem Launch Live Translator (GUI). Pass --cli ... for headless mode.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [setup] creating virtual environment ...
    python -m venv .venv || goto :fail
    ".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :fail
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
    where nvidia-smi >nul 2>nul && ".venv\Scripts\python.exe" -m pip install -r requirements-cuda.txt
)
".venv\Scripts\python.exe" -m live_translator %*
exit /b %errorlevel%
:fail
echo Setup failed. See the messages above.
exit /b 1
