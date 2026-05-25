@echo off
setlocal EnableDelayedExpansion
echo.
echo =========================================
echo   Isaac Sim NemoClaw Bridge
echo =========================================
echo.
echo   Sandbox name for SSH (must match: openshell sandbox list). Default: physiclaw
REM Always set for this session so a stale global NEMOCLAW_SANDBOX=human cannot win.
set "NEMOCLAW_SANDBOX=physiclaw"
echo     NEMOCLAW_SANDBOX=%NEMOCLAW_SANDBOX%
echo   To use another sandbox, edit this .bat line or run: set NEMOCLAW_SANDBOX=yours ^& start.bat
echo   PREREQUISITE: that sandbox must be up (nemoclaw connect / port 18789 forward).
echo.
echo -----------------------------------------
echo.
cd /d "%~dp0"

echo [1/3] Installing Python dependencies...
pip install -r requirements.txt -q
if errorlevel 1 (
    echo ERROR: pip install failed.
    pause
    exit /b 1
)
echo       OK.
echo.

echo [2/3] Freeing port 8002 if in use...
REM netstat for-loop is unreliable; kill any LISTENING process on 8002 via PowerShell.
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-NetTCPConnection -LocalPort 8002 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"
timeout /t 2 /nobreak 1>nul
echo       OK.
echo.

echo [3/3] Starting FastAPI bridge on http://localhost:8002 ...
echo.
echo   Extension may use POST /generate or /generate-human on port 8002
echo   WSL helper uses bash -i, openshell ssh-config, openclaw agent
echo.
python -m uvicorn main:app --host 127.0.0.1 --port 8002 --log-level info

pause
endlocal
