@echo off
echo.
echo PhysiClaw / Isaac Sim + NemoClaw — quick reference
echo See SETUP_GUIDE.md in this folder for full steps.
echo.
echo Typical flow:
echo   1. WSL: onboard NemoClaw once, then keep sandbox up (nemoclaw ^<sandbox^> connect).
echo   2. WSL: run API  —  cd /mnt/c/PhysiClaw/isaac-sim-backend ^&^& python3 wsl_nemoclaw_api.py
echo   3. Win: start.bat  (optional; extension defaults to http://127.0.0.1:8010)
echo   4. Isaac Sim: enable NemoClaw Connector, Send to NIM.
echo.
pause
