@echo off
:: build_isaacsim.bat — Build Isaac Sim from the cloned isaacsim/ tree.
:: Requirements: Git LFS, MSVC (Desktop C++ workload), internet, EULA acceptance.
:: See isaacsim/README.md for full prerequisites.
setlocal
set "ROOT=%~dp0.."
if not exist "%ROOT%\isaacsim\build.bat" (
  echo [ERROR] isaacsim\build.bat not found.
  echo   Clone Isaac Sim into isaacsim\ first (see isaacsim/README.md).
  exit /b 1
)
cd /d "%ROOT%\isaacsim"
call build.bat %*
endlocal
