@echo off
:: setup_isaac_paths.bat
:: Sets ISAAC_PY to the correct Isaac Sim Python launcher and exports it for
:: the current session so other scripts can call %ISAAC_PY% directly.
::
:: Usage:  call scripts\setup_isaac_paths.bat
::         (call, not just run, to inherit the exported variable)

setlocal EnableDelayedExpansion
set "ROOT=%~dp0.."

:: Preferred: built python.bat inside this repo's isaacsim tree.
set "CANDIDATE_1=%ROOT%\isaacsim\_build\windows-x86_64\release\python.bat"
:: Fallback: standard Omniverse Launcher install location.
set "CANDIDATE_2=%LOCALAPPDATA%\ov\pkg\isaac_sim-latest\python.bat"

if exist "%CANDIDATE_1%" (
    set "ISAAC_PY=%CANDIDATE_1%"
    echo [setup] Isaac Sim Python found (repo build):
    echo   %CANDIDATE_1%
    goto :Found
)

:: Search the ov/pkg tree for any isaac_sim-* python.bat
for /d %%D in ("%LOCALAPPDATA%\ov\pkg\isaac_sim-*") do (
    if exist "%%D\python.bat" (
        set "ISAAC_PY=%%D\python.bat"
        echo [setup] Isaac Sim Python found (Omniverse Launcher):
        echo   %%D\python.bat
        goto :Found
    )
)

echo [ERROR] Isaac Sim Python not found.
echo   Build from source:  build_isaacsim.bat
echo   Or install via:     https://docs.isaacsim.omniverse.nvidia.com
exit /b 1

:Found
:: Export into the PARENT shell (only works with CALL from the parent .bat).
endlocal & set "ISAAC_PY=%ISAAC_PY%"
echo [setup] ISAAC_PY=%ISAAC_PY%
exit /b 0
