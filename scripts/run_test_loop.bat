@echo off
:: =============================================================================
:: run_test_loop.bat
:: NemoForge — Reason-Act-Reflect loop launcher
::
:: PURPOSE
::   Launches test_loop.py using Isaac Sim Python when available, falling back
::   to plain Python for the standalone RAR loop (no live physics, no extension).
::
:: WHAT IS THE RAR LOOP?
::   The Reason-Act-Reflect loop is the core of NemoForge.  Each iteration:
::     1. REASON  — build_prompt() constructs a structured LLM prompt that
::                  includes the current task and any prior physics violations.
::     2. ACT     — The FastAPI bridge at :8010 calls the NemoClaw agent and
::                  returns a JSON placement (position, rotation, scale).
::     3. REFLECT — get_physics_report() runs a PhysX simulation and returns
;;                  collision depths and unstable objects.  PhysicsReflector
::                  decides whether to continue or stop.
::   The loop stops when two consecutive reports are clean, or after 8 iterations.
::
:: PREREQUISITES
::   - FastAPI bridge running:
::       cd isaac-sim-backend && uvicorn main:app --port 8010
::   - (Optional) Isaac Sim Python for live extension support.
::       Runs with plain Python if Isaac Sim is not found.
::
:: USAGE
::   scripts\run_test_loop.bat                    Run a single sample task
::   scripts\run_test_loop.bat --batch            Run all 50 NF-Core tasks
::   scripts\run_test_loop.bat --nf-core          Alias for --batch
::   scripts\run_test_loop.bat --batch --headless  Headless (no GUI, no HUD)
::   scripts\run_test_loop.bat --batch --visuals   Save viewport screenshots
::   scripts\run_test_loop.bat --batch --headless --results-dir D:\output
::   scripts\run_test_loop.bat --max-iter 5        Limit iterations per task
::
:: FLAGS (forwarded directly to test_loop.py)
::   --batch          Run all tasks in src/nemoforge/core/nf_core_50.json
::   --nf-core        Same as --batch
::   --headless       Disable Isaac Sim HUD overlay and viewport capture.
::                    Use this for server/CI runs or when no display is available.
::   --visuals        Save one PNG screenshot per task per iteration to
::                    results/visuals/.  Requires Isaac Sim viewport to be open.
::   --task-file PATH Use a custom JSON task file instead of nf_core_50.json.
::   --results-dir DIR Write CSV and screenshots to DIR (default: results/).
::   --max-iter N     Maximum RAR iterations per task (default: 8).
::
:: OUTPUTS
::   results/benchmark_report.csv          Per-task metrics (batch mode only)
::   results/visuals/NF_<ID>_Iter<N>.png   Viewport screenshots (--visuals only)
::
:: HOW TO START THE BRIDGE (run in a separate terminal before this script)
::   cd C:\NemoForge\isaac-sim-backend
::   uvicorn main:app --host 127.0.0.1 --port 8010 --reload
:: =============================================================================

setlocal

:: ---------------------------------------------------------------------------
:: Resolve the NemoForge repo root.
:: %~dp0 is the directory containing this .bat file (scripts\).
:: ..  steps up one level to the repo root (C:\NemoForge\).
:: ---------------------------------------------------------------------------
set "ROOT=%~dp0.."

:: Absolute path to the Python script being launched.
set "SCRIPT=%ROOT%\src\nemoforge\core\test_loop.py"

:: ---------------------------------------------------------------------------
:: Auto-detect Isaac Sim Python via the shared helper.
:: setup_isaac_paths.bat sets the ISAAC_PY variable if a valid launcher is found.
:: It checks, in order:
::   1. isaacsim\_build\windows-x86_64\release\python.bat  (repo build)
::   2. %LOCALAPPDATA%\ov\pkg\isaac_sim-*\python.bat        (Omniverse Launcher)
:: ---------------------------------------------------------------------------
call "%ROOT%\scripts\setup_isaac_paths.bat"

:: ---------------------------------------------------------------------------
:: Choose the Python runner.
:: The RAR test loop can run with plain Python because extension.py falls back
:: gracefully to inline stubs when omni.* modules are not available.
:: The FastAPI bridge, however, must be running in a separate terminal.
:: ---------------------------------------------------------------------------
if "%ISAAC_PY%"=="" (
    echo [warn] Isaac Sim Python not found — using plain python.
    echo        HUD and viewport capture will be disabled automatically.
    echo        Physics simulation will use the built-in convergence model.
    set "RUNNER=python"
) else (
    set "RUNNER=%ISAAC_PY%"
)

:: Confirm the target script exists before proceeding.
if not exist "%SCRIPT%" (
    echo [ERROR] Script not found:
    echo   %SCRIPT%
    echo   Verify the repository structure with: python src\nemoforge\utils\paths.py
    exit /b 1
)

:: ---------------------------------------------------------------------------
:: Launch the test loop, forwarding all arguments supplied to this batch file.
:: For example:
::   run_test_loop.bat --batch --headless
:: becomes:
::   python.bat src\nemoforge\core\test_loop.py --batch --headless
:: ---------------------------------------------------------------------------
echo.
echo [NemoForge] Starting RAR test loop
echo   Runner : %RUNNER%
echo   Script : %SCRIPT%
echo   Args   : %*
echo.

call "%RUNNER%" "%SCRIPT%" %*

:: Capture the exit code from the Python process and return it to the caller.
set "RC=%ERRORLEVEL%"
endlocal & exit /b %RC%
