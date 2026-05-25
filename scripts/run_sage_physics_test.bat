@echo off
:: =============================================================================
:: run_sage_physics_test.bat
:: NemoForge — SAGE-10k physics validity baseline launcher
::
:: PURPOSE
::   Runs the SAGE-10k physics validity benchmark.  For each selected scene:
::     1. If results/sage_usd_cache/<scene_id>/sage_composed.usd exists and
::        is newer than the source .zip, it is used directly (cache hit).
;;     2. Otherwise the .zip is extracted to a temporary folder, the layout JSON
::        is located, and the NVIDIA kit script (kits/export_usd.py logic) is
::        called to convert the scene to per-part USD files.  These are then
::        composed into a single sage_composed.usd and saved in the cache.
::     3. load_scene_usd() (from the NemoClaw extension) loads the USD into the
::        active Isaac Sim stage.
::     4. get_physics_report() runs a short PhysX simulation and returns:
::          - initial_collisions  : number of intersecting object pairs
::          - unstable_count      : number of objects that moved after settling
::          - max_penetration_cm  : deepest overlap in centimetres
::          - validity            : PASS (no collisions, no unstable objects)
::
:: PREREQUISITES
::   - Isaac Sim Python is REQUIRED (plain Python will not work here because
::     this script uses real omni.* and pxr APIs via extension.py).
::   - SAGE-10k dataset must be at:  dataset\SAGE-10k\scenes\*.zip
::     Download with:
::       %CONDA_PREFIX%\Scripts\hf.exe auth login --token YOUR_TOKEN
::       %CONDA_PREFIX%\Scripts\hf.exe download nvidia/SAGE-10k ^
::           --repo-type dataset ^
::           --local-dir C:\NemoForge\dataset\SAGE-10k
::   - SAGE kits dependencies (for the USD conversion step):
::       pip install numpy trimesh
::
:: USAGE
::   scripts\run_sage_physics_test.bat
::       Default: 10 random scenes, seed 42, headless, kit export enabled.
::
::   scripts\run_sage_physics_test.bat --n-scenes 50
::       Evaluate 50 randomly selected scenes.
::
::   scripts\run_sage_physics_test.bat --n-scenes 5 --seed 7
::       Reproduce a specific random selection.
::
::   scripts\run_sage_physics_test.bat --headless
::       Disable Isaac Sim HUD and viewport capture.
::       Recommended for server / CI environments.
::
::   scripts\run_sage_physics_test.bat --keep-extract
::       Do not delete the temporary extraction folder after each scene.
::       Useful for debugging zip contents or USD conversion issues.
::
::   scripts\run_sage_physics_test.bat --no-kit-export
::       Skip USD conversion.  Only use USD files that already exist in the
::       cache (results\sage_usd_cache\) or directly inside the zip.
::
::   scripts\run_sage_physics_test.bat --force-export
::       Ignore the USD cache and re-convert from layout JSON every time.
::       Use this when the kit conversion code has been updated.
::
::   scripts\run_sage_physics_test.bat --settle-time 1.5
::       Override the physics settle time in seconds (default: from PHYSICS_CONFIG).
::
::   scripts\run_sage_physics_test.bat --results-dir D:\output
::       Write output JSON and USD cache to a custom directory.
::
:: OUTPUTS
::   results\sage_validity_baseline.json    Per-scene physics validity records.
::   results\sage_usd_cache\<scene_id>\     Cached sage_composed.usd files.
::
:: OUTPUT FORMAT (results\sage_validity_baseline.json)
::   {
::     "metadata": { "n_scenes": 10, "data_root": "..." },
::     "records": [
::       {
::         "scene_zip":          "20251213_020526_layout_84b703fb.zip",
::         "usd_path_used":      "results/sage_usd_cache/.../sage_composed.usd",
::         "initial_collisions": 0,
::         "unstable_count":     0,
::         "max_penetration_cm": 0.0,
::         "validity":           "PASS",
::         "error":              null
::       }
::     ]
::   }
::
:: TROUBLESHOOTING
::   "Could not import SimulationApp"
::       You are running with plain Python, not Isaac Sim's python.bat.
::       Check that setup_isaac_paths.bat found a valid launcher.
::
::   "FATAL: cannot import extension"
::       The NemoClaw connector extension is not on sys.path.
::       Run:  python src\nemoforge\utils\paths.py
::       and verify EXTENSION_DIR shows [OK].
::
::   "No *.zip files under ..."
::       The SAGE-10k dataset has not been downloaded or is in the wrong location.
::       Expected path: C:\NemoForge\dataset\SAGE-10k\scenes\*.zip
::
::   "kit export failed" / "No layout JSON in zip"
::       The zip archive may be corrupted, or the kit dependencies are missing.
::       Run:  pip install numpy trimesh
::       Then retry with:  --force-export --keep-extract
:: =============================================================================

setlocal

:: ---------------------------------------------------------------------------
:: Resolve the NemoForge repo root.
:: %~dp0 is the directory of this .bat file (scripts\).
:: ..  steps up to the repo root.
:: ---------------------------------------------------------------------------
set "ROOT=%~dp0.."

:: Absolute path to the Python script being launched.
set "SCRIPT=%ROOT%\src\nemoforge\core\sage_10k_physic_test.py"

:: Default dataset path.  Can be overridden by passing --data-root to this script.
set "DATA=%ROOT%\dataset\SAGE-10k"

:: ---------------------------------------------------------------------------
:: Auto-detect Isaac Sim Python via the shared helper.
:: This script REQUIRES Isaac Sim Python — no plain-Python fallback.
:: ---------------------------------------------------------------------------
call "%ROOT%\scripts\setup_isaac_paths.bat"

if "%ISAAC_PY%"=="" (
    echo.
    echo [ERROR] Isaac Sim Python is required for the physics baseline test.
    echo         This script uses real omni.*, pxr, and PhysX APIs.
    echo.
    echo   Option A: Build from source
    echo     scripts\build_isaacsim.bat
    echo.
    echo   Option B: Install via Omniverse Launcher
    echo     https://docs.isaacsim.omniverse.nvidia.com
    echo.
    exit /b 1
)

:: Confirm the Python script exists.
if not exist "%SCRIPT%" (
    echo [ERROR] Script not found:
    echo   %SCRIPT%
    echo   Check repository structure: python src\nemoforge\utils\paths.py
    exit /b 1
)

:: ---------------------------------------------------------------------------
:: Warn (but do not abort) if the dataset scenes folder is missing.
:: The script itself will handle the error gracefully.
:: ---------------------------------------------------------------------------
if not exist "%DATA%\scenes" (
    echo.
    echo [WARN] SAGE-10k scenes folder not found: %DATA%\scenes
    echo        The script will exit with an error unless --data-root is overridden.
    echo        To download the dataset:
    echo          %%CONDA_PREFIX%%\Scripts\hf.exe auth login --token YOUR_HF_TOKEN
    echo          %%CONDA_PREFIX%%\Scripts\hf.exe download nvidia/SAGE-10k ^
    echo              --repo-type dataset ^
    echo              --local-dir "%DATA%"
    echo.
)

:: ---------------------------------------------------------------------------
:: Launch the SAGE physics baseline test.
:: Default flags: 10 scenes, seed 42.  All additional arguments (%*) are
:: forwarded, allowing the caller to override n-scenes, headless, etc.
::
:: Example expansions:
::   run_sage_physics_test.bat --n-scenes 5
::   => python.bat sage_10k_physic_test.py --data-root "..." --n-scenes 5 --seed 42
::
::   run_sage_physics_test.bat --n-scenes 5 --headless --force-export
::   => python.bat sage_10k_physic_test.py --data-root "..." --n-scenes 5 --seed 42
::                                         --headless --force-export
:: ---------------------------------------------------------------------------
echo.
echo [NemoForge] Starting SAGE-10k physics validity baseline
echo   Runner  : %ISAAC_PY%
echo   Script  : %SCRIPT%
echo   Dataset : %DATA%
echo   Args    : --data-root "%DATA%" --n-scenes 10 --seed 42 %*
echo.

call "%ISAAC_PY%" "%SCRIPT%" --data-root "%DATA%" --n-scenes 10 --seed 42 %*

:: Capture the Python process exit code and return it to the caller.
set "RC=%ERRORLEVEL%"
endlocal & exit /b %RC%
