# NemoForge — Agentic Reason-Act-Reflect Physics Pipeline

**NemoForge** is a PhD research project that implements an agentic
**Reason → Act → Reflect (RAR)** loop for robot scene placement inside
NVIDIA Isaac Sim.  A language model proposes object placements; a PhysX
physics simulation evaluates them; the model then reflects on the collision
report and self-corrects, repeating until the scene is physically valid.

The project also includes a large-scale physics validity benchmark over the
**SAGE-10k** dataset (10 000 indoor scenes, 325 GB) to measure how often
automatically generated scenes are simulation-ready.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Prerequisites](#2-prerequisites)
3. [Project Structure](#3-project-structure)
4. [How to Build / Setup](#4-how-to-build--setup)
5. [How to Run the Main Test Loop](#5-how-to-run-the-main-test-loop)
6. [How to Run the SAGE-10k Physics Baseline](#6-how-to-run-the-sage-10k-physics-baseline)
7. [How to Launch Everything Inside Isaac Sim](#7-how-to-launch-everything-inside-isaac-sim)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Project Overview

### What is the RAR loop?

```
┌─────────────────────────────────────────────────────────────────┐
│  User task: "Place the chair next to the table without overlap." │
│                                                                  │
│   REASON   →  build_prompt()  → structured LLM prompt           │
│   ACT      →  FastAPI bridge  → NemoClaw agent (JSON placement)  │
│   REFLECT  →  get_physics_report()  → PhysX contacts + stability │
│               PhysicsReflector  → decide: continue or stop       │
│               loop back to REASON with updated physics context   │
└─────────────────────────────────────────────────────────────────┘
```

The loop runs up to **8 iterations** and stops early when two consecutive
physics reports are clean (zero collisions, zero unstable objects).

### What is the SAGE-10k benchmark?

Each of the 10 000 scenes is stored as a `.zip` archive in `dataset/SAGE-10k/scenes/`.
The baseline script unpacks each scene, converts it to USD using the official
NVIDIA `kits/` conversion tools, loads it into Isaac Sim, runs a physics
simulation, and records the initial validity metrics.

---

## 2. Prerequisites

### Hardware
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA RTX (any) | RTX 4080 or higher |
| RAM | 32 GB | 64 GB |
| Disk | 50 GB (repo + build) | 400 GB (+ full SAGE-10k) |
| VRAM | 8 GB | 16 GB+ |

### Software (Windows 10/11)
- **Git** and **Git LFS** (for the Isaac Sim submodule)
- **Microsoft Visual Studio 2019 or 2022** with the
  *Desktop development with C++* workload (required only to build Isaac Sim from source)
- **Windows SDK** (installed alongside MSVC)
- **NVIDIA GPU Driver** ≥ 537 (see [NVIDIA driver requirements](https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html))
- **Python 3.10+** (for the FastAPI bridge and standalone RAR loop)
- **pip packages** for the bridge:
  ```
  pip install fastapi uvicorn requests pydantic
  ```

### Isaac Sim Python
All physics-related scripts **must** run with Isaac Sim's bundled Python
(not Anaconda or system Python).  After building or installing Isaac Sim,
the correct launcher is:

```
isaacsim\_build\windows-x86_64\release\python.bat   (built from source)
%LOCALAPPDATA%\ov\pkg\isaac_sim-<version>\python.bat (Omniverse Launcher)
```

The helper script `scripts\setup_isaac_paths.bat` finds this automatically.

---

## 3. Project Structure

```
C:\NemoForge\
├── NemoClaw/                   # NemoClaw robot agent (do NOT move)
├── isaac-sim-backend/          # FastAPI bridge — kept at repo root
│   ├── main.py                 #   FastAPI app: /health /generate
│   ├── prompt_builder.py       #   Builds RAR prompts
│   ├── nemo_client.py          #   NemoClaw API client
│   └── asset_resolver.py       #   Maps agent output to USD assets
├── reference_repos/            # Third-party clones (do NOT move)
├── dataset/                    # SAGE-10k dataset (git-ignored, 325 GB)
│   └── SAGE-10k/
│       ├── scenes/             #   *.zip scene archives
│       └── kits/               #   Official NVIDIA conversion scripts
├── isaacsim/                   # Isaac Sim cloned source tree (do NOT move)
│   └── _build/windows-x86_64/release/
│       ├── python.bat          #   <-- Isaac Sim Python launcher
│       └── isaac-sim.bat       #   <-- Isaac Sim GUI launcher
│
├── src/
│   └── nemoforge/              # Main Python package
│       ├── agent/              #   LLM agent logic (future extension)
│       ├── simulation/         #   Isaac Sim helpers (future extension)
│       ├── bridge/             #   Thin wrapper over isaac-sim-backend
│       ├── core/
│       │   ├── test_loop.py            # RAR orchestrator (main entry point)
│       │   ├── sage_10k_physic_test.py # SAGE-10k physics baseline
│       │   └── nf_core_50.json         # 50 benchmark tasks
│       └── utils/
│           ├── paths.py                # Central path registry
│           └── verify_sage_structure.py
│
├── scripts/                    # Windows launcher scripts (operational)
│   ├── run_test_loop.bat               # Launch RAR test loop
│   ├── run_sage_physics_test.bat       # Launch SAGE-10k baseline
│   ├── setup_isaac_paths.bat           # Auto-detect Isaac Sim Python
│   └── build_isaacsim.bat              # Build Isaac Sim from source
│
├── config/                     # YAML / JSON config overrides (future)
├── docs/                       # Research documentation
├── results/                    # Output CSV / JSON (git-ignored)
├── pyproject.toml              # Package metadata and dev dependencies
├── .gitignore
└── README.md
```

---

## 4. How to Build / Setup

### Step 1 — Clone and initialise

```bat
git clone <this-repo> C:\NemoForge
cd C:\NemoForge
git lfs install
git lfs pull
```

### Step 2 — Install Python dependencies for the FastAPI bridge

```bat
cd C:\NemoForge\isaac-sim-backend
pip install -r requirements.txt
```

### Step 3a — Build Isaac Sim from source (if not using Omniverse Launcher)

> This takes 20–40 minutes and requires Visual Studio with C++ tools.

```bat
cd C:\NemoForge
scripts\build_isaacsim.bat
```

After a successful build, the Isaac Sim Python launcher will be at:
`isaacsim\_build\windows-x86_64\release\python.bat`

### Step 3b — Alternatively, install Isaac Sim via Omniverse Launcher

1. Download and install the [Omniverse Launcher](https://www.nvidia.com/en-us/omniverse/).
2. In the launcher, install **Isaac Sim** (any 4.x or 5.x version).
3. `scripts\setup_isaac_paths.bat` will find it automatically under `%LOCALAPPDATA%\ov\pkg\`.

### Step 4 — Verify all paths resolve

```bat
python src\nemoforge\utils\paths.py
```

Expected output: all entries show `[OK]`.

### Step 5 — Download the SAGE-10k dataset (optional — 325 GB)

```bat
:: Login with your Hugging Face token
%CONDA_PREFIX%\Scripts\hf.exe auth login --token YOUR_HF_TOKEN

:: Download the full dataset
%CONDA_PREFIX%\Scripts\hf.exe download nvidia/SAGE-10k ^
    --repo-type dataset ^
    --local-dir C:\NemoForge\dataset\SAGE-10k
```

---

## 5. How to Run the Main Test Loop

### Start the FastAPI bridge first (Terminal 1)

The RAR loop requires the bridge to be running before any test is started.

```bat
cd C:\NemoForge\isaac-sim-backend
uvicorn main:app --host 127.0.0.1 --port 8010 --reload
```

Verify it is healthy:
```bat
curl http://127.0.0.1:8010/health
```

### Run a single task (Terminal 2)

```bat
:: Using Isaac Sim Python (full extension support)
scripts\run_test_loop.bat

:: OR with plain Python (RAR logic only, no live physics)
python src\nemoforge\core\test_loop.py
```

### Run the full NF-Core benchmark (50 tasks)

```bat
:: Headless — fast, no GUI, logs only
scripts\run_test_loop.bat --batch --headless

:: With Isaac Sim viewport and HUD overlay visible
scripts\run_test_loop.bat --batch

:: Save one viewport screenshot per iteration per task
scripts\run_test_loop.bat --batch --visuals

:: Custom task file and output directory
scripts\run_test_loop.bat --batch --task-file config\my_tasks.json --results-dir D:\output
```

### Output

| File | Contents |
|------|----------|
| `results/benchmark_report.csv` | Per-task: task ID, category, iterations, success, collision counts |
| `results/visuals/NF_<TaskID>_Iter<N>.png` | Viewport screenshots (only with `--visuals`) |

---

## 6. How to Run the SAGE-10k Physics Baseline

This test requires the full SAGE-10k dataset and Isaac Sim Python.

```bat
:: Default: 10 random scenes, seed 42, headless
scripts\run_sage_physics_test.bat

:: More scenes, keep extracted files for inspection
scripts\run_sage_physics_test.bat --n-scenes 50 --keep-extract

:: Force re-conversion from layout JSON (ignore USD cache)
scripts\run_sage_physics_test.bat --force-export

:: With Isaac Sim viewport visible
scripts\run_sage_physics_test.bat --n-scenes 5
```

### Output

```json
// results/sage_validity_baseline.json
{
  "metadata": { "n_scenes": 10, "data_root": "C:\\NemoForge\\dataset\\SAGE-10k" },
  "records": [
    {
      "scene_zip":          "20251213_020526_layout_84b703fb.zip",
      "usd_path_used":      "results/sage_usd_cache/.../sage_composed.usd",
      "initial_collisions": 0,
      "unstable_count":     0,
      "max_penetration_cm": 0.0,
      "validity":           "PASS",
      "error":              null
    }
  ]
}
```

---

## 7. How to Launch Everything Inside Isaac Sim

### Using Isaac Sim Python (recommended for physics tests)

```bat
:: Verify the launcher path is found
call scripts\setup_isaac_paths.bat
echo %ISAAC_PY%

:: Run any script with the Isaac Python environment
"%ISAAC_PY%" src\nemoforge\core\sage_10k_physic_test.py --headless

:: Or use the helper batch files (they call setup_isaac_paths.bat automatically)
scripts\run_sage_physics_test.bat --headless
```

### Using the Isaac Sim GUI (for interactive debugging)

```bat
:: Launch Isaac Sim with a visible viewport
isaacsim\_build\windows-x86_64\release\isaac-sim.bat
```

In the Isaac Sim GUI, open the Script Editor and run:
```python
exec(open(r"C:\NemoForge\src\nemoforge\core\test_loop.py").read())
```

### Warm up shader cache (first run only — reduces startup time)

```bat
isaacsim\_build\windows-x86_64\release\warmup.bat
```

---

## 8. Troubleshooting

### "Could not import SimulationApp"

**Cause:** The script is being run with plain Python (Anaconda, system Python) instead
of Isaac Sim's bundled Python.

**Fix:**
1. Check that `scripts\setup_isaac_paths.bat` prints `[setup] Isaac Sim Python found`.
2. Use `scripts\run_sage_physics_test.bat` instead of calling `python` directly.
3. If using the Omniverse Launcher install, verify the path:
   ```bat
   dir "%LOCALAPPDATA%\ov\pkg\isaac_sim-*\python.bat"
   ```
4. If using a repo build, verify the build completed:
   ```bat
   dir isaacsim\_build\windows-x86_64\release\python.bat
   ```

---

### "ModuleNotFoundError: No module named 'omni.usd'"

**Cause:** Isaac Sim Python was found but the Kit session is not yet initialised.

**Fix:** The physics test scripts call `SimulationApp({"headless": True})` automatically.
If you are running interactively, make sure you are inside an active Kit session.

---

### "FATAL: cannot import extension"

**Cause:** `src/nemoforge/utils/paths.py` did not register the extension directory on `sys.path`,
or the NemoClaw connector has not been built.

**Fix:**
```bat
:: Check all paths are valid
python src\nemoforge\utils\paths.py

:: Confirm extension.py exists
dir isaacsim\source\extensions\isaacsim.nemoclaw.connector\isaacsim\nemoclaw\connector\extension.py
```

---

### "FastAPI bridge did not respond" / "BRIDGE_ERROR"

**Cause:** The FastAPI bridge is not running when `test_loop.py` makes its request.

**Fix:** Start the bridge in a separate terminal *before* running the test loop:
```bat
cd isaac-sim-backend
uvicorn main:app --port 8010
```

Confirm it is healthy:
```bat
curl http://127.0.0.1:8010/health
```

---

### `huggingface-cli` not found on Windows

The newer `huggingface_hub` package installs `hf.exe`, not `huggingface-cli.exe`.
Use the correct launcher for your environment:

```bat
:: Conda (care-ai) environment
%CONDA_PREFIX%\Scripts\hf.exe download nvidia/SAGE-10k ...

:: Or via Python module
python -m huggingface_hub.commands.huggingface_cli download nvidia/SAGE-10k ...
```

---

### Unicode / encoding errors in PowerShell

Isaac Sim's `python.bat` uses the Windows code page (cp1252).
This project has replaced all non-ASCII characters (arrows, emoji) in batch-facing
code.  If you add new code, avoid Unicode characters in strings that reach the terminal.

---

## Citation

If you use NemoForge or the SAGE-10k benchmark in your research, please cite:

```bibtex
@article{xia2026sage,
  title   = {SAGE: Scalable Agentic 3D Scene Generation for Embodied AI},
  author  = {Xia, Hongchi and Li, Xuan and ...},
  journal = {arXiv preprint arXiv:2602.10116},
  year    = {2026}
}
```

---

## License

This project is released under the **Apache License 2.0**.
See `NemoClaw/LICENSE` and `dataset/SAGE-10k/README.md` for third-party terms.
