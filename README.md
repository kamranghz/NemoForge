# NemoForge — Agentic Reason-Act-Reflect Physics Correction

**NemoForge** is a PhD research pipeline implementing an agentic
**Reason → Act → Reflect (RAR)** loop for physics-based correction of
3D scenes reconstructed from video (USD format).  A language model proposes
geometry corrections; an ovphysx simulation evaluates physical validity;
the model reflects on the failure report and self-corrects, repeating until
the scene is simulation-ready.

The pipeline also includes a large-scale physics validity **SAGE-10k baseline**
that measures how often automatically generated scenes pass a PhysX simulation
out of the box.

> **Branch:** `v2-clean-architecture` — flat `src/` layout, standalone ovphysx
> (no Isaac Sim required to run the RAR loop or baseline).

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Repository Structure](#2-repository-structure)
3. [Prerequisites](#3-prerequisites)
4. [Setup](#4-setup)
5. [Running the Pipeline](#5-running-the-pipeline)
6. [Reference Repositories](#6-reference-repositories)
7. [Troubleshooting](#7-troubleshooting)
8. [Citation](#8-citation)
9. [License](#9-license)

---

## 1. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Input: video-reconstructed USD scene                            │
│                                                                  │
│   REASON   → prompt_builder.py  → structured LLM prompt         │
│   ACT      → nemoclaw_client.py → NemoClaw / NIM / Ollama       │
│   REFLECT  → physics_critic.py  → ovphysx failure score         │
│              correction_engine.py → loop until score < 0.5      │
│                                                                  │
│  Output: corrected USD + per-iteration correction record         │
└──────────────────────────────────────────────────────────────────┘
```

### Modules

| Module | File | Role |
|--------|------|------|
| **Simulation** | `src/simulation/physics_critic.py` | Runs ovphysx 0.4.9: 120 steps at 1/60 s, returns a `failure_score` |
| **Agent** | `src/agent/nemoclaw_client.py` | LLM client: tries NemoClaw `:8642` → NVIDIA NIM → Ollama |
| **Correction** | `src/correction/correction_engine.py` | Orchestrates RAR loop; stops when `failure_score < 0.5` |
| **Correction** | `src/correction/prompt_builder.py` | Builds physics-aware LLM prompts from `PhysicsReport` |
| **Benchmark** | `src/benchmark/sage_baseline.py` | SAGE-10k scene iterator → ovphysx → JSON output |
| **API** | `src/api/bridge.py` | FastAPI `:8010` — `/health`, `/correct`, `/generate` |
| **Utils** | `src/utils/paths.py` | Canonical path registry (`REPO_ROOT`, dataset, results) |
| **Utils** | `src/utils/json_to_usd.py` | JSON scene → USD conversion helper |

---

## 2. Repository Structure

```
NemoForge/
├── src/
│   ├── agent/
│   │   └── nemoclaw_client.py      # LLM agent client
│   ├── api/
│   │   └── bridge.py               # FastAPI server (:8010)
│   ├── benchmark/
│   │   └── sage_baseline.py        # SAGE-10k physics baseline runner
│   ├── correction/
│   │   ├── correction_engine.py    # RAR loop orchestrator
│   │   └── prompt_builder.py       # LLM prompt construction
│   ├── simulation/
│   │   └── physics_critic.py       # ovphysx physics evaluator
│   └── utils/
│       ├── paths.py                # Path registry
│       └── json_to_usd.py          # JSON → USD conversion
│
├── pyproject.toml                  # Package metadata & dependencies
├── README.md
└── .gitignore

# --- NOT in this repo --- set up locally as described below ---
# dataset/SAGE-10k/               325 GB — download from Hugging Face
# NemoClaw/                       NVIDIA NemoClaw agent (optional local)
# isaacsim/                       Isaac Sim source / build (optional)
# reference_repos/                Third-party research clones (optional)
# results/                        Output CSV / JSON (git-ignored)
```

---

## 3. Prerequisites

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA RTX (any) | RTX 4080 or higher |
| RAM | 32 GB | 64 GB |
| Disk | 10 GB (repo + venv) | 400 GB (+ full SAGE-10k) |
| VRAM | 8 GB | 16 GB+ |

### Python

Python **3.10** or higher (tested on 3.10 – 3.13).  Isaac Sim is **not** required
to run the RAR loop or the SAGE baseline; ovphysx runs as a standalone Python
package.

### Python packages

```bash
pip install fastapi uvicorn[standard] requests pydantic numpy
pip install ovphysx==0.4.9          # physics simulation (PyPI)
```

### LLM backend (choose one)

| Option | Setup |
|--------|-------|
| **NemoClaw local** | Clone and start [NemoClaw](https://github.com/NVIDIA/NemoClaw); runs on `:8642` |
| **NVIDIA NIM cloud** | Set `NVIDIA_API_KEY` env var; no local model needed |
| **Ollama** | Install [Ollama](https://ollama.com), pull a model (e.g. `ollama pull llama3`) |

The agent client tries all three in order and falls back automatically.

### SAGE-10k dataset (optional — 325 GB)

Required only for the SAGE baseline, not the correction pipeline.

```bash
# Install Hugging Face CLI if needed
pip install huggingface_hub

# Authenticate
huggingface-cli login --token YOUR_HF_TOKEN

# Download (streams in parts — safe to resume)
huggingface-cli download nvidia/SAGE-10k \
    --repo-type dataset \
    --local-dir dataset/SAGE-10k
```

Expected layout after download:

```
dataset/SAGE-10k/
├── scenes/       # *.zip archives (~10 000 scenes)
└── kits/         # NVIDIA USD conversion scripts
```

---

## 4. Setup

```bash
# 1. Clone this repo
git clone https://github.com/<your-org>/NemoForge.git
cd NemoForge

# 2. Install Python dependencies
pip install -e ".[dev]"       # installs fastapi, uvicorn, requests, pydantic
pip install ovphysx==0.4.9    # physics engine (not in pyproject.toml — GPU wheel)

# 3. Set PYTHONPATH so all src/ packages are importable
#    PowerShell:
$env:PYTHONPATH = "$PWD\src"
#    bash / zsh:
export PYTHONPATH="$PWD/src"

# 4. (Optional) Set LLM credentials
$env:NVIDIA_API_KEY = "nvapi-..."
```

Verify the path registry resolves:

```bash
python src/utils/paths.py
```

---

## 5. Running the Pipeline

### 5a. RAR correction loop — single USD scene

```bash
python src/correction/correction_engine.py \
    --usd path/to/scene.usd \
    --scene-id my_scene_001
```

The loop stops when `failure_score < 0.5` or after 8 iterations.
Results are written to `results/correction_results.csv`.

### 5b. SAGE-10k physics baseline

```bash
python src/benchmark/sage_baseline.py \
    --mode baseline \
    --n-scenes 10 \
    --seed 42
```

Output: `results/sage_validity_baseline.json`

```json
{
  "metadata": { "n_scenes": 10, "data_root": "dataset/SAGE-10k" },
  "records": [
    {
      "scene_zip":     "20251213_020526_layout_84b703fb.zip",
      "failure_score": 0.12,
      "validity":      "PASS",
      "error":         null
    }
  ]
}
```

### 5c. FastAPI bridge (standalone server)

```bash
python src/api/bridge.py
# or
uvicorn api.bridge:app --host 0.0.0.0 --port 8010 --reload
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/correct` | Run RAR correction on a USD path |
| `POST` | `/generate` | Single LLM placement call |

### 5d. All three options require PYTHONPATH

Always export `PYTHONPATH` before running any `src/` script directly:

```powershell
# PowerShell — one-liner
$env:PYTHONPATH = "C:\NemoPhys\NemoForge\src"; python src/correction/correction_engine.py ...
```

---

## 6. Reference Repositories

The following repositories were used as research references.
They are **not** required to run NemoForge but are listed here for reproducibility.
Clone them locally if you want to read or compare their implementations.

| Repo | URL | Purpose |
|------|-----|---------|
| SAGE | https://github.com/NVIDIA/SAGE | Scene generation benchmark + kits |
| GenManip | https://github.com/GenManip/GenManip | Manipulation scene generation |
| SceneWeaver | — | 3D scene composition reference |
| NemoClaw | https://github.com/NVIDIA/NemoClaw | NVIDIA agentic robot framework |

---

## 7. Troubleshooting

### `ModuleNotFoundError: No module named 'simulation'`

`src/` is not on `PYTHONPATH`.  Export it first:

```powershell
$env:PYTHONPATH = "C:\NemoPhys\NemoForge\src"
```

---

### `ModuleNotFoundError: No module named 'ovphysx'`

Install the GPU wheel:

```bash
pip install ovphysx==0.4.9
```

If the package is not yet on PyPI in your environment, check the
[NVIDIA developer portal](https://developer.nvidia.com) for the correct wheel URL.

---

### `ConnectionRefusedError` from `nemoclaw_client.py`

The client tries NemoClaw (`:8642`) first.  If no local agent is running it
automatically falls back to NVIDIA NIM (requires `NVIDIA_API_KEY`) then Ollama.
If all three fail, set at least one of:

```powershell
$env:NVIDIA_API_KEY = "nvapi-..."   # NIM cloud
# or start Ollama:  ollama serve
```

---

### `FileNotFoundError: dataset/SAGE-10k/scenes`

The SAGE-10k dataset has not been downloaded.  See [Prerequisites](#3-prerequisites).

---

### `huggingface-cli not found`

```bash
pip install huggingface_hub
python -m huggingface_hub.commands.huggingface_cli download nvidia/SAGE-10k ...
```

---

### Isaac Sim / NemoClaw extension (`extension.py`)

`isaacsim/source/extensions/isaacsim.nemoclaw.connector/` contains a
1700-line Kit extension for interactive Isaac Sim use.  It is **not** required
for the v2 Python pipeline.  If you need it:

1. Install Isaac Sim via the [Omniverse Launcher](https://www.nvidia.com/en-us/omniverse/)
   or build from source.
2. Set `ISAAC_SIM_PATH` and run with Isaac Sim Python:
   ```bat
   isaacsim\_build\windows-x86_64\release\python.bat src/...
   ```

---

## 8. Citation

If you use NemoForge or the SAGE-10k benchmark results in your research, please cite:

```bibtex
@article{xia2026sage,
  title   = {SAGE: Scalable Agentic 3D Scene Generation for Embodied AI},
  author  = {Xia, Hongchi and Li, Xuan and others},
  journal = {arXiv preprint arXiv:2602.10116},
  year    = {2026}
}
```

---

## 9. License

This project is released under the **Apache License 2.0**.
See `NemoClaw/LICENSE` and `dataset/SAGE-10k/README.md` for third-party terms.
