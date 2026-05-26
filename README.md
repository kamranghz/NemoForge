# NemoForge

**Agentic physics correction of video-reconstructed 3D scenes using LLM reasoning and PhysX simulation**

NemoForge implements an agentic **Reason → Act → Reflect (RAR)** loop that automatically corrects physics violations in 3D scenes reconstructed from video. A language model proposes geometry corrections, an ovphysx simulation evaluates physical validity, and the model reflects on the failure report and self-corrects — repeating until the scene is simulation-ready.

The pipeline also includes a large-scale physics validity **SAGE-10K baseline** that measures how often automatically generated scenes pass a PhysX simulation without manual correction.

> **Branch:** `v2-clean-architecture` — flat `src/` layout, standalone ovphysx. Isaac Sim is not required to run the RAR loop or the SAGE baseline.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Architecture](#2-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Prerequisites](#4-prerequisites)
5. [Setup](#5-setup)
6. [Running the Pipeline](#6-running-the-pipeline)
7. [Reference Repositories](#7-reference-repositories)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. The Problem

Neural reconstruction methods such as DUSt3R, MASt3R, and VGGT produce visually accurate scenes but cannot guarantee physical validity. Reconstructed scenes frequently contain:

- **Floating objects** — caused by depth uncertainty during reconstruction
- **Interpenetrating geometry** — from incomplete multi-view coverage
- **Non-manifold surfaces** — from mesh extraction artifacts

These violations are invisible to photometric metrics (PSNR, SSIM) but catastrophic in physics simulation. Manual correction takes tens of engineering hours per scene. NemoForge automates this process.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NemoForge Pipeline                           │
│                                                                       │
│  Input: video-reconstructed USD scene (with physics violations)      │
│                                                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │    REASON    │ →  │     ACT      │ →  │       REFLECT         │  │
│  │              │    │              │    │                       │  │
│  │ prompt       │    │ NemoClaw     │    │  ovphysx PhysX sim    │  │
│  │ builder      │    │ NIM cloud    │    │  failure score F      │  │
│  │              │    │ Ollama       │    │  violation report     │  │
│  └──────────────┘    └──────────────┘    └──────────┬────────────┘  │
│         ↑                                            │               │
│         └──────────── action history ────────────────┘               │
│                                                                       │
│  Stops when:  F < 0.5  OR  max iterations reached                    │
│                                                                       │
│  Output: corrected USD · correction log · RFPCR metric               │
└─────────────────────────────────────────────────────────────────────┘
```

### Physics Failure Score

$$\mathcal{F}(S_t) = \sum_{(i,j)} d_{ij} + 2.0\,\sum_i h_i + 0.5\,\sum_i v_i + \eta$$

| Term | Symbol | Source |
|------|--------|--------|
| Penetration depth | $d_{ij}$ | Contact report |
| Floating offset | $h_i$ | Position tensor — Y axis |
| Post-settle velocity | $v_i$ | Velocity tensor |
| Non-manifold ratio | $\eta$ | Mesh topology |

### Evaluation Metric — RFPCR

**Reconstructed-scene Physics Correction Rate** measures whether a corrected scene is both physically valid and visually similar to the original reconstruction:

$$\text{RFPCR} = \frac{1}{N} \sum_{n=1}^{N} \mathbf{1}\!\left[\, \mathcal{F}(S^*_n) < \varepsilon \;\wedge\; \text{SSIM}(S^*_n,\,S_n) > 1 - \delta \,\right]$$

### Module Reference

| Module | File | Responsibility |
|--------|------|----------------|
| **Physics Critic** | `simulation/physics_critic.py` | Loads USD, settles 120 PhysX steps at 1/60 s, returns `failure_score` and violation breakdown |
| **LLM Agent** | `agent/nemoclaw_client.py` | Queries NemoClaw local `:8642` → NVIDIA NIM cloud → Ollama `:11434` in priority order |
| **Correction Engine** | `correction/correction_engine.py` | Orchestrates the RAR loop with rollback, per-iteration logging, and CSV output |
| **Prompt Builder** | `correction/prompt_builder.py` | Builds structured correction prompts from `PhysicsReport` with full action history |
| **SAGE Benchmark** | `benchmark/sage_baseline.py` | Extracts SAGE-10K zips, converts to USD, runs physics measurement in batch |
| **API Bridge** | `api/bridge.py` | FastAPI server — `/health`, `/correct`, `/generate` |
| **Path Registry** | `utils/paths.py` | Central path definitions for all local dependencies |
| **USD Converter** | `utils/json_to_usd.py` | Converts SAGE layout JSON to physics-ready USD |

---

## 3. Repository Structure

### What is on GitHub

```
NemoForge/
├── src/
│   ├── agent/
│   │   └── nemoclaw_client.py      # LLM agent client
│   ├── api/
│   │   └── bridge.py               # FastAPI server (:8010)
│   ├── benchmark/
│   │   └── sage_baseline.py        # SAGE-10K physics baseline runner
│   ├── correction/
│   │   ├── correction_engine.py    # RAR loop orchestrator
│   │   └── prompt_builder.py       # LLM prompt construction
│   ├── simulation/
│   │   └── physics_critic.py       # ovphysx physics evaluator
│   └── utils/
│       ├── paths.py                # Path registry
│       ├── json_to_usd.py          # JSON → USD conversion
│       └── verify_sage_structure.py
├── pyproject.toml
├── README.md
└── .gitignore
```

### What you create locally (not committed)

```
NemoForge/
├── dataset/
│   └── SAGE-10k/           # ~325 GB — download from Hugging Face
│       ├── scenes/          # *.zip scene archives
│       └── kits/            # NVIDIA USD conversion scripts
├── NemoClaw/                # NVIDIA NemoClaw agent (optional)
├── isaacsim/                # Isaac Sim source (optional)
├── reference_repos/         # Research reference clones (optional)
└── results/                 # CSV / JSON outputs (git-ignored)
```

---

## 4. Prerequisites

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA RTX (any) | RTX 4080 or higher |
| RAM | 32 GB | 64 GB |
| Disk | 10 GB (repo + venv) | 400 GB (+ full SAGE-10K) |
| VRAM | 8 GB | 16 GB+ |

### Python

Python **3.10** or higher. Isaac Sim is **not** required to run the RAR loop or the SAGE baseline — ovphysx runs as a standalone Python package.

### Python packages

```bash
pip install -e ".[dev]"
pip install numpy "ovphysx==0.4.9" --extra-index-url https://pypi.nvidia.com
```

### LLM backend — choose one

| Option | Setup |
|--------|-------|
| **NemoClaw local** | Clone [NemoClaw](https://github.com/NVIDIA/NemoClaw) and start it — runs on port **8642** |
| **NVIDIA NIM cloud** | Set `NVIDIA_API_KEY` environment variable — no local model required |
| **Ollama local** | Install [Ollama](https://ollama.com), run `ollama serve`, pull a model |

The agent client probes all three in priority order and falls back automatically.

### SAGE-10K dataset (optional — 325 GB)

Required only for the SAGE baseline.

```bash
pip install huggingface_hub
huggingface-cli login --token YOUR_HF_TOKEN

huggingface-cli download nvidia/SAGE-10k \
    --repo-type dataset \
    --local-dir dataset/SAGE-10k
```

---

## 5. Setup

```bash
# 1. Clone
git clone https://github.com/kamranghz/NemoForge.git
cd NemoForge

# 2. Install dependencies
pip install -e ".[dev]"
pip install numpy "ovphysx==0.4.9" --extra-index-url https://pypi.nvidia.com

# 3. Set PYTHONPATH
# Linux / WSL / macOS
export PYTHONPATH="$PWD/src"

# Windows PowerShell
$env:PYTHONPATH = "$PWD\src"

# 4. Set NVIDIA API key (for cloud inference)
export NVIDIA_API_KEY="nvapi-your-key-here"

# 5. Verify
python src/utils/paths.py
```

---

## 6. Running the Pipeline

> **Always set `PYTHONPATH` before running any script.**

### Physics measurement — single scene

```bash
python src/simulation/physics_critic.py path/to/scene.usd
```

Output:

```
========================================================
  PHYSICS MEASUREMENT REPORT
========================================================
  Objects found     : 59
  Penetration pairs : 4
  Floating objects  : 54
  Unstable objects  : 0
  Failure score F   : 88.5200
========================================================
```

### RAR correction loop — single USD scene

```bash
python src/correction/correction_engine.py \
    --usd path/to/scene.usd \
    --scene-id scene_001 \
    --max-iter 10
```

Stops when `failure_score < 0.5` or after `--max-iter` iterations.

Output files:

| File | Contents |
|------|---------|
| `results/correction_results.csv` | Per-scene summary — initial F, final F, RFPCR, iterations |
| `results/correction_logs/scene_001.json` | Per-iteration action, F score, rollback status |

### SAGE-10K physics baseline

```bash
# Measure initial violations on 10 scenes
python src/benchmark/sage_baseline.py \
    --mode baseline \
    --n-scenes 10 \
    --seed 42

# Run full correction on 5 scenes
python src/benchmark/sage_baseline.py \
    --mode correction \
    --n-scenes 5 \
    --seed 42
```

Output: `results/sage_baseline.csv`

### Start the API bridge

```bash
python src/api/bridge.py
# or
uvicorn api.bridge:app --host 127.0.0.1 --port 8010 --reload
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check + active LLM backend |
| `POST` | `/correct` | Physics report → correction action |
| `POST` | `/generate` | Raw LLM call |

---

## 7. Reference Repositories

The following repositories were used as research references during development. They are not required to run NemoForge.

| Repository | URL |
|------------|-----|
| SAGE | https://github.com/NVIDIA-Omniverse/SAGE |
| NemoClaw | https://github.com/NVIDIA/NemoClaw |
| Isaac Sim | https://github.com/isaac-sim/IsaacSim |
| GenManip | https://github.com/GenManip/GenManip |

---

## 8. Troubleshooting

**`ModuleNotFoundError: No module named 'simulation'`**

`src/` is not on `PYTHONPATH`:
```bash
export PYTHONPATH="$PWD/src"
```

---

**`ModuleNotFoundError: No module named 'ovphysx'`**

```bash
pip install "ovphysx==0.4.9" --extra-index-url https://pypi.nvidia.com
```

---

**`RuntimeError: No LLM backend available`**

Start at least one backend:
```bash
# Option 1 — Ollama
ollama serve && ollama pull llama3

# Option 2 — NVIDIA NIM cloud
export NVIDIA_API_KEY="nvapi-..."
```

---

**`No scenes found in dataset/SAGE-10k/scenes`**

Download the dataset — see [Prerequisites](#4-prerequisites).

---

**`PhysX warning: cuCtxGetDevice failed`**

ovphysx falls back to CPU simulation. This is normal in WSL2. Physics results remain valid.

---

**`Could not import SimulationApp` / `No module named 'omni'`**

You are running an Isaac-specific script with plain Python. The core pipeline (`correction_engine.py`, `sage_baseline.py`) does not require Isaac Sim.

---

**`huggingface-cli not found`**

```bash
pip install huggingface_hub
python -m huggingface_hub.commands.huggingface_cli download nvidia/SAGE-10k \
    --repo-type dataset --local-dir dataset/SAGE-10k
```
