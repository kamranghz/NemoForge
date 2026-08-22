<div align="center">

# NemoForge

**Agentic Physics Correction of Video-Reconstructed 3D Scenes**

*A four-component agentic pipeline that measures physics violations in reconstructed 3D scenes, reasons about corrections with a language model, applies targeted geometry fixes, and evaluates convergence with the RFPCR metric.*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![ovphysx](https://img.shields.io/badge/ovphysx-0.4.9-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![NemoClaw](https://img.shields.io/badge/NemoClaw-Early%20Preview-76B900?logo=nvidia&logoColor=white)](https://github.com/NVIDIA/NemoClaw)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.x%2F5.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/isaac-sim)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![Paper](https://img.shields.io/badge/Paper-coming%20soon-lightgrey) ![Project Page](https://img.shields.io/badge/Project%20Page-coming%20soon-lightgrey) [![Dataset](https://img.shields.io/badge/🤗%20Dataset-SAGE--10K-yellow)](https://huggingface.co/datasets/nvidia/SAGE-10k) [![Code](https://img.shields.io/badge/Code-GitHub-181717?logo=github)](https://github.com/kamranghz/NemoForge)

[Overview](#overview) · [Key Features](#key-features) · [Architecture](#architecture) · [Repository Structure](#repository-structure) · [Installation](#installation) · [Quick Start](#quick-start) · [Running the Pipeline](#running-the-pipeline) · [Status](#project-status) · [Citation](#citation)

</div>

---

<p align="center">
  <img src="docs/assets/pipeline_architecture.png" alt="NemoForge pipeline: Physics Critic to Prompt Builder to NemoClaw Agent to Correction Engine" width="850">
  <br>
  <sub><!-- TODO: drop the real pipeline figure at docs/assets/pipeline_architecture.png, or tell me the intended path and I'll update this embed --> Figure: the four-component Reason → Act → Reflect pipeline.</sub>
</p>

### 60-second quick start

```bash
git clone https://github.com/kamranghz/NemoForge.git && cd NemoForge
pip install -e ".[dev]" && pip install numpy "ovphysx==0.4.9" --extra-index-url https://pypi.nvidia.com
export PYTHONPATH="$PWD/src"
python src/simulation/physics_critic.py path/to/scene.usd   # no LLM backend needed for this step
```

See [Installation](#installation) for the full setup, including an LLM backend for the correction loop.

---

## Overview

Neural reconstruction methods such as DUSt3R, MASt3R, and VGGT produce visually accurate 3D scenes from video — but they optimize photometric loss only and cannot guarantee physical validity. Reconstructed scenes frequently contain violations that are invisible to image-based metrics yet catastrophic in simulation:

| Violation | Cause | Effect in simulation |
|-----------|-------|---------------------|
| **Floating objects** | Depth uncertainty | Objects fall at runtime — unstable sim |
| **Interpenetrating geometry** | Incomplete multi-view coverage | Explosive forces at t=0 |
| **Non-manifold surfaces** | Mesh extraction artifacts | Invalid collision geometry |

Manual correction of these violations takes tens of engineering hours per scene. NemoForge automates this with an agentic **Reason → Act → Reflect (RAR)** loop: a language model proposes corrections, an ovphysx simulation measures physical validity, and the model reflects on the failure report and self-corrects until the scene is simulation-ready.

> **No Isaac Sim required** to run the physics-measurement step or the SAGE-10K baseline. ovphysx runs as a standalone Python package; Isaac Sim is only needed for optional visualization.

## Key Features

- **Four-stage agentic pipeline** — physics measurement, LLM-driven reasoning, geometry correction, and quantitative convergence scoring, each independently runnable.
- **Verified ovphysx integration** — the physics critic's API usage (`step_n_sync`, contact reports, rigid-body tensor bindings) is validated against a real measured baseline (F = 88.52, 59 objects, one SAGE-10K scene), not just against the SDK docs.
- **Backend-agnostic LLM reasoning** — probes NemoClaw → NVIDIA NIM → Ollama in order, so the loop runs whether you have a local GPU, a cloud API key, or neither is fully set up.
- **Bounded, auditable correction actions** — six named geometry operations (`snap_to_surface`, `resolve_penetration`, `recompute_hull`, `repair_manifold`, `stack_on`, `remove_object` as last resort) instead of unconstrained edits, with rollback whenever a step makes F worse.
- **A standardized benchmark** — the SAGE-10K baseline (Component 4) reports initial failure scores across N scenes without requiring an LLM, giving a reproducible B1 number independent of which LLM backend you use.

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                           NemoForge Pipeline                            │
│                                                                          │
│   Input: video-reconstructed USD scene (with physics violations)        │
│                                                                          │
│   ┌──────────────────┐                                                  │
│   │  Physics Critic  │  ovphysx · 120 settle steps · failure score F   │
│   └────────┬─────────┘                                                  │
│            │  F = Σd_ij + 2·Σh_i + 0.5·Σv_i + η                      │
│            ▼                                                             │
│   ┌──────────────────┐                                                  │
│   │  Prompt Builder  │  PhysicsReport → structured correction prompt    │
│   └────────┬─────────┘                                                  │
│            │  prompt + violation context + action history               │
│            ▼                                                             │
│   ┌──────────────────┐                                                  │
│   │  NemoClaw Agent  │  NemoClaw :8642 → NVIDIA NIM → Ollama :11434   │
│   └────────┬─────────┘                                                  │
│            │  {action, parameters, reasoning}                           │
│            ▼                                                             │
│   ┌──────────────────┐                                                  │
│   │  Correction      │  snap_to_surface · resolve_penetration ·        │
│   │  Engine (RAR)    │  recompute_hull · repair_manifold · stack_on    │
│   └────────┬─────────┘                                                  │
│            │  rollback if F worsens · stop if F < 0.5                  │
│            ▼                                                             │
│   Output: corrected USD · correction log · RFPCR metric                │
└────────────────────────────────────────────────────────────────────────┘
```

### Physics Failure Score

$$\mathcal{F}(S_t) = \underbrace{\sum_{(i,j)} d_{ij}}_{\text{penetration}} + \;2.0\underbrace{\sum_i h_i}_{\text{floating}} + \;0.5\underbrace{\sum_i v_i}_{\text{instability}} + \underbrace{\eta}_{\text{non-manifold}}$$

### Evaluation Metric — RFPCR

**Reconstructed-scene Physics Correction Rate** — the fraction of scenes where the corrected scene is both physically valid and visually similar to the original:

$$\text{RFPCR} = \frac{1}{N}\sum_{n=1}^{N}\mathbf{1}\!\left[\mathcal{F}(S^*_n) < \varepsilon \;\wedge\; \text{SSIM}(S^*_n, S_n) > 1-\delta\right]$$

### Component Details

#### Component 1 — Physics Critic (`simulation/physics_critic.py`)

Loads a USD scene, runs 120 PhysX settle steps (2 seconds at 60 Hz), and returns a structured violation report using the verified ovphysx 0.4.9 API:

```
px = PhysX()
→  add_usd(usd_path)
→  step_n_sync(n=1, dt=1/60, current_time=i/60) × 120
→  get_contact_report()                          # penetration_pairs
→  create_tensor_binding(RIGID_BODY_POSE)        # floating_objects
→  create_tensor_binding(RIGID_BODY_VELOCITY)    # unstable_objects
→  failure_score F
```

#### Component 2 — LLM Agent (`agent/nemoclaw_client.py`)

Sends the structured physics report to a language model and parses the correction action. Probes backends in priority order with automatic fallback:

| Priority | Backend | Endpoint |
|----------|---------|----------|
| 1 | NemoClaw local | `http://127.0.0.1:8642/v1` |
| 2 | NVIDIA NIM cloud | `https://integrate.api.nvidia.com/v1` |
| 3 | Ollama local | `http://127.0.0.1:11434` |

#### Component 3 — Correction Engine (`correction/correction_engine.py`)

Orchestrates the RAR loop with rollback on F-worsening, per-iteration JSON logging, and CSV output. Action vocabulary is constrained to six operations:

```
snap_to_surface   resolve_penetration   recompute_hull
repair_manifold   stack_on              remove_object (last resort)
```

#### Component 4 — SAGE-10K Benchmark (`benchmark/sage_baseline.py`)

Extracts scene zips, converts layout JSON to USD, and runs the physics critic in batch. Reports initial F scores across N random scenes — the B1 baseline.

---

## Repository Structure

```
NemoForge/
├── src/
│   ├── simulation/
│   │   └── physics_critic.py       # Component 1 — ovphysx physics measurement
│   ├── agent/
│   │   └── nemoclaw_client.py      # Component 2 — NemoClaw / NIM / Ollama client
│   ├── correction/
│   │   ├── correction_engine.py    # Component 3 — RAR loop orchestrator
│   │   └── prompt_builder.py       # Physics-aware LLM prompt construction
│   ├── benchmark/
│   │   └── sage_baseline.py        # Component 4 — SAGE-10K evaluation pipeline
│   ├── api/
│   │   └── bridge.py               # FastAPI bridge (:8010)
│   └── utils/
│       ├── paths.py                # Path registry
│       ├── json_to_usd.py          # SAGE JSON → USD converter
│       └── verify_sage_structure.py
│
├── docs/
│   └── assets/
│       └── pipeline_architecture.png   # pipeline figure (see top of this file)
│
├── pyproject.toml
├── README.md
└── .gitignore

# ── NOT committed — install locally ──────────────────────────────
# dataset/SAGE-10k/           ~325 GB — download from Hugging Face
# NemoClaw/                   NVIDIA NemoClaw agent (optional)
# isaacsim/                   Isaac Sim source (optional)
# reference_repos/            Research reference clones (optional)
# results/                    CSV / JSON outputs (git-ignored)
```

---

## Installation

**Prerequisites:** Python 3.10+, NVIDIA GPU recommended

```bash
# 1. Clone
git clone https://github.com/kamranghz/NemoForge.git
cd NemoForge

# 2. Install Python dependencies
pip install -e ".[dev]"
pip install numpy "ovphysx==0.4.9" --extra-index-url https://pypi.nvidia.com

# 3. Set PYTHONPATH
export PYTHONPATH="$PWD/src"          # Linux / WSL / macOS
# $env:PYTHONPATH = "$PWD\src"        # Windows PowerShell

# 4. Set NVIDIA API key (enables NIM cloud inference; optional, see LLM Backend below)
export NVIDIA_API_KEY="nvapi-your-key-here"

# 5. Verify setup
python src/utils/paths.py
```

## Dataset Preparation

NemoForge doesn't ship scene data — it evaluates against NVIDIA's SAGE-10K synthetic-scene dataset. Required only if you're running the SAGE baseline (Component 4); the physics critic and correction loop work on any USD scene you already have.

```bash
pip install huggingface_hub
huggingface-cli login --token YOUR_HF_TOKEN
huggingface-cli download nvidia/SAGE-10k \
    --repo-type dataset \
    --local-dir dataset/SAGE-10k
```

Expected layout:

```
dataset/SAGE-10k/
├── scenes/     # *.zip archives — one per scene
└── kits/       # NVIDIA USD conversion scripts
```

## Model / LLM Backend Setup

NemoForge doesn't train or ship its own model weights — the reasoning step calls an existing LLM backend. Pick one:

```bash
# Option A — Ollama local (recommended for development)
ollama serve && ollama pull qwen3:14b

# Option B — NVIDIA NIM cloud
export NVIDIA_API_KEY="nvapi-..."

# Option C — NemoClaw local
git clone https://github.com/NVIDIA/NemoClaw.git NemoClaw
cd NemoClaw && npm install && npm start
```

The physics critic (Component 1) and the SAGE baseline's `--mode baseline` need none of this — only the correction loop's reasoning step does.

---

## Quick Start

### Measure physics violations

```python
from simulation.physics_critic import measure

report = measure("path/to/scene.usd")

print(f"Failure score F  : {report['failure_score']:.4f}")
print(f"Penetration pairs: {len(report['penetration_pairs'])}")
print(f"Floating objects : {len(report['floating_objects'])}")
print(f"Unstable objects : {len(report['unstable_objects'])}")
```

### Run the correction loop

```python
from correction.correction_engine import CorrectionEngine

engine = CorrectionEngine(
    usd_path="path/to/scene.usd",
    scene_id="scene_001",
    max_iter=10
)
result = engine.run()

print(f"Initial F : {result['initial_F']:.4f}")
print(f"Final F   : {result['final_F']:.4f}")
print(f"RFPCR     : {result['rfpcr']}")
print(f"Status    : {result['status']}")
```

See [Project Status](#project-status) below before relying on this loop's convergence behavior.

### Check LLM backend

```python
from agent.nemoclaw_client import NemoClawClient

client = NemoClawClient()
ok, backend = client.is_available()
print(f"Backend: {backend} | Available: {ok}")
```

---

## Running the Pipeline

> Always set `PYTHONPATH="$PWD/src"` before running any script.

### Physics measurement — single scene

```bash
python src/simulation/physics_critic.py path/to/scene.usd
```

```
=======================================================
  PHYSICS MEASUREMENT REPORT
=======================================================
  Objects found     : 59
  Penetration pairs : 4
  Floating objects  : 54
  Unstable objects  : 0
  Failure score F   : 88.5200
=======================================================
```

### RAR correction — single scene

```bash
python src/correction/correction_engine.py \
    --usd path/to/scene.usd \
    --scene-id scene_001 \
    --max-iter 10
```

### SAGE-10K baseline

```bash
# Physics baseline — no LLM required
python src/benchmark/sage_baseline.py --mode baseline --n-scenes 10 --seed 42

# Correction baseline — LLM required
python src/benchmark/sage_baseline.py --mode correction --n-scenes 20 --seed 42
```

### API bridge

```bash
python src/api/bridge.py
# or
uvicorn api.bridge:app --host 127.0.0.1 --port 8010 --reload
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness + active LLM backend |
| `POST` | `/correct` | Physics report → correction action |
| `POST` | `/generate` | Raw LLM call (legacy) |

## Expected Outputs

| File | Contents |
|------|---------|
| `results/sage_baseline.csv` | Per-scene initial F, violation counts |
| `results/correction_results.csv` | Initial F, final F, RFPCR, iterations |
| `results/correction_logs/<id>.json` | Per-iteration action, delta-F, rollback status |

`results/` is git-ignored; every run regenerates it locally.

---

## Project Status

Component 1 (Physics Critic) is implemented and verified against a real ovphysx 0.4.9 run — the `PHYSICS MEASUREMENT REPORT` example above is an actual measured baseline, not illustrative. Components 2 and 4 (LLM Agent, SAGE-10K baseline) are functional. Component 3 (Correction Engine) is under active development — geometry-correction actions and the SSIM half of RFPCR are being finalized, so treat the RAR loop's convergence numbers as in-progress rather than final until this note is updated.

## Troubleshooting

**`ModuleNotFoundError: No module named 'simulation'`**
```bash
export PYTHONPATH="$PWD/src"
```

**`ModuleNotFoundError: No module named 'ovphysx'`**
```bash
pip install "ovphysx==0.4.9" --extra-index-url https://pypi.nvidia.com
```

**`RuntimeError: No LLM backend available`**
```bash
ollama serve && ollama pull qwen3:14b    # local
export NVIDIA_API_KEY="nvapi-..."        # cloud
```

**`No scenes found in dataset/SAGE-10k/scenes`**

Download the dataset — see [Dataset Preparation](#dataset-preparation).

**`PhysX warning: cuCtxGetDevice failed`**

ovphysx falls back to CPU simulation automatically. Normal in WSL2. Results remain valid.

**`Could not import SimulationApp` / `No module named 'omni'`**

The core pipeline does not require Isaac Sim. Use `correction_engine.py` and `sage_baseline.py` directly.

**`TypeError: step_n_sync() missing argument 'current_time'`**

`current_time` is required in ovphysx 0.4.9:
```python
for i in range(120):
    px.step_n_sync(n=1, dt=1.0/60, current_time=i/60.0)
```

---

## Citation

The paper for this work is not yet public. If you'd like to reference the system in the meantime, cite the thesis it's drawn from:

```bibtex
@mastersthesis{gholizadeh2026nemoforge,
  title  = {Post-Reconstruction Physical Rehabilitation of Neural 3D Scenes via
            Agentic Simulator-in-the-Loop Reasoning with Learned Convergence},
  author = {Gholizadeh HamlAbadi, Kamran},
  school = {University of Ottawa},
  year   = {2026},
  note   = {NemoForge system: \url{https://github.com/kamranghz/NemoForge}}
}
```

This will be replaced with the paper citation once it's available.

## License

Released under the [MIT License](LICENSE). *(Suggested default, matching this author's other public repositories — confirm before publishing if you want something else.)*

## Contact

Kamran Gholizadeh HamlAbadi — MCRLab, University of Ottawa — `kghol072@uottawa.ca`

For bugs or questions, please open a [GitHub issue](https://github.com/kamranghz/NemoForge/issues).

---

<div align="center">
<sub>NemoForge · ovphysx 0.4.9 · NemoClaw · NVIDIA NIM · FastAPI · Python 3.10+</sub>
</div>
