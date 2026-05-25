"""
NemoForge — centralised path registry.

All project scripts import from here instead of recomputing paths inline.
Only this file needs to be updated when the repo is moved or renamed.

Usage:
    from nemoforge.utils.paths import REPO_ROOT, EXTENSION_DIR, DATASET_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root detection
# Resolve from this file's location:
#   src/nemoforge/utils/paths.py -> parents[3] = NemoForge/
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Core project directories
# ---------------------------------------------------------------------------
SRC_DIR: Path = REPO_ROOT / "src"

RESULTS_DIR: Path = REPO_ROOT / "results"
CONFIG_DIR: Path = REPO_ROOT / "config"
DOCS_DIR: Path = REPO_ROOT / "docs"
SCRIPTS_DIR: Path = REPO_ROOT / "scripts"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_ROOT: Path = REPO_ROOT / "dataset"
SAGE_10K_ROOT: Path = DATASET_ROOT / "SAGE-10K"
SAGE_SCENES_DIR: Path = SAGE_10K_ROOT / "scenes"
SAGE_KITS_DIR: Path = SAGE_10K_ROOT / "kits"

# ---------------------------------------------------------------------------
# Benchmark task files
# ---------------------------------------------------------------------------
NF_CORE_50_JSON: Path = REPO_ROOT / "src" / "nemoforge" / "core" / "nf_core_50.json"

# ---------------------------------------------------------------------------
# FastAPI bridge  (kept at repo root per project design decision)
# ---------------------------------------------------------------------------
BACKEND_DIR: Path = REPO_ROOT / "isaac-sim-backend"

# ---------------------------------------------------------------------------
# NemoClaw (standalone sub-project — do NOT move contents)
# ---------------------------------------------------------------------------
NEMOCLAW_DIR: Path = REPO_ROOT / "NemoClaw"

# ---------------------------------------------------------------------------
# Reference repos  (third-party clones — do NOT move)
# ---------------------------------------------------------------------------
REFERENCE_REPOS_DIR: Path = REPO_ROOT / "reference_repos"

# ---------------------------------------------------------------------------
# Isaac Sim (cloned source tree — do NOT move)
# ---------------------------------------------------------------------------
ISAACSIM_SOURCE: Path = REPO_ROOT / "isaacsim"
ISAACSIM_BUILD: Path = ISAACSIM_SOURCE / "_build" / "windows-x86_64" / "release"
ISAACSIM_PYTHON_BAT: Path = ISAACSIM_BUILD / "python.bat"
ISAACSIM_EXE: Path = ISAACSIM_BUILD / "isaac-sim.bat"

# NemoClaw connector extension (Python modules live here inside the build tree)
EXTENSION_DIR: Path = (
    ISAACSIM_SOURCE
    / "source"
    / "extensions"
    / "isaacsim.nemoclaw.connector"
    / "isaacsim"
    / "nemoclaw"
    / "connector"
)


# ---------------------------------------------------------------------------
# sys.path helper — call from any entry point that needs backend + extension
# ---------------------------------------------------------------------------
def register_project_paths() -> None:
    """
    Prepend BACKEND_DIR and EXTENSION_DIR to sys.path so that
    `import extension`, `import prompt_builder`, etc. resolve correctly.

    Safe to call multiple times (idempotent).
    """
    for p in (BACKEND_DIR, EXTENSION_DIR):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    # Also add src/ so `from nemoforge.xxx import yyy` works when the package
    # is not installed via pip (editable install covers this automatically).
    s_src = str(SRC_DIR)
    if s_src not in sys.path:
        sys.path.insert(0, s_src)


if __name__ == "__main__":
    # Quick sanity print — run with any Python to verify paths.
    width = 28
    print("NemoForge path registry")
    print("=" * 50)
    items = [
        ("REPO_ROOT", REPO_ROOT),
        ("BACKEND_DIR", BACKEND_DIR),
        ("EXTENSION_DIR", EXTENSION_DIR),
        ("ISAACSIM_PYTHON_BAT", ISAACSIM_PYTHON_BAT),
        ("DATASET_ROOT", DATASET_ROOT),
        ("SAGE_10K_ROOT", SAGE_10K_ROOT),
        ("NF_CORE_50_JSON", NF_CORE_50_JSON),
    ]
    for name, path in items:
        exists = "OK" if path.exists() else "MISSING"
        print(f"  {name:<{width}} [{exists}]  {path}")
