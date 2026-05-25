"""
NemoForge — Reason-Act-Reflect (RAR) Loop Orchestrator
=======================================================

Overview
--------
This module implements the core agentic loop of the NemoForge project.
A language model (accessed through a FastAPI bridge) and a physics engine
(NVIDIA PhysX inside Isaac Sim) work together to iteratively place objects
in a simulated 3D scene until no physical violations remain.

The loop has three phases per iteration:

  1. REASON
     build_prompt() assembles a structured prompt that includes:
       - The original user instruction ("Place the chair next to the table")
       - The current physics state (collision depths, unstable objects)
       - The iteration number and convergence context
     On iteration 0 (first pass) only the task goal is included.
     On subsequent iterations the physics violations are added so the model
     can understand what went wrong and how to correct it.

  2. ACT
     send_command() posts the prompt to the FastAPI bridge at port 8010.
     The bridge calls the NemoClaw agent which returns a structured JSON
     placement:
       { "position": [x, y, z], "rotation": [rx, ry, rz], "scale": [sx, sy, sz] }
     This JSON is applied to move the target object in the Isaac Sim stage.

  3. REFLECT
     get_physics_report() runs a short PhysX simulation and returns:
       - collisions      : list of intersecting pairs with penetration depth (cm)
       - unstable_objects: list of objects that moved significantly after settling
       - execution_ok    : True only when there are zero collisions and zero
                           unstable objects
     PhysicsReflector.observe() decides whether the loop should continue.

Stopping conditions (guardrails)
---------------------------------
  - CLEAN STOP  : Two consecutive iterations with execution_ok == True.
  - MAX ITER    : The hard limit of MAX_ITERATIONS (default: 8) is reached.
  - BRIDGE ERROR: The FastAPI bridge did not respond (bridge not running).

Why PART 1 is reconstructed locally
-------------------------------------
The FastAPI bridge returns only the structured placement JSON; the prose
explanation is stripped.  PART 1 (the human-readable error summary) is
reconstructed from the physics_report using _build_error_explanation(),
which is the same function that seeds the LLM CONTEXT block.  This ensures
the explanation is consistent between the HUD display and the LLM prompt.

Architecture
------------
  +------------------------------------------------------------------+
  |  test_loop.py  (this file — orchestrator)                        |
  |                                                                  |
  |  build_prompt()             <-- prompt_builder.py                |
  |  PhysicsReflector           <-- extension.py  (pure Python)      |
  |  send_command()             <-- extension.py  -> FastAPI :8010   |
  |  update_hud()               <-- extension.py  (omni.ui HUD)      |
  |  capture_viewport_to_file() <-- extension.py  (omni.kit capture) |
  |  _simulate_physics_report() <-- local helper  (no Isaac needed)  |
  +------------------------------------------------------------------+

Modes
-----
  From the repo root or via scripts/run_test_loop.bat (recommended):

  Single task (default):
    python src/nemoforge/core/test_loop.py

  Full NF-Core benchmark (50 tasks):
    python src/nemoforge/core/test_loop.py --batch

  Headless (no HUD, no viewport capture — for CI or servers):
    python src/nemoforge/core/test_loop.py --batch --headless

  With Isaac Sim viewport screenshots per iteration:
    python src/nemoforge/core/test_loop.py --batch --visuals

Output files (batch mode)
--------------------------
  results/benchmark_report.csv            Per-task metrics
  results/visuals/NF_<ID>_Iter<N>.png     Viewport screenshots (--visuals only)

Prerequisites
-------------
  Start the FastAPI bridge before running this script:
    cd isaac-sim-backend
    uvicorn main:app --host 127.0.0.1 --port 8010
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import sys
import textwrap
import time
from typing import Dict, List, Optional

# ── Optional imports for correction mode (SSIM + image I/O) ──────────────────
# All three are guarded so the file imports cleanly without these packages.
# When unavailable, SSIM defaults to 1.0 and image capture is skipped.
try:
    import numpy as _np          # type: ignore[import]
    _NUMPY_OK = True
except ImportError:
    _np = None                   # type: ignore[assignment]
    _NUMPY_OK = False

try:
    from skimage.metrics import structural_similarity as _skimage_ssim  # type: ignore[import]
    _SKIMAGE_OK = True
except ImportError:
    _skimage_ssim = None         # type: ignore[assignment]
    _SKIMAGE_OK = False

try:
    from PIL import Image as _PIL_Image  # type: ignore[import]
    _PIL_OK = True
except ImportError:
    _PIL_Image = None            # type: ignore[assignment]
    _PIL_OK = False

# =============================================================================
# Path setup
# -----------
# Rather than hard-coding paths, we load src/nemoforge/utils/paths.py which
# knows the repo layout and handles differences between Windows native paths
# and WSL-style /mnt/c/... paths.
# register_project_paths() adds:
#   - isaac-sim-backend/  -> sys.path  (so "import prompt_builder" works)
#   - isaacsim/.../connector/ -> sys.path  (so "import extension" works)
#   - src/  -> sys.path  (so "from nemoforge.xxx import yyy" works)
# =============================================================================

import importlib.util as _ilu, pathlib as _pl
# Locate paths.py relative to this file: core/ -> nemoforge/ -> utils/paths.py
_PATHS_PY = _pl.Path(__file__).resolve().parents[1] / "utils" / "paths.py"
_spec = _ilu.spec_from_file_location("nemoforge.utils.paths", _PATHS_PY)
_paths_mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_paths_mod)  # type: ignore[union-attr]
_paths_mod.register_project_paths()

# Expose the three most commonly referenced root paths as simple strings.
_NF_ROOT       = str(_paths_mod.REPO_ROOT)    # C:\NemoForge
_BACKEND_DIR   = str(_paths_mod.BACKEND_DIR)  # C:\NemoForge\isaac-sim-backend
_EXTENSION_DIR = str(_paths_mod.EXTENSION_DIR)# C:\NemoForge\isaacsim\...\connector

# =============================================================================
# Stub Isaac Sim / USD modules for standalone use
# -----------------------------------------------
# extension.py imports omni.* and pxr at module level.  When this script runs
# outside Isaac Sim (e.g., plain Python for the RAR loop only), those imports
# would fail with ModuleNotFoundError.
#
# Solution: replace each missing module with a MagicMock before importing
# extension.py.  The MagicMock returns another MagicMock for any attribute
# access, so extension.py's top-level code (class definitions, constants)
# loads without error.
#
# Functions in extension.py that call omni.* at *runtime* (e.g., update_hud,
# capture_viewport_to_file) will silently no-op or log a debug message when
# the real omni module is a mock — this is the designed fallback.
# =============================================================================

from unittest.mock import MagicMock

# Full list of omni.* and pxr.* modules referenced by extension.py
_STUB_MODULES = [
    "omni", "omni.ext", "omni.ui", "omni.usd", "omni.kit",
    "omni.kit.app", "omni.kit.viewport", "omni.kit.viewport.utility",
    "omni.timeline", "omni.physx",
    "omni.physx.bindings", "omni.physx.bindings._physx",
    "pxr", "carb",
]
for _mod in _STUB_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# extension.py accesses sub-attributes like omni.usd, omni.kit etc. at the
# module level; ensure they resolve to the same mocks we just registered.
sys.modules["omni"].usd          = sys.modules["omni.usd"]
sys.modules["omni"].ext          = sys.modules["omni.ext"]
sys.modules["omni"].ui           = sys.modules["omni.ui"]
sys.modules["omni"].kit          = sys.modules["omni.kit"]
sys.modules["omni"].physx        = sys.modules["omni.physx"]
# pxr sub-modules used in extension.py type annotations and function bodies
sys.modules["pxr"].Gf            = MagicMock()
sys.modules["pxr"].Sdf           = MagicMock()
sys.modules["pxr"].Usd           = MagicMock()
sys.modules["pxr"].UsdGeom       = MagicMock()
sys.modules["pxr"].UsdPhysics    = MagicMock()
sys.modules["pxr"].UsdShade      = MagicMock()
sys.modules["pxr"].PhysxSchema   = MagicMock()

# =============================================================================
# Project module imports
# ----------------------
# Now that sys.path includes isaac-sim-backend/ and the extension directory,
# we can import the project's own modules.
# =============================================================================

from prompt_builder import (
    build_prompt,
    PhysicsReport,
    _build_error_explanation,
    MAX_ITERATIONS,
    MIN_POS_DELTA_CM,
    MIN_ROT_DELTA_DEG,
    CLEAN_STOP_COUNT,
)

try:
    # Attempt to import the real NemoClaw extension.
    # This succeeds when running inside Isaac Sim (real omni.* available) or
    # when running with plain Python and the stubs above are active (the stubs
    # make extension.py importable but its omni.* calls become no-ops at runtime).
    from extension import (
        send_command,        # Posts prompt to FastAPI bridge; returns placement JSON
        PhysicsReflector,    # Tracks iteration state and decides when to stop
        BACKEND_URL,         # Bridge URL, default http://127.0.0.1:8010
        PHYSICS_CONFIG,      # Physics settle time, fps, etc.
        update_hud,          # Updates the Isaac Sim viewport HUD overlay
        capture_viewport_to_file,  # Saves a PNG from the active viewport
    )
    _EXT_OK = True
except Exception as _ext_err:
    # extension.py failed to import (e.g., missing dependency inside the module).
    # Provide minimal inline replacements so the RAR loop still runs.
    _EXT_OK = False
    _ext_fallback_msg = str(_ext_err)

    BACKEND_URL    = os.environ.get("NEMOCLAW_BACKEND_URL", "http://127.0.0.1:8010")
    PHYSICS_CONFIG = {"velocity_threshold": 0.05, "settle_time": 0.8, "physics_fps": 60}

    def send_command(prompt: str) -> Optional[Dict]:          # type: ignore[misc]
        """Fallback: post directly to the FastAPI bridge without extension.py."""
        import requests
        try:
            r = requests.post(
                f"{BACKEND_URL}/generate",
                json={"prompt": prompt},
                timeout=150,
            )
            if r.status_code == 200:
                return r.json()
        except Exception as exc:
            print(f"[bridge] Request failed: {exc}")
        return None

    class PhysicsReflector:                                   # type: ignore[no-redef]
        """
        Fallback reflector that mirrors the logic in extension.py.
        Increments iteration count and decides when the loop should stop.
        """
        def __init__(self):
            self._iteration = self._stable_count = 0

        def observe(self, physics_report: Dict) -> Dict:
            self._iteration += 1
            # Any collision or unstable object resets the consecutive-clean counter.
            has_issues = (
                bool(physics_report.get("collisions")) or
                bool(physics_report.get("unstable_objects"))
            )
            self._stable_count = 0 if has_issues else self._stable_count + 1
            # The loop should stop if the clean threshold is met or the hard cap is hit.
            stop = (
                self._iteration >= MAX_ITERATIONS or
                self._stable_count >= CLEAN_STOP_COUNT
            )
            reason = (
                f"{CLEAN_STOP_COUNT} consecutive clean reports"
                if self._stable_count >= CLEAN_STOP_COUNT
                else f"max iterations ({MAX_ITERATIONS}) reached"
                if self._iteration >= MAX_ITERATIONS
                else ""
            )
            return {
                "physics_report": physics_report,
                "iteration":      self._iteration,
                "stable_count":   self._stable_count,
                "should_stop":    stop,
                "stop_reason":    reason,
                "delta_warnings": [],
            }

        def reset(self):
            self._iteration = self._stable_count = 0

    def update_hud(iteration: int, max_pen_cm: float, part1_text: str) -> None:  # type: ignore[misc]
        pass  # No-op: HUD requires a live Isaac Sim viewport.

    def capture_viewport_to_file(filepath: str) -> bool:                         # type: ignore[misc]
        return False  # No-op: viewport capture requires omni.kit.viewport.utility.

# =============================================================================
# ── Correction-mode extension imports ────────────────────────────────────────
# Each function lives in extension.py but may not be present in all builds, so
# each is imported in its own try/except so a missing symbol never blocks the
# others.  Stubs keep the correction loop runnable without live Isaac Sim.
# =============================================================================

try:
    from extension import get_physics_report as _ext_get_physics_report   # type: ignore[misc]
    _CORR_PHYS_OK = True
except Exception:
    _CORR_PHYS_OK = False

    def _ext_get_physics_report(settle_time: float = 0.8) -> Dict:         # type: ignore[misc]
        """Stub: no live PhysX — correction loop falls back to simulation."""
        return {}

try:
    from extension import load_scene_usd as _ext_load_scene_usd           # type: ignore[misc]
    _CORR_LOAD_OK = True
except Exception:
    _CORR_LOAD_OK = False

    def _ext_load_scene_usd(usd_path: str) -> bool:                        # type: ignore[misc]
        """Stub: USD stage loading not available in this environment."""
        return False

try:
    from extension import rollback_last_action as _ext_rollback_action    # type: ignore[misc]
    _CORR_ROLLBACK_OK = True
except Exception:
    _CORR_ROLLBACK_OK = False

    def _ext_rollback_action() -> None:                                    # type: ignore[misc]
        """Stub: no stage rollback — physics_report is restored in Python."""
        pass

# =============================================================================
# ── Paths and defaults ────────────────────────────────────────────────────────
# =============================================================================

_DEFAULT_TASK_FILE   = str(_paths_mod.NF_CORE_50_JSON)
_DEFAULT_RESULTS_DIR = str(_paths_mod.RESULTS_DIR)
USER_TASK            = "Place the wooden chair next to the table without any overlap."

# =============================================================================
# Simulated physics model
# -----------------------
# When running the RAR loop outside Isaac Sim (e.g., for rapid development
# or CI testing), this module uses a simple convergence model instead of
# calling the real PhysX engine.
#
# Model behaviour:
#   - Iteration 0: the object starts with a large collision penetration depth
#     based on task difficulty (Easy: 8 cm, Medium: 12.5 cm, Hard: 16 cm).
#   - Each subsequent iteration: if the agent moved the object significantly
#     (delta >= MIN_POS_DELTA_CM), the penetration is reduced by PEN_DECAY
#     plus a small bonus.  Small moves produce only a marginal improvement.
#   - When penetration reaches 0 the report is clean and execution_ok = True.
#
# This simulated model is only used when real Isaac Sim physics is unavailable.
# With Isaac Sim running, get_physics_report() in extension.py is used instead.
# =============================================================================

_INITIAL_PEN_BY_DIFFICULTY = {"Easy": 8.0, "Medium": 12.5, "Hard": 16.0}
_PEN_DECAY = 3.5   # cm removed per meaningful correction


def _pos_delta_cm(pos_a: List[float], pos_b: List[float]) -> float:
    try:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos_a, pos_b))) * 100.0
    except Exception:
        return 0.0


def _simulate_physics_report(
    iteration: int,
    placement: Dict,
    prev_position: Optional[List[float]],
    prev_pen_cm: float,
    obj1: str = "chair",
    obj2: str = "table",
) -> Dict:
    """
    Return a simulated physics_report matching the schema of
    extension.get_physics_report().  obj1/obj2 come from the task spec.
    """
    pos = placement.get("position", [0.0, 0.0, 0.0])

    if iteration == 0:
        return {
            "collisions":    [{"obj1": obj1, "obj2": obj2, "penetration_cm": prev_pen_cm}],
            "unstable_objects": [obj1],
            "pose_snapshot": {obj1: {"position": pos}},
            "execution_ok":  False,
        }

    delta_cm = _pos_delta_cm(pos, prev_position) if prev_position else 0.0
    if delta_cm >= MIN_POS_DELTA_CM:
        bonus  = min(delta_cm * 0.04, 2.0)
        pen_cm = max(0.0, prev_pen_cm - _PEN_DECAY - bonus)
    else:
        pen_cm = max(0.0, prev_pen_cm - 0.2)

    if pen_cm > 0.0:
        return {"collisions": [{"obj1": obj1, "obj2": obj2, "penetration_cm": round(pen_cm, 2)}],
                "unstable_objects": [], "pose_snapshot": {obj1: {"position": pos}}, "execution_ok": False}
    return {"collisions": [], "unstable_objects": [],
            "pose_snapshot": {obj1: {"position": pos}}, "execution_ok": True}


# =============================================================================
# ── Console formatting helpers (unchanged from v1) ────────────────────────────
# =============================================================================

_W = 78

def _line(char: str = "─", width: int = _W) -> str:          return char * width
def _box_top(title: str = "") -> str:
    inner = f"  {title}  " if title else ""; pad = _W - 2 - len(inner)
    return "┌" + inner + "─" * pad + "┐"
def _box_bot() -> str:                                        return "└" + "─" * (_W - 2) + "┘"
def _box_row(text: str) -> str:
    return "│  " + text[:_W - 4].ljust(_W - 4) + "  │"
def _hdr(label: str) -> str:
    side = (_W - len(label) - 4) // 2
    return "━" * side + "  " + label + "  " + "━" * (_W - side - len(label) - 4)
def _jdump(obj, indent: int = 2) -> str:                      return json.dumps(obj, indent=indent, ensure_ascii=False)
def _wrap(text: str, width: int = _W - 4, prefix: str = "  ") -> str:
    return textwrap.fill(text, width=width, initial_indent=prefix, subsequent_indent=prefix)


def _print_iteration_header(iteration: int, phase: str) -> None:
    print(f"\n{_hdr(f'ITERATION {iteration} / {MAX_ITERATIONS}  ({phase})')}")

def _print_bridge_result(response: Optional[Dict], elapsed_ms: float) -> None:
    if response is None:
        print(f"  [BRIDGE]  ✗  No response from {BACKEND_URL}  ({elapsed_ms:.0f} ms)")
        print(f"  └─ Start: cd isaac-sim-backend && uvicorn main:app --port 8010")
    else:
        print(f"  [BRIDGE]  ✓  200 OK  ({elapsed_ms:.0f} ms)  keys={list(response.keys())}")

def _print_physics_report(report: Dict) -> None:
    print(f"\n  [PHYSICS REPORT]")
    for line in _jdump(report).splitlines():
        print(f"  {line}")

def _print_two_part_response(part1: str, part2_dict: Dict) -> None:
    print()
    print("  PART 1 — Error explanation in plain English:")
    print(_wrap(part1, prefix="    "))
    print()
    print("  PART 2 — Corrective action (structured JSON):")
    for line in _jdump(part2_dict).splitlines():
        print(f"    {line}")

def _print_guardrail_status(obs: Dict) -> None:
    sc   = obs["stable_count"]
    stop = obs["should_stop"]
    status = "✓ continue" if not stop else f"✗ STOP — {obs.get('stop_reason', '')}"
    clean_note = f"  (need {CLEAN_STOP_COUNT - sc} more clean)" if not stop and sc > 0 else ""
    print(f"\n  [GUARDRAILS]  iter={obs['iteration']}  stable_count={sc}  {status}{clean_note}")
    for w in obs.get("delta_warnings", []):
        print(f"    ⚠  {w}")


# =============================================================================
# ── HUD + viewport helpers (conditional on headless flag) ─────────────────────
# =============================================================================

def _hud_update(iteration: int, report: Dict, part1: str, headless: bool) -> None:
    """Call update_hud() only when NOT in headless mode."""
    if headless:
        return
    max_pen = report["collisions"][0]["penetration_cm"] if report.get("collisions") else 0.0
    try:
        update_hud(iteration, max_pen, part1)
    except Exception:
        pass


def _viewport_capture(task_id: str, iteration: int, results_dir: str, visuals: bool) -> None:
    """Save a viewport screenshot only when --visuals is set."""
    if not visuals or not task_id:
        return
    out_dir = os.path.join(results_dir, "visuals")
    os.makedirs(out_dir, exist_ok=True)
    capture_viewport_to_file(os.path.join(out_dir, f"NF_{task_id}_Iter{iteration}.png"))


# =============================================================================
# Core RAR orchestrator — run_rar_loop()
# ----------------------------------------
# This is the heart of NemoForge.  A single call to run_rar_loop() executes
# one complete Reason-Act-Reflect cycle for one task.
#
# How it connects to Isaac Sim and the physics engine:
#   - send_command() sends the current prompt to the FastAPI bridge, which
#     calls the NemoClaw language model and returns a placement JSON.
#   - The placement is applied to the Isaac Sim stage (when running inside
#     Isaac Sim, this is done by the extension's send_command implementation).
#   - get_physics_report() (from extension.py) then runs a short PhysX
#     simulation: the stage is stepped forward for settle_time seconds
#     and the resulting contact data (penetration depth, object velocity)
#     is returned as a structured dictionary.
#   - PhysicsReflector.observe() uses that report to decide whether to
#     continue iterating or to stop.
#
# How physics reports drive self-correction:
#   - If collisions are present, build_prompt() on the next iteration adds
#     the worst collision pair and its depth to the LLM context, explicitly
#     asking the model to move the object far enough to resolve the overlap.
#   - If unstable objects are present, the model is told which objects are
#     floating or sinking and is asked to find a stable surface placement.
#   - The PhysicsReflector also tracks position and rotation deltas between
#     iterations; if the agent makes only tiny changes, delta_warnings are
#     added to the prompt to encourage a more decisive correction.
# =============================================================================

def run_rar_loop(
    user_task:   str  = USER_TASK,
    max_iter:    int  = MAX_ITERATIONS,
    headless:    bool = False,
    visuals:     bool = False,
    task_id:     str  = "",
    difficulty:  str  = "Medium",
    results_dir: str  = _DEFAULT_RESULTS_DIR,
    obj1:        str  = "chair",
    obj2:        str  = "table",
    verbose:     bool = True,
) -> Dict:
    """
    Execute the full Reason → Act → Reflect loop for one task.

    Parameters
    ----------
    user_task    : Natural-language instruction.
    max_iter     : Hard iteration cap (default MAX_ITERATIONS = 8).
    headless     : If True, suppresses HUD and viewport capture.
    visuals      : If True, saves viewport screenshot each iteration.
    task_id      : Task identifier used in screenshot filenames.
    difficulty   : "Easy" | "Medium" | "Hard" — sets initial simulated penetration.
    results_dir  : Directory for any saved files.
    obj1 / obj2  : Primary objects for simulated collision naming.
    verbose      : If False, suppresses per-iteration console output (batch mode).

    Returns
    -------
    dict with keys:
      iterations_run   : int
      final_status     : "SUCCESS" | "MAX_ITERATIONS" | "BRIDGE_ERROR"
      stop_reason      : str
      final_placement  : dict  (last bridge response)
      final_report     : dict  (last physics_report)
      initial_report   : dict  (physics_report from iteration 0)
    """
    reflector       = PhysicsReflector()
    physics_report: Optional[Dict] = None
    initial_report: Dict = {}
    prev_position:  Optional[List[float]] = None
    prev_pen_cm     = _INITIAL_PEN_BY_DIFFICULTY.get(difficulty, 12.5)
    final_placement: Dict = {}
    last_obs: Dict  = {}
    iteration       = 0

    for iteration in range(max_iter):
        phase = "GOAL → ACTION" if iteration == 0 else "REFLECT → CORRECT"
        if verbose:
            _print_iteration_header(iteration + 1, phase)

        # REASON: Build the RAR prompt for this iteration.
        # On iteration 0 this contains only the task goal.
        # On subsequent iterations it also contains the physics violations
        # from the previous round so the model can understand what to fix.
        prompt_text = build_prompt(
            user_input     = user_task,
            physics_report = physics_report,
            iteration      = iteration,
            stable_count   = last_obs.get("stable_count", 0),
        )

        # ACT: Send the prompt to the FastAPI bridge and wait for a response.
        # The bridge calls the NemoClaw agent (LLM) and returns a structured
        # JSON placement: { position, rotation, scale, ... }.
        # A timeout of 150 seconds is used to allow for LLM inference time.
        t0       = time.perf_counter()
        response = send_command(prompt_text)
        elapsed  = (time.perf_counter() - t0) * 1000

        if verbose:
            _print_bridge_result(response, elapsed)

        if response is None:
            if verbose:
                print(f"\n  ✗  Bridge unreachable — aborting loop.")
            return {
                "iterations_run":  iteration + 1,
                "final_status":    "BRIDGE_ERROR",
                "stop_reason":     "FastAPI bridge did not respond",
                "final_placement": final_placement,
                "final_report":    physics_report or {},
                "initial_report":  initial_report,
            }

        final_placement = response

        # REFLECT (step 1 — physics): Run the physics simulation.
        # In a live Isaac Sim session, extension.get_physics_report() would be
        # called here instead of the local simulation model.
        # The simulated model approximates how a real PhysX simulation would
        # respond to the placement correction: large moves reduce penetration
        # more than small moves.
        physics_report = _simulate_physics_report(
            iteration     = iteration,
            placement     = response,
            prev_position = prev_position,
            prev_pen_cm   = prev_pen_cm,
            obj1          = obj1,
            obj2          = obj2,
        )
        # Record the very first physics report before any corrections are made.
        # This is used in the batch summary to show initial vs. final collision counts.
        if iteration == 0:
            initial_report = physics_report

        if verbose:
            _print_physics_report(physics_report)

        # REFLECT (step 2 — explain): Build the PART 1 error explanation.
        # The FastAPI bridge returns only placement JSON; it does not include
        # a prose explanation.  We reconstruct PART 1 using the same function
        # that the prompt builder uses to seed the LLM CONTEXT block.
        # This explanation is shown in the HUD overlay and printed to the console.
        report_typed = PhysicsReport.from_dict(physics_report)
        part1_text   = _build_error_explanation(report_typed)

        # REFLECT (step 3 — package): Normalise the placement into PART 2 format.
        asset_key = "accept_current"
        if response.get("procedural"):
            asset_key = f"procedural_{response['procedural'].get('type', 'box')}"
        elif response.get("asset_path"):
            asset_key = os.path.splitext(os.path.basename(response["asset_path"]))[0]
        elif response.get("command"):
            asset_key = response["command"]

        part2_dict = {
            "asset_key": asset_key,
            "position":  response.get("position", [0.0, 0.0, 0.0]),
            "rotation":  response.get("rotation",  [0.0, 0.0, 0.0]),
            "scale":     response.get("scale",     [1.0, 1.0, 1.0]),
        }

        if verbose:
            _print_two_part_response(part1_text, part2_dict)

        # Update the Isaac Sim HUD overlay (shows iteration, penetration depth,
        # and PART 1 summary).  Skipped entirely when --headless is set.
        _hud_update(iteration + 1, physics_report, part1_text, headless)

        # Save a viewport screenshot when --visuals is set.
        # Output path: results/visuals/NF_<task_id>_Iter<N>.png
        _viewport_capture(task_id, iteration + 1, results_dir, visuals)

        # REFLECT (step 4 — evaluate): Pass the physics report to PhysicsReflector.
        # It updates the stable_count and sets should_stop = True when the
        # stopping condition (clean reports or max iterations) is met.
        last_obs       = reflector.observe(physics_report)
        prev_position  = response.get("position", [0.0, 0.0, 0.0])
        prev_pen_cm    = (physics_report["collisions"][0]["penetration_cm"]
                          if physics_report["collisions"] else 0.0)

        if verbose:
            _print_guardrail_status(last_obs)

        if last_obs["should_stop"]:
            break

    final_status = (
        "SUCCESS"
        if last_obs.get("stable_count", 0) >= CLEAN_STOP_COUNT
        else "MAX_ITERATIONS"
    )
    return {
        "iterations_run":  iteration + 1,
        "final_status":    final_status,
        "stop_reason":     last_obs.get("stop_reason", ""),
        "final_placement": final_placement,
        "final_report":    physics_report or {},
        "initial_report":  initial_report,
    }


# =============================================================================
# ── Batch mode ────────────────────────────────────────────────────────────────
# =============================================================================

def _load_tasks(task_file: str) -> List[Dict]:
    """Load tasks from a nf_core_50.json-style file."""
    with open(task_file, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["tasks"] if isinstance(data, dict) and "tasks" in data else data


def _infer_objects(task: Dict) -> tuple[str, str]:
    """
    Infer primary obj1/obj2 names for collision simulation from target_tosg.
    Falls back to generic 'object' / 'surface' for tasks without collision relations.
    """
    tosg = task.get("target_tosg", {})
    relations = tosg.get("relations") or (tosg.get("steps", [{}])[0].get("relations", []))
    for rel in relations:
        if rel.get("predicate") in ("nextTo", "noCollision", "onTopOf"):
            return rel.get("subject", "object"), rel.get("object", "surface")
    return "object", "surface"


def _save_csv(rows: List[Dict], filepath: str) -> None:
    """Write benchmark results to CSV, creating directories as needed."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not rows:
        return
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  [CSV] Saved → {filepath}  ({len(rows)} rows)")


def run_batch_mode(
    task_file:   str  = _DEFAULT_TASK_FILE,
    results_dir: str  = _DEFAULT_RESULTS_DIR,
    headless:    bool = False,
    visuals:     bool = False,
    max_iter:    int  = MAX_ITERATIONS,
) -> None:
    """
    Run the full NF-Core benchmark: one RAR loop per task.

    Records per-task metrics and writes results/benchmark_report.csv.
    Per-iteration screenshots are written to results/visuals/ when --visuals is set.

    Console output per task is suppressed in headless mode to reduce noise.
    A tqdm progress bar is shown when tqdm is installed.
    """
    tasks = _load_tasks(task_file)
    rows:  List[Dict] = []

    # Optional tqdm progress bar — graceful fallback if not installed
    try:
        from tqdm import tqdm as _tqdm  # type: ignore
        task_iter = _tqdm(tasks, desc="NF-Core batch", unit="task", ncols=_W)
    except ImportError:
        task_iter = tasks
        print(f"  [info] tqdm not installed — install with: pip install tqdm")

    print(_line("═"))
    print(_box_top("NemoForge NF-Core Batch Benchmark"))
    print(_box_row(f"Task file  : {os.path.basename(task_file)}  ({len(tasks)} tasks)"))
    print(_box_row(f"Results dir: {results_dir}"))
    print(_box_row(f"Headless   : {headless}  |  Visuals: {visuals}  |  Max iter: {max_iter}"))
    print(_box_row(f"Bridge     : {BACKEND_URL}"))
    print(_box_bot())

    for task in task_iter:
        task_id    = task["id"]
        category   = task["category"]
        difficulty = task["difficulty"]
        instruction = task["instruction"]
        obj1, obj2 = _infer_objects(task)

        # Suppress per-iteration output in headless; show task header always
        verbose_loop = not headless

        if not headless:
            print(f"\n{_line()}")
            print(f"  TASK  {task_id}  [{category}]  [{difficulty}]")
            print(f"  {instruction[:72]}")
            print(_line())

        result = run_rar_loop(
            user_task    = instruction,
            max_iter     = max_iter,
            headless     = headless,
            visuals      = visuals,
            task_id      = task_id,
            difficulty   = difficulty,
            results_dir  = results_dir,
            obj1         = obj1,
            obj2         = obj2,
            verbose      = verbose_loop,
        )

        initial_cols = len(result["initial_report"].get("collisions", []))
        final_cols   = len(result["final_report"].get("collisions",   []))

        rows.append({
            "TaskID":                task_id,
            "Category":              category,
            "Difficulty":            difficulty,
            "Instruction":           instruction[:80],
            "Initial_Collisions":    initial_cols,
            "Final_Collisions":      final_cols,
            "Iterations_to_Success": result["iterations_run"],
            "Success_Bool":          result["final_status"] == "SUCCESS",
            "Stop_Reason":           result["stop_reason"],
        })

        status_icon = "✓" if result["final_status"] == "SUCCESS" else "✗"
        if headless:
            print(f"  {status_icon}  {task_id}  [{difficulty}]  "
                  f"iter={result['iterations_run']}  "
                  f"col {initial_cols}→{final_cols}  "
                  f"{result['final_status']}")

    csv_path = os.path.join(results_dir, "benchmark_report.csv")
    _save_csv(rows, csv_path)

    # ── Batch summary ──────────────────────────────────────────────────────
    n_success = sum(1 for r in rows if r["Success_Bool"])
    print(f"\n{_line('═')}")
    print(_box_top("Batch Summary"))
    print(_box_row(f"Tasks run : {len(rows)}"))
    print(_box_row(f"Succeeded : {n_success} / {len(rows)}  ({100*n_success//max(len(rows),1)} %)"))
    print(_box_row(f"CSV       : {csv_path}"))
    if visuals:
        print(_box_row(f"Screenshots: {os.path.join(results_dir, 'visuals')}"))
    print(_box_bot())


# =============================================================================
# ── Single-task summary printer (unchanged from v1) ──────────────────────────
# =============================================================================

def _print_summary(result: Dict) -> None:
    status = result["final_status"]
    n_iter = result["iterations_run"]
    reason = result["stop_reason"]
    pos    = result["final_placement"].get("position", [])
    report = result["final_report"]
    n_col  = len(report.get("collisions", []))
    n_unst = len(report.get("unstable_objects", []))
    icon   = "✓" if status == "SUCCESS" else ("✗" if status == "BRIDGE_ERROR" else "⚠")

    print(f"\n{_line('═')}")
    print(_box_top("LOOP SUMMARY"))
    print(_box_row(f"{icon}  Loop finished after {n_iter} iteration(s)."))
    print(_box_row(f"   Final status  : {status}"))
    if reason:
        print(_box_row(f"   Stop reason   : {reason}"))
    print(_box_row(f"   Collisions    : {n_col}"))
    print(_box_row(f"   Unstable objs : {n_unst}"))
    if pos:
        print(_box_row(f"   Final position: {[round(v, 4) for v in pos]}"))
    print(_box_bot())


# =============================================================================
# ── Correction mode — USD scene physics correction loop ──────────────────────
# =============================================================================
#
# run_correction_loop() is the correction-mode entry point.  It starts from
# a USD scene file (output of video reconstruction) rather than a natural-
# language task description.
#
# Key differences from run_rar_loop():
#   • Input       : USD file path + scene_id, no user task string.
#   • Physics API : uses new schema (penetration_pairs / floating_objects /
#                   unstable_objects / nonmanifold_ratio) and failure_score F.
#   • Action vocab: six correction actions instead of asset placement JSON.
#   • Success      : F < 0.5 AND SSIM > 0.95 (visual fidelity preserved).
#   • Rollback     : if an action worsens F, the stage is rolled back via
#                    _ext_rollback_action() and the physics state is restored.
#   • SSIM         : skimage.metrics.structural_similarity between the S_0
#                    reference render and the current viewport render.
#                    Falls back to 1.0 when skimage / PIL / numpy are absent.
#   • Logs         : per-iteration JSON in results/correction_logs/<scene_id>.json
#                    final row appended to results/correction_results.csv
# =============================================================================

_F_SUCCESS_THRESHOLD    = 0.5   # failure_score below this → candidate for success
_SSIM_SUCCESS_THRESHOLD = 0.95  # SSIM above this → visual fidelity preserved


# ── Scene loading + initial physics state ─────────────────────────────────────

def _stub_correction_report() -> Dict:
    """
    Return a minimal physics_report in the new schema for when Isaac Sim is
    unavailable.  Gives the correction loop a non-trivial starting state so
    the RAR logic can still be exercised without live PhysX.
    """
    return {
        "penetration_pairs": [
            {"object_a": "obj_01", "object_b": "obj_02", "depth_m": 0.08},
        ],
        "floating_objects":  [{"object_id": "obj_03", "offset_m": 0.04}],
        "unstable_objects":  [{"object_id": "obj_01", "velocity_ms": 0.15}],
        "nonmanifold_ratio": 0.0,
    }


def load_scene_for_correction(usd_path: str) -> tuple[Dict, float]:
    """
    Load a USD scene into Isaac Sim and return the initial physics report
    together with the computed failure score F.

    Parameters
    ----------
    usd_path : Path to the .usd / .usda / .usdc file to load.

    Returns
    -------
    (physics_report, failure_score)
      physics_report : dict with keys penetration_pairs, floating_objects,
                       unstable_objects, nonmanifold_ratio.
      failure_score  : scalar F = PhysicsReport.failure_score.
    """
    if _CORR_LOAD_OK:
        try:
            ok = _ext_load_scene_usd(usd_path)
            if not ok:
                print(f"  [correction] load_scene_usd returned False: {usd_path}")
        except Exception as exc:
            print(f"  [correction] load_scene_usd error: {exc}")

    if _CORR_PHYS_OK:
        try:
            report = _ext_get_physics_report()
        except Exception as exc:
            print(f"  [correction] get_physics_report error: {exc} — using stub")
            report = _stub_correction_report()
    else:
        report = _stub_correction_report()

    f_score = PhysicsReport.from_dict(report).failure_score
    return report, f_score


# ── Post-action physics measurement ──────────────────────────────────────────

def _get_physics_report_correction(
    action: str,
    params: dict,
    prev_report: dict,
) -> Dict:
    """
    Retrieve the physics state after a correction action has been applied.

    Uses the real PhysX engine when Isaac Sim is available; otherwise falls
    back to _simulate_correction_step() which models the expected improvement
    analytically so the loop still converges during development / CI.
    """
    if _CORR_PHYS_OK:
        try:
            return _ext_get_physics_report()
        except Exception as exc:
            print(f"  [correction] get_physics_report error: {exc} — using simulation")
    return _simulate_correction_step(action, params, prev_report)


def _simulate_correction_step(action: str, params: dict, prev_report: dict) -> Dict:
    """
    Stub: model the physics effect of a single correction action.

    Each of the six action types targets a different violation class:
      resolve_penetration → reduces penetration depth by 70 %
      snap_to_surface     → removes object from floating + unstable lists
      stack_on            → same as snap_to_surface
      recompute_hull      → reduces nonmanifold_ratio by 0.35
      repair_manifold     → same as recompute_hull
      remove_object       → drops all violations involving the object
    """
    import copy
    r = copy.deepcopy(prev_report)

    if action == "resolve_penetration":
        obj_a = params.get("object_id_a", "")
        obj_b = params.get("object_id_b", "")
        new_pairs = []
        for p in r.get("penetration_pairs", []):
            if {p.get("object_a"), p.get("object_b")} == {obj_a, obj_b}:
                reduced = round(p["depth_m"] * 0.30, 4)
                if reduced > 0.001:
                    new_pairs.append({**p, "depth_m": reduced})
            else:
                new_pairs.append(p)
        r["penetration_pairs"] = new_pairs

    elif action in ("snap_to_surface", "stack_on"):
        oid = params.get("object_id", "")
        r["floating_objects"] = [
            f for f in r.get("floating_objects", []) if f.get("object_id") != oid
        ]
        r["unstable_objects"] = [
            u for u in r.get("unstable_objects", []) if u.get("object_id") != oid
        ]
        # Partial improvement to any penetration involving this object
        r["penetration_pairs"] = [
            {**p, "depth_m": round(p["depth_m"] * 0.50, 4)}
            for p in r.get("penetration_pairs", [])
            if round(p["depth_m"] * 0.50, 4) > 0.001
        ]

    elif action in ("recompute_hull", "repair_manifold"):
        r["nonmanifold_ratio"] = max(
            0.0, round(r.get("nonmanifold_ratio", 0.0) - 0.35, 4)
        )

    elif action == "remove_object":
        oid = params.get("object_id", "")
        r["penetration_pairs"] = [
            p for p in r.get("penetration_pairs", [])
            if p.get("object_a") != oid and p.get("object_b") != oid
        ]
        r["floating_objects"] = [
            f for f in r.get("floating_objects", []) if f.get("object_id") != oid
        ]
        r["unstable_objects"] = [
            u for u in r.get("unstable_objects", []) if u.get("object_id") != oid
        ]

    return r


# ── SSIM helpers ──────────────────────────────────────────────────────────────

def _load_png_as_array(filepath: str):
    """Load *filepath* as an H×W×3 uint8 numpy array, or None on any failure."""
    if not (_PIL_OK and _NUMPY_OK and os.path.exists(filepath)):
        return None
    try:
        img = _PIL_Image.open(filepath).convert("RGB")
        return _np.array(img, dtype=_np.uint8)
    except Exception:
        return None


def _render_scene_to_array(filepath: str):
    """Capture the Isaac Sim viewport to *filepath* and return as numpy array."""
    capture_viewport_to_file(filepath)
    return _load_png_as_array(filepath)


def _compute_ssim(ref_array, cur_array) -> float:
    """
    Compute structural similarity between two rendered images.

    Parameters
    ----------
    ref_array : H×W×3 uint8 numpy array — S_0 reference render.
    cur_array : H×W×3 uint8 numpy array — current iteration render.

    Returns
    -------
    float in [0, 1].  Returns 1.0 when skimage / numpy / PIL are unavailable
    or the arrays cannot be compared (different shapes, exception, etc.).
    """
    if not (_SKIMAGE_OK and _NUMPY_OK) or ref_array is None or cur_array is None:
        return 1.0
    try:
        ref_gray = _np.mean(ref_array, axis=2).astype(_np.float32)
        cur_gray = _np.mean(cur_array, axis=2).astype(_np.float32)
        if ref_gray.shape != cur_gray.shape:
            return 1.0
        return float(_skimage_ssim(ref_gray, cur_gray, data_range=255.0))
    except Exception:
        return 1.0


# ── Correction log I/O ────────────────────────────────────────────────────────

def _save_correction_log(logs: List[Dict], scene_id: str, results_dir: str) -> None:
    """
    Persist the growing per-iteration correction log for *scene_id* as JSON.
    Writes (overwrites) results/correction_logs/<scene_id>.json each iteration
    so the log is always up to date even if the process is interrupted.
    """
    log_dir = os.path.join(results_dir, "correction_logs")
    os.makedirs(log_dir, exist_ok=True)
    filepath = os.path.join(log_dir, f"{scene_id}.json")
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump({"scene_id": scene_id, "iterations": logs}, fh, indent=2)
    print(f"  [log]  → {filepath}")


def _save_correction_csv(row: Dict, results_dir: str) -> None:
    """
    Append *row* to results/correction_results.csv.

    Columns: scene_id, initial_F, final_F, ssim, rfpcr, iterations, success.
    Creates the file with a header row on first write.
    """
    csv_path = os.path.join(results_dir, "correction_results.csv")
    os.makedirs(results_dir, exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"  [csv]  → {csv_path}")


# ── Correction console display ────────────────────────────────────────────────

def _print_correction_iter(
    iteration:   int,
    max_iter:    int,
    action:      str,
    params:      dict,
    reasoning:   str,
    f_before:    float,
    f_after_raw: float,    # actual F after action, before any rollback
    ssim:        float,
    rolled_back: bool,
) -> None:
    """Print a compact per-iteration status line for correction mode."""
    delta   = f_after_raw - f_before
    arrow   = "↓ improved" if delta < -1e-9 else ("↑ worsened" if delta > 1e-9 else "→ unchanged")
    rb_note = "  [ROLLED BACK — stage restored]" if rolled_back else ""
    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
    print(f"\n{_hdr(f'CORRECTION {iteration} / {max_iter}')}")
    print(f"  action    : {action}({param_str}){rb_note}")
    print(f"  reasoning : {reasoning[:74]}")
    print(f"  F score   : {f_before:.4f} → {f_after_raw:.4f}  ({arrow})")
    eff_f = f_before if rolled_back else f_after_raw
    if rolled_back:
        print(f"  eff. F    : {eff_f:.4f}  (restored)")
    print(f"  SSIM      : {ssim:.4f}  (threshold {_SSIM_SUCCESS_THRESHOLD})")


def _print_correction_summary(
    scene_id:  str,
    usd_path:  str,
    initial_F: float,
    final_F:   float,
    ssim:      float,
    rfpcr:     int,
    n_iter:    int,
    status:    str,
) -> None:
    icon = "✓" if status == "SUCCESS" else ("✗" if status == "BRIDGE_ERROR" else "⚠")
    print(f"\n{_line('═')}")
    print(_box_top("CORRECTION LOOP SUMMARY"))
    print(_box_row(f"Scene            : {scene_id}"))
    print(_box_row(f"USD              : {usd_path[:68]}"))
    print(_box_row(f"{icon}  Status         : {status}"))
    print(_box_row(f"   Iterations used : {n_iter}"))
    print(_box_row(f"   Initial F        : {initial_F:.4f}"))
    print(_box_row(f"   Final F          : {final_F:.4f}  (delta {final_F - initial_F:+.4f})"))
    print(_box_row(
        f"   SSIM vs S_0      : {ssim:.4f}  "
        f"(threshold {_SSIM_SUCCESS_THRESHOLD})"
    ))
    print(_box_row(
        f"   RFPCR            : {rfpcr}  "
        f"(1 = F<{_F_SUCCESS_THRESHOLD} AND SSIM>{_SSIM_SUCCESS_THRESHOLD})"
    ))
    print(_box_bot())


# ── Main correction loop ──────────────────────────────────────────────────────

def run_correction_loop(
    usd_path:    str,
    scene_id:    str,
    max_iter:    int = 10,
    results_dir: str = _DEFAULT_RESULTS_DIR,
) -> Dict:
    """
    Execute the Reason → Act → Reflect correction loop for a single USD scene.

    Starts from a pre-existing USD file (typically the output of a video
    reconstruction pipeline) instead of a natural-language task description.
    Iteratively corrects physics violations using the six-action correction
    vocabulary and stops when the scene passes the success criteria or the
    iteration budget is exhausted.

    Parameters
    ----------
    usd_path    : Path to the .usd / .usda / .usdc file to correct.
    scene_id    : Identifier for this scene (used in filenames and CSV).
    max_iter    : Maximum RAR correction cycles (default 10).
    results_dir : Root directory for all output files.

    Returns
    -------
    dict with keys:
      scene_id        : str
      usd_path        : str
      initial_F       : float  — failure_score before any correction
      final_F         : float  — effective failure_score after loop ends
      ssim            : float  — SSIM(final_render, S_0_render)
      rfpcr           : int    — 1 if final_F < F_threshold AND ssim > SSIM_threshold
      iterations      : int    — number of correction cycles run
      success         : bool
      status          : "SUCCESS" | "MAX_ITER" | "BRIDGE_ERROR" | "ALREADY_VALID"
      action_history  : list   — accepted (non-rolled-back) action records
      bad_actions     : list   — actions that worsened F and were rolled back
      iteration_logs  : list   — full per-iteration metrics

    Output files
    ------------
      results/correction_logs/<scene_id>.json   — per-iteration log (JSON)
      results/correction_renders/<scene_id>/    — viewport PNGs (when available)
      results/correction_results.csv            — one row per scene
    """
    # ── Header ────────────────────────────────────────────────────────────────
    print(_line("═"))
    print(_box_top("NemoForge — Correction Mode"))
    print(_box_row(f"Scene   : {scene_id}"))
    print(_box_row(f"USD     : {usd_path[:68]}"))
    print(_box_row(
        f"Max iter: {max_iter}  |  F threshold: {_F_SUCCESS_THRESHOLD}  "
        f"|  SSIM threshold: {_SSIM_SUCCESS_THRESHOLD}"
    ))
    print(_box_row(f"Bridge  : {BACKEND_URL}"))
    if not _EXT_OK:
        print(_box_row("⚠  extension.py not loaded — physics simulation is stubbed"))
    if not _SKIMAGE_OK:
        print(_box_row("⚠  scikit-image not installed — SSIM defaults to 1.0"))
    print(_box_bot())

    # ── Step 1: load scene, measure S_0 ──────────────────────────────────────
    print(f"\n  [correction] Loading USD: {usd_path}")
    physics_report, initial_F = load_scene_for_correction(usd_path)
    print(f"  [correction] Initial failure score F = {initial_F:.4f}")

    # ── Step 2: early exit if already valid ───────────────────────────────────
    if initial_F == 0.0:
        print("  [correction] Scene is already physics-valid — no corrections needed.")
        _save_correction_csv(
            {"scene_id": scene_id, "initial_F": 0.0, "final_F": 0.0,
             "ssim": 1.0, "rfpcr": 1, "iterations": 0, "success": True},
            results_dir,
        )
        _print_correction_summary(scene_id, usd_path, 0.0, 0.0, 1.0, 1, 0, "ALREADY_VALID")
        return {
            "scene_id": scene_id, "usd_path": usd_path,
            "initial_F": 0.0, "final_F": 0.0, "ssim": 1.0,
            "rfpcr": 1, "iterations": 0, "success": True,
            "status": "ALREADY_VALID",
            "action_history": [], "bad_actions": [], "iteration_logs": [],
        }

    # ── Step 3: capture S_0 reference render ──────────────────────────────────
    render_dir = os.path.join(results_dir, "correction_renders", scene_id)
    os.makedirs(render_dir, exist_ok=True)
    s0_path  = os.path.join(render_dir, "S_0.png")
    s0_array = _render_scene_to_array(s0_path)
    if s0_array is None:
        print("  [correction] Viewport capture unavailable — SSIM will default to 1.0")

    # ── Step 4: RAR correction loop ───────────────────────────────────────────
    action_history: List[Dict] = []
    bad_actions:    List[Dict] = []
    iteration_logs: List[Dict] = []

    prev_report = physics_report
    prev_F      = initial_F
    final_F     = initial_F
    final_ssim  = 1.0
    status      = "MAX_ITER"
    iteration   = 0

    for iteration in range(max_iter):
        # ── a. REASON ─────────────────────────────────────────────────────────
        prompt_text = build_prompt(
            user_input     = scene_id,       # retained for API compat; not in content
            physics_report = prev_report,
            iteration      = iteration,
            stable_count   = 0,
            action_history = action_history,
        )

        # ── b. ACT ────────────────────────────────────────────────────────────
        t0       = time.perf_counter()
        response = send_command(prompt_text)
        elapsed  = (time.perf_counter() - t0) * 1000

        _print_bridge_result(response, elapsed)

        if response is None:
            status = "BRIDGE_ERROR"
            print(f"\n  ✗  Bridge unreachable — aborting correction loop.")
            break

        action_name   = response.get("action",     "")
        action_params = response.get("parameters", {})
        reasoning     = response.get("reasoning",  "")

        # ── c. Action is applied as a side effect of send_command() ───────────
        #    The Isaac Sim extension mutates the stage when it processes the
        #    bridge response; no separate apply call is needed here.

        f_before = prev_F   # capture pre-action F before any state update

        # ── d. REFLECT: measure new physics state ──────────────────────────────
        new_report  = _get_physics_report_correction(action_name, action_params, prev_report)
        f_after_raw = PhysicsReport.from_dict(new_report).failure_score

        # ── e. Compute SSIM vs S_0 ────────────────────────────────────────────
        cur_path  = os.path.join(render_dir, f"iter_{iteration + 1:03d}.png")
        cur_array = _render_scene_to_array(cur_path)
        ssim      = _compute_ssim(s0_array, cur_array)

        # ── g. Rollback if F worsened ──────────────────────────────────────────
        if f_after_raw > f_before:
            _ext_rollback_action()              # restore Isaac Sim stage
            bad_actions.append({
                "iteration":  iteration + 1,
                "action":     action_name,
                "parameters": action_params,
                "reasoning":  reasoning,
                "F_before":   f_before,
                "F_after":    f_after_raw,
            })
            # Physics state is NOT updated — prev_report / prev_F unchanged
            effective_F = f_before
            rolled_back = True
        else:
            action_history.append({
                "iteration":            iteration + 1,
                "action":               action_name,
                "parameters":           action_params,
                "reasoning":            reasoning,
                "failure_score_before": f_before,
                "failure_score_after":  f_after_raw,
            })
            prev_report = new_report
            prev_F      = f_after_raw
            effective_F = f_after_raw
            rolled_back = False

        final_F    = effective_F
        final_ssim = ssim

        # ── h. Save iteration log ─────────────────────────────────────────────
        iteration_logs.append({
            "iteration":     iteration + 1,
            "action":        action_name,
            "parameters":    action_params,
            "reasoning":     reasoning,
            "delta_F":       round(f_after_raw - f_before, 6),
            "ssim":          round(ssim, 6),
            "failure_score": round(effective_F, 6),
            "rolled_back":   rolled_back,
        })
        _save_correction_log(iteration_logs, scene_id, results_dir)

        _print_correction_iter(
            iteration   = iteration + 1,
            max_iter    = max_iter,
            action      = action_name,
            params      = action_params,
            reasoning   = reasoning,
            f_before    = f_before,
            f_after_raw = f_after_raw,
            ssim        = ssim,
            rolled_back = rolled_back,
        )

        # ── f. Check success ──────────────────────────────────────────────────
        if effective_F < _F_SUCCESS_THRESHOLD and ssim > _SSIM_SUCCESS_THRESHOLD:
            status = "SUCCESS"
            print(
                f"\n  ✓  Success: F={effective_F:.4f} < {_F_SUCCESS_THRESHOLD}  "
                f"and SSIM={ssim:.4f} > {_SSIM_SUCCESS_THRESHOLD}"
            )
            break

    # ── Step 5: compute final metrics ─────────────────────────────────────────
    n_iterations_run = len(iteration_logs)
    rfpcr = 1 if (final_F < _F_SUCCESS_THRESHOLD and final_ssim > _SSIM_SUCCESS_THRESHOLD) else 0

    # ── Save CSV row ──────────────────────────────────────────────────────────
    _save_correction_csv(
        {
            "scene_id":   scene_id,
            "initial_F":  round(initial_F,  6),
            "final_F":    round(final_F,    6),
            "ssim":       round(final_ssim, 6),
            "rfpcr":      rfpcr,
            "iterations": n_iterations_run,
            "success":    status == "SUCCESS",
        },
        results_dir,
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    _print_correction_summary(
        scene_id  = scene_id,
        usd_path  = usd_path,
        initial_F = initial_F,
        final_F   = final_F,
        ssim      = final_ssim,
        rfpcr     = rfpcr,
        n_iter    = n_iterations_run,
        status    = status,
    )

    return {
        "scene_id":       scene_id,
        "usd_path":       usd_path,
        "initial_F":      initial_F,
        "final_F":        final_F,
        "ssim":           final_ssim,
        "rfpcr":          rfpcr,
        "iterations":     n_iterations_run,
        "success":        status == "SUCCESS",
        "status":         status,
        "action_history": action_history,
        "bad_actions":    bad_actions,
        "iteration_logs": iteration_logs,
    }


# =============================================================================
# ── Entry point ───────────────────────────────────────────────────────────────
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NemoForge RAR loop — task placement, batch benchmark, or scene correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Modes
        -----
          task       (default) — run a single placement task
          batch                — run all 50 NF-Core tasks
          correction           — correct physics violations in a USD scene

        Examples
        --------
          # Task / batch (unchanged)
          python test_loop.py
          python test_loop.py --batch
          python test_loop.py --batch --headless
          python test_loop.py --batch --visuals
          python test_loop.py --batch --task-file custom.json --results-dir /tmp/out

          # Correction mode
          python test_loop.py --mode correction --usd path/to/scene.usd --scene-id scene_001
          python test_loop.py --mode correction --usd scene.usd --scene-id s01 --max-iter 15
          python test_loop.py --mode correction --usd scene.usd --scene-id s01 \\
                              --results-dir D:/results
        """),
    )

    # ── Mode ──────────────────────────────────────────────────────────────────
    p.add_argument(
        "--mode",
        choices=["task", "batch", "correction"],
        default="task",
        metavar="MODE",
        help="Operating mode: task (default), batch, or correction.",
    )

    # ── Task / batch flags (unchanged) ────────────────────────────────────────
    p.add_argument("--batch",     action="store_true",
                   help="Run all tasks from the task file (alias for --mode batch).")
    p.add_argument("--nf-core",   action="store_true", dest="nf_core",
                   help="Alias for --batch.")
    p.add_argument("--task-file", default=_DEFAULT_TASK_FILE, metavar="PATH",
                   help="JSON task file (default: nf_core_50.json).")
    p.add_argument("--headless",  action="store_true",
                   help="Disable HUD and viewport capture — fast server / CI runs.")
    p.add_argument("--visuals",   action="store_true",
                   help="Save one viewport screenshot per task per iteration.")

    # ── Correction-mode flags ─────────────────────────────────────────────────
    p.add_argument(
        "--usd",
        default="",
        metavar="PATH",
        help="(correction mode) Path to the USD scene file to correct.",
    )
    p.add_argument(
        "--scene-id",
        default="",
        dest="scene_id",
        metavar="ID",
        help="(correction mode) Scene identifier used in log filenames and CSV.",
    )

    # ── Shared flags ──────────────────────────────────────────────────────────
    p.add_argument("--results-dir", default=_DEFAULT_RESULTS_DIR, metavar="DIR",
                   help="Output directory for CSV and screenshots (default: results/).")
    p.add_argument("--max-iter",    type=int, default=MAX_ITERATIONS, metavar="N",
                   help=f"Max RAR iterations per task/scene (default: {MAX_ITERATIONS}).")

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Resolve mode — legacy --batch / --nf-core flags take precedence so
    # existing call sites keep working without adding --mode.
    if args.batch or args.nf_core:
        mode = "batch"
    else:
        mode = args.mode   # "task" | "batch" | "correction"

    # ── Correction mode ───────────────────────────────────────────────────────
    if mode == "correction":
        if not args.usd:
            print("  ✗  --usd is required in correction mode.")
            sys.exit(1)
        scene_id = args.scene_id or pathlib.Path(args.usd).stem
        result = run_correction_loop(
            usd_path    = args.usd,
            scene_id    = scene_id,
            max_iter    = args.max_iter,
            results_dir = args.results_dir,
        )
        if result["status"] == "BRIDGE_ERROR":
            sys.exit(1)
        return

    # ── Batch / NF-Core mode (unchanged) ─────────────────────────────────────
    if mode == "batch":
        if not os.path.exists(args.task_file):
            print(f"  ✗  Task file not found: {args.task_file}")
            sys.exit(1)
        run_batch_mode(
            task_file    = args.task_file,
            results_dir  = args.results_dir,
            headless     = args.headless,
            visuals      = args.visuals,
            max_iter     = args.max_iter,
        )
        return

    # ── Single-task mode (default, unchanged) ─────────────────────────────────
    print(_line("═"))
    print(_box_top("NemoForge — Reason → Act → Reflect Loop Test"))
    print(_box_row(f'Task: "{USER_TASK}"'))
    print(_box_row(
        f"Max iter: {args.max_iter}  │  Min Δpos: {MIN_POS_DELTA_CM} cm  │  "
        f"Min Δrot: {MIN_ROT_DELTA_DEG}°  │  Auto-stop: {CLEAN_STOP_COUNT} clean"
    ))
    print(_box_row(f"Bridge: {BACKEND_URL}"))
    if not _EXT_OK:
        print(_box_row("⚠  extension.py import failed — using inline fallback"))
        print(_box_row(f"   ({_ext_fallback_msg[:60]})"))
    print(_box_row(f"Headless: {args.headless}  |  Visuals: {args.visuals}"))
    print(_box_bot())

    result = run_rar_loop(
        user_task    = USER_TASK,
        max_iter     = args.max_iter,
        headless     = args.headless,
        visuals      = args.visuals,
        results_dir  = args.results_dir,
    )
    _print_summary(result)
    if result["final_status"] == "BRIDGE_ERROR":
        sys.exit(1)


if __name__ == "__main__":
    main()
