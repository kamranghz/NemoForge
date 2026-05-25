"""
NemoForge Prompt Builder — Scene Correction via Reason → Act → Reflect (RAR)
=============================================================================
Builds correction prompts for physics violations detected in USD scenes
reconstructed from real-world video.  No user task description is involved;
the input is always a structured physics failure report from the physics critic.

Architecture Diagram
────────────────────
              physics_report (from critic)
                  │
                  ▼
       ┌──────────────────────┐
       │       PLANNER        │ ◄── action_history (from previous iterations)
       │    (this module)     │
       │  OBSERVE → PLAN →    │
       │  ACT → REFLECT       │
       └──────────┬───────────┘
                  │ prompt string
                  ▼
       ┌──────────────────────┐
       │      EXECUTOR        │  extension.py → FastAPI :8010 (bridge unchanged)
       │  send_command()      │  get_physics_report() → PhysX contacts + velocity
       └──────────┬───────────┘
                  │ physics_report dict
                  ▼
       ┌──────────────────────┐
       │      REFLECTOR       │  extension.PhysicsReflector
       │  report → observation│  enforces guardrails, tracks stable_count
       └──────────────────────┘
                  │ observation dict
                  └──────────────────► back to PLANNER (next iteration)

Guardrails (enforced in both prompt text and PhysicsReflector):
  • Max 8 iterations before scene is locked as-is.
  • Minimum adjustment: ≥ 2 cm position  /  ≥ 5° rotation.
  • Auto-stop if no violations for 2 consecutive observations.

Correction output contract (all cycles):
  The LLM MUST respond with JSON only — no prose, no markdown, no extra keys:
    {
      "action":     "<action_name>",
      "parameters": {"object_id": "<id>", ...},
      "reasoning":  "<one sentence>"
    }
  Violations are always presented sorted worst-first (highest depth_m first).
  The failure score F is shown each iteration so the LLM can track progress.

ORBIT-Surgical hook:
  Set env var  NEMOCLAW_ORBIT_MODE=1  to activate surgical asset vocabulary
  and sub-centimetre precision guidance in all prompts.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

# ── ORBIT-Surgical compatibility flag ─────────────────────────────────────────
# When True, precision language shifts for surgical simulation contexts.
_ORBIT_MODE: bool = os.environ.get("NEMOCLAW_ORBIT_MODE", "0") == "1"

# ── Guardrail constants ────────────────────────────────────────────────────────
MAX_ITERATIONS:    int   = 8    # hard cap on Reason-Act-Reflect cycles
MIN_POS_DELTA_CM:  float = 2.0  # ignore corrections smaller than 2 cm
MIN_ROT_DELTA_DEG: float = 5.0  # ignore rotational corrections smaller than 5°
CLEAN_STOP_COUNT:  int   = 2    # consecutive violation-free steps → auto-stop

# ── Action vocabulary ──────────────────────────────────────────────────────────
# These are the ONLY six actions the correction agent may choose from.
# remove_object is last resort only — always prefer non-destructive actions.
_ACTIONS = """\
  snap_to_surface(object_id, surface_id)
      Move object_id flush onto surface_id, resolving floating or sinking.

  resolve_penetration(object_id_a, object_id_b)
      Push object_id_a and object_id_b apart to eliminate their overlap.

  recompute_hull(object_id)
      Rebuild the collision hull for object_id (use when hull is stale/oversized).

  repair_manifold(object_id)
      Fix non-manifold geometry for object_id to restore mesh integrity.

  stack_on(object_id, target_id)
      Place object_id neatly on top of target_id.

  remove_object(object_id)
      LAST RESORT — remove object_id from the scene entirely.
      Only emit this if all other actions have been tried and failed.\
"""

# ── Mode-detection regexes (retained for API compatibility) ───────────────────
_CLEAR_WORDS  = r"\b(clear|reset|empty|clean|remove all|delete all|wipe)\b"
_CHANGE_WORDS = r"\b(change|switch|swap|replace|set environment|load environment|open|environment)\b"


def detect_mode(user_input: str) -> str:
    """Return 'clear', 'change', or 'add'. Retained for compatibility."""
    t = user_input.lower()
    if re.search(_CLEAR_WORDS,  t): return "clear"
    if re.search(_CHANGE_WORDS, t): return "change"
    return "add"


# ── Physics report typed wrapper ───────────────────────────────────────────────

@dataclass
class PhysicsReport:
    """
    Typed view of the dict returned by extension.get_physics_report().

    Expected dict schema
    --------------------
    {
      "penetration_pairs": [
          {"object_a": str, "object_b": str, "depth_m": float},
          ...
      ],
      "floating_objects": [
          {"object_id": str, "offset_m": float},
          ...
      ],
      "unstable_objects": [
          {"object_id": str, "velocity_ms": float},
          ...
      ],
      "nonmanifold_ratio": float   # fraction of non-manifold faces, 0.0–1.0
    }

    penetration_pairs are re-sorted descending by depth_m in from_dict() so
    callers never need to worry about ordering.
    """
    penetration_pairs: list = field(default_factory=list)  # [{object_a, object_b, depth_m}]
    floating_objects:  list = field(default_factory=list)  # [{object_id, offset_m}]
    unstable_objects:  list = field(default_factory=list)  # [{object_id, velocity_ms}]
    nonmanifold_ratio: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "PhysicsReport":
        raw_pairs = d.get("penetration_pairs", [])
        sorted_pairs = sorted(raw_pairs, key=lambda c: c.get("depth_m", 0.0), reverse=True)
        return cls(
            penetration_pairs = sorted_pairs,
            floating_objects  = d.get("floating_objects",  []),
            unstable_objects  = d.get("unstable_objects",  []),
            nonmanifold_ratio = float(d.get("nonmanifold_ratio", 0.0)),
        )

    @property
    def has_issues(self) -> bool:
        return (
            bool(self.penetration_pairs)
            or bool(self.floating_objects)
            or bool(self.unstable_objects)
            or self.nonmanifold_ratio > 0.0
        )

    @property
    def issue_count(self) -> int:
        nm_count = 1 if self.nonmanifold_ratio > 0.0 else 0
        return (
            len(self.penetration_pairs)
            + len(self.floating_objects)
            + len(self.unstable_objects)
            + nm_count
        )

    @property
    def failure_score(self) -> float:
        """
        Scalar failure score F (lower is better, 0.0 = fully clean).

        Weighting:
          penetration  : sum of depth_m values      (heaviest — direct overlap)
          floating     : sum of offset_m values     (objects hovering above ground)
          unstable     : sum of velocity_ms values  (objects still moving)
          nonmanifold  : nonmanifold_ratio × 10     (geometry defect multiplier)
        """
        pen  = sum(p.get("depth_m",     0.0) for p in self.penetration_pairs)
        fl   = sum(f.get("offset_m",    0.0) for f in self.floating_objects)
        unst = sum(u.get("velocity_ms", 0.0) for u in self.unstable_objects)
        nm   = self.nonmanifold_ratio * 10.0
        return round(pen + fl + unst + nm, 4)


# ── Public formatting helper ───────────────────────────────────────────────────

def format_physics_report(report: dict) -> str:
    """
    Format a raw physics_report dict into a human-readable text block.

    This is the canonical public helper for converting the dict returned by
    extension.get_physics_report() into readable text for prompt injection,
    logging, or debugging.

    Parameters
    ----------
    report : dict with keys penetration_pairs, floating_objects,
             unstable_objects, nonmanifold_ratio.

    Returns
    -------
    str : Multi-line formatted string.
    """
    return _format_physics_block(PhysicsReport.from_dict(report))


# ── Main entry point ──────────────────────────────────────────────────────────

def build_prompt(
    user_input: str,
    physics_report: Optional[dict] = None,
    iteration: int = 0,
    stable_count: int = 0,
    action_history: Optional[list] = None,
) -> str:
    """
    Build the full correction prompt for the NemoClaw LLM.

    Parameters
    ----------
    user_input     : Retained for API compatibility; not used in prompt content.
    physics_report : Structured dict from PhysicsReflector.observe() or
                     extension.get_physics_report(), or None on the first call.
                     Expected keys: penetration_pairs, floating_objects,
                     unstable_objects, nonmanifold_ratio.
    iteration      : Current RAR cycle index (0-based).
    stable_count   : Consecutive clean (no-issue) observation count so far.
    action_history : Optional list of previous action records, each containing:
                       {
                         "iteration":            int,
                         "action":               str,
                         "parameters":           dict,
                         "reasoning":            str,
                         "failure_score_before": float,
                         "failure_score_after":  float,
                       }

    Returns
    -------
    str : Complete prompt string to send to /generate.

    Output contract (all cycles)
    ─────────────────────────────
    The LLM MUST respond with JSON only — no prose, no markdown, no extra keys:
      {
        "action":     "<action_name>",
        "parameters": {"object_id": "<id>", ...},
        "reasoning":  "<one sentence>"
      }
    """
    # Guardrail: iteration hard cap
    if iteration >= MAX_ITERATIONS:
        return _prompt_max_iter_reached(iteration)

    # Guardrail: auto-stop on consecutive clean observations
    if stable_count >= CLEAN_STOP_COUNT and physics_report is not None:
        report = PhysicsReport.from_dict(physics_report)
        if not report.has_issues:
            return _prompt_success(stable_count)

    # First call — emit initial correction prompt
    if physics_report is None or iteration == 0:
        report = PhysicsReport.from_dict(physics_report) if physics_report else PhysicsReport()
        return _prompt_initial_action(report)

    # Subsequent calls — emit corrective ACTION with reflection on history
    report = PhysicsReport.from_dict(physics_report)
    if report.has_issues:
        return _prompt_reflect_and_correct(report, iteration, action_history or [])

    # Physics clean this cycle, not yet reached clean_stop_count
    return _prompt_confirm_clean(iteration, stable_count)


# ── Individual prompt constructors ────────────────────────────────────────────

def _prompt_clear() -> str:
    """Retained for compatibility — not used in the correction workflow."""
    return (
        'Scene clear requested. '
        'Return ONLY this JSON:\n'
        '{"action": "remove_object", "parameters": {"object_id": "all"}, '
        '"reasoning": "User requested full scene clear."}'
    )


def _prompt_change(user_input: str) -> str:
    """Retained for compatibility — not used in the correction workflow."""
    return (
        f'Scene environment change requested: "{user_input}". '
        'Return ONLY this JSON (no other text):\n'
        '{"action": "snap_to_surface", "parameters": {"object_id": "env", "surface_id": "ground"}, '
        '"reasoning": "Environment swap requested."}'
    )


def _prompt_initial_action(report: PhysicsReport) -> str:
    """
    First RAR cycle: initial correction prompt.

    No action history yet; the agent receives the full physics violation
    report and must choose the single best correction action.
    """
    precision_note = (
        "\n  Precision mode active (ORBIT-Surgical). "
        "Use sub-millimetre positioning; all depth values in metres."
        if _ORBIT_MODE else ""
    )
    physics_block = _format_physics_block(report)
    f_score       = report.failure_score
    worst_hint    = _build_worst_violation_hint(report)

    return f"""\
You are a physics correction agent. This scene was reconstructed from \
real-world video. The physics critic has detected violations. Your job \
is to make the MINIMUM correction that fixes the worst violation while \
keeping objects as close as possible to their original positions.\
{precision_note}

── PHYSICS VIOLATIONS (iteration 1 / {MAX_ITERATIONS}) ────────────────────────
Failure score F = {f_score:.4f}  (0.0 = fully clean; lower is better)

{physics_block}

── CORRECTION TASK ─────────────────────────────────────────────────────────────
{worst_hint}
Choose exactly ONE action that resolves the worst violation above.
Prefer non-destructive actions. Use remove_object only as a last resort.
Corrections smaller than {MIN_POS_DELTA_CM} cm are silently ignored.
Maximum correction cycles: {MAX_ITERATIONS}.

── AVAILABLE ACTIONS ────────────────────────────────────────────────────────────
{_ACTIONS}

── RESPONSE FORMAT ──────────────────────────────────────────────────────────────
Return ONLY valid JSON — no prose, no markdown, no extra keys:
{{"action": "<action_name>", "parameters": {{"object_id": "<id>", ...}}, "reasoning": "<one sentence>"}}\
"""


def _prompt_reflect_and_correct(
    report: PhysicsReport,
    iteration: int,
    action_history: List[dict],
) -> str:
    """
    REFLECT + corrective ACTION prompt for iterations > 0.

    Shows the full physics violation state, the complete action history
    (with F before/after each step so the LLM can see what is working),
    and a contextual hint about the worst remaining violation.
    """
    remaining     = MAX_ITERATIONS - iteration
    physics_block = _format_physics_block(report)
    f_score       = report.failure_score
    history_block = _format_action_history(action_history)
    worst_hint    = _build_worst_violation_hint(report)
    example_blk   = _format_response_example(report)

    return f"""\
You are a physics correction agent. This scene was reconstructed from \
real-world video. The physics critic has detected violations. Your job \
is to make the MINIMUM correction that fixes the worst violation while \
keeping objects as close as possible to their original positions.

── REFLECT (iteration {iteration + 1} / {MAX_ITERATIONS}) ─────────────────────
Physics critic found {report.issue_count} violation(s).
Failure score F = {f_score:.4f}  (0.0 = fully clean; lower is better)

{physics_block}

── CORRECTION HISTORY ───────────────────────────────────────────────────────────
{history_block}

── WORST REMAINING VIOLATION ────────────────────────────────────────────────────
{worst_hint}

── GUARDRAILS ───────────────────────────────────────────────────────────────────
  • Choose the action that addresses the WORST violation listed above.
  • Prefer non-destructive actions; use remove_object only as a last resort.
  • {remaining} correction attempt(s) remaining before scene is locked as-is.
  • Auto-stop: {CLEAN_STOP_COUNT} consecutive violation-free reports ends the loop.

── AVAILABLE ACTIONS ────────────────────────────────────────────────────────────
{_ACTIONS}

{example_blk}\
── RESPONSE FORMAT ──────────────────────────────────────────────────────────────
Return ONLY valid JSON — no prose, no markdown, no extra keys:
{{"action": "<action_name>", "parameters": {{"object_id": "<id>", ...}}, "reasoning": "<one sentence>"}}\
"""


def _prompt_confirm_clean(
    iteration: int,
    stable_count: int,
) -> str:
    """
    Called when physics is clean this cycle but auto-stop hasn't triggered yet.
    Gives the LLM the option to confirm or make a final refinement.
    """
    need = CLEAN_STOP_COUNT - stable_count - 1
    return f"""\
You are a physics correction agent.

── REFLECT (iteration {iteration + 1} / {MAX_ITERATIONS}) ─────────────────────
Physics check PASSED — no violations detected.
Clean observations so far: {stable_count + 1} / {CLEAN_STOP_COUNT} \
(need {need} more to auto-stop).

The scene is currently valid. If no further correction is needed, confirm
with the no-op response below. Otherwise choose one refinement action.

Return ONLY valid JSON (no other text):
{{"action": "snap_to_surface", "parameters": {{"object_id": "none", "surface_id": "none"}}, "reasoning": "Scene is clean — no correction required."}}\
"""


def _prompt_max_iter_reached(iteration: int) -> str:
    return (
        f"Maximum correction cycles ({MAX_ITERATIONS}) reached. "
        "Accepting current scene state as final. "
        "Return ONLY this JSON to confirm:\n"
        '{"action": "snap_to_surface", "parameters": {"object_id": "none", "surface_id": "none"}, '
        '"reasoning": "Max iterations reached — scene locked as-is."}'
    )


def _prompt_success(stable_count: int) -> str:
    return (
        f"Scene is violation-free for {stable_count} consecutive checks. "
        "Correction loop completed successfully. "
        "Return ONLY this JSON to confirm:\n"
        '{"action": "snap_to_surface", "parameters": {"object_id": "none", "surface_id": "none"}, '
        '"reasoning": "All violations resolved — correction complete."}'
    )


# ── Physics formatting helpers ────────────────────────────────────────────────

def _format_physics_block(report: PhysicsReport) -> str:
    """
    Render the full physics violation block for prompt injection.

    Sections:
      PENETRATION PAIRS  — sorted worst-first by depth_m
      FLOATING OBJECTS   — sorted by offset_m descending
      UNSTABLE OBJECTS   — sorted by velocity_ms descending
      NON-MANIFOLD       — shown only when nonmanifold_ratio > 0
    """
    return "\n".join([
        _format_penetration_pairs(report.penetration_pairs),
        _format_floating(report.floating_objects),
        _format_unstable(report.unstable_objects),
        _format_nonmanifold(report.nonmanifold_ratio),
    ])


def _format_penetration_pairs(pairs: list) -> str:
    """
    Render the ranked penetration-pair table.

    The first entry carries a '◀ WORST' marker so the LLM knows
    where to focus.  Depths are shown in both metres and centimetres
    so severity is immediately legible.
    """
    if not pairs:
        return "PENETRATION PAIRS   : none"

    n      = len(pairs)
    header = f"PENETRATION PAIRS — {n} pair(s), sorted worst-first:"
    rows: List[str] = []
    for i, p in enumerate(pairs):
        tag   = "◀ WORST    " if i == 0 else "           "
        obj_a = p.get("object_a", "?")
        obj_b = p.get("object_b", "?")
        depth = p.get("depth_m", 0.0)
        rows.append(
            f"  #{i+1} {tag}{obj_a}  ↔  {obj_b}    "
            f"{depth:.4f} m  ({depth * 100:.2f} cm)"
        )
    return header + "\n" + "\n".join(rows)


def _format_floating(floating: list) -> str:
    if not floating:
        return "FLOATING OBJECTS    : none"
    n    = len(floating)
    rows = [
        f"  • {f.get('object_id', '?')}    "
        f"offset = {f.get('offset_m', 0.0):.4f} m  "
        f"({f.get('offset_m', 0.0) * 100:.2f} cm above nearest surface)"
        for f in floating
    ]
    return f"FLOATING OBJECTS — {n}:\n" + "\n".join(rows)


def _format_unstable(unstable: list) -> str:
    if not unstable:
        return "UNSTABLE OBJECTS    : none"
    n    = len(unstable)
    rows = [
        f"  • {u.get('object_id', '?')}    "
        f"velocity = {u.get('velocity_ms', 0.0):.4f} m/s  "
        f"(still moving after settle period)"
        for u in unstable
    ]
    return f"UNSTABLE OBJECTS — {n}:\n" + "\n".join(rows)


def _format_nonmanifold(ratio: float) -> str:
    if ratio <= 0.0:
        return "NON-MANIFOLD GEOMETRY: none"
    pct = ratio * 100.0
    return (
        f"NON-MANIFOLD GEOMETRY: {pct:.1f}% of faces are non-manifold  "
        f"(repair_manifold or recompute_hull recommended)"
    )


# ── Worst-violation context builder ──────────────────────────────────────────

def _build_worst_violation_hint(report: PhysicsReport) -> str:
    """
    Return a one-paragraph natural-language description of the single worst
    remaining violation.  Used as the CONTEXT block to prime the LLM towards
    the correct action before it reads the guardrails.

    Priority order: penetration_pairs > floating_objects > unstable_objects
    > nonmanifold geometry.
    """
    lines: List[str] = []

    if report.penetration_pairs:
        worst = report.penetration_pairs[0]
        a, b  = worst.get("object_a", "?"), worst.get("object_b", "?")
        d_m   = worst.get("depth_m", 0.0)
        d_cm  = d_m * 100.0
        lines.append(
            f"Worst penetration: {a} overlaps {b} by {d_m:.4f} m ({d_cm:.2f} cm). "
            f"Suggested fix: resolve_penetration({a}, {b}) to push them apart, or "
            f"snap_to_surface({a}, <surface>) to reposition {a} onto a valid surface."
        )
        secondary = report.penetration_pairs[1:]
        if secondary:
            sec_parts = [
                f"{p['object_a']} ↔ {p['object_b']} ({p['depth_m'] * 100:.2f} cm)"
                for p in secondary
            ]
            lines.append(f"Additional penetrations: {', '.join(sec_parts)}.")

    elif report.floating_objects:
        worst = report.floating_objects[0]
        oid   = worst.get("object_id", "?")
        off_m = worst.get("offset_m", 0.0)
        lines.append(
            f"Worst floating: {oid} is {off_m:.4f} m ({off_m * 100:.2f} cm) "
            f"above the nearest surface. "
            f"Suggested fix: snap_to_surface({oid}, <surface>) to ground it."
        )

    elif report.unstable_objects:
        worst = report.unstable_objects[0]
        oid   = worst.get("object_id", "?")
        vel   = worst.get("velocity_ms", 0.0)
        lines.append(
            f"Worst instability: {oid} is moving at {vel:.4f} m/s after the settle period. "
            f"Suggested fix: snap_to_surface({oid}, <surface>) or "
            f"stack_on({oid}, <target>) to stabilise it."
        )

    elif report.nonmanifold_ratio > 0.0:
        pct = report.nonmanifold_ratio * 100.0
        lines.append(
            f"Non-manifold geometry: {pct:.1f}% of faces are non-manifold. "
            f"Suggested fix: repair_manifold(<object_id>) to restore mesh integrity, "
            f"or recompute_hull(<object_id>) to rebuild the collision hull."
        )

    if not lines:
        lines.append("No active physics violations detected.")

    return "\n".join(lines)


# ── Action history formatter ──────────────────────────────────────────────────

def _format_action_history(history: List[dict]) -> str:
    """
    Render the correction history, showing each attempted action, its
    parameters, and whether failure score F improved (↓) or worsened (↑).

    Each entry in history should contain:
      {
        "iteration":            int,
        "action":               str,
        "parameters":           dict,
        "reasoning":            str,
        "failure_score_before": float,
        "failure_score_after":  float,
      }
    """
    if not history:
        return "(no prior corrections — this is the first corrective step)"

    rows: List[str] = []
    for h in history:
        it       = h.get("iteration", "?")
        action   = h.get("action", "?")
        params   = h.get("parameters", {})
        f_before = h.get("failure_score_before", 0.0)
        f_after  = h.get("failure_score_after",  0.0)
        delta    = f_after - f_before
        if delta < -1e-9:
            arrow = f"↓ improved  (Δ = {delta:+.4f})"
        elif delta > 1e-9:
            arrow = f"↑ worsened  (Δ = {delta:+.4f})"
        else:
            arrow = "→ unchanged"
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        rows.append(
            f"  iter {it}: {action}({param_str})\n"
            f"            F {f_before:.4f} → {f_after:.4f}  {arrow}"
        )

    return "\n".join(rows)


# ── Response example block ────────────────────────────────────────────────────

def _format_response_example(report: PhysicsReport) -> str:
    """
    Append a concrete filled-in example of the required JSON response so
    the LLM has an unambiguous template to follow.

    The example is populated from the actual worst violation in the current
    report, making it contextually relevant rather than generic.
    """
    if not report.has_issues:
        return ""

    if report.penetration_pairs:
        w      = report.penetration_pairs[0]
        action = "resolve_penetration"
        params = (
            f'"object_id_a": "{w.get("object_a", "obj_a")}", '
            f'"object_id_b": "{w.get("object_b", "obj_b")}"'
        )
        reason = (
            f"{w.get('object_a', 'obj_a')} overlaps "
            f"{w.get('object_b', 'obj_b')} by "
            f"{w.get('depth_m', 0.0) * 100:.2f} cm — pushing apart resolves the deepest penetration."
        )
    elif report.floating_objects:
        w      = report.floating_objects[0]
        action = "snap_to_surface"
        params = f'"object_id": "{w.get("object_id", "obj")}", "surface_id": "floor"'
        reason = (
            f"{w.get('object_id', 'obj')} is floating "
            f"{w.get('offset_m', 0.0) * 100:.2f} cm above the ground — "
            "snapping to surface resolves the offset."
        )
    elif report.unstable_objects:
        w      = report.unstable_objects[0]
        action = "snap_to_surface"
        params = f'"object_id": "{w.get("object_id", "obj")}", "surface_id": "floor"'
        reason = (
            f"{w.get('object_id', 'obj')} is moving at "
            f"{w.get('velocity_ms', 0.0):.4f} m/s — "
            "snapping to a surface stabilises it."
        )
    else:
        action = "repair_manifold"
        params = '"object_id": "mesh_obj"'
        reason = "Non-manifold geometry detected — repairing manifold to fix mesh integrity."

    example_json = (
        f'{{"action": "{action}", '
        f'"parameters": {{{params}}}, '
        f'"reasoning": "{reason}"}}'
    )

    return (
        "── EXAMPLE OF A VALID RESPONSE ──────────────────────────────────────────────\n"
        f"{example_json}\n"
        "── (end of example) ─────────────────────────────────────────────────────────\n\n"
    )


# ── Legacy helpers (retained for compatibility with test_loop.py) ─────────────

def _build_error_explanation(report: PhysicsReport) -> str:
    """
    Retained for callers in test_loop.py that reconstruct the PART 1 text.
    Delegates to _build_worst_violation_hint() with the new schema.
    """
    return _build_worst_violation_hint(report)


# ── Public demo utility ───────────────────────────────────────────────────────

def render_reflection_example(
    user_input: str = "scene correction from video reconstruction",
    sample_report: Optional[dict] = None,
    iteration: int = 1,
) -> str:
    """
    Render the full correction prompt that would be sent to the LLM for a
    given physics_report.  Useful for testing, documentation, and debugging.

    Parameters
    ----------
    user_input    : Retained for API compatibility; not used in prompt content.
    sample_report : physics_report dict (from get_physics_report()).
                    Defaults to a built-in example if not provided.
    iteration     : RAR cycle index (1-based for the first reflection).

    Returns
    -------
    str : The exact prompt string that build_prompt() would produce.

    Example
    -------
    >>> print(render_reflection_example())
    """
    if sample_report is None:
        sample_report = {
            "penetration_pairs": [
                {"object_a": "chair_01", "object_b": "table_02", "depth_m": 0.082},
                {"object_a": "lamp_03",  "object_b": "shelf_04", "depth_m": 0.011},
            ],
            "floating_objects": [
                {"object_id": "box_05", "offset_m": 0.035},
            ],
            "unstable_objects": [
                {"object_id": "chair_01", "velocity_ms": 0.12},
            ],
            "nonmanifold_ratio": 0.0,
        }

    sample_history: List[dict] = []
    if iteration > 1:
        sample_history = [
            {
                "iteration":            1,
                "action":               "resolve_penetration",
                "parameters":           {"object_id_a": "chair_01", "object_id_b": "table_02"},
                "reasoning":            "chair_01 overlaps table_02 by 8.20 cm.",
                "failure_score_before": 0.2450,
                "failure_score_after":  0.1630,
            }
        ]

    return build_prompt(
        user_input     = user_input,
        physics_report = sample_report,
        iteration      = iteration,
        stable_count   = 0,
        action_history = sample_history,
    )
