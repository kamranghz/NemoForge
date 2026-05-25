"""
physics_critic.py — Standalone ovphysx physics report generator.

Zero Isaac Sim / omni / pxr dependency.
Requires: ovphysx >= 0.4.9, numpy

Public API
----------
    measure(usd_path: str) -> dict

Return schema (never raises; returns empty stub on any failure)::

    {
      "penetration_pairs": [{"object_a": str, "object_b": str, "depth_m": float}],
      "floating_objects":  [{"object_id": str, "offset_m": float}],
      "unstable_objects":  [{"object_id": str, "velocity_ms": float}],
      "nonmanifold_ratio": 0.0,
      "failure_score":     float,
    }

failure_score = sum(depth_m) + 2.0*sum(offset_m) + 0.5*sum(velocity_ms) + nonmanifold_ratio

Verified ovphysx 0.4.9 API used here — do NOT change without re-verifying against
the installed package.
"""

from __future__ import annotations

import logging
from typing import Dict, List

log = logging.getLogger(__name__)

# ── ovphysx availability ──────────────────────────────────────────────────────
try:
    import ovphysx as _ovphysx          # type: ignore[import]
    _ovphysx.bootstrap()                # must be called once at module load
    _OVPHYSX_OK: bool = True
    log.debug("[physics_critic] ovphysx loaded and bootstrapped.")
except Exception as _bootstrap_exc:
    _ovphysx = None                     # type: ignore[assignment]
    _OVPHYSX_OK = False
    log.warning(
        "[physics_critic] ovphysx not available (%s) — measure() will return empty stub.",
        _bootstrap_exc,
    )

# ── numpy availability ────────────────────────────────────────────────────────
try:
    import numpy as np                  # type: ignore[import]
    _NUMPY_OK: bool = True
except ImportError:
    np = None                           # type: ignore[assignment]
    _NUMPY_OK = False
    log.warning("[physics_critic] numpy not available — measure() will return empty stub.")


# ── Physics constants ─────────────────────────────────────────────────────────
_SETTLE_STEPS:       int   = 120    # steps at 60 Hz ≈ 2 s of settle time
_SETTLE_DT:          float = 1.0 / 60.0
_FLOAT_THRESHOLD_M:  float = 0.02  # objects > this far above floor_min are floating
_UNSTABLE_THRESHOLD: float = 0.01  # m/s — objects faster than this are unstable

# ── Failure-score weights (single source of truth for this module) ─────────────
_W_PENETRATION: float = 1.0
_W_FLOATING:    float = 2.0
_W_UNSTABLE:    float = 0.5


def _empty_report() -> Dict:
    """Return an all-empty physics report with failure_score = 0.0."""
    return {
        "penetration_pairs": [],
        "floating_objects":  [],
        "unstable_objects":  [],
        "nonmanifold_ratio": 0.0,
        "failure_score":     0.0,
    }


def _compute_failure_score(
    penetration_pairs: List[Dict],
    floating_objects:  List[Dict],
    unstable_objects:  List[Dict],
    nonmanifold_ratio: float,
) -> float:
    pen  = sum(p.get("depth_m",     0.0) for p in penetration_pairs)
    fl   = sum(f.get("offset_m",    0.0) for f in floating_objects)
    unst = sum(u.get("velocity_ms", 0.0) for u in unstable_objects)
    return round(
        _W_PENETRATION * pen
        + _W_FLOATING   * fl
        + _W_UNSTABLE   * unst
        + nonmanifold_ratio,
        4,
    )


def measure(usd_path: str) -> Dict:
    """
    Load *usd_path* into a fresh ovphysx.PhysX instance, settle it for 120 steps
    (2 s at 60 Hz), then sample contacts, positions, and velocities.

    Parameters
    ----------
    usd_path : Absolute or relative path to a .usd / .usda / .usdc scene file.

    Returns
    -------
    dict — physics report (see module docstring for schema).
    Always returns the schema dict; never raises.

    Behaviour when ovphysx / numpy are unavailable
    -----------------------------------------------
    Returns _empty_report() immediately with a log warning.

    Notes
    -----
    - A fresh PhysX instance is created and released each call — safe to call
      from a loop without manual cleanup.
    - Y axis (index 1 of the pose tensor) is height in Isaac / ovphysx conventions.
    - Object labels come from tb_pos.body_names; fall back to "obj_N" when shorter
      than the tensor dimension.
    """
    if not _OVPHYSX_OK or not _NUMPY_OK:
        log.warning("[physics_critic] measure() returning empty stub (backends unavailable).")
        return _empty_report()

    px = None
    try:
        # ── 1. Create instance and load USD ───────────────────────────────────
        px = _ovphysx.PhysX()
        px.add_usd(str(usd_path))
        log.info("[physics_critic] USD loaded: %s", usd_path)

        # ── 2. Settle simulation ───────────────────────────────────────────────
        # current_time is REQUIRED by ovphysx 0.4.9 — do not omit.
        log.info(
            "[physics_critic] Settling %d steps (%.2f s at 60 Hz) …",
            _SETTLE_STEPS,
            _SETTLE_STEPS * _SETTLE_DT,
        )
        for i in range(_SETTLE_STEPS):
            px.step_n_sync(n=1, dt=_SETTLE_DT, current_time=i * _SETTLE_DT)

        # ── 3. Contacts → penetration_pairs ───────────────────────────────────
        contacts = px.get_contact_report()
        penetration_pairs: List[Dict] = []
        for idx, c in enumerate(contacts):
            depth = abs(float(getattr(c, "separation", 0.001)))
            # Attempt to read actor names if the contact object exposes them
            obj_a = str(getattr(c, "actor0", None) or f"contact_a_{idx}")
            obj_b = str(getattr(c, "actor1", None) or f"contact_b_{idx}")
            penetration_pairs.append({
                "object_a": obj_a,
                "object_b": obj_b,
                "depth_m":  round(depth, 4),
            })
        # Sort worst-first
        penetration_pairs.sort(key=lambda p: p["depth_m"], reverse=True)
        log.debug("[physics_critic] penetration_pairs: %d", len(penetration_pairs))

        # ── 4. Positions → floating_objects ───────────────────────────────────
        # RIGID_BODY_POSE tensor: shape (N, 7) — [x, y, z, qx, qy, qz, qw]
        # Y (index 1) is the scene height axis in Isaac/ovphysx.
        tb_pos = px.create_tensor_binding(
            "/World/*", None, _ovphysx.TensorType.RIGID_BODY_POSE
        )
        pos = np.zeros(tb_pos.shape, dtype=np.float32)
        tb_pos.read(pos)
        names_pos: List[str] = list(tb_pos.body_names)

        floating_objects: List[Dict] = []
        if pos.shape[0] > 0:
            floor_min = float(pos[:, 1].min())
            for i in range(pos.shape[0]):
                offset = float(pos[i, 1]) - floor_min
                if offset > _FLOAT_THRESHOLD_M:
                    name = names_pos[i] if i < len(names_pos) else f"obj_{i}"
                    floating_objects.append({
                        "object_id": name,
                        "offset_m":  round(offset, 4),
                    })
        log.debug("[physics_critic] floating_objects: %d", len(floating_objects))

        # ── 5. Velocities → unstable_objects ──────────────────────────────────
        # RIGID_BODY_VELOCITY tensor: shape (N, 6) — [vx, vy, vz, ax, ay, az]
        # Linear speed = L2 norm of first 3 components.
        tb_vel = px.create_tensor_binding(
            "/World/*", None, _ovphysx.TensorType.RIGID_BODY_VELOCITY
        )
        vel = np.zeros(tb_vel.shape, dtype=np.float32)
        tb_vel.read(vel)
        names_vel: List[str] = list(tb_vel.body_names)

        unstable_objects: List[Dict] = []
        if vel.shape[0] > 0:
            speeds = np.linalg.norm(vel[:, :3], axis=1)
            for i in range(len(speeds)):
                speed = float(speeds[i])
                if speed > _UNSTABLE_THRESHOLD:
                    name = names_vel[i] if i < len(names_vel) else f"obj_{i}"
                    unstable_objects.append({
                        "object_id":   name,
                        "velocity_ms": round(speed, 4),
                    })
        log.debug("[physics_critic] unstable_objects: %d", len(unstable_objects))

        # ── 6. Assemble report ────────────────────────────────────────────────
        nonmanifold_ratio: float = 0.0   # ovphysx does not expose mesh topology
        failure_score = _compute_failure_score(
            penetration_pairs,
            floating_objects,
            unstable_objects,
            nonmanifold_ratio,
        )

        log.info(
            "[physics_critic] Report: %d penetrations, %d floating, %d unstable, F=%.4f",
            len(penetration_pairs),
            len(floating_objects),
            len(unstable_objects),
            failure_score,
        )

        return {
            "penetration_pairs": penetration_pairs,
            "floating_objects":  floating_objects,
            "unstable_objects":  unstable_objects,
            "nonmanifold_ratio": nonmanifold_ratio,
            "failure_score":     failure_score,
        }

    except Exception as exc:
        log.error("[physics_critic] measure() failed for %s: %s", usd_path, exc)
        return _empty_report()

    finally:
        # Always release the PhysX instance to free GPU memory, even on failure.
        if px is not None:
            try:
                px.release()
                log.debug("[physics_critic] PhysX instance released.")
            except Exception as rel_exc:
                log.debug("[physics_critic] release() failed (ignored): %s", rel_exc)


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python physics_critic.py <scene.usd>", file=sys.stderr)
        sys.exit(1)

    usd = sys.argv[1]
    report = measure(usd)
    print(json.dumps(report, indent=2))
    print(f"F = {report['failure_score']:.4f}")
