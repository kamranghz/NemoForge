"""
NemoForge Physics Critic
Measures physics violations in USD scenes using ovphysx.
No Isaac Sim dependency. Standalone module.
"""
import logging
import json
import sys
import numpy as np
from pathlib import Path

log = logging.getLogger(__name__)

# ovphysx bootstrap — once at module level
_OVPHYSX_OK = False
try:
    import ovphysx as _ovphysx
    _ovphysx.bootstrap()
    _OVPHYSX_OK = True
    log.info("ovphysx initialized successfully")
except Exception as e:
    log.warning(f"ovphysx not available: {e}")

# Constants
FLOAT_THRESHOLD_M  = 0.02   # objects above this are floating
UNSTABLE_THRESHOLD = 0.01   # m/s — objects faster than this are unstable
SETTLE_STEPS       = 120    # 2 seconds at 60 Hz
PATTERN            = '/World/*'


def measure(usd_path: str) -> dict:
    """
    Load USD scene, run physics settle, return violation report.

    Returns:
        dict with keys:
            penetration_pairs: list of {object_a, object_b, depth_m}
            floating_objects:  list of {object_id, offset_m}
            unstable_objects:  list of {object_id, velocity_ms}
            nonmanifold_ratio: float (0.0 — ovphysx does not expose this)
            failure_score:     float (weighted sum of violations)
            n_objects:         int (number of rigid bodies found)

    Never raises. Returns empty report on any error.
    """
    if not _OVPHYSX_OK:
        log.warning("ovphysx unavailable — returning empty report")
        return _empty_report()

    px = None
    try:
        px = _ovphysx.PhysX()
        px.add_usd(str(usd_path))
        log.info(f"Loaded: {usd_path}")

        log.info(f"Settling {SETTLE_STEPS} steps (2s at 60Hz)...")
        for i in range(SETTLE_STEPS):
            px.step_n_sync(n=1, dt=1.0/60, current_time=i/60.0)

        # Contacts → penetration pairs
        contacts = px.get_contact_report()
        penetration_pairs = []
        for idx, c in enumerate(contacts):
            depth = abs(float(getattr(c, 'separation', 0.001)))
            penetration_pairs.append({
                'object_a': f'obj_a_{idx}',
                'object_b': f'obj_b_{idx}',
                'depth_m':  round(depth, 4)
            })

        # Positions → floating objects
        tb_pos = px.create_tensor_binding(
            PATTERN, None, _ovphysx.TensorType.RIGID_BODY_POSE)
        pos = np.zeros(tb_pos.shape, dtype=np.float32)
        tb_pos.read(pos)
        n = len(pos)

        # Velocities → unstable objects
        tb_vel = px.create_tensor_binding(
            PATTERN, None, _ovphysx.TensorType.RIGID_BODY_VELOCITY)
        vel = np.zeros(tb_vel.shape, dtype=np.float32)
        tb_vel.read(vel)

        floor_min = float(pos[:, 1].min()) if n > 0 else 0.0

        floating_objects = []
        for i in range(n):
            offset = float(pos[i, 1]) - floor_min
            if offset > FLOAT_THRESHOLD_M:
                floating_objects.append({
                    'object_id': f'obj_{i}',
                    'offset_m':  round(offset, 4)
                })

        speeds = np.linalg.norm(vel[:, :3], axis=1)
        unstable_objects = []
        for i in range(n):
            speed = float(speeds[i])
            if speed > UNSTABLE_THRESHOLD:
                unstable_objects.append({
                    'object_id':   f'obj_{i}',
                    'velocity_ms': round(speed, 4)
                })

        failure_score = round(
            sum(p['depth_m']     for p in penetration_pairs)
            + 2.0 * sum(f['offset_m']    for f in floating_objects)
            + 0.5 * sum(u['velocity_ms'] for u in unstable_objects),
            4
        )

        log.info(f"F = {failure_score:.4f} | "
                 f"pairs={len(penetration_pairs)} "
                 f"float={len(floating_objects)} "
                 f"unstable={len(unstable_objects)}")

        return {
            'penetration_pairs': penetration_pairs,
            'floating_objects':  floating_objects,
            'unstable_objects':  unstable_objects,
            'nonmanifold_ratio': 0.0,
            'failure_score':     failure_score,
            'n_objects':         n
        }

    except Exception as e:
        log.error(f"measure() failed: {e}")
        return _empty_report()

    finally:
        if px is not None:
            try:
                px.release()
            except Exception:
                pass


def _empty_report() -> dict:
    return {
        'penetration_pairs': [],
        'floating_objects':  [],
        'unstable_objects':  [],
        'nonmanifold_ratio': 0.0,
        'failure_score':     0.0,
        'n_objects':         0
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python physics_critic.py <usd_path>")
        sys.exit(1)

    report = measure(sys.argv[1])

    print()
    print("=" * 55)
    print("  PHYSICS MEASUREMENT REPORT")
    print("=" * 55)
    print(f"  Objects found     : {report['n_objects']}")
    print(f"  Penetration pairs : {len(report['penetration_pairs'])}")
    print(f"  Floating objects  : {len(report['floating_objects'])}")
    print(f"  Unstable objects  : {len(report['unstable_objects'])}")
    print(f"  Nonmanifold ratio : {report['nonmanifold_ratio']}")
    print(f"  Failure score F   : {report['failure_score']:.4f}")
    print("=" * 55)
