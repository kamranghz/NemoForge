"""
SAGE-10k physics validity baseline: local zip scenes -> USD (cache or kits) ->
extension.load_scene_usd -> extension.get_physics_report.

Run with Isaac Sim's Python (this repo's _build/.../python.bat or Omniverse Kit).
No Hugging Face streaming, no datasets package, no mocks for extension APIs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import types
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── Register project paths via central registry ─────────────────────────────
import importlib.util as _ilu
_PATHS_PY = Path(__file__).resolve().parents[1] / "utils" / "paths.py"
_spec = _ilu.spec_from_file_location("nemoforge.utils.paths", _PATHS_PY)
_paths_mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_paths_mod)  # type: ignore[union-attr]
_paths_mod.register_project_paths()

_DEFAULT_DATA_ROOT: Path = _paths_mod.SAGE_10K_ROOT
_DEFAULT_N_SCENES = 10
_DEFAULT_SEED = 42
_DEFAULT_RESULTS: Path = _paths_mod.RESULTS_DIR

_EXCLUDED_JSON_NAMES = frozenset(
    {
        "rigid_object_property_dict.json",
        "rigid_object_transform_dict.json",
        "package.json",
        "metadata.json",
    }
)

_SIM_APP: Any = None


def _ensure_isaac_simulation_app(*, kit_headless: bool) -> None:
    """Spawn Kit if needed. ``kit_headless`` matches ``--headless`` (no window when True)."""
    global _SIM_APP
    try:
        import omni.kit.app  # type: ignore

        get_app = getattr(omni.kit.app, "get_app", None)
        if callable(get_app) and get_app() is not None:
            return
    except Exception:
        pass

    try:
        from isaacsim import SimulationApp  # type: ignore
    except ImportError:
        try:
            from omni.isaac.kit import SimulationApp  # type: ignore
        except ImportError as exc:
            print(
                "FATAL: Could not import SimulationApp. Use Isaac Sim Python.",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc

    try:
        import omni.isaac.kit  # type: ignore  # noqa: F401
    except ImportError:
        pass

    _SIM_APP = SimulationApp({"headless": kit_headless})


def _close_simulation_app_if_started() -> None:
    global _SIM_APP
    if _SIM_APP is not None:
        try:
            _SIM_APP.close()
        except Exception:
            pass
        _SIM_APP = None


def _apply_headless_extension_patches(extension_module: types.ModuleType) -> None:
    """Avoid HUD / viewport capture without editing extension.py."""

    def _noop_hud(*_a: Any, **_k: Any) -> None:
        return None

    def _noop_capture(*_a: Any, **_k: Any) -> bool:
        return False

    extension_module.update_hud = _noop_hud  # type: ignore[assignment]
    extension_module.capture_viewport_to_file = _noop_capture  # type: ignore[assignment]


def normalize_data_root(raw: str) -> Path:
    s = raw.strip()
    if sys.platform == "win32":
        norm = s.replace("\\", "/")
        if norm.lower().startswith("/mnt/") and len(norm) > 6:
            drive = norm[5:6]
            rest = norm[7:].lstrip("/")
            if drive.isalpha() and rest:
                return Path(f"{drive.upper()}:\\" + rest.replace("/", "\\")).resolve()
    return Path(s).expanduser().resolve()


def scenes_zip_dir(root: Path) -> Path:
    return root / "scenes"


def list_scene_zips(root: Path) -> List[Path]:
    sdir = scenes_zip_dir(root)
    if not sdir.is_dir():
        return []
    return sorted(p for p in sdir.glob("*.zip") if p.is_file())


def extract_scene_zip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)


def find_layout_json_in_tree(root: Path) -> Optional[Path]:
    cands: List[Tuple[Any, ...]] = []
    for p in root.rglob("*.json"):
        parts_lower = {x.lower() for x in p.parts}
        if "__macosx" in parts_lower:
            continue
        if p.name in _EXCLUDED_JSON_NAMES:
            continue
        if "rigid_object" in p.name.lower():
            continue
        rel = p.relative_to(root)
        if len(rel.parts) > 6:
            continue
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        cands.append((len(rel.parts), "layout" not in p.name.lower(), -sz, p.name, p))
    if not cands:
        return None
    cands.sort()
    return cands[0][4]


def find_existing_scene_usd(scene_dir: Path) -> Optional[Path]:
    for name in ("sage_composed.usd", "scene.usd", "layout.usd", "room.usd"):
        for p in sorted(scene_dir.rglob(name)):
            if p.is_file():
                return p
    for p in sorted(scene_dir.rglob("*.usdz")):
        if p.is_file():
            return p
    loose = [p for p in scene_dir.rglob("*.usd") if p.name != "sage_composed.usd"]
    if len(loose) == 1:
        return loose[0]
    return None


def _matrix4d_from_json(val: Any) -> Any:
    from pxr import Gf  # type: ignore

    if val is None:
        return Gf.Matrix4d(1.0)
    if isinstance(val, list) and len(val) == 4 and isinstance(val[0], (list, tuple)):
        r = val
        return Gf.Matrix4d(
            float(r[0][0]),
            float(r[0][1]),
            float(r[0][2]),
            float(r[0][3]),
            float(r[1][0]),
            float(r[1][1]),
            float(r[1][2]),
            float(r[1][3]),
            float(r[2][0]),
            float(r[2][1]),
            float(r[2][2]),
            float(r[2][3]),
            float(r[3][0]),
            float(r[3][1]),
            float(r[3][2]),
            float(r[3][3]),
        )
    if isinstance(val, list) and len(val) == 16:
        return Gf.Matrix4d(*[float(x) for x in val])
    return Gf.Matrix4d(1.0)


def compose_sage_export_to_single_usd(export_dir: Path, composed_path: Path) -> None:
    from pxr import Usd, UsdGeom  # type: ignore

    rigid_tf: Dict[str, Any] = {}
    tf_json = export_dir / "rigid_object_transform_dict.json"
    if tf_json.is_file():
        with tf_json.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        for k, v in raw.items():
            rigid_tf[str(k)] = _matrix4d_from_json(v)

    composed_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(composed_path))
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    try:
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    except Exception:
        pass

    usd_files = sorted(p for p in export_dir.glob("*.usd") if p.stem != "sage_composed")
    if not usd_files:
        raise FileNotFoundError(f"No part .usd files under {export_dir}")

    for usd_file in usd_files:
        stem = usd_file.stem
        abs_path = usd_file.resolve().as_posix()
        is_door = stem.startswith("door_") and not stem.endswith("_frame")
        if is_door:
            xf = UsdGeom.Xform.Define(stage, f"/World/__door_{stem}")
            xf.GetPrim().GetReferences().AddReference(abs_path, "/World")
        else:
            internal = f"/World/{stem}"
            xf = UsdGeom.Xform.Define(stage, internal)
            xf.GetPrim().GetReferences().AddReference(abs_path, internal)
            if stem in rigid_tf:
                xformable = UsdGeom.Xformable(xf)
                try:
                    xformable.ClearXformOpOrder()
                except Exception:
                    pass
                xformable.AddTransformOp().Set(rigid_tf[stem])

    stage.GetRootLayer().Save()


def run_sage_kit_export(layout_json: Path, kits_dir: Path, out_dir: Path) -> None:
    if not kits_dir.is_dir():
        raise FileNotFoundError(f"kits folder not found: {kits_dir}")
    kstr = str(kits_dir.resolve())
    if kstr not in sys.path:
        sys.path.insert(0, kstr)
    from isaacsim_utils import get_room_layout_scene_usd_separate_from_layout  # type: ignore

    out_dir.mkdir(parents=True, exist_ok=True)
    result = get_room_layout_scene_usd_separate_from_layout(
        str(layout_json.resolve()),
        str(out_dir.resolve()),
    )
    if result.get("status") != "success":
        raise RuntimeError(result.get("message", "kit export failed"))


def resolve_usd_from_extract(
    scene_id: str,
    extract_root: Path,
    layout_json: Path,
    kits_dir: Path,
    cache_root: Path,
    use_kit: bool,
    force_export: bool,
) -> Tuple[Path, str]:
    existing = find_existing_scene_usd(extract_root)
    if existing and existing.name != "sage_composed.usd":
        return existing, "prebuilt_in_zip"

    cache_scene = cache_root / scene_id
    composed = cache_scene / "sage_composed.usd"

    if composed.is_file() and not force_export:
        try:
            if composed.stat().st_mtime >= layout_json.stat().st_mtime:
                return composed, "cached_sage_composed"
        except OSError:
            return composed, "cached_sage_composed"

    if not use_kit:
        if composed.is_file():
            return composed, "cached_sage_composed"
        raise FileNotFoundError("No USD in zip and --no-kit-export set.")

    cache_scene.mkdir(parents=True, exist_ok=True)
    export_dir = cache_scene / "parts"
    if force_export and export_dir.exists():
        shutil.rmtree(export_dir, ignore_errors=True)

    run_sage_kit_export(layout_json, kits_dir, export_dir)
    compose_sage_export_to_single_usd(export_dir, composed)
    return composed, "kit_export_plus_compose"


def cached_usd_if_valid(
    zip_path: Path,
    scene_id: str,
    cache_root: Path,
    force_export: bool,
) -> Optional[Path]:
    """Use results/sage_usd_cache/<scene_id>/sage_composed.usd when fresh vs zip mtime."""
    if force_export:
        return None
    p = cache_root / scene_id / "sage_composed.usd"
    if not p.is_file():
        return None
    try:
        if p.stat().st_mtime >= zip_path.stat().st_mtime:
            return p
    except OSError:
        return p
    return None


def max_penetration_cm(report: Dict) -> float:
    cols = report.get("collisions") or []
    if not cols:
        return 0.0
    return max(float(c.get("penetration_cm", 0.0)) for c in cols)


def _make_record(
    scene_zip: str,
    usd_path: Optional[str],
    collisions: int,
    unstable: int,
    max_pen: float,
    validity: str,
    error: Optional[str],
) -> Dict[str, Any]:
    return {
        "scene_zip": scene_zip,
        "usd_path_used": usd_path,
        "initial_collisions": collisions,
        "unstable_count": unstable,
        "max_penetration_cm": round(max_pen, 4),
        "validity": validity,
        "error": error,
    }


def evaluate_one_zip(
    zip_path: Path,
    scene_id: str,
    extract_root: Path,
    kits_dir: Path,
    cache_root: Path,
    settle_time: float,
    use_kit: bool,
    force_export: bool,
    load_scene_usd: Callable[[str], Optional[str]],
    get_physics_report: Callable[..., Dict],
    cleanup_extract: bool,
) -> Dict[str, Any]:
    name = zip_path.name
    t0 = time.perf_counter()

    def finish(
        usd: Optional[str],
        cols: int,
        unst: int,
        mx: float,
        ok: bool,
        err: Optional[str],
    ) -> Dict[str, Any]:
        rec = _make_record(
            name,
            usd,
            cols,
            unst,
            mx,
            "PASS" if ok else "FAIL",
            err,
        )
        rec["_elapsed_s"] = round(time.perf_counter() - t0, 3)
        return rec

    try:
        hit = cached_usd_if_valid(zip_path, scene_id, cache_root, force_export)
        if hit is not None:
            prim = load_scene_usd(str(hit))
            if prim is None:
                return finish(str(hit), 0, 0, 0.0, False, "load_scene_usd returned None")
            rep = get_physics_report(settle_time=settle_time)
            cols = len(rep.get("collisions") or [])
            unst = len(rep.get("unstable_objects") or [])
            mx = max_penetration_cm(rep)
            ok = cols == 0 and unst == 0
            return finish(str(hit), cols, unst, mx, ok, None)

        if extract_root.exists():
            shutil.rmtree(extract_root, ignore_errors=True)
        extract_root.mkdir(parents=True, exist_ok=True)
        extract_scene_zip(zip_path, extract_root)

        layout = find_layout_json_in_tree(extract_root)
        if layout is None:
            return finish(None, 0, 0, 0.0, False, "No layout JSON in zip")

        usd_path, _src = resolve_usd_from_extract(
            scene_id, extract_root, layout, kits_dir, cache_root, use_kit, force_export
        )
        prim = load_scene_usd(str(usd_path))
        if prim is None:
            return finish(str(usd_path), 0, 0, 0.0, False, "load_scene_usd returned None")

        rep = get_physics_report(settle_time=settle_time)
        cols = len(rep.get("collisions") or [])
        unst = len(rep.get("unstable_objects") or [])
        mx = max_penetration_cm(rep)
        ok = cols == 0 and unst == 0
        return finish(str(usd_path), cols, unst, mx, ok, None)

    except zipfile.BadZipFile as exc:
        return finish(None, 0, 0, 0.0, False, f"BadZipFile: {exc}")
    except Exception as exc:
        return finish(None, 0, 0, 0.0, False, f"{type(exc).__name__}: {exc}")
    finally:
        if cleanup_extract:
            try:
                if extract_root.exists():
                    shutil.rmtree(extract_root, ignore_errors=True)
            except Exception:
                pass


def run_correction_baseline(
    n_scenes:    int            = 20,
    data_root:   Optional[Path] = None,
    results_dir: Optional[Path] = None,
    seed:        int            = _DEFAULT_SEED,
) -> None:
    """
    Run the RAR physics correction loop on a random sample of SAGE-10k scenes
    and record initial/final failure scores, SSIM, RFPCR, and iteration counts.

    For each sampled scene zip the function:
      1. Extracts the zip to a fresh temp directory.
      2. Finds the layout JSON and runs export_usd.py (subprocess) to produce
         USD part files; composes them into sage_composed.usd if needed.
      3. Calls run_correction_loop() from test_loop.py (imported lazily to
         avoid circular imports).
      4. Appends one result row to results/sage_correction_baseline.csv
         immediately after each scene so partial results survive crashes.
      5. Cleans up the temp directory.

    Scene failures (bad zip, USD export error, bridge error, etc.) are caught
    individually — a row with rfpcr=0 and all metrics=None is saved and the run
    continues to the next scene.

    Parameters
    ----------
    n_scenes    : Number of scenes to sample (capped to available zip count).
    data_root   : Path to the SAGE-10k root (default: _DEFAULT_DATA_ROOT).
    results_dir : Directory for CSV and correction logs (default: _DEFAULT_RESULTS).
    seed        : Random seed for reproducibility (default 42).
    """
    # ── Lazy import to avoid circular dependency ───────────────────────────────
    from nemoforge.core.test_loop import run_correction_loop  # noqa: PLC0415

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_root    = data_root    or _DEFAULT_DATA_ROOT
    results_dir  = results_dir  or _DEFAULT_RESULTS
    kits_dir     = data_root / "kits"
    export_script = kits_dir / "export_usd.py"
    csv_path     = results_dir / "sage_correction_baseline.csv"

    CSV_COLS: List[str] = [
        "scene_id", "initial_F", "final_F", "ssim", "rfpcr", "iterations"
    ]

    # ── 1. Discover and sample scenes ─────────────────────────────────────────
    zips = list_scene_zips(data_root)
    if not zips:
        print(
            f"[ERROR] No *.zip files found in {scenes_zip_dir(data_root)}",
            file=sys.stderr,
        )
        return

    rng = random.Random(seed)
    n   = min(n_scenes, len(zips))
    chosen = rng.sample(zips, n)

    print(
        f"\n[correction-baseline] Sampling {n} / {len(zips)} scenes  "
        f"seed={seed}  →  {csv_path}"
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    all_rows: List[Dict[str, Any]] = []

    # Build the PYTHONPATH used by the export_usd.py subprocess so that
    # isaacsim_utils (which lives in kits/) can be imported.
    kits_str = str(kits_dir.resolve())
    sub_env  = os.environ.copy()
    sub_env["PYTHONPATH"] = (
        kits_str + os.pathsep + sub_env["PYTHONPATH"]
        if sub_env.get("PYTHONPATH")
        else kits_str
    )

    # ── 2. Per-scene loop ──────────────────────────────────────────────────────
    for idx, zip_path in enumerate(chosen, 1):
        scene_id = zip_path.stem
        print(f"\n  [{idx}/{n}] {scene_id}")
        tmp_dir: Optional[Path] = None

        try:
            # 2a. Extract zip into a fresh temp directory
            tmp_dir = Path(tempfile.mkdtemp(prefix=f"sagecorr_{scene_id}_"))
            extract_scene_zip(zip_path, tmp_dir)

            # 2b. Find layout JSON and convert to USD via export_usd.py
            layout_json = find_layout_json_in_tree(tmp_dir)
            if layout_json is None:
                raise FileNotFoundError("No layout JSON found inside zip")

            usd_out_dir = tmp_dir / "usd_out"
            usd_out_dir.mkdir(parents=True, exist_ok=True)

            if not export_script.is_file():
                raise FileNotFoundError(f"export_usd.py not found at {export_script}")

            proc = subprocess.run(
                [
                    sys.executable,
                    str(export_script),
                    str(layout_json),
                    str(usd_out_dir),
                ],
                capture_output=True,
                text=True,
                timeout=300,
                env=sub_env,
                cwd=str(kits_dir),
            )
            if proc.returncode != 0:
                err_snippet = (proc.stderr or proc.stdout or "").strip()[:300]
                raise RuntimeError(
                    f"export_usd.py exited {proc.returncode}: {err_snippet}"
                )

            # Locate the produced USD; compose part files if necessary
            usd_path = find_existing_scene_usd(usd_out_dir)
            if usd_path is None:
                composed = usd_out_dir / "sage_composed.usd"
                compose_sage_export_to_single_usd(usd_out_dir, composed)
                usd_path = composed

            if not usd_path.is_file():
                raise FileNotFoundError(
                    f"No USD file produced under {usd_out_dir}"
                )

            # 2c. Run the RAR correction loop
            loop_result = run_correction_loop(
                usd_path    = str(usd_path),
                scene_id    = scene_id,
                max_iter    = 10,
                results_dir = str(results_dir),
            )

            # 2d. Extract metrics from the returned dict
            row: Dict[str, Any] = {
                "scene_id":   scene_id,
                "initial_F":  loop_result.get("initial_F"),
                "final_F":    loop_result.get("final_F"),
                "ssim":       loop_result.get("ssim"),
                "rfpcr":      loop_result.get("rfpcr"),
                "iterations": loop_result.get("iterations"),
            }
            icon = "✓" if row["rfpcr"] == 1 else "✗"
            iF = row["initial_F"]
            fF = row["final_F"]
            ss = row["ssim"]
            print(
                f"    {icon}  F {iF:.4f} → {fF:.4f}  "
                f"SSIM={ss:.4f}  RFPCR={row['rfpcr']}  "
                f"iter={row['iterations']}"
            )

        except Exception as exc:
            print(
                f"    [FAIL] {scene_id}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            row = {
                "scene_id":   scene_id,
                "initial_F":  None,
                "final_F":    None,
                "ssim":       None,
                "rfpcr":      0,
                "iterations": None,
            }

        finally:
            # 2e. Remove temp directory regardless of outcome
            if tmp_dir is not None:
                try:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass

        # 3. Append row to CSV immediately so partial results survive crashes
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLS)
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerow(row)

        all_rows.append(row)

    # ── 4. Summary ────────────────────────────────────────────────────────────
    def _mean(key: str) -> float:
        vals = [r[key] for r in all_rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    n_succeeded  = sum(1 for r in all_rows if r.get("rfpcr") == 1)
    mean_rfpcr   = n_succeeded / n if n else 0.0
    mean_init_F  = _mean("initial_F")
    mean_final_F = _mean("final_F")
    delta_F      = mean_final_F - mean_init_F
    mean_iter    = _mean("iterations")

    _W = 43
    print()
    print("═" * _W)
    print(f"SAGE-10K CORRECTION BASELINE — {n} scenes")
    print("─" * _W)
    print(f"Mean RFPCR        : {mean_rfpcr:.2f}")
    print(f"Mean initial F    : {mean_init_F:.4f}")
    print(f"Mean final F      : {mean_final_F:.4f}  (delta: {delta_F:+.4f})")
    print(f"Mean iterations   : {mean_iter:.1f}")
    print(f"Scenes succeeded  : {n_succeeded} / {n}")
    print(f"Results saved to  : {csv_path}")
    print("═" * _W)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "SAGE-10k physics test suite.  "
            "Modes: 'baseline' (physics validity) or 'correction' (RAR correction loop)."
        ),
    )
    p.add_argument(
        "--mode",
        choices=["baseline", "correction"],
        default="baseline",
        metavar="MODE",
        help="Operating mode: baseline (default) or correction.",
    )
    # ── Shared flags ──────────────────────────────────────────────────────────
    p.add_argument("--data-root",   type=str, default=str(_DEFAULT_DATA_ROOT))
    p.add_argument("--n-scenes",    type=int, default=_DEFAULT_N_SCENES)
    p.add_argument("--seed",        type=int, default=_DEFAULT_SEED)
    p.add_argument("--results-dir", type=str, default=str(_DEFAULT_RESULTS))
    p.add_argument(
        "--headless",
        action="store_true",
        help="Headless Kit + disable extension HUD and viewport capture.",
    )
    # ── Baseline-only flags (ignored in correction mode) ──────────────────────
    p.add_argument("--settle-time",   type=float, default=None)
    p.add_argument("--no-kit-export", action="store_true")
    p.add_argument("--force-export",  action="store_true")
    p.add_argument("--keep-extract",  action="store_true")
    return p.parse_args()


def main() -> None:
    args     = _parse_args()
    headless = bool(args.headless)

    _ensure_isaac_simulation_app(kit_headless=headless)

    # ── Correction mode ───────────────────────────────────────────────────────
    if args.mode == "correction":
        run_correction_baseline(
            n_scenes    = args.n_scenes,
            data_root   = normalize_data_root(args.data_root),
            results_dir = Path(args.results_dir).expanduser().resolve(),
            seed        = args.seed,
        )
        _close_simulation_app_if_started()
        return

    # ── Baseline mode (run_physics_baseline — completely unchanged) ───────────
    try:
        import extension as ext  # type: ignore

        load_scene_usd     = ext.load_scene_usd
        get_physics_report = ext.get_physics_report
        PHYSICS_CONFIG     = ext.PHYSICS_CONFIG
    except Exception as exc:
        print(f"FATAL: cannot import extension: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if headless:
        _apply_headless_extension_patches(ext)

    settle      = float(args.settle_time or PHYSICS_CONFIG.get("settle_time", 0.8))
    data_root   = normalize_data_root(args.data_root)
    kits_dir    = data_root / "kits"
    results_dir = Path(args.results_dir).expanduser().resolve()
    cache_root  = results_dir / "sage_usd_cache"
    extract_parent = results_dir / "sage_zip_extract"
    use_kit     = not args.no_kit_export

    zips = list_scene_zips(data_root)
    if not zips:
        print(f"[ERROR] No *.zip under {scenes_zip_dir(data_root)}", file=sys.stderr)
        _close_simulation_app_if_started()
        sys.exit(1)

    extract_parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    n      = min(args.n_scenes, len(zips))
    chosen = random.sample(zips, n)

    records_out: List[Dict[str, Any]] = []
    for zip_path in chosen:
        sid = zip_path.stem
        if args.keep_extract:
            extract_root = extract_parent / sid
        else:
            extract_root = Path(
                tempfile.mkdtemp(prefix=f"sage_{sid}_", dir=str(extract_parent))
            )

        rec = evaluate_one_zip(
            zip_path,
            sid,
            extract_root,
            kits_dir,
            cache_root,
            settle,
            use_kit,
            args.force_export,
            load_scene_usd,
            get_physics_report,
            cleanup_extract=not args.keep_extract,
        )
        elapsed_s = rec.pop("_elapsed_s", 0.0)
        records_out.append(rec)

        ok   = rec["validity"] == "PASS"
        em   = rec.get("error") or ""
        tail = f"  err={(em[:72] + '...') if len(em) > 72 else em}" if em else ""
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {rec['scene_zip']}"
            f"  col={rec['initial_collisions']}"
            f"  unst={rec['unstable_count']}"
            f"  pen={rec['max_penetration_cm']}"
            f"  {elapsed_s}s{tail}"
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / "sage_validity_baseline.json"
    payload  = {
        "metadata": {
            "n_scenes": len(records_out),
            "data_root": str(data_root),
        },
        "records": records_out,
    }
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"Wrote {out_json}")
    _close_simulation_app_if_started()


if __name__ == "__main__":
    main()
