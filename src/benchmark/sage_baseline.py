"""
NemoForge SAGE-10K Benchmark
Runs physics measurement on N random SAGE-10K scenes.
"""
import csv
import json
import logging
import random
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

SAGE_ROOT   = Path("dataset/SAGE-10k")
SCENES_DIR  = SAGE_ROOT / "scenes"
RESULTS_DIR = Path("results")
SEED        = 42


def run_baseline(n_scenes:    int           = 5,
                 scenes_dir:  Optional[Path] = None,
                 results_dir: Optional[Path] = None,
                 seed:        int           = SEED) -> None:
    """
    Measure initial physics violations on N SAGE-10K scenes.
    No LLM needed. Pure physics measurement.
    Saves to results/sage_baseline.csv.
    """
    from simulation.physics_critic import measure
    from utils.json_to_usd import convert_layout_to_usd

    scenes_dir  = Path(scenes_dir  or SCENES_DIR)
    results_dir = Path(results_dir or RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    zips = sorted(scenes_dir.glob("*.zip"))
    if not zips:
        log.error(f"No scenes found in {scenes_dir}")
        return

    rng      = random.Random(seed)
    selected = rng.sample(zips, min(n_scenes, len(zips)))
    log.info(f"Selected {len(selected)} scenes (seed={seed})")

    csv_path     = results_dir / "sage_baseline.csv"
    write_header = not csv_path.exists()
    rows         = []

    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "scene_id", "n_objects",
                "penetration_pairs", "floating_objects",
                "unstable_objects", "failure_score"
            ])

        for idx, zip_path in enumerate(selected):
            scene_id = zip_path.stem[:12]
            print(f"\n[{idx+1}/{len(selected)}] {scene_id}")

            tmp = None
            try:
                tmp = Path(tempfile.mkdtemp())

                with zipfile.ZipFile(zip_path) as z:
                    z.extractall(tmp)

                jsons = list(tmp.rglob("layout_*.json"))
                if not jsons:
                    raise FileNotFoundError("No layout JSON")

                usd_path = str(tmp / f"{scene_id}.usd")
                convert_layout_to_usd(str(jsons[0]), usd_path)

                if not Path(usd_path).exists():
                    raise FileNotFoundError("USD not created")

                report = measure(usd_path)

                n_obj  = report.get("n_objects", 0)
                n_pen  = len(report["penetration_pairs"])
                n_flt  = len(report["floating_objects"])
                n_unst = len(report["unstable_objects"])
                F      = report["failure_score"]

                print(f"  Objects    : {n_obj}")
                print(f"  Penetration: {n_pen}")
                print(f"  Floating   : {n_flt}")
                print(f"  Unstable   : {n_unst}")
                print(f"  F score    : {F:.4f}")

                row = [scene_id, n_obj, n_pen, n_flt, n_unst, F]
                w.writerow(row)
                f.flush()
                rows.append(row)

            except Exception as e:
                log.error(f"  FAILED: {e}")
                w.writerow([scene_id, 0, 0, 0, 0, 0.0])
                f.flush()

            finally:
                if tmp and tmp.exists():
                    shutil.rmtree(tmp, ignore_errors=True)

    # Summary
    valid = [r for r in rows if r[1] > 0]
    if valid:
        mean_F   = sum(r[5] for r in valid) / len(valid)
        mean_flt = sum(r[3] for r in valid) / len(valid)
        mean_pen = sum(r[2] for r in valid) / len(valid)

        print()
        print("=" * 55)
        print(f"  SAGE-10K BASELINE — {len(valid)}/{len(selected)} scenes")
        print("=" * 55)
        print(f"  Mean F score    : {mean_F:.4f}")
        print(f"  Mean floating   : {mean_flt:.1f} objects")
        print(f"  Mean penetration: {mean_pen:.1f} pairs")
        print(f"  Results saved   : {csv_path}")
        print("=" * 55)
        print()
        print("TABLE 1 — Initial Physics Violations (B1 Baseline):")
        print(f"{'Scene':<15} {'Obj':>5} {'Pen':>5} "
              f"{'Flt':>5} {'Unst':>6} {'F':>10}")
        print("-" * 55)
        for r in rows:
            print(f"{r[0]:<15} {r[1]:>5} {r[2]:>5} "
                  f"{r[3]:>5} {r[4]:>6} {r[5]:>10.4f}")


def run_correction_baseline(
        n_scenes:    int           = 5,
        scenes_dir:  Optional[Path] = None,
        results_dir: Optional[Path] = None,
        seed:        int           = SEED) -> None:
    """
    Run full correction loop on N SAGE-10K scenes.
    Requires LLM backend.
    """
    from correction.correction_engine import run as correct
    from utils.json_to_usd import convert_layout_to_usd

    scenes_dir  = Path(scenes_dir  or SCENES_DIR)
    results_dir = Path(results_dir or RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    zips     = sorted(scenes_dir.glob("*.zip"))
    rng      = random.Random(seed)
    selected = rng.sample(zips, min(n_scenes, len(zips)))

    for idx, zip_path in enumerate(selected):
        scene_id = zip_path.stem[:12]
        print(f"\n[{idx+1}/{len(selected)}] {scene_id}")
        tmp = None
        try:
            tmp = Path(tempfile.mkdtemp())
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(tmp)
            jsons = list(tmp.rglob("layout_*.json"))
            if not jsons:
                continue
            usd_path = str(tmp / f"{scene_id}.usd")
            convert_layout_to_usd(str(jsons[0]), usd_path)
            correct(usd_path, scene_id,
                    max_iter=10,
                    results_dir=str(results_dir))
        except Exception as e:
            log.error(f"  FAILED: {e}")
        finally:
            if tmp and tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(
        description="NemoForge SAGE-10K Benchmark")
    p.add_argument("--mode",
        choices=["baseline", "correction"],
        default="baseline")
    p.add_argument("--n-scenes", type=int, default=5)
    p.add_argument("--seed",     type=int, default=SEED)
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = p.parse_args()

    if args.mode == "baseline":
        run_baseline(
            n_scenes=args.n_scenes,
            seed=args.seed,
            results_dir=Path(args.results_dir)
        )
    else:
        run_correction_baseline(
            n_scenes=args.n_scenes,
            seed=args.seed,
            results_dir=Path(args.results_dir)
        )
