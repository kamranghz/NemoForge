"""
NemoForge Correction Engine
Runs the Reason → Act → Reflect loop on a USD scene.
"""
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

from simulation.physics_critic import measure
from agent.nemoclaw_client import get_client
from correction.prompt_builder import build_prompt

F_THRESHOLD      = 0.5
MAX_ITER_DEFAULT = 10
RESULTS_DIR      = Path("results")


class CorrectionEngine:

    def __init__(self,
                 usd_path:    str,
                 scene_id:    str,
                 max_iter:    int  = MAX_ITER_DEFAULT,
                 results_dir: Path = RESULTS_DIR):
        self.usd_path    = Path(usd_path)
        self.scene_id    = scene_id
        self.max_iter    = max_iter
        self.results_dir = Path(results_dir)
        self.llm         = get_client()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "correction_logs").mkdir(
            parents=True, exist_ok=True)

    def run(self) -> dict:
        log.info(f"Correction start: {self.scene_id}")

        # Initial measurement
        initial_report = measure(str(self.usd_path))
        initial_F      = initial_report['failure_score']

        self._print_header(initial_report)

        if initial_F == 0.0:
            log.info("Scene already valid")
            return self._finish(initial_F, initial_F, 0, [], True)

        ok, backend = self.llm.is_available()
        if not ok:
            log.error("No LLM backend — aborting")
            return self._finish(
                initial_F, initial_F, 0, [], False,
                status="NO_LLM")

        log.info(f"LLM backend: {backend}")

        report         = initial_report
        current_F      = initial_F
        action_history = []
        iter_logs      = []

        for iteration in range(self.max_iter):
            log.info(
                f"Iter {iteration+1}/{self.max_iter} | F={current_F:.4f}")

            # REASON
            prompt = build_prompt(
                user_input=self.scene_id,
                physics_report=report,
                iteration=iteration,
                action_history=action_history
            )

            # ACT
            try:
                action = self.llm.correct(prompt)
                log.info(
                    f"  Action: {action['action']} — "
                    f"{action['reasoning']}")
            except Exception as e:
                log.error(f"  LLM error: {e}")
                iter_logs.append({
                    "iteration": iteration,
                    "error": str(e),
                    "F": current_F
                })
                continue

            # REFLECT — re-measure
            new_report = measure(str(self.usd_path))
            new_F      = new_report['failure_score']
            delta_F    = new_F - current_F

            log.info(
                f"  New F={new_F:.4f} (delta {delta_F:+.4f})")

            if new_F > current_F:
                log.info("  F worsened — skipping")
                iter_logs.append({
                    "iteration": iteration,
                    "action": action,
                    "F_before": current_F,
                    "F_after": new_F,
                    "rolled_back": True
                })
                continue

            action_history.append({
                "action":     action['action'],
                "parameters": action['parameters'],
                "reasoning":  action['reasoning'],
                "delta_F":    round(delta_F, 4),
                "outcome":    "improved" if delta_F < 0 else "unchanged"
            })
            iter_logs.append({
                "iteration": iteration,
                "action": action,
                "F_before": current_F,
                "F_after": new_F,
                "delta_F": delta_F,
                "rolled_back": False
            })

            report    = new_report
            current_F = new_F

            if current_F < F_THRESHOLD:
                log.info(f"  SUCCESS — F={current_F:.4f}")
                break

        # Save logs
        log_path = (self.results_dir / "correction_logs" /
                    f"{self.scene_id}.json")
        log_path.write_text(json.dumps(iter_logs, indent=2))

        success = current_F < F_THRESHOLD
        return self._finish(
            initial_F, current_F, len(action_history),
            action_history, success)

    def _finish(self,
                initial_F:  float,
                final_F:    float,
                iterations: int,
                history:    list,
                success:    bool,
                status:     str = "") -> dict:

        rfpcr = 1 if success else 0
        summary = {
            "scene_id":       self.scene_id,
            "usd_path":       str(self.usd_path),
            "initial_F":      round(initial_F, 4),
            "final_F":        round(final_F, 4),
            "delta_F":        round(final_F - initial_F, 4),
            "iterations":     iterations,
            "rfpcr":          rfpcr,
            "success":        success,
            "status":         status or ("SUCCESS" if success
                                         else "PARTIAL"),
            "action_history": history
        }

        # Save CSV
        csv_path     = self.results_dir / "correction_results.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "scene_id", "usd_path", "initial_F",
                "final_F", "delta_F", "iterations",
                "rfpcr", "success", "status"
            ])
            if write_header:
                w.writeheader()
            w.writerow({k: summary[k] for k in [
                "scene_id", "usd_path", "initial_F",
                "final_F", "delta_F", "iterations",
                "rfpcr", "success", "status"
            ]})

        self._print_summary(summary)
        return summary

    def _print_header(self, report: dict) -> None:
        print()
        print("=" * 60)
        print(f"  Scene    : {self.scene_id}")
        print(f"  USD      : {self.usd_path}")
        print(f"  Objects  : {report.get('n_objects', '?')}")
        print(f"  Initial F: {report['failure_score']:.4f}")
        print(f"  Penetr.  : {len(report['penetration_pairs'])}")
        print(f"  Floating : {len(report['floating_objects'])}")
        print(f"  Unstable : {len(report['unstable_objects'])}")
        print("=" * 60)

    def _print_summary(self, s: dict) -> None:
        print()
        print("=" * 60)
        print(f"  RESULT: {s['scene_id']}")
        print("=" * 60)
        print(f"  Initial F  : {s['initial_F']:.4f}")
        print(f"  Final F    : {s['final_F']:.4f}")
        print(f"  Delta F    : {s['delta_F']:+.4f}")
        print(f"  Iterations : {s['iterations']}")
        print(f"  RFPCR      : {s['rfpcr']}")
        print(f"  Status     : {s['status']}")
        print("=" * 60)


def run(usd_path:    str,
        scene_id:    str,
        max_iter:    int  = MAX_ITER_DEFAULT,
        results_dir: str  = str(RESULTS_DIR)) -> dict:
    engine = CorrectionEngine(
        usd_path, scene_id, max_iter, Path(results_dir))
    return engine.run()


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--usd",        required=True)
    p.add_argument("--scene-id",   required=True)
    p.add_argument("--max-iter",   type=int, default=MAX_ITER_DEFAULT)
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = p.parse_args()
    result = run(args.usd, args.scene_id,
                 args.max_iter, args.results_dir)
    print(json.dumps(
        {k: v for k, v in result.items()
         if k != "action_history"}, indent=2))
