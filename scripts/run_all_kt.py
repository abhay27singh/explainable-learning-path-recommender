#!/usr/bin/env python
"""Run every Stage 1 configuration in one go, resumably.

All runs use identical flags and identical code, so Table I is internally
comparable. Runs already present in results/ are skipped, which means that after a
Colab disconnect you re-run this same command and it picks up where it stopped.

    python scripts/run_all_kt.py --backup /content/drive/MyDrive/elpr_results
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ROOT / "artifacts" / "models"

# (results-file stem, extra CLI arguments)
RUNS: list[tuple[str, list[str]]] = [
    # The proposed model and its two component ablations.
    ("kt_proposed",           ["--variant", "proposed"]),
    ("kt_no_graph",           ["--variant", "no_graph"]),
    ("kt_no_time",            ["--variant", "no_time"]),
    # The cold-concept pair: the only test that can detect prerequisite propagation,
    # because ordinary AUC is measured solely on concepts that already have
    # abundant direct supervision.
    ("kt_proposed_cold0.3",   ["--variant", "proposed", "--cold-concepts", "0.3"]),
    ("kt_no_graph_cold0.3",   ["--variant", "no_graph", "--cold-concepts", "0.3"]),
    # Table I baselines.
    ("kt_dkt",                ["--variant", "dkt"]),
    ("kt_sakt",               ["--variant", "sakt"]),
    ("kt_gnn",                ["--variant", "gnn"]),
    ("kt_majority",           ["--variant", "majority", "--epochs", "1"]),
]

COMMON = ["--folds", "5", "--epochs", "12", "--workers", "2", "--no-amp"]


def backup(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in (RESULTS, MODELS):
        if source.exists():
            shutil.copytree(source, destination / source.name, dirs_exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup", default="", help="directory to copy results into after each run")
    parser.add_argument("--force", action="store_true", help="re-run configurations already present")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    destination = Path(args.backup) if args.backup else None

    for i, (stem, extra) in enumerate(RUNS, start=1):
        target = RESULTS / f"{stem}.json"
        if target.exists() and not args.force:
            print(f"[{i}/{len(RUNS)}] {stem}: already done, skipping")
            continue

        # Later flags win in argparse, so COMMON's --epochs is overridden where a run
        # supplies its own.
        command = [sys.executable, str(ROOT / "scripts" / "05_train_kt.py")] + COMMON + extra
        print(f"\n[{i}/{len(RUNS)}] {stem}\n  {' '.join(command[1:])}", flush=True)

        start = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT)
        elapsed = (time.perf_counter() - start) / 60

        if result.returncode != 0:
            print(f"  FAILED after {elapsed:.1f} min (exit {result.returncode})")
            print("  Continuing; re-run this script to retry.")
            continue

        print(f"  done in {elapsed:.1f} min")
        if destination:
            backup(destination)
            print(f"  backed up to {destination}")

    print("\n" + "=" * 68)
    print(f"{'run':<24}{'AUC':>18}{'RMSE':>18}{'ECE':>8}")
    print("=" * 68)
    for stem, _ in RUNS:
        target = RESULTS / f"{stem}.json"
        if not target.exists():
            print(f"{stem:<24}{'MISSING':>18}")
            continue
        data = json.loads(target.read_text())
        print(
            f"{stem:<24}"
            f"{data['auc_mean']:>10.4f} ± {data['auc_std']:.4f}"
            f"{data['rmse_mean']:>10.4f} ± {data['rmse_std']:.4f}"
            f"{data['ece_calibrated_mean']:>8.4f}"
        )
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
