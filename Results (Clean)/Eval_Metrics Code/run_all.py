#!/usr/bin/env python3
"""Run the whole evaluation pipeline in dependency order.

The stage order below is not cosmetic -- it is the actual DAG:

  Stage 1  RMSE Metric/ and SERA Metric/ read raw_predictions_by_dataset/ and
           WRITE raw_rmse_by_dataset/ and raw_sera_by_dataset/. Every later
           stage reads those, so these must run first.
  Stage 2  Per-metric appendix tables (F1 needs only the raw F1 iterations).
  Stage 3  Rankings and significance -- CD, Bayes, Summary. Heatmaps and
           Rank Distribution import load_metric_long() from CD Metrics, so CD
           must be importable (it is; the import does not require CD to have
           been run first, but keeping the order makes failures read in order).
  Stage 4  Regression, then Causal -- Causal imports from Regression.
  Stage 5  Paper Comparison -- independent of 2-4, run last so its console
           summary is the final thing printed.

Usage:
    python3 run_all.py              # everything
    python3 run_all.py --stage 3    # one stage
    python3 run_all.py --list       # show the DAG without running
    python3 run_all.py --tests      # run the test suite instead

Each generator is run as a subprocess with its own directory as cwd, because
they resolve sibling modules with sys.path.insert(parent) and expect that.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from paths import ROOT

CODE = Path(__file__).resolve().parent

# (stage, directory, script)
PIPELINE = [
    (1, "RMSE Metric", "generate_rmse_tables.py"),
    (1, "SERA Metric", "generate_sera_tables.py"),
    (2, "F1 Metric", "generate_f1_tables.py"),
    (3, "CD Metrics", "generate_cd_diagrams.py"),
    (3, "Bayes Signed-Rank", "generate_bayes_diagrams.py"),
    (3, "Summary Tables", "generate_summary_tables.py"),
    (3, "Rank Distribution", "generate_rank_boxplots.py"),
    (3, "Heatmaps", "generate_rank_heatmap.py"),
    (3, "Heatmaps", "generate_value_heatmap.py"),
    (3, "Heatmaps", "generate_delta_heatmap.py"),
    (3, "Heatmaps", "generate_bayes_heatmap.py"),
    (3, "Heatmaps", "generate_win_count_heatmap.py"),
    (3, "Overview Figures", "generate_overview_figures.py"),
    (3, "Overview Figures", "generate_overview_figures_sera.py"),
    (3, "Metric Diagnostics", "generate_metric_disagreement.py"),
    (3, "Metric Diagnostics", "generate_precision_recall.py"),
    (4, "Regression", "generate_regression.py"),
    (4, "Causal", "generate_causal_effects.py"),
    (4, "Causal", "generate_causal_tables.py"),
    (4, "Causal", "generate_causal_figures.py"),
    (4, "Causal", "generate_effect_forest_plots.py"),
    (5, "Paper Comparison", "generate_paper_comparison.py"),
]


def run_tests() -> int:
    """Every test_*.py under Eval_Metrics Code/. Run this BEFORE regenerating
    anything on new results -- the guards catch the hardcoded-20 class of bug
    that silently drops added datasets."""
    print("=== test suite ===")
    r = subprocess.run([sys.executable, "-m", "pytest", str(CODE), "-q"], cwd=CODE)
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, help="run only this stage")
    ap.add_argument("--list", action="store_true", help="print the DAG and exit")
    ap.add_argument("--tests", action="store_true", help="run the test suite and exit")
    args = ap.parse_args()

    if args.tests:
        return run_tests()

    if args.list:
        for stage, d, s in PIPELINE:
            print(f"  stage {stage}  {d}/{s}")
        return 0

    todo = [p for p in PIPELINE if args.stage is None or p[0] == args.stage]
    print(f"ROOT = {ROOT}")
    print(f"Running {len(todo)} generators\n")

    failures = []
    for stage, d, script in todo:
        wd = CODE / d
        t0 = time.time()
        print(f"[stage {stage}] {d}/{script} ... ", end="", flush=True)
        r = subprocess.run([sys.executable, script], cwd=wd,
                           capture_output=True, text=True)
        if r.returncode == 0:
            print(f"ok ({time.time()-t0:.1f}s)")
        else:
            print("FAILED")
            print(r.stdout[-2000:])
            print(r.stderr[-2000:])
            failures.append(f"{d}/{script}")

    print()
    if failures:
        print(f"{len(failures)} FAILED: " + ", ".join(failures))
        return 1
    print(f"All {len(todo)} generators completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
