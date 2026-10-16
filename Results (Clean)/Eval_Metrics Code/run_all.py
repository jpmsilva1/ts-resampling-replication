#!/usr/bin/env python3
"""Run the whole evaluation pipeline in dependency order.

The stage order below is not cosmetic -- it is the actual DAG:

  Stage 1  RMSE Metric/ and SERA Metric/ read raw_predictions_by_dataset/ and
           WRITE raw_rmse_by_dataset/ and raw_sera_by_dataset/. Every later
           stage reads those, so these must run first.
  Stage 2  Per-metric appendix tables (F1 needs only the raw F1 iterations).
           A generator whose script is not present (local-only, not in the
           repo) is skipped with a note rather than failing the run.
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
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
import sys
import time
from pathlib import Path

from paths import ROOT

CODE = Path(__file__).resolve().parent

# (stage, directory, script)
PIPELINE = [
    (1, "RMSE Metric", "generate_rmse_tables.py"),
    (1, "SERA Metric", "generate_sera_tables.py"),
    (1, "Region Analysis", "run_region_analysis.py"),
    (2, "F1 Metric", "generate_f1_tables.py"),
    # verify_region checks the region split against stage 1's published RMSE,
    # so it must follow it; generate_region_tables consumes both.
    (2, "Region Analysis", "verify_region.py"),
    (3, "Region Analysis", "generate_region_tables.py"),
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
    (3, "Metric Diagnostics", "generate_bias_ts_baselines.py"),
    (4, "Regression", "generate_regression.py"),
    (4, "Causal", "generate_causal_effects.py"),
    (4, "Causal", "generate_causal_tables.py"),
    (4, "Causal", "generate_causal_figures.py"),
    (4, "Causal", "generate_effect_forest_plots.py"),
    (4, "Causal", "generate_fold_median_check.py"),
    (5, "Paper Comparison", "generate_paper_comparison.py"),
]


# Optional machine-local generators, same (stage, dir, script) shape, kept out of
# the repo; merged in stage order (sort is stable, so they run after the stage's
# repo generators).
_LOCAL = CODE / "local_pipeline.json"
if _LOCAL.exists():
    PIPELINE = sorted(PIPELINE + [tuple(x) for x in json.loads(_LOCAL.read_text())], key=lambda p: p[0])


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
    ap.add_argument("--jobs", type=int, default=1,
                    help="generators of the SAME stage to run concurrently (stages stay in order)")
    args = ap.parse_args()

    if args.tests:
        return run_tests()

    if args.list:
        for stage, d, s in PIPELINE:
            print(f"  stage {stage}  {d}/{s}")
        return 0

    todo = [p for p in PIPELINE if args.stage is None or p[0] == args.stage]
    absent = [p for p in todo if not (CODE / p[1] / p[2]).exists()]
    todo = [p for p in todo if p not in absent]
    print(f"ROOT = {ROOT}")
    if absent:
        print(f"Skipping {len(absent)} generator(s) not present in this checkout")
    print(f"Running {len(todo)} generators\n")

    def run_one(item):
        stage, d, script = item
        t0 = time.time()
        r = subprocess.run([sys.executable, script], cwd=CODE / d,
                           capture_output=True, text=True)
        return item, r, time.time() - t0

    failures = []
    for stage in sorted({p[0] for p in todo}):
        with ThreadPoolExecutor(args.jobs) as ex:
            for (st, d, script), r, dt in ex.map(run_one, [p for p in todo if p[0] == stage]):
                if r.returncode == 0:
                    print(f"[stage {st}] {d}/{script} ... ok ({dt:.1f}s)", flush=True)
                else:
                    print(f"[stage {st}] {d}/{script} ... FAILED")
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
