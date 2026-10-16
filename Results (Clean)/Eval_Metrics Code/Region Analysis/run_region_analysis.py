#!/usr/bin/env python3
"""Stage 1 of the relevance-region analysis: compute the per-fold, per-region
metrics for every dataset, model family and strategy.

Writes, following the raw_*_by_dataset convention the RMSE and SERA
generators already use:
  Results Data/raw_region_by_dataset/DSxx.csv   per (family, strategy, fold, region)
  Results Data/raw_region_by_dataset/diag.csv   per dataset rare-region diagnostics
  Results Data/raw_region_by_dataset/quantiles.csv  rare-region error quantile ratios

Usage: python3 run_region_analysis.py [DS01 ...]   (default: every dataset
present; EXP002_WORKERS processes, default 6)
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from region_metrics import (  # noqa: E402
    build_fold_cache, error_quantiles, load_baseline, load_cell, rare_diagnostics,
    region_fold_metrics,
)
from paths import ROOT  # noqa: E402
from tables_common import MODEL_FAMILIES, STRATEGIES, datasets_on_disk  # noqa: E402

DATA_DIR = ROOT / "Results (Clean)/Results Data/raw_predictions_by_dataset"
OUT = ROOT / "Results (Clean)/Results Data/raw_region_by_dataset"
FAMILIES = list(MODEL_FAMILIES)


def run_dataset(ds: str) -> str:
    t0 = time.time()
    rows, quant, diag = [], [], None
    for fam in FAMILIES:
        base_preds, ctrl = load_baseline(DATA_DIR, ds, fam)
        cache = build_fold_cache(ctrl)
        for strat in STRATEGIES[1:]:
            cell = load_cell(DATA_DIR, ds, fam, strat, base_preds, cache)
            m = region_fold_metrics(cell)
            m.insert(0, "strategy", strat); m.insert(0, "family", fam)
            rows.append(m)
            q = error_quantiles(cell)
            if q:
                quant.append({"dataset": ds, "family": fam, "strategy": strat, **q})
            if diag is None:
                diag = {"dataset": ds, **rare_diagnostics(cell)}
    out = pd.concat(rows, ignore_index=True)
    out.insert(0, "dataset", ds)
    out.to_csv(OUT / f"{ds}.csv", index=False)
    pd.DataFrame([diag]).to_csv(OUT / f"diag_{ds}.csv", index=False)
    pd.DataFrame(quant).to_csv(OUT / f"quant_{ds}.csv", index=False)
    return f"{ds} {time.time() - t0:.0f}s"


def main(argv):
    OUT.mkdir(parents=True, exist_ok=True)
    todo = argv or datasets_on_disk(DATA_DIR)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=int(os.environ.get("EXP002_WORKERS", 6))) as ex:
        for msg in ex.map(run_dataset, todo):
            print(msg, flush=True)
    # Merge the per-dataset shards, then drop them: the workers write one each
    # so they never contend for a file, but only the merged form is an input
    # to anything downstream, and leaving 48 redundant CSVs in a tracked
    # directory just makes the diffs noisy.
    for name, pat in (("diag.csv", "diag_DS??.csv"), ("quantiles.csv", "quant_DS??.csv")):
        shards = sorted(OUT.glob(pat))
        pd.concat([pd.read_csv(f) for f in shards]).to_csv(OUT / name, index=False)
        for f in shards:
            f.unlink()
    print(f"done {len(todo)} datasets in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
