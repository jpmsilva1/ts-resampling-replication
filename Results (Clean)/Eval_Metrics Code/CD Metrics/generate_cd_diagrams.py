#!/usr/bin/env python3
"""Generates critical-difference diagrams -- one per (metric, model family)
-- matching the visual style of the second paper's Figs. 6/7 (see cd_plot.py
for the exact reference). For each of our 5 model families (LM/SVM/MARS/RF/
RPART), ranks the 10 resampling strategies against each other across the 20
datasets (Friedman/Nemenyi setup: 20 datasets = blocks, 10 strategies =
treatments), using that dataset's MEAN value across the 50 Monte Carlo
iterations -- the same aggregate each strategy's table cell already shows.

Source data:
  F1:   Results (Clean)/Results Data/raw_iterations_by_dataset_v2/*.csv (data_v2 run,
        same run as RMSE/SERA -- see CLAUDE.md "2026-08-28 -- F1 re-extracted from data_v2")
  RMSE: Results (Clean)/Results Data/raw_rmse_by_dataset/*.csv
  SERA: Results (Clean)/Results Data/raw_sera_by_dataset/*.csv
Output: Results (Clean)/Evaluation Metrics/Figures/CD/cd_{metric}_{family}.png
"""
from pathlib import Path

import pandas as pd

from cd_stats import average_ranks, friedman_p, nemenyi_cd
from cd_plot import plot_cd_diagram

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # make Eval_Metrics Code/ importable when run directly
from paths import ROOT  # noqa: E402
RESULTS_DATA = ROOT / "Results (Clean)/Results Data"
OUT_DIR = ROOT / "Results (Clean)/Evaluation Metrics/Figures/CD"

MODEL_FAMILIES = {"lm": "Linear Model (LM)", "svm": "SVM", "mars": "MARS",
                  "rf": "Random Forest (RF)", "rpart": "CART (RPART)"}
STRATEGIES = ["baseline", "UNDERB", "UNDERT", "UNDERTPhi", "OVERB", "OVERT",
              "OVERTPhi", "SMOTEB", "SMOTET", "SMOTETPhi"]
# Derived per-matrix inside main(), never hardcoded: Nemenyi's critical
# distance scales as 1/sqrt(n), so a stale N silently produces a WRONG CD
# bar rather than an error. See EXPERIMENTAL_PROTOCOL.md, 'N is derived'.
ALPHA = 0.05

# metric -> (value column, higher_is_better, how to load one combined
# long-format frame with dataset_id/workflow/value columns)
METRICS = {"F1": {"higher_is_better": True}, "RMSE": {"higher_is_better": False},
           "SERA": {"higher_is_better": False}}


def _parse_workflow(wf: str):
    x = wf.removeprefix("mc.")
    model_family = x.split("_")[0]
    strategy = x.split("_", 1)[1] if "_" in x else "baseline"
    return model_family, strategy


def load_f1_long() -> pd.DataFrame:
    files = sorted((RESULTS_DATA / "raw_iterations_by_dataset_v2").glob("DS??_*.csv"))
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df[["model_family", "strategy"]] = df["workflow"].apply(lambda w: pd.Series(_parse_workflow(w)))
    return df.rename(columns={"F1": "value"})


def load_metric_long(metric: str) -> pd.DataFrame:
    if metric == "F1":
        return load_f1_long()
    subdir = {"RMSE": "raw_rmse_by_dataset", "SERA": "raw_sera_by_dataset"}[metric]
    files = sorted((RESULTS_DATA / subdir).glob("DS??.csv"))
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    value_col = metric.lower()
    return df.rename(columns={value_col: "value"})


def dataset_by_strategy_matrix(long_df: pd.DataFrame, family: str) -> pd.DataFrame:
    sub = long_df[long_df["model_family"] == family]
    per_ds = sub.groupby(["dataset_id", "strategy"])["value"].mean().reset_index()
    matrix = per_ds.pivot(index="dataset_id", columns="strategy", values="value")
    return matrix[STRATEGIES]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for metric, cfg in METRICS.items():
        long_df = load_metric_long(metric)
        for family, family_label in MODEL_FAMILIES.items():
            matrix = dataset_by_strategy_matrix(long_df, family)
            n_datasets = matrix.shape[0]
            assert matrix.shape[1] == len(STRATEGIES), matrix.shape
            assert n_datasets >= 5, f"only {n_datasets} datasets -- Nemenyi is not meaningful"

            # Demsar's procedure: the Nemenyi post-hoc is only licensed once
            # the Friedman omnibus null is rejected. Report p on every figure
            # so a non-significant panel can't be misread as showing real
            # differences (F1/rpart currently sits at p=.056 -- see CLAUDE.md).
            p = friedman_p(matrix)
            avg_ranks = average_ranks(matrix, higher_is_better=cfg["higher_is_better"])
            cd = nemenyi_cd(k=len(STRATEGIES), n=n_datasets, alpha=0.05)

            if p >= ALPHA:
                print(f"  WARNING: {metric}/{family} Friedman p = {p:.4f} >= {ALPHA} "
                      f"-- omnibus null NOT rejected; CD bars are not licensed here.")

            p_str = f"p < 0.001" if p < 0.001 else f"p = {p:.3f}"
            flag = "" if p < ALPHA else " -- NOT SIGNIFICANT"
            out_file = OUT_DIR / f"cd_{metric.lower()}_{family}.png"
            plot_cd_diagram(
                avg_ranks.to_dict(), cd,
                title=(f"{metric} -- {family_label} (CD = {cd:.3f}, N = {n_datasets} "
                       f"datasets, Friedman {p_str}{flag})"),
                out_path=str(out_file), k=len(STRATEGIES),
            )
            print(f"Wrote {out_file}")
    print("Done.")


if __name__ == "__main__":
    main()
