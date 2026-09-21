#!/usr/bin/env python3
"""Bayesian signed-rank charts -- one per (metric, model family) -- comparing
each of the 9 resampling strategies against the 'baseline' (no resampling),
mirroring Cerqueira, Torgo & Mozetic (2020) Figs 12/14. Complements the CD
diagrams (Friedman/Nemenyi omnibus ranking) with a pairwise, ungated
probability that a given strategy practically beats doing nothing.

Source data: same as CD Metrics/generate_cd_diagrams.py
  F1:   Results (Clean)/Results Data/raw_iterations_by_dataset_v2/*.csv
  RMSE: Results (Clean)/Results Data/raw_rmse_by_dataset/*.csv
  SERA: Results (Clean)/Results Data/raw_sera_by_dataset/*.csv
Output: Results (Clean)/Evaluation Metrics/Figures/Bayes/bayes_{metric}_{family}.png
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from bayes_stats import bayes_signedrank_probs, log_ratio_diffs

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # make Eval_Metrics Code/ importable when run directly
from paths import ROOT  # noqa: E402
CD_METRICS_DIR = ROOT / "Results (Clean)/Eval_Metrics Code/CD Metrics"
sys.path.insert(0, str(CD_METRICS_DIR))
sys.path.insert(0, str(ROOT / "Results (Clean)/Eval_Metrics Code"))
from generate_cd_diagrams import load_metric_long, dataset_by_strategy_matrix, MODEL_FAMILIES, STRATEGIES
from plot_style import apply_style, save, C_DARK, C_TEAL, C_GOLD, C_GREY

apply_style()

OUT_DIR = ROOT / "Results (Clean)/Evaluation Metrics/Figures/Bayes"

# metric -> (higher_is_better, ROPE on the comparison scale used below)
METRICS = {
    "F1": {"higher_is_better": True, "rope": 0.01},
    "RMSE": {"higher_is_better": False, "rope": np.log(1.05)},
    "SERA": {"higher_is_better": False, "rope": np.log(1.05)},
}
STRATEGY_LABELS = [s for s in STRATEGIES if s != "baseline"]


def diffs_for(matrix, strategy: str, higher_is_better: bool) -> np.ndarray:
    if higher_is_better:
        return (matrix[strategy] - matrix["baseline"]).values
    return log_ratio_diffs(strategy=matrix[strategy].values, baseline=matrix["baseline"].values)


def plot_bayes_chart(probs: dict, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    strategies = list(probs.keys())
    baseline_wins = [probs[s][0] for s in strategies]
    draws = [probs[s][1] for s in strategies]
    strategy_wins = [probs[s][2] for s in strategies]

    y = np.arange(len(strategies))
    ax.barh(y, strategy_wins, color=C_TEAL, label="Strategy wins",
            edgecolor="white", linewidth=0.5)
    ax.barh(y, draws, left=strategy_wins, color=C_GOLD, label="Draw (ROPE)",
            edgecolor="white", linewidth=0.5)
    left2 = [s + d for s, d in zip(strategy_wins, draws)]
    ax.barh(y, baseline_wins, left=left2, color=C_GREY, label="Baseline wins",
            edgecolor="white", linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(strategies, fontsize=14)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Posterior probability", fontsize=14)
    ax.tick_params(axis="x", labelsize=13)
    ax.grid(axis="x", alpha=0.15)
    ax.grid(axis="y", visible=False)
    ax.set_title(title, fontsize=17, fontweight="bold")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3,
              frameon=False, fontsize=13)
    fig.tight_layout()
    save(fig, out_path)
    print(f"Wrote {out_path}")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for metric, cfg in METRICS.items():
        long_df = load_metric_long(metric)
        for family, family_label in MODEL_FAMILIES.items():
            matrix = dataset_by_strategy_matrix(long_df, family)
            probs = {}
            for strategy in STRATEGY_LABELS:
                diffs = diffs_for(matrix, strategy, cfg["higher_is_better"])
                probs[strategy] = bayes_signedrank_probs(diffs, rope=cfg["rope"], seed=0)
            out_file = OUT_DIR / f"bayes_{metric.lower()}_{family}.png"
            plot_bayes_chart(probs, f"{metric} vs. Baseline -- {family_label}", out_file)
    print("Done.")


if __name__ == "__main__":
    main()
