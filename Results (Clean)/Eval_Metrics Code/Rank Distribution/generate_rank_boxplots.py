#!/usr/bin/env python3
"""Rank-distribution boxplots -- one figure per metric, 5 panels (one per
model family) -- showing each resampling strategy's distribution of
per-dataset ranks across the 20 datasets. Complements the CD diagrams, which
only show each strategy's MEAN rank +/- the shared CD bar: this shows whether
a strategy is consistently mid-pack or swings between best and worst.

Source data: same as CD Metrics/generate_cd_diagrams.py
Output: Results (Clean)/Evaluation Metrics/Figures/Rank Distribution/rank_boxplot_{metric}.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # make Eval_Metrics Code/ importable when run directly
from paths import ROOT  # noqa: E402
CD_METRICS_DIR = ROOT / "Results (Clean)/Eval_Metrics Code/CD Metrics"
sys.path.insert(0, str(CD_METRICS_DIR))
sys.path.insert(0, str(ROOT / "Results (Clean)/Eval_Metrics Code"))
from generate_cd_diagrams import load_metric_long, dataset_by_strategy_matrix, MODEL_FAMILIES, STRATEGIES
from plot_style import apply_style, save, C_DARK, C_TEAL, C_CORAL

apply_style()

OUT_DIR = ROOT / "Results (Clean)/Evaluation Metrics/Figures/Rank Distribution"
METRICS = {"F1": True, "RMSE": False, "SERA": False}  # higher_is_better


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for metric, higher_is_better in METRICS.items():
        long_df = load_metric_long(metric)
        # Sized for a full-width single-figure landscape page (one metric per
        # page) -- not stacked 3-to-a-page, which forces text back down no
        # matter how large the source fonts are.
        fig, axes = plt.subplots(1, 5, figsize=(26, 8.5), sharey=True)
        fig.suptitle(f"{metric} rank distribution by model family (1 = best)",
                     fontsize=27, fontweight="bold")
        for ax, (family, family_label) in zip(axes, MODEL_FAMILIES.items()):
            matrix = dataset_by_strategy_matrix(long_df, family)
            ranks = matrix.rank(axis=1, ascending=not higher_is_better)
            bp = ax.boxplot([ranks[s].values for s in STRATEGIES], tick_labels=STRATEGIES,
                             patch_artist=True, widths=0.6)
            for box in bp["boxes"]:
                box.set(facecolor=C_TEAL, alpha=0.35, edgecolor=C_DARK, linewidth=1.6)
            for med in bp["medians"]:
                med.set(color=C_CORAL, linewidth=3.0)
            for part in ("whiskers", "caps"):
                for line in bp[part]:
                    line.set(color=C_DARK, linewidth=1.6)
            for fl in bp["fliers"]:
                fl.set(markeredgecolor=C_DARK, markersize=7)
            ax.set_title(family_label, fontsize=22, fontweight="bold")
            ax.tick_params(axis="x", rotation=90, labelsize=19)
            ax.tick_params(axis="y", labelsize=19)
            ax.grid(True, alpha=0.15, axis="y")
        axes[0].set_ylabel("Rank", fontsize=22)
        axes[0].invert_yaxis()
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out_file = OUT_DIR / f"rank_boxplot_{metric.lower()}.png"
        save(fig, out_file)
        print(f"Wrote {out_file}")
        plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
