#!/usr/bin/env python3
"""Heatmap D: compresses the 15 separate Bayes signed-rank bar charts
(Bayes Signed-Rank/generate_bayes_diagrams.py, 5 families x 3 metrics) into
3 compact heatmap panels -- one glance instead of 15 panels.

Same 9x5 grid as heatmap C (9 non-baseline strategies x 5 families), one
panel per metric. Categorical/status encoding (3 discrete outcomes), reusing
the *exact* palette already established in the Bayes bar charts
(C_TEAL=strategy wins, C_GOLD=practical draw, C_GREY=baseline wins) so this
reads as the same finding, just compressed -- not a new color language to
learn. Each cell is colored by whichever outcome has the highest posterior
probability, annotated with that probability.

Output: Results (Clean)/Evaluation Metrics/Figures/Heatmaps/heatmap_bayes.{pdf,png}
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # make Eval_Metrics Code/ importable when run directly
from paths import FIG_DIR  # noqa: E402
from heatmap_common import FAM, FAMLAB, STRAT, add_minor_grid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "CD Metrics"))
from generate_cd_diagrams import load_metric_long, dataset_by_strategy_matrix  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Bayes Signed-Rank"))
from bayes_stats import bayes_signedrank_probs, log_ratio_diffs  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_style import save, C_TEAL, C_GOLD, C_GREY  # noqa: E402

STRAT_NB = STRAT[1:]
METRICS = [("$F_1^{\\phi}$", "F1", True, 0.01),
           ("SERA", "SERA", False, np.log(1.05)),
           ("RMSE", "RMSE", False, np.log(1.05))]
OUTCOME_COLORS = [C_GREY, C_GOLD, C_TEAL]  # index 0=baseline wins,1=draw,2=strategy wins


def cell_probs(metric: str, higher_is_better: bool, rope: float):
    """9 (strategy) x 5 (family) array of (p_baseline, p_rope, p_strategy)
    triples -- same bayes_signedrank_probs() call generate_bayes_diagrams.py
    already makes per (metric, family, strategy), just laid out as a grid
    instead of 5 separate bar-chart figures."""
    long_df = load_metric_long(metric)
    probs = np.zeros((len(STRAT_NB), len(FAM), 3))
    for j, fam in enumerate(FAM):
        m = dataset_by_strategy_matrix(long_df, fam)
        for i, s in enumerate(STRAT_NB):
            if higher_is_better:
                diffs = (m[s] - m["baseline"]).values
            else:
                diffs = log_ratio_diffs(m[s].values, m["baseline"].values)
            probs[i, j] = bayes_signedrank_probs(diffs, rope=rope, seed=0)
    return probs


def main():
    OUT_DIR = FIG_DIR / "Heatmaps"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # No fig.suptitle -- the LaTeX caption already carries this (see
    # heatmap_value's docstring note for why).
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.3))

    for ax, (title, metric, hib, rope) in zip(axes, METRICS):
        probs = cell_probs(metric, hib, rope)
        dominant = probs.argmax(axis=2)
        rgb = np.zeros((*dominant.shape, 3))
        for k in range(3):
            color = np.array(plt.matplotlib.colors.to_rgb(OUTCOME_COLORS[k]))
            rgb[dominant == k] = color
        ax.imshow(rgb, aspect="auto")
        for i in range(dominant.shape[0]):
            for j in range(dominant.shape[1]):
                p = probs[i, j, dominant[i, j]]
                txt_color = "white" if dominant[i, j] != 1 else "#222"
                ax.text(j, i, f"{p:.2f}", ha="center", va="center",
                        fontsize=9.0, color=txt_color)
        ax.set_xticks(range(len(FAM)))
        ax.set_xticklabels(FAMLAB, fontsize=10)
        ax.set_title(title, fontsize=13, pad=8)
        add_minor_grid(ax, len(STRAT_NB), len(FAM))

    axes[0].set_yticks(range(len(STRAT_NB)))
    axes[0].set_yticklabels(STRAT_NB, fontsize=10)
    for ax in axes[1:]:
        ax.set_yticks(range(len(STRAT_NB)))
        ax.set_yticklabels([])

    handles = [mpatches.Patch(color=C_TEAL, label="Strategy wins"),
               mpatches.Patch(color=C_GOLD, label="Draw (ROPE)"),
               mpatches.Patch(color=C_GREY, label="Baseline wins")]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=10)

    fig.tight_layout()
    save(fig, OUT_DIR / "heatmap_bayes.png")
    print("Wrote", OUT_DIR / "heatmap_bayes.pdf")


if __name__ == "__main__":
    main()
