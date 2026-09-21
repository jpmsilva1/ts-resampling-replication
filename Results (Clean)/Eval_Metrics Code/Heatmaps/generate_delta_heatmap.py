#!/usr/bin/env python3
"""Heatmap C: change vs. baseline per (strategy, family) -- the magnitude
companion to the Bayes charts (which show *probability* of a win, not
*size*) and the CD diagrams (which show *rank*, not magnitude).

Same 10x5 grid as Figure 1, but only the 9 non-baseline strategy rows (a
strategy's delta against itself is trivially 0). True diverging colormap:
0 = neutral white midpoint, positive (green) = improvement over baseline,
negative (red) = regression -- the textbook diverging case for polarity
data (dataviz skill's color-formula rule).

F1 uses the raw difference (bounded [0,1], no scale issue). SERA/RMSE reuse
Bayes Signed-Rank/bayes_stats.py's log_ratio_diffs() -- the same log-ratio
convention already established for these two scale-heterogeneous metrics
elsewhere in the project (Bayes charts' ROPE), so this heatmap is directly
comparable to those.

Output: Results (Clean)/Evaluation Metrics/Figures/Heatmaps/heatmap_delta.{pdf,png}
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # make Eval_Metrics Code/ importable when run directly
from paths import FIG_DIR  # noqa: E402
from heatmap_common import FAM, FAMLAB, STRAT, add_minor_grid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "CD Metrics"))
from generate_cd_diagrams import load_metric_long, dataset_by_strategy_matrix  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Bayes Signed-Rank"))
from bayes_stats import log_ratio_diffs  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_style import save  # noqa: E402

STRAT_NB = STRAT[1:]  # non-baseline strategies (9 rows)
METRICS = [("$\\Delta F_1^{\\phi}$ vs. baseline", "F1", "{:+.3f}"),
           ("$\\Delta$SERA vs. baseline (log ratio)", "SERA", "{:+.2f}"),
           ("$\\Delta$RMSE vs. baseline (log ratio)", "RMSE", "{:+.2f}")]


def build_matrix(metric: str) -> np.ndarray:
    """9 (non-baseline strategy) x 5 (family) delta array, in STRAT_NB/FAM
    order. Positive always means "better than baseline" regardless of the
    metric's own raw direction."""
    long_df = load_metric_long(metric)
    cols = []
    for fam in FAM:
        m = dataset_by_strategy_matrix(long_df, fam)
        if metric == "F1":
            delta = m[STRAT_NB].mean(axis=0) - m["baseline"].mean()
        else:
            # log_ratio_diffs is per-dataset paired; average across datasets
            # after computing the per-dataset log ratio, matching the Bayes
            # charts' own treatment (not diff-of-means, which would reopen
            # the scale-heterogeneity problem heatmap A's docstring explains).
            delta = np.array([log_ratio_diffs(m[s].values, m["baseline"].values).mean()
                               for s in STRAT_NB])
        cols.append(np.asarray(delta))
    return np.column_stack(cols)


def main():
    OUT_DIR = FIG_DIR / "Heatmaps"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # No fig.suptitle -- the LaTeX caption already carries this (see
    # heatmap_value's docstring note for why: a suptitle above the axes
    # ends up inside the tight bounding box, shrinking the actual heatmap).
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.3))

    for ax, (title, metric, fmt) in zip(axes, METRICS):
        M = build_matrix(metric)
        vmax = np.nanmax(np.abs(M))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(M, cmap="RdYlGn", norm=norm, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                frac = abs(norm(v) - 0.5) * 2  # 0 at center, 1 at either pole
                txt_color = "white" if frac > 0.55 else "#222"
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=9.0, color=txt_color)
        ax.set_xticks(range(len(FAM)))
        ax.set_xticklabels(FAMLAB, fontsize=10)
        ax.set_title(title, fontsize=11.5, pad=8)
        add_minor_grid(ax, len(STRAT_NB), len(FAM))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)

    axes[0].set_yticks(range(len(STRAT_NB)))
    axes[0].set_yticklabels(STRAT_NB, fontsize=10)
    for ax in axes[1:]:
        ax.set_yticks(range(len(STRAT_NB)))
        ax.set_yticklabels([])

    fig.tight_layout()
    save(fig, OUT_DIR / "heatmap_delta.png")
    print("Wrote", OUT_DIR / "heatmap_delta.pdf")


if __name__ == "__main__":
    main()
