#!/usr/bin/env python3
"""Heatmap A: mean F1/SERA/RMSE value per (strategy, family) -- the
magnitude companion to fig_rank_heatmap (the report's Figure 1), which only
shows ordinal rank and so can't distinguish a strategy narrowly edging out
another from one that crushes it.

Same 10x5 grid as Figure 1 (10 strategies x 5 model families), one panel per
metric, baseline row outlined. Colormap direction flips per metric (green =
better, red = worse, in that metric's own direction) since F1 is
higher-is-better while SERA/RMSE are lower-is-better -- unlike Figure 1's
rank panels, which can share one direction because rank 1 always means best.

Source data: same loaders as CD Metrics/generate_cd_diagrams.py.
Output: Results (Clean)/Evaluation Metrics/Figures/Heatmaps/heatmap_value.{pdf,png}
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heatmap_common import FAM, FAMLAB, STRAT, SLAB, OUT_DIR, add_minor_grid, outline_baseline_row

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "CD Metrics"))
from generate_cd_diagrams import load_metric_long, dataset_by_strategy_matrix  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_style import save  # noqa: E402

METRICS = [("$F_1^{\\phi}$", "F1", True, "{:.3f}", False),
           ("SERA", "SERA", False, "{:.2g}", True),
           ("RMSE", "RMSE", False, "{:.2g}", True)]


def build_matrix(metric: str, geometric: bool) -> np.ndarray:
    """10 (strategy) x 5 (family) array of the metric's average value across
    the 20 datasets, in STRAT/FAM order.

    SERA and RMSE span several orders of magnitude across datasets of very
    different scale (e.g. baseline SERA ranges 0.004 to 1.5e8 across the 20
    datasets for a single strategy/family) -- an arithmetic mean would be
    dominated entirely by the 1-2 largest-scale datasets and wouldn't
    represent the typical cell. Use the geometric mean instead (same fix
    the log-scale y-axis in the Overview figures and the log-ratio ROPE in
    the Bayes charts already apply to this exact problem). F1 is bounded
    [0,1] and doesn't have this issue, so it stays a plain arithmetic mean.
    """
    long_df = load_metric_long(metric)
    cols = []
    for fam in FAM:
        per_dataset = dataset_by_strategy_matrix(long_df, fam)[STRAT]
        if geometric:
            cols.append(np.exp(np.log(per_dataset.clip(lower=1e-12)).mean(axis=0)))
        else:
            cols.append(per_dataset.mean(axis=0))
    return np.column_stack(cols)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # No fig.suptitle -- the LaTeX caption already carries this, and a
    # suptitle sitting well above the axes gets included in the tight
    # bounding box, shrinking the actual heatmap once placed at a fixed
    # LaTeX width.
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.8))

    for ax, (title, metric, hib, fmt, geometric) in zip(axes, METRICS):
        M = build_matrix(metric, geometric)
        cmap = "RdYlGn" if hib else "RdYlGn_r"
        norm = (Normalize(vmin=np.nanmin(M), vmax=np.nanmax(M)) if hib
                else LogNorm(vmin=np.nanmin(M[M > 0]), vmax=np.nanmax(M)))
        im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                frac = norm(v)
                txt_color = "white" if (frac < 0.28 or frac > 0.82) else "#222"
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=9.2, color=txt_color)
        ax.set_xticks(range(len(FAM)))
        ax.set_xticklabels(FAMLAB, fontsize=10)
        ax.set_title(title, fontsize=13, pad=8)
        add_minor_grid(ax, len(STRAT), len(FAM))
        outline_baseline_row(ax, len(FAM))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)

    axes[0].set_yticks(range(len(STRAT)))
    axes[0].set_yticklabels(SLAB, fontsize=10)
    for ax in axes[1:]:
        ax.set_yticks(range(len(STRAT)))
        ax.set_yticklabels([])

    fig.tight_layout()
    save(fig, OUT_DIR / "heatmap_value.png")
    print("Wrote", OUT_DIR / "heatmap_value.pdf")


if __name__ == "__main__":
    main()
