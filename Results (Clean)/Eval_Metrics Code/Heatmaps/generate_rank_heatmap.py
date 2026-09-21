#!/usr/bin/env python3
"""Figure 1 of the report: mean rank per (strategy, family) under all three
metrics.

This figure existed as a hand-built artifact for months with no generator in
the pipeline, so it silently kept pre-fix, 20-dataset numbers through two
regenerations. It is now derived from the same loaders every other figure
uses, and registered in run_all.py, so it can no longer go stale.

Rank 1 = best on that metric (F1 higher-is-better, SERA/RMSE lower), so all
three panels share one colour direction -- unlike heatmap_value, whose raw
magnitudes force a per-metric flip.

Output: Evaluation Metrics/Figures/Heatmaps/heatmap_rank.{pdf,png}
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heatmap_common import FAM, FAMLAB, STRAT, SLAB, OUT_DIR, add_minor_grid, outline_baseline_row

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "CD Metrics"))
from generate_cd_diagrams import load_metric_long, dataset_by_strategy_matrix  # noqa: E402
from cd_stats import average_ranks  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_style import save  # noqa: E402

METRICS = [("$F_1^{\\phi}$", "F1", True),
           ("SERA", "SERA", False),
           ("RMSE", "RMSE", False)]


def build_matrix(metric: str, higher_is_better: bool) -> np.ndarray:
    """10 (strategy) x 5 (family) array of mean rank, in STRAT/FAM order."""
    long_df = load_metric_long(metric)
    cols = []
    for fam in FAM:
        ranks = average_ranks(dataset_by_strategy_matrix(long_df, fam), higher_is_better)
        cols.append(ranks.reindex(STRAT).to_numpy())
    return np.column_stack(cols)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.8))
    norm = Normalize(vmin=1, vmax=len(STRAT))

    for ax, (title, metric, hib) in zip(axes, METRICS):
        M = build_matrix(metric, hib)
        im = ax.imshow(M, cmap="RdYlGn_r", norm=norm, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                frac = norm(M[i, j])
                txt_color = "white" if (frac < 0.15 or frac > 0.85) else "#222"
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                        fontsize=9.2, color=txt_color)
        ax.set_xticks(range(len(FAM)))
        ax.set_xticklabels(FAMLAB, fontsize=10)
        ax.set_title(title, fontsize=13, pad=8)
        add_minor_grid(ax, len(STRAT), len(FAM))
        outline_baseline_row(ax, len(FAM))

    axes[0].set_yticks(range(len(STRAT)))
    axes[0].set_yticklabels(SLAB, fontsize=10)
    for ax in axes[1:]:
        ax.set_yticks(range(len(STRAT)))
        ax.set_yticklabels([])

    cbar = fig.colorbar(im, ax=axes, fraction=0.020, pad=0.02)
    cbar.set_label(f"mean rank of {len(STRAT)} (1 = best)", fontsize=10)
    save(fig, OUT_DIR / "heatmap_rank.png")
    print("Wrote", OUT_DIR / "heatmap_rank.pdf")


if __name__ == "__main__":
    main()
