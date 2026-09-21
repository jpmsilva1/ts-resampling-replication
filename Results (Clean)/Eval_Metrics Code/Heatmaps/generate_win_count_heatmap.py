#!/usr/bin/env python3
"""Heatmap B: win-count heatmap -- how many of the 20 datasets each
(strategy, family) combo is the single best performer for. Direct
heatmap-ification of tables/Wins_Loss/{f1,sera}_table_best_counts.tex (2
panels: F1, SERA -- RMSE has no win-count table in this project, see
CLAUDE.md's Summary Tables entry).

Sequential single-hue colormap (win count is a pure 0-20 magnitude with no
"bad" pole, unlike heatmap A/C's diverging/direction-flipped treatment), a
teal ramp matching the project's Ocean Dusk palette.

Reuses the exact win-counting logic from Summary Tables/generate_summary_tables.py
(`_win_count_table`) rather than recomputing it -- same numbers as
f1_table_best_counts.tex/sera_table_best_counts.tex by construction.

Output: Results (Clean)/Evaluation Metrics/Figures/Heatmaps/heatmap_win_count.{pdf,png}
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heatmap_common import FAM, FAMLAB, STRAT, SLAB, OUT_DIR, add_minor_grid, outline_baseline_row

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Summary Tables"))
from generate_summary_tables import _win_count_table  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_style import save, C_DARK, C_TEAL  # noqa: E402

CMAP = LinearSegmentedColormap.from_list("ocean_teal", ["#FFFFFF", C_TEAL, C_DARK])
METRICS = [("$F_1^{\\phi}$ win count (of {n} datasets)", "F1", True),
           ("SERA win count (of {n} datasets)", "SERA", False)]


def build_matrix(metric: str, higher_is_better: bool) -> np.ndarray:
    """10 (strategy) x 5 (family) win-count array, in STRAT/FAM order."""
    table = _win_count_table(metric, higher_is_better)  # rows=family, cols=strategy
    return table.loc[FAM, STRAT].values.T


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # No fig.suptitle -- the LaTeX caption already carries this (see
    # heatmap_value's docstring note for why).
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.8))

    n_ds = int(round(_win_count_table("F1", True).loc[FAM[0]].sum()))
    norm = Normalize(vmin=0, vmax=n_ds)
    for ax, (title, metric, hib) in zip(axes, METRICS):
        title = title.replace("{n}", str(n_ds))
        M = build_matrix(metric, hib)
        im = ax.imshow(M, cmap=CMAP, norm=norm, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                txt_color = "white" if norm(v) > 0.55 else "#222"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=9.5, color=txt_color)
        ax.set_xticks(range(len(FAM)))
        ax.set_xticklabels(FAMLAB, fontsize=10)
        ax.set_title(title, fontsize=11.5, pad=8)
        add_minor_grid(ax, len(STRAT), len(FAM))
        outline_baseline_row(ax, len(FAM))

    axes[0].set_yticks(range(len(STRAT)))
    axes[0].set_yticklabels(SLAB, fontsize=10)
    axes[1].set_yticks(range(len(STRAT)))
    axes[1].set_yticklabels([])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=axes,
                 fraction=0.03, pad=0.03, label="datasets won (of 20)")

    save(fig, OUT_DIR / "heatmap_win_count.png")
    print("Wrote", OUT_DIR / "heatmap_win_count.pdf")


if __name__ == "__main__":
    main()
