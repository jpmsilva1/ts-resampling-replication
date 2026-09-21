"""Shared grid-drawing helpers for the strategy x family heatmap family
(value/win-count/delta/Bayes-outcome), matching the visual grammar of
fig_rank_heatmap() in Papers/unified_paper/figures/gen_figures.py: a 10x5
grid (10 strategies x 5 model families), white minor gridlines between
cells, and an outlined baseline row.

Reuses generate_cd_diagrams.py's MODEL_FAMILIES/STRATEGIES rather than
redefining the family/strategy universe -- these are the same 10 strategies
and 5 families every table and figure in this project already uses.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # make Eval_Metrics Code/ importable when run directly
from paths import ROOT  # noqa: E402
sys.path.insert(0, str(ROOT / "Results (Clean)/Eval_Metrics Code"))
sys.path.insert(0, str(ROOT / "Results (Clean)/Eval_Metrics Code/CD Metrics"))
from plot_style import apply_style, save, C_DARK  # noqa: E402
from generate_cd_diagrams import MODEL_FAMILIES, STRATEGIES  # noqa: E402

apply_style()

FAM = list(MODEL_FAMILIES)
FAMLAB = ["LM", "SVM", "MARS", "RF", "CART"]
STRAT = STRATEGIES
SLAB = ["Baseline"] + STRATEGIES[1:]

OUT_DIR = ROOT / "Results (Clean)/Evaluation Metrics/Figures/Heatmaps"


def add_minor_grid(ax, n_rows: int, n_cols: int) -> None:
    ax.set_xticks(np.arange(-.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_rows, 1), minor=True)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="minor", color="white", lw=1.1, alpha=1.0)
    ax.grid(False)


def outline_baseline_row(ax, n_cols: int, row_idx: int = 0) -> None:
    ax.add_patch(plt.Rectangle((-.5, row_idx - .5), n_cols, 1, fill=False,
                                ec=C_DARK, lw=1.6, zorder=5))
