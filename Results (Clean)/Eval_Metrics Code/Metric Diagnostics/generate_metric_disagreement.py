#!/usr/bin/env python3
"""fig_02: how many (dataset, family) cells each strategy wins, under F1 vs.
under SERA -- the report's headline "the metric decides the verdict" chart.

Like fig_07 and fig_09, this figure was a hand-built artifact with no
generator, so it kept pre-fix 20-dataset numbers through two regenerations.
It now derives from the same loaders as every other figure and is registered
in run_all.py.

A cell is one (dataset, model family) pair; the winner is the strategy with
the best mean metric value in that cell, ties split evenly, so each metric's
counts sum to n_datasets x 5.

Output: Evaluation Metrics/Figures/Metric Diagnostics/fig_02_metric_disagreement.{pdf,png}
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "CD Metrics"))

from paths import FIG_DIR  # noqa: E402
from plot_style import apply_style, save, C_TEAL, C_CORAL  # noqa: E402
from tables_common import MODEL_FAMILIES, STRATEGIES  # noqa: E402
from generate_cd_diagrams import load_metric_long, dataset_by_strategy_matrix  # noqa: E402

apply_style()

OUT_DIR = FIG_DIR / "Metric Diagnostics"
FAM = list(MODEL_FAMILIES)


def win_counts(metric: str, higher_is_better: bool) -> tuple[np.ndarray, int]:
    """Per-strategy win count over all (dataset, family) cells, ties split."""
    long_df = load_metric_long(metric)
    counts = dict.fromkeys(STRATEGIES, 0.0)
    n_cells = 0
    for fam in FAM:
        mat = dataset_by_strategy_matrix(long_df, fam)[STRATEGIES]
        for _, row in mat.iterrows():
            best = row.max() if higher_is_better else row.min()
            tied = row[row == best].index
            for t in tied:
                counts[t] += 1 / len(tied)
            n_cells += 1
    return np.array([counts[s] for s in STRATEGIES]), n_cells


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    f1, n_cells = win_counts("F1", higher_is_better=True)
    sera, _ = win_counts("SERA", higher_is_better=False)

    x = np.arange(len(STRATEGIES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.6, 4.4))

    # Highlight the baseline column: the whole point of the figure is that this
    # one bar pair reverses between the two metrics.
    ax.axvspan(-0.5, 0.5, color=C_CORAL, alpha=0.08, zorder=0)

    colors_f1 = [C_CORAL] + [C_TEAL] * (len(STRATEGIES) - 1)
    b1 = ax.bar(x - w / 2, f1, w, color=colors_f1, label="$F_1^{\\phi}$ wins")
    b2 = ax.bar(x + w / 2, sera, w, color=colors_f1, hatch="///",
                edgecolor="white", linewidth=0.6, label="SERA wins")
    for bars, vals in ((b1, f1), (b2, sera)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + n_cells * 0.012,
                    f"{v:g}", ha="center", va="bottom", fontsize=9)

    ax.annotate(f"baseline: {f1[0]:g} under $F_1^{{\\phi}}$,\n{sera[0]:g} under SERA",
                xy=(0.5, sera[0]), xytext=(1.35, sera[0] * 0.97),
                fontsize=11, color=C_CORAL, va="top",
                arrowprops=dict(arrowstyle="-", color=C_CORAL, lw=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline"] + STRATEGIES[1:], rotation=20, ha="right")
    ax.set_ylabel(f"Datasets won (of {n_cells} dataset$\\times$family cells)")
    ax.set_ylim(0, max(f1.max(), sera.max()) * 1.22)
    ax.legend(ncol=2, loc="upper right")
    fig.tight_layout()
    save(fig, OUT_DIR / "fig_02_metric_disagreement.png")
    print("Wrote", OUT_DIR / "fig_02_metric_disagreement.pdf")


if __name__ == "__main__":
    main()
