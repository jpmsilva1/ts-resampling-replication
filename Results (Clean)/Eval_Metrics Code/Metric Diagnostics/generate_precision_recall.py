#!/usr/bin/env python3
"""fig_09: the utility precision-recall plane, one point per
(dataset, family, strategy) cell, with per-strategy-group means overlaid.

Tests the original paper's Sec. 5.1 claim that resampling's F1 gain comes
mostly from higher precision rather than higher recall. The companion table
(Latex Tables/Paper Comparison/cmp_precision_recall.tex) reports the same
deltas per family in numbers; this figure shows the spread behind them.

Like fig_02 and fig_07, this figure had no generator and so kept pre-fix
20-dataset numbers through two regenerations. It now reads the same
raw_iterations_by_dataset_v2 CSVs as the paper-comparison audit.

Output: Evaluation Metrics/Figures/Metric Diagnostics/fig_09_precision_recall.{pdf,png}
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))

from paths import FIG_DIR, RAW_F1, require  # noqa: E402
from plot_style import apply_style, save, C_DARK, C_TEAL, C_SAND, C_CORAL  # noqa: E402
from tables_common import parse_workflow  # noqa: E402

apply_style()

OUT_DIR = FIG_DIR / "Metric Diagnostics"

# strategy prefix -> (legend label, colour). Baseline is drawn separately.
GROUPS = [("UNDER", C_TEAL), ("OVER", C_SAND), ("SMOTE", C_CORAL)]


def cell_means() -> pd.DataFrame:
    """One row per (dataset, family, strategy): mean prec/rec over the folds."""
    frames = []
    for csv in sorted(require(RAW_F1).glob("DS??_*.csv")):
        frames.append(pd.read_csv(csv, usecols=["dataset_id", "workflow", "prec", "rec"]))
    df = pd.concat(frames, ignore_index=True)
    parsed = df["workflow"].map(parse_workflow)
    df["family"] = [p[0] for p in parsed]
    df["strategy"] = [p[1] for p in parsed]
    # arima/BDES have no resampling variants and are not part of this plane.
    df = df[df["family"].isin(["lm", "svm", "mars", "rf", "rpart"])]
    return df.groupby(["dataset_id", "family", "strategy"], as_index=False)[["prec", "rec"]].mean()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = cell_means()
    base = cells[cells["strategy"] == "baseline"]
    resampled = cells[cells["strategy"] != "baseline"]

    fig, ax = plt.subplots(figsize=(7.6, 6.0))

    for prefix, color in GROUPS:
        g = resampled[resampled["strategy"].str.startswith(prefix)]
        ax.scatter(g["rec"], g["prec"], s=26, color=color, alpha=0.30,
                   linewidths=0, zorder=2)
    ax.scatter(base["rec"], base["prec"], s=26, color=C_DARK, alpha=0.30,
               linewidths=0, zorder=2)

    handles = [ax.scatter(base["rec"].mean(), base["prec"].mean(), marker="*",
                          s=260, color=C_DARK, zorder=5,
                          label=f"Baseline (n={len(base)} cells)")]
    for prefix, color in GROUPS:
        g = resampled[resampled["strategy"].str.startswith(prefix)]
        handles.append(ax.scatter(g["rec"].mean(), g["prec"].mean(), marker="D",
                                  s=150, color=color, edgecolor="white", lw=1.0,
                                  zorder=5, label=f"{prefix} (n={len(g)} cells)"))

    d_prec = resampled["prec"].mean() - base["prec"].mean()
    d_rec = resampled["rec"].mean() - base["rec"].mean()
    ax.text(0.98, 0.03,
            f"mean $\\Delta$precision = {d_prec:+.3f}, mean $\\Delta$recall = {d_rec:+.3f}\n"
            f"vs. baseline, pooled across all 9 strategies",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color="#555")

    ax.set_xlabel("recall$_\\phi$ (one point per dataset$\\times$family cell)")
    ax.set_ylabel("precision$_\\phi$")
    ax.set_title(f"Precision-recall plane, all {len(cells)} raw cells (faded) "
                 "with strategy-group means (solid)", fontsize=12)
    ax.legend(handles=handles, loc="upper left", fontsize=11)
    fig.tight_layout()
    save(fig, OUT_DIR / "fig_09_precision_recall.png")
    print("Wrote", OUT_DIR / "fig_09_precision_recall.pdf")


if __name__ == "__main__":
    main()
