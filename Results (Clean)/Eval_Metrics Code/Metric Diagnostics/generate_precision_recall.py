#!/usr/bin/env python3
"""fig_09: (a) each resampled (dataset, family, strategy) cell's change in
utility precision and recall vs. the baseline of the same dataset and family,
with quadrant shares; (b) the mean F1 change split into precision and recall
parts.

Tests the original paper's Sec. 5.1 claim that resampling's F1 gain comes
mostly from higher precision rather than higher recall. The companion table
(Latex Tables/Paper Comparison/cmp_precision_recall.tex) reports the same
deltas per family in numbers; this figure shows the per-cell changes behind them.

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
from plot_style import apply_style, save, C_DARK, C_TEAL, C_CORAL, C_GOLD, C_GREY  # noqa: E402
from tables_common import FAMILY_LABELS, parse_workflow  # noqa: E402

sys.path.insert(0, str(CODE / "Paper Comparison"))
from generate_paper_comparison import f1_decomposition  # noqa: E402

apply_style()

OUT_DIR = FIG_DIR / "Metric Diagnostics"

# strategy prefix -> (legend label, colour). Baseline is drawn separately.
C_PREC, C_REC = C_GOLD, C_GREY  # panel (b): kept apart from panel (a)'s group colours
GROUPS = [("UNDER", C_DARK), ("OVER", C_TEAL), ("SMOTE", C_CORAL)]


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


def paired_changes(cells: pd.DataFrame) -> pd.DataFrame:
    """Each resampled cell's change vs. the baseline of the same (dataset,
    family): d_prec, d_rec."""
    base = cells[cells["strategy"] == "baseline"].set_index(["dataset_id", "family"])[["prec", "rec"]]
    res = cells[cells["strategy"] != "baseline"].join(base, on=["dataset_id", "family"], rsuffix="0")
    res["d_prec"] = res["prec"] - res["prec0"]
    res["d_rec"] = res["rec"] - res["rec0"]
    return res


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ch = paired_changes(cell_means())

    fig, (ax, axd) = plt.subplots(1, 2, figsize=(15.0, 6.5), gridspec_kw={"width_ratios": [1.6, 1]})

    handles = []
    for prefix, color in GROUPS:
        g = ch[ch["strategy"].str.startswith(prefix)]
        ax.scatter(g["d_prec"], g["d_rec"], s=22, color=color, alpha=0.35, linewidths=0, zorder=2)
        handles.append(ax.scatter(g["d_prec"].mean(), g["d_rec"].mean(), marker="D", s=150,
                                  color=color, edgecolor="white", lw=1.0, zorder=5,
                                  label=f"{prefix} mean ({len(g)} cells)"))
    ax.axhline(0, color="#777", lw=0.9, zorder=1)
    ax.axvline(0, color="#777", lw=0.9, zorder=1)
    # tint the two unambiguous quadrants: both improve (green), both worse (red)
    (x0, x1), (y0, _) = ax.get_xlim(), ax.get_ylim()
    y1 = 0.6  # recall axis runs past the data so the equal-scale panel is not a thin strip
    ax.fill_between([0, x1], 0, y1, color="#2E9E44", alpha=0.10, lw=0, zorder=0)
    ax.fill_between([x0, 0], y0, 0, color="#D64545", alpha=0.10, lw=0, zorder=0)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    # both axes are changes in a [0,1] metric: same unit length, so the y = x diagonal means equal gain
    ax.set_aspect("equal", adjustable="box")
    # equal-gain diagonal (meaningful because the axes share one scale): below it, precision gained more
    lo, hi = max(x0, y0), min(x1, y1)
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color="#555", zorder=1)
    ax.text(0.47, 0.47, "equal gain", rotation=45, rotation_mode="anchor", ha="center", va="bottom",
            fontsize=9, style="italic", color="#555", zorder=6)

    up_p, up_r = ch["d_prec"] > 0, ch["d_rec"] > 0
    quads = [((0.98, 0.97), "right", "top", up_p & up_r, "Both improve"),
             ((0.02, 0.97), "left", "top", ~up_p & up_r, "Only recall improves"),
             ((0.98, 0.03), "right", "bottom", up_p & ~up_r, "Only precision improves"),
             ((0.02, 0.03), "left", "bottom", ~up_p & ~up_r, "Both worse")]
    for (x, y), ha, va, mask, label in quads:
        ax.text(x, y, f"{label}\n{mask.mean():.0%} of cells", transform=ax.transAxes, ha=ha, va=va,
                fontsize=11, color="#333", fontweight="bold", zorder=6)

    ax.set_xlabel("$\\Delta$precision$_\\phi$ vs. own baseline")
    ax.set_ylabel("$\\Delta$recall$_\\phi$ vs. own baseline")
    ax.set_title(f"(a) Change vs. the baseline of the same dataset and family ({len(ch)} cells)",
                 fontsize=12)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
              fontsize=10, handletextpad=0.3, columnspacing=1.2)
    decomposition_panel(axd)
    fig.tight_layout()
    save(fig, OUT_DIR / "fig_09_precision_recall.png")
    print("Wrote", OUT_DIR / "fig_09_precision_recall.pdf")


def decomposition_panel(ax):
    """(b) mean F1 change vs. baseline per family, split into the part due to
    precision and the part due to recall (same per-fold Shapley split as
    Table cmp_precision_recall)."""
    frames = [pd.read_csv(c, usecols=["dataset_id", "workflow", "iteration", "prec", "rec"])
              for c in sorted(require(RAW_F1).glob("DS??_*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    fams = ["lm", "svm", "mars", "rf", "rpart"]
    decs = [f1_decomposition(df[df.workflow.str.startswith(f"mc.{f}")], f"mc.{f}") for f in fams]
    labels = [FAMILY_LABELS[f] for f in fams] + ["All families"]
    parts = [(d.dF_prec.mean(), d.dF_rec.mean()) for d in decs]
    alld = pd.concat(decs)
    parts.append((alld.dF_prec.mean(), alld.dF_rec.mean()))
    y = range(len(labels))[::-1]
    for yi, lab, (p, r) in zip(y, labels, parts):
        # stack each part away from zero on its own side so negatives stay readable
        ax.barh(yi, p, color=C_PREC, height=0.6)
        ax.barh(yi, r, left=p if (p >= 0) == (r >= 0) else 0, color=C_REC, height=0.6)
        share = p / (p + r)
        ax.text(max(p, 0) + max(r, 0) + 0.004, yi, f"{share:.0%} precision",
                va="center", fontsize=10, color="#444")
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_yticks(list(y), labels)
    ax.set_xlim(right=max(p + r for p, r in parts) * 1.45)
    ax.set_xlabel("mean $\\Delta F_1^{\\phi}$ vs. baseline (per fold)")
    ax.set_title("(b) Where the $F_1^{\\phi}$ gain comes from", fontsize=12)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_PREC), plt.Rectangle((0, 0), 1, 1, color=C_REC)],
              labels=["due to precision", "due to recall"], loc="lower right", fontsize=10)


if __name__ == "__main__":
    main()
