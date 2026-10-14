#!/usr/bin/env python3
"""fig_10: H.2 and H.3 in one figure.

(a) H.2 -- effect of temporal (T) and temporal+relevance (TPhi) bias on F1_phi,
    as the paired difference vs. the non-biased variant (B) of the same
    resampling family, model family and dataset. Mean over the 120 (dataset,
    family) cells with a 95% percentile bootstrap CI that resamples the 24
    DATASETS (the 5 family cells of a dataset are not independent, so resampling
    cells would give intervals that are too narrow).
(b) H.3 -- per-dataset F1_phi of the two time-series-specific models (BDES,
    ARIMA) next to each of the 5 general-purpose learners with resampling (mean
    over its 9 strategies). Outliers are labelled with their dataset ID.

Self-contained on purpose (registered only in local_pipeline.json): deleting
this file, its test and fig_10_* removes the figure without touching anything
else.

Output: Evaluation Metrics/Figures/Metric Diagnostics/fig_10_bias_ts_baselines.{pdf,png}
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))

from paths import FIG_DIR, RAW_F1, require  # noqa: E402
from plot_style import apply_style, save, C_DARK, C_TEAL, C_CORAL, C_GOLD, C_SAND  # noqa: E402
from tables_common import parse_workflow  # noqa: E402

OUT_DIR = FIG_DIR / "Metric Diagnostics"
FAMILIES = ["lm", "svm", "mars", "rf", "rpart"]
RESAMP = [("UNDER", "U", "Undersampling", C_DARK),
          ("OVER", "O", "Oversampling", C_TEAL),
          ("SMOTE", "SM", "SMOTE", C_CORAL)]
VARIANTS = ["T", "TPhi"]
N_BOOT, SEED = 10_000, 0


def cell_f1() -> pd.DataFrame:
    """Mean F1_phi over the folds, one row per dataset, columns (family, strategy)."""
    frames = [pd.read_csv(c, usecols=["dataset_id", "workflow", "F1"])
              for c in sorted(require(RAW_F1).glob("DS??_*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    parsed = df["workflow"].map(parse_workflow)
    df["family"] = [p[0] for p in parsed]
    df["strategy"] = [p[1] for p in parsed]
    return df.groupby(["dataset_id", "family", "strategy"])["F1"].mean().unstack(["family", "strategy"])


def paired_effect(diff: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED):
    """diff: rows = datasets, columns = families (paired differences).
    Returns (mean over all cells, CI lo, CI hi), bootstrapping datasets."""
    per_ds = diff.mean(axis=1).to_numpy()  # balanced design: mean of these = mean of all cells
    rng = np.random.default_rng(seed)
    boots = per_ds[rng.integers(0, len(per_ds), (n_boot, len(per_ds)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return per_ds.mean(), lo, hi


def panel_bias(ax, f1: pd.DataFrame):
    rows = []
    for rf, code, _, color in RESAMP:
        for v in VARIANTS:
            diff = pd.DataFrame({fam: f1[(fam, rf + v)] - f1[(fam, rf + "B")] for fam in FAMILIES})
            m, lo, hi = paired_effect(diff)
            rows.append((f"{code}_{v}", v, color, m, lo, hi))
            print(f"  H.2 {rf:5s} {v:4s} - B: {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")

    # same convention as Causal/generate_effect_forest_plots.py forest_plot()
    m, lo, hi = (np.array([r[i] for r in rows]) for i in (3, 4, 5))
    y = np.arange(len(rows))[::-1]
    ax.errorbar(m, y, xerr=[m - lo, hi - m], fmt="o", color=C_DARK, ecolor=C_DARK,
                capsize=3, markersize=6, lw=1.6)
    ax.axvline(0, color="#888", linestyle="--", linewidth=1, label="No change vs. _B variant")
    ax.set_yticks(y, [r[0] for r in rows])  # Moniz et al. (2017) strategy names
    ax.set_xlabel(r"mean($F_1^{\phi}$$_{,biased}$ $-$ $F_1^{\phi}$$_{,B}$) per (dataset, learner) pair")
    ax.set_title("(a) H.2: effect of bias on $F_1^{\\phi}$ (95% Bootstrap CI)")
    ax.legend(loc="upper left")
    ax.grid(axis="x", linestyle=":", alpha=0.4)


def _resamp(f1):
    return [s for s in f1.columns.get_level_values(1).unique() if s.startswith(("UNDER", "OVER", "SMOTE"))]


def panel_ts(ax, f1: pd.DataFrame):
    """One box per learner (each dataset's F1 averaged over the learner's 9 resampling
    strategies), next to ARIMA and BDES. Learners are not averaged together: H.3, like
    the paper's Table 5, compares each learner with the time-series models."""
    resamp = _resamp(f1)
    cols = [("BDES", f1[("BDES", "baseline")], C_SAND), ("ARIMA", f1[("arima", "baseline")], C_SAND)]
    cols += [(fam.upper(), f1.loc[:, [(fam, s) for s in resamp]].mean(axis=1), C_GOLD) for fam in FAMILIES]
    bp = ax.boxplot([c[1].to_numpy() for c in cols], positions=range(len(cols)), widths=0.55,
                    patch_artist=True, showmeans=True,
                    medianprops=dict(color="#333", lw=1.5),
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="#333", markersize=6),
                    flierprops=dict(marker="o", markersize=5, markerfacecolor="none", markeredgecolor="#666"))
    for x, (box, fl, (label, vals, color)) in enumerate(zip(bp["boxes"], bp["fliers"], cols)):
        box.set(facecolor=color, edgecolor="#333", lw=1.0)
        print(f"  H.3 {label:6s} mean {vals.mean():.3f}  median {vals.median():.3f}")
        for yv in fl.get_ydata():  # name the dataset behind each outlier
            ax.annotate(vals.index[np.isclose(vals.to_numpy(), yv)][0], (x, yv), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8.5, color="#555")
    ax.axvline(1.5, color="#bbb", lw=1.0)
    ax.text(0.5, 1.02, "time-series models", ha="center", va="bottom", fontsize=9.5, color="#555")
    ax.text(4.0, 1.02, "learners with resampling", ha="center", va="bottom", fontsize=9.5, color="#555")
    ax.set_xticks(range(len(cols)), [c[0] for c in cols], fontsize=10)
    ax.set_ylim(-0.03, 1.08)
    ax.set_ylabel(r"$F_1^{\phi}$ per dataset (mean over folds)")
    ax.set_title("(b) H.3: time-series models vs. resampled learners")
    ax.grid(axis="x", visible=False)


def main():
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    f1 = cell_f1()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.5, 5.0), gridspec_kw={"width_ratios": [1.2, 1]})
    panel_bias(ax_a, f1)
    panel_ts(ax_b, f1)
    fig.tight_layout(w_pad=1.5)
    save(fig, OUT_DIR / "fig_10_bias_ts_baselines.png")
    print("Wrote", OUT_DIR / "fig_10_bias_ts_baselines.pdf")


if __name__ == "__main__":
    main()
