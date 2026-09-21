#!/usr/bin/env python3
"""Plain effect-size forest plots + leave-one-source-out sensitivity plot for
Exp_Report_Main -- same underlying estimator (bootstrap CI on the mean
paired difference vs. baseline) as generate_causal_figures.py, but relabeled
without causal-inference framing/notation ("Delta" instead of "tau",
"Effect vs. baseline" instead of "Causal Effect") and matching this
project's plot_style.py convention (Ocean Dusk palette, PDF+PNG via save()),
since the causal-experiments language was scoped out of that report.

Source: same panel-building/estimator code as generate_causal_figures.py
(causal_effects.py's bootstrap_ci/leave_one_source_out), reused unchanged --
only the plot labels differ.
Output: Results (Clean)/Evaluation Metrics/Figures/Metric Diagnostics/
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # make Eval_Metrics Code/ importable when run directly
from paths import ROOT  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Regression"))
from generate_regression import build_panel  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_effects import bootstrap_ci, leave_one_source_out, tau_k, SOURCE_MAP  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_style import apply_style, save, C_DARK, C_TEAL, C_GOLD, C_SAND, C_CORAL  # noqa: E402

OUT_DIR = ROOT / "Results (Clean)/Evaluation Metrics/Figures/Metric Diagnostics"
N_BOOT = 2000
SEED = 0

STRATEGY_ORDER = ["UNDERB", "UNDERT", "UNDERTPhi", "OVERB", "OVERT",
                  "OVERTPhi", "SMOTEB", "SMOTET", "SMOTETPhi"]
SOURCE_COLORS = {"bike_daily": C_DARK, "bike_hourly": C_TEAL,
                  "icelandic_river": C_GOLD, "istanbul_stock": C_CORAL,
                  "porto_weather": C_SAND}


def forest_plot(ci: pd.DataFrame, title: str, xlabel: str, out_path: Path):
    ci = ci.reindex(STRATEGY_ORDER)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    y = np.arange(len(ci))[::-1]
    err_lo = ci["tau"] - ci["ci_lo"]
    err_hi = ci["ci_hi"] - ci["tau"]
    ax.errorbar(ci["tau"], y, xerr=[err_lo, err_hi], fmt="o", color=C_DARK,
                ecolor=C_DARK, capsize=3, markersize=6, lw=1.6)
    ax.axvline(0, color="#888", linestyle="--", linewidth=1, label="No change vs. baseline")
    ax.set_yticks(y)
    ax.set_yticklabels(ci.index)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    save(fig, out_path)


def jackknife_plot(f1_panel, sera_panel, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, panel, dv, title, ylabel in [
        (axes[0], f1_panel, "value", "$F_1^{\\phi}$", r"$\Delta$ vs. baseline (raw $F_1^{\phi}$ points)"),
        (axes[1], sera_panel, "log_value", "SERA (log ratio)", r"$\Delta$ vs. baseline (log SERA)"),
    ]:
        pooled = tau_k(panel, dv=dv).reindex(STRATEGY_ORDER)
        jk = leave_one_source_out(panel, dv=dv, source_map=SOURCE_MAP).reindex(columns=STRATEGY_ORDER)

        x = np.arange(len(STRATEGY_ORDER))
        ax.scatter(x, pooled.to_numpy(), color=C_DARK, zorder=5,
                   label="Pooled (all datasets)", marker="D", s=50)
        for source in jk.index:
            ax.scatter(x, jk.loc[source].to_numpy(), alpha=0.75, s=30,
                       color=SOURCE_COLORS.get(source, "#999"),
                       label=f"Excl. {source}")
        ax.axhline(0, color="#888", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(STRATEGY_ORDER, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    fig.suptitle("Sensitivity to dropping each dataset source: pooled effect "
                 "vs. leave-one-source-out")
    fig.tight_layout()
    save(fig, out_path)


def main():
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    f1_panel = build_panel("F1")
    f1_ci = bootstrap_ci(f1_panel, dv="value", n_boot=N_BOOT, seed=SEED)
    forest_plot(f1_ci, title="Effect on $F_1^{\\phi}$ vs. Baseline (95% Bootstrap CI)",
                xlabel=r"mean($F_1^{\phi}$$_{,strategy}$ $-$ $F_1^{\phi}$$_{,baseline}$)",
                out_path=OUT_DIR / "fig_effect_forest_f1.png")
    print(f"Wrote {OUT_DIR / 'fig_effect_forest_f1.pdf'}")

    sera_panel = build_panel("SERA")
    sera_panel["log_value"] = np.log(sera_panel["value"])
    sera_ci = bootstrap_ci(sera_panel, dv="log_value", n_boot=N_BOOT, seed=SEED)
    forest_plot(sera_ci, title="Effect on log(SERA) vs. Baseline (95% Bootstrap CI)",
                xlabel=r"mean(log SERA$_{strategy}$ $-$ log SERA$_{baseline}$); positive = worse",
                out_path=OUT_DIR / "fig_effect_forest_sera.png")
    print(f"Wrote {OUT_DIR / 'fig_effect_forest_sera.pdf'}")

    jackknife_plot(f1_panel, sera_panel, OUT_DIR / "fig_effect_jackknife.png")
    print(f"Wrote {OUT_DIR / 'fig_effect_jackknife.pdf'}")
    print("Done.")


if __name__ == "__main__":
    main()
