#!/usr/bin/env python3
"""Figures for the causal-effect report: forest plots of tau_k +/- 95%
bootstrap CI (one per metric) and a leave-one-source-out sensitivity plot
(both metrics, matching this project's multi-panel-per-file convention used
by Rank Distribution/Overview).

Source: same panel as generate_causal_effects.py.
Output: Results (Clean)/Evaluation Metrics/Figures/Causal/
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # make Eval_Metrics Code/ importable when run directly
from paths import ROOT  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Regression"))
from generate_regression import build_panel

from causal_effects import tau_k, bootstrap_ci, leave_one_source_out, SOURCE_MAP

plt.rcParams["font.family"] = "serif"  # matches cd_plot.py's academic-figure convention

OUT_DIR = ROOT / "Results (Clean)/Evaluation Metrics/Figures/Causal"
N_BOOT = 2000
SEED = 0

STRATEGY_ORDER = ["UNDERB", "UNDERT", "UNDERTPhi", "OVERB", "OVERT",
                  "OVERTPhi", "SMOTEB", "SMOTET", "SMOTETPhi"]


def forest_plot(ci: pd.DataFrame, title: str, xlabel: str, out_path: Path, zero_line_label: str):
    ci = ci.reindex(STRATEGY_ORDER)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    y = np.arange(len(ci))[::-1]
    err_lo = ci["tau"] - ci["ci_lo"]
    err_hi = ci["ci_hi"] - ci["tau"]
    ax.errorbar(ci["tau"], y, xerr=[err_lo, err_hi], fmt="o", color="black",
                ecolor="black", capsize=3, markersize=5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, label=zero_line_label)
    ax.set_yticks(y)
    ax.set_yticklabels(ci.index)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def jackknife_plot(f1_panel, sera_panel, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, panel, dv, title, ylabel in [
        (axes[0], f1_panel, "value", "F1", r"$\tau_k$ (raw F1 points)"),
        (axes[1], sera_panel, "log_value", "SERA (log)", r"$\tau_k$ (log SERA)"),
    ]:
        pooled = tau_k(panel, dv=dv).reindex(STRATEGY_ORDER)
        jk = leave_one_source_out(panel, dv=dv, source_map=SOURCE_MAP).reindex(columns=STRATEGY_ORDER)

        x = np.arange(len(STRATEGY_ORDER))
        ax.scatter(x, pooled.to_numpy(), color="black", zorder=5, label="Pooled (all datasets)", marker="D", s=40)
        for source in jk.index:
            ax.scatter(x, jk.loc[source].to_numpy(), alpha=0.6, s=25,
                       label=f"Excl. {source}")
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(STRATEGY_ORDER, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7)
    fig.suptitle("Leave-one-source-out jackknife: pooled $\\tau_k$ vs. excluding each dataset source")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    f1_panel = build_panel("F1")
    f1_ci = bootstrap_ci(f1_panel, dv="value", n_boot=N_BOOT, seed=SEED)
    forest_plot(f1_ci, title="Causal Effect on F1 vs. Baseline (95% Bootstrap CI)",
                xlabel=r"$\tau_k$ = mean(F1$_{strategy}$ $-$ F1$_{baseline}$)",
                out_path=OUT_DIR / "fig_forest_f1.png", zero_line_label="No effect")
    print(f"Wrote {OUT_DIR / 'fig_forest_f1.png'}")

    sera_panel = build_panel("SERA")
    sera_panel["log_value"] = np.log(sera_panel["value"])
    sera_ci = bootstrap_ci(sera_panel, dv="log_value", n_boot=N_BOOT, seed=SEED)
    forest_plot(sera_ci, title="Causal Effect on log(SERA) vs. Baseline (95% Bootstrap CI)",
                xlabel=r"$\tau_k$ = mean(log SERA$_{strategy}$ $-$ log SERA$_{baseline}$); positive = worse",
                out_path=OUT_DIR / "fig_forest_sera.png", zero_line_label="No effect")
    print(f"Wrote {OUT_DIR / 'fig_forest_sera.png'}")

    jackknife_plot(f1_panel, sera_panel, OUT_DIR / "fig_jackknife_sensitivity.png")
    print(f"Wrote {OUT_DIR / 'fig_jackknife_sensitivity.png'}")
    print("Done.")


if __name__ == "__main__":
    main()
