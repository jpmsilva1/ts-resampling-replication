#!/usr/bin/env python3
"""LaTeX tables for the causal-effect report, matching this project's
existing booktabs/resizebox aesthetic (tables_common.table_star).

Source: same panel as generate_causal_effects.py.
Output: Results (Clean)/Evaluation Metrics/Latex Tables/Causal/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # make Eval_Metrics Code/ importable when run directly
from paths import ROOT  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Regression"))
from tables_common import table_star
from generate_regression import build_panel

from causal_effects import (tau_k, randomization_inference, bootstrap_ci, win_probability,
                             rank_shift, leave_one_source_out, holm_adjust, SOURCE_MAP)

OUT_DIR = ROOT / "Results (Clean)/Evaluation Metrics/Latex Tables/Causal"
N_PERM = 10000
N_BOOT = 2000
SEED = 0

STRATEGY_ORDER = ["UNDERB", "UNDERT", "UNDERTPhi", "OVERB", "OVERT",
                  "OVERTPhi", "SMOTEB", "SMOTET", "SMOTETPhi"]
STRATEGY_HDR = [rf"\texttt{{{s}}}" for s in STRATEGY_ORDER]


def _pfmt(p: float) -> str:
    return "$<0.001$" if p < 0.001 else f"{p:.3f}"


def pooled_table(panel: pd.DataFrame, dv: str, caption: str, label: str,
                  higher_is_better: bool, with_secondary: bool) -> str:
    ri = randomization_inference(panel, dv=dv, n_perm=N_PERM, seed=SEED)
    ri["p_holm"] = holm_adjust(ri["p_value"])
    ci = bootstrap_ci(panel, dv=dv, n_boot=N_BOOT, seed=SEED)

    rows = []
    for s in STRATEGY_ORDER:
        tau, p_holm = ri.loc[s, "tau"], ri.loc[s, "p_holm"]
        ci_lo, ci_hi = ci.loc[s, "ci_lo"], ci.loc[s, "ci_hi"]
        tau_cell = rf"\textbf{{{tau:.4f}}}" if p_holm < 0.05 else f"{tau:.4f}"
        cells = [rf"\texttt{{{s}}}", tau_cell, f"[{ci_lo:.4f}, {ci_hi:.4f}]",
                 _pfmt(ri.loc[s, "p_value"]), _pfmt(p_holm)]
        if with_secondary:
            wp = win_probability(panel, value_col=dv, higher_is_better=higher_is_better)
            rk = rank_shift(panel, value_col=dv, higher_is_better=higher_is_better)
            cells += [f"{wp[s]:.2f}", f"{rk[s]:.2f}"]
        rows.append(" & ".join(cells))

    header_cells = [r"\textbf{Strategy}", r"\textbf{$\tau_k$}", r"\textbf{95\% CI}",
                     r"\textbf{RI $p$}", r"\textbf{Holm $p$}"]
    col_spec = "lccc c"
    if with_secondary:
        header_cells += [r"\textbf{Win Prob.}", r"\textbf{Avg. Rank}"]
        col_spec = "lcccccc"
    header = " & ".join(header_cells)
    body = " \\\\\n".join(rows) + r" \\"

    return table_star(caption=caption, label=label, col_spec=col_spec, header=header,
                       body=body, resize="\\textwidth" if with_secondary else None)


def jackknife_table(panel: pd.DataFrame, dv: str, caption: str, label: str) -> str:
    pooled = tau_k(panel, dv=dv)
    jk = leave_one_source_out(panel, dv=dv, source_map=SOURCE_MAP)
    sources = sorted(jk.index)

    rows = []
    for s in STRATEGY_ORDER:
        cells = [rf"\texttt{{{s}}}", f"{pooled[s]:.4f}"] + [f"{jk.loc[src, s]:.4f}" for src in sources]
        rows.append(" & ".join(cells))

    header = " & ".join([r"\textbf{Strategy}", r"\textbf{Pooled}"] +
                         [rf"\textbf{{Excl.\ {src.replace('_', ' ').title()}}}" for src in sources])
    col_spec = "l" + "c" * (1 + len(sources))
    body = " \\\\\n".join(rows) + r" \\"
    return table_star(caption=caption, label=label, col_spec=col_spec, header=header,
                       body=body, resize="\\textwidth")


def svm_sensitivity_table(f1_panel, sera_panel) -> str:
    f1_pooled = tau_k(f1_panel, dv="value")
    f1_excl = tau_k(f1_panel[f1_panel["model_family"] != "svm"], dv="value")
    sera_pooled = tau_k(sera_panel, dv="log_value")
    sera_excl = tau_k(sera_panel[sera_panel["model_family"] != "svm"], dv="log_value")

    rows = []
    for s in STRATEGY_ORDER:
        cells = [rf"\texttt{{{s}}}", f"{f1_pooled[s]:.4f}", f"{f1_excl[s]:.4f}",
                 f"{sera_pooled[s]:.4f}", f"{sera_excl[s]:.4f}"]
        rows.append(" & ".join(cells))

    header = (r"\textbf{Strategy} & \multicolumn{2}{c}{\textbf{F1}} & "
              r"\multicolumn{2}{c}{\textbf{SERA (log)}} \\"
              r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}"
              r" & \textbf{Pooled} & \textbf{Excl.\ SVM} & \textbf{Pooled} & \textbf{Excl.\ SVM}")
    body = " \\\\\n".join(rows) + r" \\"
    return table_star(
        caption=r"\textbf{Sensitivity to the SVM Training-Window Co-Treatment}: pooled $\tau_k$ "
                r"with and without the SVM model family, which alone is subject to "
                r"\texttt{cap\_svm\_train()}'s asymmetric training-window truncation (see Section on Robustness).",
        label="tab:causal_svm_sensitivity", col_spec="lcccc", header=header, body=body, resize=None)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    f1_panel = build_panel("F1")
    sera_panel = build_panel("SERA")
    n_ds = f1_panel["dataset_id"].nunique()
    n_blk = f1_panel.groupby(["dataset_id", "model_family"]).ngroups
    sera_panel["log_value"] = np.log(sera_panel["value"])

    (OUT_DIR / "causal_table_f1_pooled.tex").write_text(pooled_table(
        f1_panel, dv="value",
        caption=rf"\textbf{{Causal Effect on $F_1^\phi$ vs.\ Baseline}} (pooled across {n_ds} datasets "
                rf"$\times$ 5 model families, $N{{=}}{n_blk}$ blocks). $\tau_k$ = mean paired difference; "
                r"bold indicates Holm-significant at $\alpha{=}0.05$. Win Prob.\ and Avg.\ Rank are "
                r"the distribution-free secondary estimand (Section~4.1).",
        label="tab:causal_f1_pooled", higher_is_better=True, with_secondary=True))
    print("Wrote causal_table_f1_pooled.tex")

    (OUT_DIR / "causal_table_sera_pooled.tex").write_text(pooled_table(
        sera_panel, dv="log_value",
        caption=rf"\textbf{{Causal Effect on $\log(\mathrm{{SERA}})$ vs.\ Baseline}} (pooled, $N{{=}}{n_blk}$ "
                rf"blocks). Positive $\tau_k$ = strategy WORSE than baseline (SERA is lower-is-better). "
                r"Bold indicates Holm-significant at $\alpha{=}0.05$.",
        label="tab:causal_sera_pooled", higher_is_better=False, with_secondary=False))
    print("Wrote causal_table_sera_pooled.tex")

    (OUT_DIR / "causal_table_jackknife_f1.tex").write_text(jackknife_table(
        f1_panel, dv="value",
        caption=r"\textbf{Leave-One-Source-Out Jackknife, F1}: pooled $\tau_k$ recomputed excluding "
                r"each of the 5 dataset sources in turn.",
        label="tab:causal_jackknife_f1"))
    print("Wrote causal_table_jackknife_f1.tex")

    (OUT_DIR / "causal_table_jackknife_sera.tex").write_text(jackknife_table(
        sera_panel, dv="log_value",
        caption=r"\textbf{Leave-One-Source-Out Jackknife, $\log(\mathrm{SERA})$}: pooled $\tau_k$ "
                r"recomputed excluding each of the 5 dataset sources in turn. Note the sign change "
                r"for UNDER strategies when excluding Istanbul Stock Exchange.",
        label="tab:causal_jackknife_sera"))
    print("Wrote causal_table_jackknife_sera.tex")

    (OUT_DIR / "causal_table_svm_sensitivity.tex").write_text(svm_sensitivity_table(f1_panel, sera_panel))
    print("Wrote causal_table_svm_sensitivity.tex")
    print("Done.")


if __name__ == "__main__":
    main()
