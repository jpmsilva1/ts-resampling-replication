#!/usr/bin/env python3
"""Panel regression: strategy effect on F1/SERA vs. baseline, with dataset
and model-family fixed effects (see CLAUDE.md for the full write-up).

IMPORTANT REFRAMING (2026-08-28, /causal-experiments): this is a fully-crossed
factorial EXPERIMENT (20 datasets x 5 families x 10 strategies, strategy
assigned by the experimenter, not selected), not an observational study. The
dataset/family terms below are BLOCKING FACTORS for precision, not confounding
controls -- there is no selection into strategy to adjust away, so the
strategy coefficients are already causal by design. For the same estimand
with exact (non-asymptotic) inference -- randomization inference in place of
these cluster-robust SEs, plus a leave-one-source-out jackknife for the
5-source-not-20-dataset clustering threat and an SVM-training-window
sensitivity check -- see Causal/generate_causal_effects.py, which reproduces
these coefficients exactly (verified: causal_effects.py's tau_k for SERA
SMOTEB = 0.1725, matching this script's coefficient to 4 decimals).

Unit of observation: one (dataset, model_family, strategy) mean, aggregated
across the 50 Monte Carlo iterations first -- regressing on raw iterations
would treat resampling noise as independent observations and overstate
significance (pseudo-replication).

  value ~ C(strategy, baseline ref) + C(dataset_id) + C(model_family)

Standard errors clustered by dataset_id (20 clusters -- small-cluster
caveat noted in the printed output; randomization inference in
Causal/generate_causal_effects.py sidesteps this entirely). SERA is
regressed in log space (matches the log-ratio ROPE already used for the
Bayes charts); F1 is regressed as-is (OLS-as-linear-probability-model,
flagged in the output).

A second, interacted model (strategy x family) is fit per metric and an
F-test reports whether the strategy effect significantly differs across
model families -- a formal version of what the 15 Bayes charts show visually.

Source data: same as CD Metrics/generate_cd_diagrams.py (F1 now from the
data_v2 run -- see CLAUDE.md "2026-08-28 -- F1 re-extracted from data_v2").
Output: Results (Clean)/Evaluation Metrics/Regression/{f1,sera}_regression_summary.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # make Eval_Metrics Code/ importable when run directly
from paths import ROOT  # noqa: E402
CD_METRICS_DIR = ROOT / "Results (Clean)/Eval_Metrics Code/CD Metrics"
sys.path.insert(0, str(CD_METRICS_DIR))
from generate_cd_diagrams import load_metric_long, MODEL_FAMILIES, STRATEGIES

OUT_DIR = ROOT / "Results (Clean)/Evaluation Metrics/Regression"


def build_panel(metric: str) -> pd.DataFrame:
    """One row per (dataset_id, model_family, strategy): mean value across
    the 50 MC iterations. Restricted to the 5 model families / 10 strategies
    the rest of this project's tables and figures use."""
    long_df = load_metric_long(metric)
    long_df = long_df[long_df["model_family"].isin(MODEL_FAMILIES) & long_df["strategy"].isin(STRATEGIES)]
    panel = long_df.groupby(["dataset_id", "model_family", "strategy"])["value"].mean().reset_index()
    return panel


def fit_and_report(panel: pd.DataFrame, dv: str, metric_label: str) -> str:
    lines = []
    formula_main = f"{dv} ~ C(strategy, Treatment('baseline')) + C(dataset_id) + C(model_family)"
    model_main = smf.ols(formula_main, data=panel).fit(
        cov_type="cluster", cov_kwds={"groups": panel["dataset_id"]}
    )
    lines.append(f"=== {metric_label}: strategy effect (dataset + family FE, SE clustered by dataset) ===")
    lines.append(str(model_main.summary()))
    lines.append("")

    formula_int = f"{dv} ~ C(strategy, Treatment('baseline')) * C(model_family) + C(dataset_id)"
    model_int = smf.ols(formula_int, data=panel).fit(
        cov_type="cluster", cov_kwds={"groups": panel["dataset_id"]}
    )
    interaction_terms = [t for t in model_int.params.index if ":" in t]
    ftest = model_int.f_test([f"{t} = 0" for t in interaction_terms])
    lines.append(f"=== {metric_label}: strategy x family interaction, joint F-test ===")
    lines.append(f"F = {float(ftest.fvalue):.3f}, p = {float(ftest.pvalue):.4g} "
                 f"({len(interaction_terms)} interaction terms)")
    lines.append("A significant p means the strategy effect genuinely differs by model")
    lines.append("family (matches the per-family split already visible across the 15")
    lines.append("Bayes charts / 15 CD diagrams) -- the pooled coefficients above then")
    lines.append("describe an average effect that no single family actually experiences.")
    lines.append("")
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    f1_panel = build_panel("F1")
    f1_report = fit_and_report(f1_panel, dv="value", metric_label="F1")
    (OUT_DIR / "f1_regression_summary.txt").write_text(f1_report)
    print(f"Wrote {OUT_DIR / 'f1_regression_summary.txt'}")

    sera_panel = build_panel("SERA")
    sera_panel["log_sera"] = np.log(sera_panel["value"])
    sera_report = fit_and_report(sera_panel, dv="log_sera", metric_label="SERA (log)")
    (OUT_DIR / "sera_regression_summary.txt").write_text(sera_report)
    print(f"Wrote {OUT_DIR / 'sera_regression_summary.txt'}")

    print("Done. NOTE: 20 dataset clusters is below the ~30-50 typically wanted")
    print("for cluster-robust SEs to behave asymptotically -- treat p-values as")
    print("indicative, not exact; a wild cluster bootstrap would tighten this.")


if __name__ == "__main__":
    main()
