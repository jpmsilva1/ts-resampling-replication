#!/usr/bin/env python3
"""Causal effect of each resampling strategy vs. baseline on F1 and SERA --
the analysis /causal-experiments recommended for this project (fully-crossed
factorial: N datasets x 5 model families x 10 strategies, strategy assigned
by the experimenter, so identification is by design; dataset/family are
blocking factors for precision, not confounding controls).

See docs/causal-plans/2026-08-28-resampling-causal-effect/plan.md for the
full design writeup (estimands, threats, why randomization inference
replaces the cluster-robust SEs in generate_regression.py).

Source data: same panel as Regression/generate_regression.py (reused via
sys.path insert, no re-derivation of the aggregation).
Output: Results (Clean)/Evaluation Metrics/Causal/{f1,sera}_causal_effects.txt
        Results (Clean)/Evaluation Metrics/Causal/causal_effects.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # make Eval_Metrics Code/ importable when run directly
from paths import ROOT  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Regression"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "CD Metrics"))
from generate_regression import build_panel
from generate_cd_diagrams import MODEL_FAMILIES

from causal_effects import (tau_k, randomization_inference, win_probability, rank_shift,
                             leave_one_source_out, holm_adjust, SOURCE_MAP, BASELINE)

OUT_DIR = ROOT / "Results (Clean)/Evaluation Metrics/Causal"
N_PERM = 10000
SEED = 0
FAMILIES = list(MODEL_FAMILIES.keys())


def pooled_effects(panel: pd.DataFrame, dv: str) -> pd.DataFrame:
    ri = randomization_inference(panel, dv=dv, n_perm=N_PERM, seed=SEED)
    ri["p_holm"] = holm_adjust(ri["p_value"])
    return ri


def per_family_effects(panel: pd.DataFrame, dv: str) -> pd.DataFrame:
    """Exploratory, unadjusted -- see plan: confirmatory multiplicity control
    (Holm) applies only to the pooled 9-strategy family of tests."""
    rows = []
    for family in FAMILIES:
        sub = panel[panel["model_family"] == family]
        ri = randomization_inference(sub, dv=dv, n_perm=N_PERM, seed=SEED)
        ri["model_family"] = family
        rows.append(ri)
    return pd.concat(rows).reset_index().rename(columns={"index": "strategy"})


def format_report(metric_label: str, pooled: pd.DataFrame, per_family: pd.DataFrame,
                   jackknife: pd.DataFrame, svm_excl: pd.DataFrame | None,
                   secondary: pd.DataFrame | None, panel: pd.DataFrame) -> str:
    # Counts are derived from the panel actually loaded, never hardcoded --
    # this file previously said "100 blocks"/"20 datasets" long after the
    # experiment grew to 24.
    n_ds = panel["dataset_id"].nunique()
    n_blocks = panel.groupby(["dataset_id", "model_family"]).ngroups
    n_src = len({SOURCE_MAP[d] for d in panel["dataset_id"].unique()})
    lines = [f"=== {metric_label}: pooled causal effect (tau_k vs. baseline) ===",
             "Randomization inference, permutation within (dataset, model_family) blocks,",
             f"N_perm={N_PERM}. Holm-adjusted p controls the family-wise error across",
             "these 9 strategies (confirmatory). Point estimate = mean paired difference",
             f"across all {n_blocks} blocks ({n_ds} datasets); identical to the two-way FE OLS",
             "coefficient under",
             "this design's full balance.", "",
             pooled.round(4).to_string(), ""]

    lines += ["=== Leave-one-source-out jackknife (pooled tau_k) ===",
              f"The {n_ds} datasets are {n_src} sources (see plan): dropping an entire source and",
              "recomputing tau_k shows how much of the pooled effect any single source",
              "(esp. the 7-dataset Istanbul Stock Exchange block) is carrying.", "",
              jackknife.round(4).to_string(), ""]

    if svm_excl is not None:
        lines += ["=== Sensitivity: pooled tau_k excluding the SVM family ===",
                  "cap_svm_train() bundles a training-window truncation into the SVM",
                  "OVER/SMOTE arms (see plan). If dropping svm barely moves tau_k, that",
                  "co-treatment isn't driving the pooled conclusion.", "",
                  svm_excl.round(4).to_string(), ""]

    if secondary is not None:
        lines += ["=== Distribution-free secondary estimand (win probability + rank shift) ===",
                  "F1's raw mean is partly a floor artifact (baseline F1 approx 0 on most",
                  "datasets). win_prob = P(strategy beats baseline on a random block);",
                  "avg_rank reuses cd_stats.average_ranks (rank 1 = best of the 10 strategies).",
                  "", secondary.round(4).to_string(), ""]

    lines += ["=== Per-family effects (exploratory, NOT multiplicity-adjusted) ===",
              per_family.round(4).to_string(), ""]
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_rows = []

    # --- F1 ---
    f1_panel = build_panel("F1")
    f1_pooled = pooled_effects(f1_panel, dv="value")
    f1_by_family = per_family_effects(f1_panel, dv="value")
    f1_jk = leave_one_source_out(f1_panel, dv="value", source_map=SOURCE_MAP)
    f1_svm_excl = pooled_effects(f1_panel[f1_panel["model_family"] != "svm"], dv="value")
    f1_secondary = pd.DataFrame({
        "win_prob": win_probability(f1_panel, value_col="value"),
        "avg_rank": rank_shift(f1_panel, value_col="value"),
    }).drop(index=BASELINE, errors="ignore")

    (OUT_DIR / "f1_causal_effects.txt").write_text(
        format_report("F1", f1_pooled, f1_by_family, f1_jk, f1_svm_excl, f1_secondary, f1_panel))
    print(f"Wrote {OUT_DIR / 'f1_causal_effects.txt'}")

    for strategy, row in f1_pooled.iterrows():
        csv_rows.append({"metric": "F1", "strategy": strategy, "model_family": "pooled",
                          "tau": row["tau"], "p_value": row["p_value"], "p_holm": row["p_holm"]})
    for _, row in f1_by_family.iterrows():
        csv_rows.append({"metric": "F1", "strategy": row["strategy"], "model_family": row["model_family"],
                          "tau": row["tau"], "p_value": row["p_value"], "p_holm": np.nan})

    # --- SERA (log scale, per project convention) ---
    sera_panel = build_panel("SERA")
    sera_panel["log_value"] = np.log(sera_panel["value"])
    sera_pooled = pooled_effects(sera_panel, dv="log_value")
    sera_by_family = per_family_effects(sera_panel, dv="log_value")
    sera_jk = leave_one_source_out(sera_panel, dv="log_value", source_map=SOURCE_MAP)
    sera_svm_excl = pooled_effects(sera_panel[sera_panel["model_family"] != "svm"], dv="log_value")

    sign_note = (
        "NOTE ON SIGN: tau = mean(log(SERA_strategy) - log(SERA_baseline)). SERA is\n"
        "lower-is-better, so a POSITIVE tau means the strategy is WORSE than baseline\n"
        "(matches generate_regression.py's coefficient sign -- e.g. SMOTEB here should\n"
        "equal that script's SMOTEB coefficient exactly). This is the OPPOSITE sign\n"
        "convention from the Bayes charts' ROPE, which is defined as\n"
        "log(baseline) - log(strategy) (positive = strategy better) -- don't compare\n"
        "signs across the two without checking which convention each uses.\n"
    )
    (OUT_DIR / "sera_causal_effects.txt").write_text(
        sign_note + "\n" + format_report("SERA (log)", sera_pooled, sera_by_family, sera_jk, sera_svm_excl, None, sera_panel))
    print(f"Wrote {OUT_DIR / 'sera_causal_effects.txt'}")

    for strategy, row in sera_pooled.iterrows():
        csv_rows.append({"metric": "SERA_log", "strategy": strategy, "model_family": "pooled",
                          "tau": row["tau"], "p_value": row["p_value"], "p_holm": row["p_holm"]})
    for _, row in sera_by_family.iterrows():
        csv_rows.append({"metric": "SERA_log", "strategy": row["strategy"], "model_family": row["model_family"],
                          "tau": row["tau"], "p_value": row["p_value"], "p_holm": np.nan})

    pd.DataFrame(csv_rows).to_csv(OUT_DIR / "causal_effects.csv", index=False)
    print(f"Wrote {OUT_DIR / 'causal_effects.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
