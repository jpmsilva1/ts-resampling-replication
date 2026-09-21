#!/usr/bin/env python3
"""Cleans the 4 reference CSVs in src/original/Results (the paper authors'
own repo output) into presentable, parseable tables under
Results (Clean)/Results Data/original_paper_reference_results/.

Python port of the original clean_original_results.R (kept as R only
where R itself was unavoidable; this step is pure CSV munging + a
hyperparameter grid replicated with itertools, so no R dependency here).
"""
import itertools
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/joaopms/Documents/002_ts_resampling_strategies_isolated")
SRC_DIR = ROOT / "src/original/Results"
OUT_DIR = ROOT / "Results (Clean)/Results Data/original_paper_reference_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_workflow(wf: pd.Series) -> pd.DataFrame:
    """'mc.lm_UNDERTPhi' -> model_family='lm', strategy='UNDERTPhi';
    'mc.lm' -> model_family='lm', strategy='baseline'."""
    x = wf.str.replace(r"^mc\.", "", regex=True)
    has_strategy = x.str.contains("_")
    model_family = x.where(~has_strategy, x.str.split("_").str[0])
    strategy = x.where(has_strategy, "baseline")
    strategy = strategy.mask(has_strategy, x.str.split("_", n=1).str[1])
    return pd.DataFrame({"model_family": model_family, "strategy": strategy})


# ---------------------------------------------------------------------
# 1. results_table.csv -> average prec/rec/F1 per workflow (52 rows)
# ---------------------------------------------------------------------
rt = pd.read_csv(SRC_DIR / "results_table.csv")
rt = rt.rename(columns={rt.columns[0]: "workflow"})
rt = pd.concat([rt[["workflow"]], parse_workflow(rt["workflow"]), rt.drop(columns=["workflow"])], axis=1)
rt = rt.sort_values("F1", ascending=False).reset_index(drop=True)
rt["rank"] = rt.index + 1
rt.to_csv(OUT_DIR / "workflow_f1_results.csv", index=False)
print(f"1/4 workflow_f1_results.csv: {len(rt)} rows")

# ---------------------------------------------------------------------
# 2. paired_comparisons.csv -> Win/sigWin/Loss/SigLoss/Tie for *_UNDERB/
#    *_OVERB/*_SMOTEB vs. their own model's un-resampled baseline
#    (only these 15 rows exist in the source file; it is a single-dataset
#    demo/sanity-check run on the repo's bundled example dataset, NOT the
#    paper's full 24-dataset benchmark).
# ---------------------------------------------------------------------
pc = pd.read_csv(SRC_DIR / "paired_comparisons.csv")
pc = pc.rename(columns={pc.columns[0]: "workflow"})
pc = pd.concat([pc[["workflow"]], parse_workflow(pc["workflow"]), pc.drop(columns=["workflow"])], axis=1)
pc.to_csv(OUT_DIR / "paired_comparisons_demo.csv", index=False)
print(f"2/4 paired_comparisons_demo.csv: {len(pc)} rows")

# ---------------------------------------------------------------------
# 3. runtime_results.csv -> tr.time/pr.time/tot.time per workflow
# ---------------------------------------------------------------------
rr = pd.read_csv(SRC_DIR / "runtime_results.csv")
rr = rr.rename(columns={rr.columns[0]: "workflow"})
rr = pd.concat([rr[["workflow"]], parse_workflow(rr["workflow"]), rr.drop(columns=["workflow"])], axis=1)
rr.to_csv(OUT_DIR / "runtime_results.csv", index=False)
print(f"3/4 runtime_results.csv: {len(rr)} rows")

# ---------------------------------------------------------------------
# 4. opt_params_results.csv -> SVM hyperparameter search, decoded.
#    Grids and enumeration order copied from R_Code/OptParmsSearch.R
#    (workflowVariants() expand.grid order: first-listed arg varies
#    fastest, verified earlier against the installed performanceEstimation
#    package).
# ---------------------------------------------------------------------
op = pd.read_csv(SRC_DIR / "opt_params_results.csv")
op = op.rename(columns={op.columns[0]: "workflow"})

UNDER_STRATEGIES = ["UNDERB", "UNDERT", "UNDERTPhi"]
OVER_STRATEGIES = ["OVERB", "OVERT", "OVERTPhi"]
SMOTE_STRATEGIES = ["SMOTEB", "SMOTET", "SMOTETPhi"]


def expand_grid(**cols: list) -> pd.DataFrame:
    """R's expand.grid() varies the FIRST-listed column fastest;
    itertools.product varies the LAST one fastest -- so reverse the
    input order into product(), then reverse each tuple back."""
    names = list(cols)
    combos = itertools.product(*(cols[n] for n in reversed(names)))
    rows = [tuple(reversed(c)) for c in combos]
    return pd.DataFrame(rows, columns=names)


def grid_of(strategy: str) -> pd.DataFrame:
    if strategy in UNDER_STRATEGIES:
        return expand_grid(cost=[10, 150, 300], gamma=[0.01, 0.001], un=[.1, .2, .4, .6, .8])
    elif strategy in OVER_STRATEGIES:
        return expand_grid(cost=[10, 150, 300], gamma=[0.01, 0.001], ov=[2, 3, 5, 10])
    else:
        return expand_grid(cost=[10, 150, 300], gamma=[0.01, 0.001], un=[.05, .1, .2, .4, .6, .8], ov=[2, 3, 5, 10])


strategies = UNDER_STRATEGIES + OVER_STRATEGIES + SMOTE_STRATEGIES
decoded_frames = []
for s in strategies:
    g = grid_of(s)
    g["strategy"] = s
    g["variant_index"] = range(1, len(g) + 1)
    g["workflow"] = "mc.svm_" + s + g["variant_index"].astype(str)
    decoded_frames.append(g)
decoded = pd.concat(decoded_frames, ignore_index=True)

op_decoded = op.merge(decoded, on="workflow", how="left")
op_decoded["model_family"] = "svm"
op_decoded["strategy"] = op_decoded["strategy"].fillna("baseline")  # the lone "mc.svm" row
op_decoded["workflow"] = pd.Categorical(op_decoded["workflow"], categories=op["workflow"], ordered=True)
op_decoded = op_decoded.sort_values("workflow").reset_index(drop=True)
op_decoded = op_decoded[["workflow", "model_family", "strategy", "variant_index",
                         "cost", "gamma", "un", "ov", "prec", "rec", "F1"]]
op_decoded.to_csv(OUT_DIR / "svm_hyperparameter_search_decoded.csv", index=False)
print(f"4/4 svm_hyperparameter_search_decoded.csv: {len(op_decoded)} rows")

print("Done.")
