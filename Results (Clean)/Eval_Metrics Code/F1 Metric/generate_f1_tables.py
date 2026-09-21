#!/usr/bin/env python3
"""Builds one F1-score LaTeX table PER MODEL FAMILY (LM, SVM, MARS, RF,
RPART), matching the layout of the online appendix's F1 tables
(Tables B1-B6): dataset rows x resampling-strategy columns, mean +/- sd
cells, best strategy per dataset row bolded, closing Overall Average
row -- rendered in this project's existing table aesthetic (booktabs,
resizebox+makebox centering, bold caption, \\texttt{} strategy headers).

Source data: Results (Clean)/Results Data/raw_iterations_by_dataset_v2/*.csv
        (data_v2 run -- same run RMSE/SERA use; see CLAUDE.md
        "2026-08-28 -- F1 re-extracted from data_v2")
Output: one .tex file per model family in
        Results (Clean)/Evaluation Metrics/Latex Tables/F1/
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tables_common import DS_LABELS, labels_for, MODEL_FAMILIES, STRATEGIES, STRATEGY_HDR, ROOT, fmt_mean_sd, table_star

DATA_DIR = ROOT / "Results (Clean)/Results Data/raw_iterations_by_dataset_v2"
OUT_DIR = ROOT / "Results (Clean)/Evaluation Metrics/Latex Tables/F1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Load & tidy all 20 raw per-iteration CSVs
# --------------------------------------------------------------------------
all_data = pd.concat((pd.read_csv(f) for f in sorted(DATA_DIR.glob("DS??_*.csv"))), ignore_index=True)

wf = all_data["workflow"].str.replace(r"^mc\.", "", regex=True)
all_data["model_family"] = wf.str.split("_").str[0]
all_data["strategy"] = wf.where(wf.str.contains("_"), "baseline")
all_data["strategy"] = all_data["strategy"].mask(wf.str.contains("_"), wf.str.split("_", n=1).str[1])

# --------------------------------------------------------------------------
# Per (dataset, model_family, strategy): mean and sd of F1 across the 50
# Monte Carlo iterations
# --------------------------------------------------------------------------
agg = (all_data.groupby(["dataset_id", "model_family", "strategy"])["F1"]
       .agg(["mean", "std"]).reset_index().rename(columns={"std": "sd"}))


def build_table(agg: pd.DataFrame, family: str, family_label: str) -> str:
    sub = agg[agg["model_family"] == family]

    row_lines = []
    col_means, col_sds = [], []
    for ds_id, ds_label in labels_for(agg["dataset_id"].unique()).items():
        row = sub[sub["dataset_id"] == ds_id].set_index("strategy").reindex(STRATEGIES)
        means, sds = row["mean"].to_numpy(), row["sd"].to_numpy()
        col_means.append(means)
        col_sds.append(sds)

        best = means.argmax()
        cells = [fmt_mean_sd(m, s, j == best) for j, (m, s) in enumerate(zip(means, sds))]
        row_lines.append(f"{ds_label} & " + " & ".join(cells))

    col_means_arr = pd.DataFrame(col_means, columns=STRATEGIES)
    col_sds_arr = pd.DataFrame(col_sds, columns=STRATEGIES)
    overall_mean = col_means_arr.mean()
    overall_sd = col_sds_arr.mean()
    best_overall = overall_mean.to_numpy().argmax()
    overall_cells = [fmt_mean_sd(overall_mean[s], overall_sd[s], j == best_overall)
                      for j, s in enumerate(STRATEGIES)]
    overall_line = r"\textbf{Overall Average} & " + " & ".join(overall_cells)

    col_spec = "l" + "c" * len(STRATEGIES)
    header = r"\textbf{Dataset} & " + " & ".join(STRATEGY_HDR)
    body = " \\\\\n".join(row_lines) + rf" \\" + "\n\\midrule\n" + overall_line + r" \\"

    caption = (rf"\textbf{{Utility $F_1^\phi$ Score (Mean $\pm$ SD) by Resampling Strategy -- {family_label}}}, "
               rf"across {len(row_lines)} imbalanced time series forecasting datasets ($50$ Monte Carlo folds per cell). "
               r"Bold values indicate the best-performing strategy for each dataset.")
    return table_star(caption=caption, label=f"tab:exp002_f1_{family}", col_spec=col_spec,
                       header=header, body=body)


for family, label in MODEL_FAMILIES.items():
    tex = build_table(agg, family, label)
    out_file = OUT_DIR / f"f1_table_{family}.tex"
    out_file.write_text(tex)
    print(f"Wrote {out_file}")
print("Done.")
