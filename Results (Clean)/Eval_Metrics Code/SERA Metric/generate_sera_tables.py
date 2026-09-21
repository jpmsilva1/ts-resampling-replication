#!/usr/bin/env python3
"""Builds one SERA LaTeX table PER MODEL FAMILY (LM, SVM, MARS, RF, RPART),
same layout/aesthetic as generate_f1_tables.py (F1 Metric), but bold marks
the LOWEST value per row -- SERA is lower-is-better, the opposite of F1.

Per-fold relevance control points (ctrl_x/ctrl_phi/ctrl_deriv, captured
from R's uba::phi.control() during the rerun) are used via method="manual"
-style reconstruction so SERA uses the exact same relevance function as
this project's existing F1 numbers -- see CLAUDE.md.

Source data: Results (Clean)/Results Data/raw_predictions_by_dataset/
             <DSxx>/<workflow>_predictions.csv + <workflow>_phi_ctrl.csv
Output:
  - one .tex file per model family in
    Results (Clean)/Evaluation Metrics/Latex Tables/SERA/
  - the raw per-(dataset, workflow, iteration) SERA values themselves
    (all 52 workflows, not just the 5-family x 10-strategy grid the
    tables use) in Results (Clean)/Results Data/raw_sera_by_dataset/
"""
import sys
from pathlib import Path

import pandas as pd

from sera_metric import sera

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tables_common import DS_LABELS, datasets_on_disk, labels_for, MODEL_FAMILIES, STRATEGIES, ROOT, fmt_mean_sd, parse_workflow, table_star

DATA_DIR = ROOT / "Results (Clean)/Results Data/raw_predictions_by_dataset"
OUT_DIR = ROOT / "Results (Clean)/Evaluation Metrics/Latex Tables/SERA"
RAW_OUT_DIR = ROOT / "Results (Clean)/Results Data/raw_sera_by_dataset"


def _workflow_stem(family: str, strategy: str) -> str:
    return f"mc.{family}" if strategy == "baseline" else f"mc.{family}_{strategy}"


def _sera_per_iteration(preds_df: pd.DataFrame, ctrl_df: pd.DataFrame) -> pd.Series:
    def sera_for_group(g):
        it = g.name
        ctrl_pts = ctrl_df.loc[ctrl_df["iteration"] == it, ["ctrl_x", "ctrl_phi", "ctrl_deriv"]].to_numpy()
        return sera(g["y_true"].to_numpy(), g["y_pred"].to_numpy(), ctrl_pts)
    return preds_df.groupby("iteration").apply(sera_for_group, include_groups=False)


def build_family_table(data_dir: Path, family: str, ds_labels: dict, strategies: list) -> str:
    strategy_hdr = [r"\textbf{Baseline}"] + [rf"\textbf{{\texttt{{{s}}}}}" for s in strategies[1:]]
    row_lines = []

    for ds_id, ds_label in ds_labels.items():
        ds_dir = data_dir / ds_id
        values = []
        for strategy in strategies:
            stem = _workflow_stem(family, strategy)
            preds_df = pd.read_csv(ds_dir / f"{stem}_predictions.csv")
            ctrl_df = pd.read_csv(ds_dir / f"{stem}_phi_ctrl.csv")
            per_it = _sera_per_iteration(preds_df, ctrl_df)
            values.append((per_it.mean(), per_it.std()))

        means = [v[0] for v in values]
        best = means.index(min(means))
        cells = [fmt_mean_sd(m, s, j == best) for j, (m, s) in enumerate(values)]
        row_lines.append(f"{ds_label} & " + " & ".join(cells))

    col_spec = "l" + "c" * len(strategies)
    header = r"\textbf{Dataset} & " + " & ".join(strategy_hdr)
    family_label = MODEL_FAMILIES.get(family, family)
    body = " \\\\\n".join(row_lines) + r" \\"

    caption = (rf"\textbf{{SERA (Mean $\pm$ SD) by Resampling Strategy -- {family_label}}}, across the "
               r"evaluated time series forecasting datasets ($50$ Monte Carlo folds per cell). Bold values "
               r"indicate the LOWEST (best) SERA for each dataset.")
    return table_star(caption=caption, label=f"tab:exp002_sera_{family}", col_spec=col_spec,
                       header=header, body=body)


def export_raw_sera(data_dir: Path, out_dir: Path, ds_labels: dict) -> None:
    """Writes one CSV per dataset with the SERA of every workflow's every
    iteration (all 52 workflows, incl. mc.arima/mc.BDES) -- the actual
    per-fold numbers behind the aggregated tables above."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds_id in ds_labels:
        ds_dir = data_dir / ds_id
        rows = []
        for preds_path in sorted(ds_dir.glob("*_predictions.csv")):
            workflow = preds_path.stem.removesuffix("_predictions")
            ctrl_path = ds_dir / f"{workflow}_phi_ctrl.csv"
            model_family, strategy = parse_workflow(workflow)

            preds_df = pd.read_csv(preds_path)
            ctrl_df = pd.read_csv(ctrl_path)
            per_it = _sera_per_iteration(preds_df, ctrl_df)
            rows.append(pd.DataFrame({
                "dataset_id": ds_id, "workflow": workflow,
                "model_family": model_family, "strategy": strategy,
                "iteration": per_it.index, "sera": per_it.to_numpy(),
            }))

        out_file = out_dir / f"{ds_id}.csv"
        pd.concat(rows, ignore_index=True).to_csv(out_file, index=False)
        print(f"Wrote {out_file}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Iterate what the predictions extraction actually produced, not the full
    # DS_LABELS catalogue -- see tables_common.labels_for().
    ds_labels = labels_for(datasets_on_disk(DATA_DIR))
    missing = [d for d in DS_LABELS if d not in ds_labels]
    if missing:
        print(f"NOTE: no predictions on disk for {missing} -- skipped.")
    for family in MODEL_FAMILIES:
        tex = build_family_table(DATA_DIR, family, ds_labels, STRATEGIES)
        out_file = OUT_DIR / f"sera_table_{family}.tex"
        out_file.write_text(tex)
        print(f"Wrote {out_file}")
    export_raw_sera(DATA_DIR, RAW_OUT_DIR, ds_labels)
    print("Done.")


if __name__ == "__main__":
    main()
