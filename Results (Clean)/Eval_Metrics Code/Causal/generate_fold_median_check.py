#!/usr/bin/env python3
"""Robustness check on how the 50 Monte Carlo folds are aggregated per cell.

Every other SERA/RMSE result scores a (dataset, family, strategy) cell by the
MEAN over its 50 folds. A model that extrapolates wildly on one fold can then
dominate the cell: e.g. DS22 / MARS / OVERT, where 4 test rows during a price
spike are predicted at ~1e8 and that single fold's SERA is ~1e9x the median.
This recomputes the pooled effect (same estimator and bootstrap as
generate_effect_forest_plots.py) with the MEDIAN over folds instead, and lists
the cells where one fold dominates the mean.

Output: Evaluation Metrics/Latex Tables/Causal/fold_median_check.{tex,txt}
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import LATEX_DIR  # noqa: E402
from tables_common import table_star  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_effects import bootstrap_ci  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CD Metrics"))
from generate_cd_diagrams import load_metric_long  # noqa: E402

FAMILIES = ["lm", "svm", "mars", "rf", "rpart"]
STRATEGY_ORDER = ["UNDERB", "UNDERT", "UNDERTPhi", "OVERB", "OVERT",
                  "OVERTPhi", "SMOTEB", "SMOTET", "SMOTETPhi"]
OUT_DIR = LATEX_DIR / "Causal"
N_BOOT, SEED = 2000, 0
TRIM = 0.10


def panel(long_df: pd.DataFrame, agg: str) -> pd.DataFrame:
    """One row per (dataset, family, strategy): `agg` over the 50 folds, logged."""
    p = (long_df[long_df["model_family"].isin(FAMILIES)]
         .groupby(["dataset_id", "model_family", "strategy"])["value"].agg(agg).reset_index())
    p["log_value"] = np.log(p["value"])
    return p


def baseline_wins(p: pd.DataFrame) -> int:
    """Cells where baseline has the lowest (best) value; same rule as Figure 3."""
    wide = p.pivot_table(index=["dataset_id", "model_family"], columns="strategy", values="value")
    return int((wide.round(4).idxmin(axis=1) == "baseline").sum())


def dominated_cells(long_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Cells where the single worst fold is >50% of the fold sum and >100x the
    median fold -- i.e. the cell's mean is essentially that one fold."""
    rows = []
    sub = long_df[long_df["model_family"].isin(FAMILIES)]
    for (ds, wf), g in sub.groupby(["dataset_id", "workflow"]):
        v = g["value"].to_numpy()
        if v.max() > 0.5 * v.sum() and v.max() > 100 * np.median(v):
            rows.append({"metric": metric, "dataset": ds, "workflow": wf,
                         "worst_over_median": v.max() / np.median(v),
                         "mean_over_mean_without_worst": v.mean() / np.delete(v, v.argmax()).mean()})
    return pd.DataFrame(rows)


def failure_rates(long_df: pd.DataFrame) -> pd.DataFrame:
    """Share of folds (%) per strategy whose value is >100x (extreme) or >10x
    (moderate) its own cell's median fold -- the failure rates quoted in the
    report's extreme-run check."""
    sub = long_df[long_df["model_family"].isin(FAMILIES)]
    ratio = sub["value"] / sub.groupby(["dataset_id", "model_family", "strategy"])["value"].transform("median")
    return (pd.DataFrame({"strategy": sub["strategy"], ">100x": ratio > 100, ">10x": ratio > 10})
            .groupby("strategy")[[">100x", ">10x"]].mean() * 100)


def trimmed_ci(p: pd.DataFrame, cut: float = TRIM) -> pd.DataFrame:
    """Between-cell robust variant: trimmed mean of the per-cell log ratios
    (drops the `cut` share of cells at each end), with the same cell-bootstrap
    as causal_effects.bootstrap_ci (whole cells resampled, pairing kept)."""
    wide = p.pivot_table(index=["dataset_id", "model_family"], columns="strategy", values="log_value")
    diffs = wide[STRATEGY_ORDER].sub(wide["baseline"], axis=0).to_numpy()
    rng = np.random.default_rng(SEED)
    boot = np.array([trim_mean(diffs[rng.integers(0, len(diffs), len(diffs))], cut, axis=0)
                     for _ in range(N_BOOT)])
    return pd.DataFrame({"tau": trim_mean(diffs, cut, axis=0),
                         "ci_lo": np.percentile(boot, 2.5, axis=0),
                         "ci_hi": np.percentile(boot, 97.5, axis=0)}, index=STRATEGY_ORDER)


def fmt_ci(row) -> str:
    return f"{row.tau:+.3f} [{row.ci_lo:.2f}, {row.ci_hi:.2f}]"


VARIANTS = [("mean", "Mean of folds"), ("median", "Median of folds"), ("trim", "Trimmed across cells")]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols, wins, dominated, fails = {}, {}, [], {}
    for metric in ["SERA", "RMSE"]:
        long_df = load_metric_long(metric)
        dominated.append(dominated_cells(long_df, metric))
        fails[metric] = failure_rates(long_df)
        for agg in ["mean", "median"]:
            p = panel(long_df, agg)
            cols[(metric, agg)] = bootstrap_ci(p, dv="log_value", n_boot=N_BOOT, seed=SEED).reindex(STRATEGY_ORDER)
            wins[(metric, agg)] = f"{baseline_wins(p)} / {len(p) // 10}"
        cols[(metric, "trim")] = trimmed_ci(panel(long_df, "mean"))
        wins[(metric, "trim")] = "--"
    dom = pd.concat(dominated, ignore_index=True)
    n_dom = dom["metric"].value_counts()

    keys = [(m, v) for m in ["SERA", "RMSE"] for v, _ in VARIANTS]
    lines = ["Fold-aggregation robustness check: pooled log ratio vs. baseline",
             "(log strategy - log baseline; positive = worse), 95% bootstrap CI over cells.",
             f"trim = mean of folds per cell, then {TRIM:.0%} trimmed mean across cells.", ""]
    lines.append(f"{'strategy':10s} " + " ".join(f"{m+' '+a:>22s}" for m, a in keys))
    for s in STRATEGY_ORDER:
        lines.append(f"{s:10s} " + " ".join(f"{fmt_ci(cols[k].loc[s]):>22s}" for k in keys))
    lines.append(f"{'base wins':10s} " + " ".join(f"{wins[k]:>22s}" for k in keys))
    lines += ["", f"Cells whose fold-mean is dominated by one fold (worst fold >50% of the fold sum "
              f"and >100x the median fold): {n_dom.get('SERA', 0)} SERA, {n_dom.get('RMSE', 0)} RMSE",
              dom.to_string(index=False, float_format=lambda v: f"{v:.3g}")]
    for metric, fr in fails.items():
        lines += ["", f"{metric}: % of folds more than 100x / 10x their cell's median fold",
                  fr.round(2).to_string()]
    (OUT_DIR / "fold_median_check.txt").write_text("\n".join(lines) + "\n")

    sub = " & ".join(lab for _ in range(2) for _, lab in VARIANTS)
    header = (r"\textbf{Strategy} & \multicolumn{3}{c}{\textbf{SERA}} & \multicolumn{3}{c}{\textbf{RMSE}} \\"
              "\n" r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" "\n" " & " + sub)
    body = [rf"\texttt{{{s}}} & " + " & ".join(
                rf"${c.tau:+.3f}$ {{\scriptsize[{c.ci_lo:.2f}, {c.ci_hi:.2f}]}}"
                for c in (cols[k].loc[s] for k in keys)) + r" \\"
            for s in STRATEGY_ORDER]
    body.append(r"\midrule")
    body.append(r"Baseline wins & " + " & ".join(wins[k] for k in keys) + r" \\")
    tex = table_star(
        caption=(r"\textbf{Robustness of the pooled effect vs.\ baseline to how folds and cells are "
                 r"aggregated.} Pooled log ratio $\log(\text{strategy}) - \log(\text{baseline})$ "
                 r"(positive = worse) with 95\% bootstrap interval, computed as in "
                 r"Figure~\ref{fig:effect_forest}. \emph{Mean of folds}: each cell scored by the mean of "
                 r"its 50 folds (the report's default). \emph{Median of folds}: by the median instead. "
                 rf"\emph{{Trimmed across cells}}: mean of folds, then a {TRIM:.0%} trimmed mean of the "
                 r"120 cell log ratios instead of the plain mean. The last row counts cells where the "
                 r"baseline has the lowest error.").replace("%", r"\%").replace(r"\\%", r"\%"),
        label="tab:exp002_fold_median_check",
        col_spec="lcccccc",
        header=header,
        body="\n".join(body),
        resize=r"\textwidth",
    )
    (OUT_DIR / "fold_median_check.tex").write_text(tex)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
