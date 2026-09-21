#!/usr/bin/env python3
"""Builds the paper's cross-cutting summary tables (Tables 6-11), F1 and
SERA only (RMSE has no paper-Table equivalent -- see CLAUDE.md):
  - Tables 6/7: per-family win-count matrix (best strategy per dataset,
    ties split 1/n) with a Total row.
  - Tables 8/9: best/worst (family, strategy) combo per dataset, across
    all 50 combos.
  - Tables 10/11: global average rank per combo across all datasets,
    sorted ascending, two-panel layout.

Reuses the CD Metrics module's data loaders (same aggregate each strategy's
per-family table cell already shows) and average_ranks().

Source data: Results (Clean)/Results Data/raw_iterations_by_dataset_v2/*.csv (F1),
             Results (Clean)/Results Data/raw_sera_by_dataset/*.csv (SERA)
Output: Results (Clean)/Evaluation Metrics/Latex Tables/Summary/*.tex
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tables_common import DS_LABELS, labels_for, MODEL_FAMILIES, STRATEGIES, ROOT, table_star

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "CD Metrics"))
from generate_cd_diagrams import load_metric_long
from cd_stats import average_ranks

OUT_DIR = ROOT / "Results (Clean)/Evaluation Metrics/Latex Tables/Summary"
METRICS = {"F1": True, "SERA": False}  # metric -> higher_is_better


def cell_means(metric: str) -> pd.DataFrame:
    """20 x 50 frame: rows = dataset_id, columns = 'family.strategy',
    values = mean of `metric` over the 50 Monte Carlo iterations for that
    combo, restricted to the 5x10 grid (drops mc.arima/mc.BDES)."""
    long_df = load_metric_long(metric)
    long_df = long_df[long_df["model_family"].isin(MODEL_FAMILIES) & long_df["strategy"].isin(STRATEGIES)]
    per_combo = (long_df.groupby(["dataset_id", "model_family", "strategy"])["value"]
                 .mean().reset_index())
    per_combo["combo"] = per_combo["model_family"] + "." + per_combo["strategy"]
    matrix = per_combo.pivot(index="dataset_id", columns="combo", values="value")
    cols = [f"{fam}.{strat}" for fam in MODEL_FAMILIES for strat in STRATEGIES]
    return matrix[cols]


def winners(row: pd.Series, higher_is_better: bool) -> list:
    """Combo names tied for best in `row` -- ties decided on the value
    rounded to 4 decimals (matches the precision the tables already
    display), not raw float equality."""
    rounded = row.round(4)
    best = rounded.max() if higher_is_better else rounded.min()
    return list(rounded[rounded == best].index)


def win_counts(means: pd.DataFrame, families: list, strategies: list, higher_is_better: bool) -> pd.DataFrame:
    """rows=family (+ 'Total'), columns=strategy, cells=win count (ties
    split 1/n) of that strategy within that family, across every row of
    `means`. Each family row sums to len(means); Total sums to
    len(means) * len(families)."""
    counts = pd.DataFrame(0.0, index=list(families), columns=list(strategies))
    for family in families:
        cols = [f"{family}.{s}" for s in strategies]
        for ds_id in means.index:
            row = means.loc[ds_id, cols]
            row.index = strategies
            win = winners(row, higher_is_better)
            for combo in win:
                counts.loc[family, combo] += 1.0 / len(win)
    counts.loc["Total"] = counts.sum()
    return counts


def fmt_value(v: float) -> str:
    if v == 0 or 1e-3 <= abs(v) < 1e5:
        return f"{v:.4f}"
    mant, exp = f"{v:.4e}".split("e")
    return rf"${mant} \times 10^{{{int(exp)}}}$"


def _win_count_table(metric: str, higher_is_better: bool, means: pd.DataFrame | None = None) -> pd.DataFrame:
    cells = means if means is not None else cell_means(metric)
    # Derived, not hardcoded: each family's win counts must sum to the number
    # of datasets actually in the data (ties split fractionally). A stale
    # constant here fired as a bare AssertionError the moment DS21 landed.
    n_datasets = cells.index.get_level_values("dataset_id").nunique()
    table = win_counts(cells, list(MODEL_FAMILIES), STRATEGIES, higher_is_better)
    for family in MODEL_FAMILIES:
        assert abs(table.loc[family].sum() - n_datasets) < 1e-9, (
            f"{family}: win counts sum to {table.loc[family].sum()}, expected {n_datasets}")
    assert abs(table.loc["Total"].sum() - n_datasets * len(MODEL_FAMILIES)) < 1e-9
    return table


def _win_count_cells(table: pd.DataFrame, row_key: str) -> str:
    return " & ".join(f"{table.loc[row_key, s]:.1f}" for s in STRATEGIES)


def win_count_tex(metric: str, higher_is_better: bool, means: pd.DataFrame | None = None) -> str:
    table = _win_count_table(metric, higher_is_better, means)
    # N is each family row's own sum -- one win per (dataset, family) cell -- so
    # the caption tracks the data instead of a hardcoded 20.
    n_ds = int(round(table.loc[next(iter(MODEL_FAMILIES))].sum()))

    row_lines = [f"{MODEL_FAMILIES[fam]} & {_win_count_cells(table, fam)}" for fam in MODEL_FAMILIES]
    total_line = rf"\textbf{{Total}} & {_win_count_cells(table, 'Total')}"
    body = " \\\\\n".join(row_lines) + r" \\" + "\n\\midrule\n" + total_line + r" \\"

    strategy_hdr = [r"\textbf{Baseline}"] + [rf"\textbf{{\texttt{{{s}}}}}" for s in STRATEGIES[1:]]
    col_spec = "l" + "c" * len(STRATEGIES)
    header = r"\textbf{Model Family} & " + " & ".join(strategy_hdr)
    caption = (rf"\textbf{{{metric} Best-Strategy Win Counts by Model Family}}, counting how often each "
               r"resampling strategy is the best performer for a dataset (ties split evenly), across the "
               rf"{n_ds} datasets. Total row sums across all 5 model families.")
    return table_star(caption=caption, label=f"tab:exp002_{metric.lower()}_win_counts",
                       col_spec=col_spec, header=header, body=body)


def _best_worst_cells(row: pd.Series, metric: str, higher_is_better: bool) -> str:
    """'Best Combo & Best <metric> & Worst Combo & Worst <metric>' for one
    dataset's row of the 50-combo frame -- the four data cells shared by
    the single-metric and side-by-side best/worst tables."""
    best_combo = row.idxmax() if higher_is_better else row.idxmin()
    worst_combo = row.idxmin() if higher_is_better else row.idxmax()
    best_val, worst_val = row[best_combo], row[worst_combo]
    best_label = rf"\texttt{{{best_combo.split('.')[0].upper()}.{best_combo.split('.', 1)[1]}}}"
    worst_label = rf"\texttt{{{worst_combo.split('.')[0].upper()}.{worst_combo.split('.', 1)[1]}}}"
    dagger = r"$^\dagger$" if metric == "F1" and worst_val < 1e-4 else ""
    return f"{best_label} & {fmt_value(best_val)} & {worst_label} & {fmt_value(worst_val)}{dagger}"


def best_worst_tex(metric: str, higher_is_better: bool, means: pd.DataFrame | None = None) -> str:
    means = means if means is not None else cell_means(metric)
    row_lines = [f"{ds_label} & {_best_worst_cells(means.loc[ds_id], metric, higher_is_better)}"
                 for ds_id, ds_label in labels_for(means.index).items()]
    body = " \\\\\n".join(row_lines) + r" \\"

    col_spec = "lllll"
    header = (r"\textbf{Dataset} & \textbf{Best Combo} & \textbf{Best " + metric +
              r"} & \textbf{Worst Combo} & \textbf{Worst " + metric + r"}")
    footnote = (r" $^\dagger$Below the utility-based $F_1^\phi$ floor ($<10^{-4}$)." if metric == "F1" else "")
    caption = (rf"\textbf{{Best and Worst (Model, Strategy) Combination per Dataset -- {metric}}}, "
               r"across all 50 model-family$\times$strategy combinations." + footnote)
    return table_star(caption=caption, label=f"tab:exp002_{metric.lower()}_best_worst",
                       col_spec=col_spec, header=header, body=body)


def _avg_rank_entries(means: pd.DataFrame, higher_is_better: bool) -> list:
    ranks = average_ranks(means, higher_is_better=higher_is_better).sort_values()
    assert len(ranks) == means.shape[1]
    return list(ranks.items())


def _fmt_rank_entry(entries: list, i: int) -> str:
    combo, rank = entries[i]
    fam, strat = combo.split(".", 1)
    label = rf"\texttt{{{fam.upper()}.{strat}}}"
    rank_str = f"{rank:.1f}"
    if i == 0:
        label, rank_str = rf"\textbf{{{label}}}", rf"\textbf{{{rank_str}}}"
    return f"{label} & {rank_str}"


def avg_rank_tex(metric: str, higher_is_better: bool, means: pd.DataFrame | None = None) -> str:
    means = means if means is not None else cell_means(metric)
    entries = _avg_rank_entries(means, higher_is_better)
    n_ds = len(means)

    half = len(entries) // 2
    row_lines = [f"{_fmt_rank_entry(entries, i)} & {_fmt_rank_entry(entries, i + half)}" for i in range(half)]
    body = " \\\\\n".join(row_lines) + r" \\"

    caption = (rf"\textbf{{Global Average Rank -- {metric}}}, ranking all 50 model-family$\times$strategy "
               rf"combinations jointly across the {n_ds} datasets (rank 1 = best; NOT a per-family ranking).")
    return table_star(caption=caption, label=f"tab:exp002_{metric.lower()}_avg_ranks",
                       col_spec="lclc", header=r"\textbf{Combo} & \textbf{Avg. Rank} & \textbf{Combo} & \textbf{Avg. Rank}",
                       body=body, resize=None)


def best_worst_side_by_side_tex(means_f1: pd.DataFrame | None = None,
                                 means_sera: pd.DataFrame | None = None) -> str:
    """f1_table_best_worst.tex and sera_table_best_worst.tex as two column
    panels of one table (F1 columns then SERA columns), one row per dataset
    -- same two-panel `resize=None` idea as avg_rank_tex, but panels are
    per-metric columns instead of a row split."""
    means_f1 = means_f1 if means_f1 is not None else cell_means("F1")
    means_sera = means_sera if means_sera is not None else cell_means("SERA")
    row_lines = [
        f"{ds_label} & {_best_worst_cells(means_f1.loc[ds_id], 'F1', METRICS['F1'])} & "
        f"{_best_worst_cells(means_sera.loc[ds_id], 'SERA', METRICS['SERA'])}"
        # Intersection: F1 comes from raw_iterations_by_dataset_v2/ while SERA
        # comes from raw_sera_by_dataset/, and the predictions extraction that
        # feeds SERA can lag the F1 extraction -- a side-by-side row needs both.
        for ds_id, ds_label in labels_for(means_f1.index.intersection(means_sera.index)).items()
    ]
    body = " \\\\\n".join(row_lines) + r" \\"

    col_spec = "l" + "llll" * 2
    header = (r"& \multicolumn{4}{c}{\textbf{F1}} & \multicolumn{4}{c}{\textbf{SERA}} \\" + "\n"
              r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}" + "\n"
              r"\textbf{Dataset} & \textbf{Best Combo} & \textbf{Best Value} & \textbf{Worst Combo} & "
              r"\textbf{Worst Value} & \textbf{Best Combo} & \textbf{Best Value} & \textbf{Worst Combo} & "
              r"\textbf{Worst Value}")
    caption = (r"\textbf{Best and Worst Combination per Dataset -- F1 and SERA Side by Side}, across all 50 "
               r"model-family$\times$strategy combinations per metric. $^\dagger$Below the utility-based "
               r"$F_1^\phi$ floor ($<10^{-4}$).")
    return table_star(caption=caption, label="tab:exp002_f1_sera_best_worst",
                       col_spec=col_spec, header=header, body=body)


def win_counts_side_by_side_tex(means_f1: pd.DataFrame | None = None,
                                 means_sera: pd.DataFrame | None = None) -> str:
    """f1_table_best_counts.tex and sera_table_best_counts.tex stacked
    vertically (F1 table, then SERA table below it) instead of merged into
    one wide side-by-side table -- each keeps its original family-row /
    strategy-column orientation."""
    return win_count_tex("F1", METRICS["F1"], means_f1) + "\n\n" + win_count_tex("SERA", METRICS["SERA"], means_sera)


def avg_ranks_side_by_side_tex(means_f1: pd.DataFrame | None = None,
                                means_sera: pd.DataFrame | None = None) -> str:
    """f1_table_avg_ranks.tex and sera_table_avg_ranks.tex as two column
    panels of one table -- each metric keeps its own two-way row split
    (rank i paired with rank i+25), and the F1 block sits beside the SERA
    block on every row."""
    means_f1 = means_f1 if means_f1 is not None else cell_means("F1")
    means_sera = means_sera if means_sera is not None else cell_means("SERA")
    entries_f1 = _avg_rank_entries(means_f1, METRICS["F1"])
    entries_sera = _avg_rank_entries(means_sera, METRICS["SERA"])
    n_ds = len(means_f1)
    half = len(entries_f1) // 2
    assert half == len(entries_sera) // 2

    row_lines = [
        f"{_fmt_rank_entry(entries_f1, i)} & {_fmt_rank_entry(entries_f1, i + half)} & "
        f"{_fmt_rank_entry(entries_sera, i)} & {_fmt_rank_entry(entries_sera, i + half)}"
        for i in range(half)
    ]
    body = " \\\\\n".join(row_lines) + r" \\"

    pair_hdr = r"\textbf{Combo} & \textbf{Avg. Rank} & \textbf{Combo} & \textbf{Avg. Rank}"
    header = (r"\multicolumn{4}{c}{\textbf{F1}} & \multicolumn{4}{c}{\textbf{SERA}} \\" + "\n"
              r"\cmidrule(lr){1-4}\cmidrule(lr){5-8}" + "\n"
              f"{pair_hdr} & {pair_hdr}")
    caption = (r"\textbf{Global Average Rank -- F1 and SERA Side by Side}, ranking all 50 "
               rf"model-family$\times$strategy combinations jointly across the {n_ds} datasets per metric "
               r"(rank 1 = best; NOT a per-family ranking).")
    return table_star(caption=caption, label="tab:exp002_f1_sera_avg_ranks",
                       col_spec="lclc" * 2, header=header, body=body)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for metric, higher_is_better in METRICS.items():
        m = metric.lower()
        for suffix, builder in (("best_counts", win_count_tex), ("best_worst", best_worst_tex),
                                 ("avg_ranks", avg_rank_tex)):
            out_file = OUT_DIR / f"{m}_table_{suffix}.tex"
            out_file.write_text(builder(metric, higher_is_better))
            print(f"Wrote {out_file}")

    for suffix, builder in (("best_worst", best_worst_side_by_side_tex), ("best_counts", win_counts_side_by_side_tex),
                             ("avg_ranks", avg_ranks_side_by_side_tex)):
        out_file = OUT_DIR / f"f1_sera_table_{suffix}.tex"
        out_file.write_text(builder())
        print(f"Wrote {out_file}")
    print("Done.")


if __name__ == "__main__":
    main()
