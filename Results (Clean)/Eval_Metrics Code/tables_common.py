#!/usr/bin/env python3
"""Shared constants and LaTeX helpers for the per-metric table generators
(F1 Metric/, RMSE Metric/, SERA Metric/, Summary Tables/).

Extracted 2026-08-28 from the three original generators, which had ~150
lines of verbatim copy-paste (dataset labels, model families, strategy
list, the \\begin{table*} skeleton, the cell formatter, and the workflow
name parser). See CLAUDE.md "Recipe for a new metric" for the layout
convention this codifies.
"""
from pathlib import Path

# Re-exported so the many `from tables_common import RESULTS_DATA, LATEX_DIR`
# callers keep working; paths.py is the single source of truth (see its
# docstring for why the old hardcoded ROOT had to go).
from paths import LATEX_DIR, RESULTS_DATA, ROOT  # noqa: F401

# Row labels used throughout the appendix-style tables (results/tables/table5_sd.tex etc.)
DS_LABELS = {
    "DS01": "DS01 (Bike Daily Temp)",       "DS02": "DS02 (Bike Daily Humidity)",
    "DS03": "DS03 (Bike Daily Windspeed)",  "DS04": "DS04 (Bike Daily Count)",
    "DS05": "DS05 (Bike Hourly Temp)",      "DS06": "DS06 (Bike Hourly Humidity)",
    "DS07": "DS07 (Bike Hourly Windspeed)", "DS08": "DS08 (Bike Hourly Count)",
    "DS09": "DS09 (Icelandic River Flow)",  "DS10": "DS10 (Porto Min Temp)",
    "DS11": "DS11 (Porto Max Temp)",        "DS12": "DS12 (Porto Max Steady Wind)",
    "DS13": "DS13 (Porto Max Wind Gust)",   "DS14": r"DS14 (Istanbul S\&P)",
    "DS15": "DS15 (Istanbul DAX)",          "DS16": "DS16 (Istanbul FTSE)",
    "DS17": "DS17 (Istanbul Nikkei)",       "DS18": "DS18 (Istanbul Bovespa)",
    "DS19": "DS19 (Istanbul EU)",           "DS20": "DS20 (Istanbul EM)",
    # DS21-24 added 2026-09-04 with the DS21-24 unblock. A dataset missing from
    # this map silently drops out of every appendix table, so it must be
    # extended whenever a dataset is added -- test_tables_common guards it.
    "DS21": "DS21 (AU Elec Demand)",        "DS22": "DS22 (AU Elec Price)",
    "DS23": "DS23 (Oporto Water Pedroucos)", "DS24": "DS24 (Oporto Water Rotunda)",
}

MODEL_FAMILIES = {"lm": "Linear Model (LM)", "svm": "SVM", "mars": "MARS",
                   "rf": "Random Forest (RF)", "rpart": "CART (RPART)"}
FAMILY_LABELS = MODEL_FAMILIES

STRATEGIES = ["baseline", "UNDERB", "UNDERT", "UNDERTPhi", "OVERB", "OVERT",
              "OVERTPhi", "SMOTEB", "SMOTET", "SMOTETPhi"]
STRATEGY_HDR = [r"\textbf{Baseline}"] + [rf"\textbf{{\texttt{{{s}}}}}" for s in STRATEGIES[1:]]


def datasets_on_disk(data_dir) -> list[str]:
    """Dataset ids actually present under `data_dir`, handling both layouts
    the pipeline uses: one subdirectory per dataset
    (raw_predictions_by_dataset/DS01/) and one CSV per dataset
    (raw_iterations_by_dataset_v2/DS01_bike_daily_temp.csv)."""
    data_dir = Path(data_dir)
    ids = {p.name for p in data_dir.glob("DS??") if p.is_dir()}
    ids |= {p.name[:4] for p in data_dir.glob("DS??_*.csv")}
    return sorted(ids)


def labels_for(ds_ids) -> dict[str, str]:
    """DS_LABELS restricted to `ds_ids`, preserving DS_LABELS' order.

    Generators must iterate THIS, not DS_LABELS directly: DS_LABELS is the
    full catalogue of every dataset that could exist, while a given run may
    hold fewer (a partial cluster run) -- and, for RMSE/SERA, the predictions
    extraction can lag the F1 extraction. Iterating the catalogue against
    partial data raises a bare KeyError deep in pandas; iterating what is
    present is correct and lets the generator report the gap itself."""
    unknown = sorted(set(ds_ids) - set(DS_LABELS))
    if unknown:
        raise KeyError(f"datasets {unknown} have no DS_LABELS entry -- add them "
                       f"to tables_common.DS_LABELS (test_tables_common guards this)")
    return {ds: DS_LABELS[ds] for ds in DS_LABELS if ds in set(ds_ids)}


def parse_workflow(workflow: str) -> tuple[str, str]:
    """'mc.rf_SMOTETPhi' -> ('rf', 'SMOTETPhi'); 'mc.lm' -> ('lm', 'baseline')."""
    wf = workflow.removeprefix("mc.")
    model_family = wf.split("_")[0]
    strategy = wf.split("_", 1)[1] if "_" in wf else "baseline"
    return model_family, strategy


def fmt_mean_sd(mean: float, sd: float, bold: bool) -> str:
    if bold:
        return rf"\textbf{{{mean:.4f}}} $\pm$ \textbf{{{sd:.4f}}}"
    return rf"{mean:.4f} $\pm$ {sd:.4f}"


def table_star(*, caption: str, label: str, col_spec: str, header: str, body: str,
                resize: str | None = "1.15\\textwidth") -> str:
    """The \\begin{table*} skeleton shared by every generator (booktabs,
    makebox+resizebox centering, bold caption). `body` is the fully-formed
    row content (each row already ending in ` \\\\`, including any closing
    \\midrule + summary row a caller wants) -- this function only owns the
    surrounding scaffold. `resize=None` skips \\resizebox, for narrow
    tables that shouldn't be force-stretched to \\textwidth."""
    tabular = rf"""\begin{{tabular}}{{{col_spec}}}
\toprule
{header} \\
\midrule
{body}
\bottomrule
\end{{tabular}}"""

    if resize is not None:
        core = rf"""\resizebox{{{resize}}}{{!}}{{%
{tabular}%
}}"""
    else:
        core = tabular

    return rf"""\begin{{table*}}[t]
\centering
\caption{{{caption}}}
\label{{{label}}}
\vspace{{4pt}}
\noindent
\makebox[\textwidth][c]{{%
{core}%
}}
\end{{table*}}
"""
