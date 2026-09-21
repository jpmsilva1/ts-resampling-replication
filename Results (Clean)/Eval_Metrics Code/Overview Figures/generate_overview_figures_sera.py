#!/usr/bin/env python3
"""SERA counterpart of generate_overview_figures.py -- same 4x5 grid layout
(rows: None/UNDER/OVER/SmoteR, columns: lm/svm/mars/rf/rpart), but plotting
SERA instead of F1. SERA is lower-is-better and unnormalized (raw squared-
error area, unlike F1's 0-1 phi-weighted scale), so unlike the F1 figures
each subplot uses its own log-scaled y-axis instead of a shared 0-1 range.

Source data:
  Results (Clean)/Results Data/raw_sera_by_dataset/*.csv
  (already-computed per-dataset/workflow/iteration SERA, see SERA Metric/generate_sera_tables.py)

Output:
  Results (Clean)/Evaluation Metrics/Figures/Overview/fig7_sera_by_dataset.png
  Results (Clean)/Evaluation Metrics/Figures/Overview/fig8_sera_by_rare_cases.png
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # make Eval_Metrics Code/ importable when run directly
from paths import ROOT  # noqa: E402
RESULTS_DATA = ROOT / "Results (Clean)/Results Data"
OUT_DIR = ROOT / "Results (Clean)/Evaluation Metrics/Figures/Overview"

sys.path.insert(0, str(ROOT / "Results (Clean)/Eval_Metrics Code"))
from plot_style import apply_style, save, C_DARK, C_TEAL, C_GOLD, C_CORAL

apply_style()

FAMILIES = ["lm", "svm", "mars", "rf", "rpart"]
GROUPS = ["None", "UNDER", "OVER", "SmoteR"]

RARE_ORDER = [
    "DS20", "DS19", "DS16", "DS18", "DS03", "DS15", "DS17", "DS02",
    "DS10", "DS01", "DS14", "DS04", "DS12", "DS13", "DS11", "DS09",
    "DS05", "DS06", "DS07", "DS08"
]
NATURAL_ORDER = [f"DS{i:02d}" for i in range(1, 21)]

# %Rare per dataset, from the paper's Table 1. Imbalance Ratio (IR) is the
# majority:minority ratio, IR = (100 - %Rare) / %Rare -- inversely related to
# %Rare, so low %Rare (few rare cases) means HIGH IR. Ties broken by dataset id.
PCT_RARE = {
    "DS01": 9.9, "DS02": 9.3, "DS03": 7.8, "DS04": 13.3, "DS05": 3.5,
    "DS06": 4.8, "DS07": 12.5, "DS08": 17.6, "DS09": 21.1, "DS10": 4.8,
    "DS11": 13.3, "DS12": 11.0, "DS13": 11.1, "DS14": 16.3, "DS15": 11.4,
    "DS16": 9.7, "DS17": 11.6, "DS18": 10.1, "DS19": 8.2, "DS20": 6.8,
}
IMBALANCE_RATIO = {ds: (100 - pct) / pct for ds, pct in PCT_RARE.items()}
IMBALANCE_ORDER = sorted(IMBALANCE_RATIO, key=lambda ds: (IMBALANCE_RATIO[ds], ds))
PCT_RARE_ORDER = sorted(PCT_RARE, key=lambda ds: (PCT_RARE[ds], ds))


def _parse_workflow(wf: str):
    """Parse workflow name into (model_family, group, bias) or special case."""
    if wf == "mc.arima":
        return ("arima", "special", "arima")
    if wf == "mc.BDES":
        return ("bdes", "special", "bdes")

    x = wf.removeprefix("mc.")
    parts = x.split("_")
    model_family = parts[0]

    if len(parts) == 1:
        return (model_family, "None", "Original")

    strategy = parts[1]
    if strategy.startswith("UNDER"):
        group = "UNDER"
        bias = strategy[5:]
    elif strategy.startswith("OVER"):
        group = "OVER"
        bias = strategy[4:]
    elif strategy.startswith("SMOTE"):
        group = "SmoteR"
        bias = strategy[5:]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return (model_family, group, bias)


def load_sera_data() -> pd.DataFrame:
    files = sorted((RESULTS_DATA / "raw_sera_by_dataset").glob("DS??.csv"))
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    parsed = df["workflow"].apply(lambda w: pd.Series(_parse_workflow(w)))
    df[["model_family", "group", "bias"]] = parsed
    agg = df.groupby(["dataset_id", "workflow", "model_family", "group", "bias"])["sera"].mean().reset_index()
    agg = agg.rename(columns={"sera": "sera_mean"})
    return agg


def plot_figures(data: pd.DataFrame):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_figure(data, NATURAL_ORDER, OUT_DIR / "fig7_sera_by_dataset.png", "SERA by Dataset (Natural Order)")
    _plot_figure(data, RARE_ORDER, OUT_DIR / "fig8_sera_by_rare_cases.png", "SERA by Dataset (Sorted by Rare Cases)")
    _plot_figure(data, IMBALANCE_ORDER, OUT_DIR / "fig9_sera_by_imbalance_ratio.png", "SERA by Dataset (Sorted by Imbalance Ratio)")
    _plot_figure(data, PCT_RARE_ORDER, OUT_DIR / "fig10_sera_by_pct_rare.png", "SERA by Dataset (Sorted by %Rare)")


def _plot_figure(data: pd.DataFrame, dataset_order: list, out_path: Path, title: str):
    # Sized for its own full-width landscape page (not stacked with its F1
    # counterpart) -- font size alone can't fix a dense 4x5 grid squeezed
    # into a shared portrait page.
    fig, axes = plt.subplots(4, 5, figsize=(22, 14))
    fig.suptitle(title, fontsize=28, fontweight='bold')
    dataset_to_pos = {ds: i + 1 for i, ds in enumerate(dataset_order)}
    line_styles = {
        "Original": ("solid", C_DARK), "B": ("dashed", C_TEAL), "T": ("dotted", C_TEAL),
        "TPhi": ("dashdot", C_TEAL), "arima": ("dotted", C_GOLD), "bdes": ("dashdot", C_CORAL),
    }
    for row_idx, group in enumerate(GROUPS):
        for col_idx, family in enumerate(FAMILIES):
            ax = axes[row_idx, col_idx]
            lines_to_plot = {}
            biases = ["Original", "arima", "bdes"] if group == "None" else ["Original", "B", "T", "TPhi", "arima", "bdes"]
            for bias in biases:
                if bias == "Original":
                    # Baseline is repeated in every row as a reference line.
                    mask = (data["group"] == "None") & (data["model_family"] == family)
                elif bias in ["B", "T", "TPhi"]:
                    mask = (data["group"] == group) & (data["model_family"] == family) & (data["bias"] == bias)
                elif bias == "arima":
                    mask = (data["model_family"] == "arima")
                elif bias == "bdes":
                    mask = (data["model_family"] == "bdes")

                subset = data[mask]
                if len(subset) > 0:
                    ordered = subset.assign(pos=subset["dataset_id"].map(dataset_to_pos)).sort_values("pos")
                    lines_to_plot[bias] = (ordered["pos"], ordered["sera_mean"])

            for bias, (positions, values) in lines_to_plot.items():
                ls, color = line_styles[bias]
                ax.plot(positions, values, linestyle=ls, color=color, marker='o', markersize=4, linewidth=2.0, label=bias)

            ax.set_xlim(0, 21)
            ax.set_yscale("log")
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.tick_params(axis="both", labelsize=16)
            ax.grid(True, alpha=0.15, which="both")
            if row_idx == 0:
                ax.set_title(family.upper(), fontsize=23, fontweight='bold')
            if col_idx == 4:
                ax.text(1.14, 0.5, group, transform=ax.transAxes, fontsize=21, fontweight='bold', va='center')
            if row_idx < 3:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Dataset Position", fontsize=19)
            if col_idx == 0:
                ax.set_ylabel("SERA (log scale)", fontsize=19)

    legend_elements = [
        Line2D([0], [0], color=C_DARK, linestyle="solid", linewidth=3.5, label="Original"),
        Line2D([0], [0], color=C_TEAL, linestyle="dashed", linewidth=3.5, label="B"),
        Line2D([0], [0], color=C_TEAL, linestyle="dotted", linewidth=3.5, label="T"),
        Line2D([0], [0], color=C_TEAL, linestyle="dashdot", linewidth=3.5, label="TPhi"),
        Line2D([0], [0], color=C_GOLD, linestyle="dotted", linewidth=3.5, label="ARIMA"),
        Line2D([0], [0], color=C_CORAL, linestyle="dashdot", linewidth=3.5, label="BDES"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.02),
               fontsize=26, frameon=False, handlelength=3.5, handletextpad=0.8, columnspacing=2.2)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save(fig, out_path)
    print(f"Wrote {out_path}")
    plt.close(fig)


def main():
    data = load_sera_data()
    plot_figures(data)
    print("Done.")


if __name__ == "__main__":
    main()
