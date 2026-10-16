#!/usr/bin/env python3
"""Tables, figures and prose macros for the relevance-region section of the
main report.

Output:
  Evaluation Metrics/Latex Tables/Region/*.tex
  Evaluation Metrics/Figures/Region/*.{pdf,png}
  Evaluation Metrics/Latex Tables/Region/region_numbers.tex   prose macros
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from region_metrics import METRICS  # noqa: E402
from region_stats import ROPE, bayes_table, effect_cells, wl_table  # noqa: E402
from paths import ROOT  # noqa: E402
from plot_style import C_GOLD, C_GREY, C_TEAL, apply_style, save  # noqa: E402
from tables_common import MODEL_FAMILIES, STRATEGIES, table_star  # noqa: E402

apply_style()

DATA = ROOT / "Results (Clean)/Results Data/raw_region_by_dataset"
TABLES = ROOT / "Results (Clean)/Evaluation Metrics/Latex Tables/Region"
FIGURES = ROOT / "Results (Clean)/Evaluation Metrics/Figures/Region"

STRATS = STRATEGIES[1:]
FAMS = list(MODEL_FAMILIES)
ORDER = ("rare", "normal")

_DIG = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
        "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
_MACROS: dict[str, str] = {}


def macro(name: str, value) -> str:
    """Register a \\newcommand. Digits and dots cannot appear in a TeX macro
    name, so digits are spelled out and dots dropped."""
    clean = "".join(_DIG.get(c, c) for c in name if c != ".")
    _MACROS[clean] = str(value)
    return clean


def table_local(**kw) -> str:
    """tables_common.table_star as a plain `table` with [!htbp].

    table_star emits `table*[t]`, which is right for the two-column papers it
    was written for. This report is one-column, where a starred float can only
    reach the top of a page, so a run of them drifts away from the prose that
    cites it. The shared helper is left untouched; it has other callers.
    """
    s = table_star(**kw)
    return (s.replace(r"\begin{table*}[t]", r"\begin{table}[!htbp]")
             .replace(r"\end{table*}", r"\end{table}"))


# ------------------------------------------------------------------ tables ---

def tab_wl(wl: pd.DataFrame, metric: str, region: str) -> str:
    sub = wl[wl.region == region].set_index(["family", "strategy"])
    cell = lambda r: f"{int(r.win)} ({int(r.sigwin)}) / {int(r.loss)} ({int(r.sigloss)})"
    body = [f"{MODEL_FAMILIES[f]} & " + " & ".join(cell(sub.loc[(f, s)]) for s in STRATS) + r" \\"
            for f in FAMS]
    tot = []
    for s in STRATS:
        x = sub.xs(s, level="strategy")
        tot.append(rf"\textbf{{{int(x.win.sum())} ({int(x.sigwin.sum())}) / "
                   rf"{int(x.loss.sum())} ({int(x.sigloss.sum())})}}")
    body += [r"\midrule", r"\textbf{Total} & " + " & ".join(tot) + r" \\"]
    n_ds = int(sub["n_datasets"].max())
    return table_local(
        caption=(rf"\textbf{{{METRICS[metric]} on the {region} region only.}} Paired Wilcoxon "
                 rf"signed-rank test of each strategy against its own un-resampled baseline, "
                 rf"over the Monte Carlo folds of each of the {n_ds} datasets, counted as Wins "
                 rf"(significant) / Losses (significant) at $\alpha = 0.05$. A win means the "
                 rf"strategy has the \emph{{lower}} {METRICS[metric]} inside the region. The "
                 rf"region is fixed by $\phi(y)$, so both sides are scored on exactly the same "
                 rf"points. Ties are omitted, so a row need not sum to {n_ds}."),
        label=f"tab:exp002_region_wl_{metric}_{region}",
        col_spec="l" + "c" * len(STRATS),
        header=r"\textbf{Model} & " + " & ".join(rf"\textbf{{\texttt{{{s}}}}}" for s in STRATS),
        body="\n".join(body))


def tab_flip(wl_all: dict) -> str:
    body = []
    for metric, lab in METRICS.items():
        cells = []
        for region in ORDER:
            sub = wl_all[metric][wl_all[metric].region == region]
            w, sw, l, sl = (int(sub.win.sum()), int(sub.sigwin.sum()),
                            int(sub.loss.sum()), int(sub.sigloss.sum()))
            cells += [f"{w} ({sw})", f"{l} ({sl})"]
            for k, v in (("W", w), ("SW", sw), ("L", l), ("SL", sl)):
                macro(f"rg{metric}{region}{k}", v)
        body.append(rf"{lab} & " + " & ".join(cells) + r" \\")
    n_cells = int(wl_all["rmse"].query("region == 'rare'")[["win", "loss", "tie"]].sum().sum())
    macro("rgNCells", n_cells)
    return table_local(
        caption=(rf"\textbf{{The verdict reverses between regions.}} Strategy wins and losses "
                 rf"against the un-resampled baseline, summed over all {n_cells} (dataset "
                 rf"$\times$ model family $\times$ strategy) comparisons, with the significant "
                 rf"count in parentheses. The same fitted models and the same predictions are "
                 rf"scored twice; only the set of instances changes."),
        label="tab:exp002_region_flip", col_spec="lcccc",
        header=(r"\textbf{Metric} & \multicolumn{2}{c}{\textbf{Rare region}} & "
                r"\multicolumn{2}{c}{\textbf{Normal region}} \\"
                "\n" r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}"
                "\n" r" & \textbf{Wins (sig)} & \textbf{Losses (sig)} & "
                r"\textbf{Wins (sig)} & \textbf{Losses (sig)}"),
        body="\n".join(body), resize=None)


def tab_effect(eff_all: dict) -> str:
    def cell(df, strat, region):
        v = df[(df.strategy == strat) & (df.region == region)].pct.median()
        return rf"\textbf{{{v:+.1f}}}" if v > 0 else f"{v:+.1f}"

    body = [rf"\texttt{{{s}}} & "
            + " & ".join(cell(eff_all[m], s, r) for m in METRICS for r in ORDER) + r" \\"
            for s in STRATS]
    allc = []
    for m in METRICS:
        for r in ORDER:
            v = eff_all[m][eff_all[m].region == r].pct.median()
            allc.append(rf"\textbf{{{v:+.1f}}}" if v > 0 else f"{v:+.1f}")
            macro(f"rgEff{m}{r}", f"{v:+.1f}")
    body += [r"\midrule", r"\textbf{All strategies} & " + " & ".join(allc) + r" \\"]
    return table_local(
        caption=(r"\textbf{Size of the trade, by strategy.} Median percentage change in each "
                 r"metric, strategy against its own baseline, over the 24 datasets $\times$ 5 "
                 r"model families. \textbf{Positive means the strategy has the lower error}, "
                 r"and bold marks the cells where the strategy is ahead. Computed on the log "
                 r"ratio, as the Bayesian test is, because these metrics span orders of "
                 r"magnitude across datasets and raw differences cannot be pooled. "
                 r"Table~\ref{tab:exp002_region_flip} reports which way each comparison goes; "
                 r"this table reports by how much."),
        label="tab:exp002_region_effect", col_spec="l" + "rr" * len(METRICS),
        header=(r"\textbf{Strategy} & "
                + " & ".join(rf"\multicolumn{{2}}{{c}}{{\textbf{{{v}}}}}" for v in METRICS.values())
                + r" \\" + "\n"
                + " ".join(rf"\cmidrule(lr){{{2*i+2}-{2*i+3}}}" for i in range(len(METRICS)))
                + "\n & " + " & ".join(r"\textbf{Rare} & \textbf{Normal}" for _ in METRICS)),
        body="\n".join(body), resize=None)


def tab_bayes(bayes_all: dict) -> str:
    body = []
    for metric, lab in METRICS.items():
        for region in ORDER:
            sub = bayes_all[metric].query("region == @region").groupby("strategy").p_strategy.mean()
            body.append(rf"{lab} & {region} & " + " & ".join(f"{sub[s]:.2f}" for s in STRATS) + r" \\")
    return table_local(
        caption=(r"\textbf{Bayesian signed-rank test: posterior probability that the strategy "
                 r"practically wins}, averaged over the five model families. The test of "
                 r"\citet{benavoli2017time} is run on the per-dataset log ratio against the "
                 rf"baseline, with a region of practical equivalence of $\pm{100*(np.exp(ROPE)-1):.0f}\%$. "
                 r"A value near $0$ means the baseline practically wins, a value near $1$ means "
                 r"the strategy does, and the remaining probability mass sits in the draw band. "
                 r"Read alongside Table~\ref{tab:exp002_region_flip}: the Wilcoxon test asks "
                 r"whether a difference exists, this one asks whether it is large enough to "
                 r"matter."),
        label="tab:exp002_region_bayes", col_spec="ll" + "c" * len(STRATS),
        header=(r"\textbf{Metric} & \textbf{Region} & "
                + " & ".join(rf"\textbf{{\texttt{{{s}}}}}" for s in STRATS)),
        body="\n".join(body))


def tab_quantiles(q: pd.DataFrame) -> str:
    cols = ["q25", "q50", "q75", "q90", "q99", "max"]
    body = []
    for strat in STRATS:
        s = q[q.strategy == strat][cols].median()
        body.append(rf"\texttt{{{strat}}} & "
                    + " & ".join(rf"\textbf{{{s[c]:.3f}}}" if s[c] < 1 else f"{s[c]:.3f}"
                                 for c in cols) + r" \\")
    allq = q[cols].median()
    body += [r"\midrule", r"\textbf{All strategies} & "
             + " & ".join(rf"\textbf{{{allq[c]:.3f}}}" if allq[c] < 1 else f"{allq[c]:.3f}"
                          for c in cols) + r" \\"]
    for c in cols:
        macro(f"rgQ{c}", f"{allq[c]:.3f}")
    macro("rgQCells", len(q))
    return table_local(
        caption=(r"\textbf{Which rare cases improve.} Quantiles of the absolute error inside "
                 r"the rare region, strategy divided by its baseline, median over the "
                 rf"{len(q)} (dataset $\times$ model family $\times$ strategy) comparisons. "
                 r"\textbf{A ratio below $1$ means the strategy is better}, and bold marks "
                 r"those cells. The ratio rises monotonically with the quantile and crosses $1$ "
                 r"between the 75th and 90th percentile: resampling shrinks the smaller "
                 r"rare-case errors and enlarges the largest ones. This is the mechanism behind "
                 r"the gap between the MAE and RMSE columns of "
                 r"Table~\ref{tab:exp002_region_effect}. Errors are pooled over a cell's folds, "
                 r"so the table describes the error distribution rather than anything the "
                 r"fold-level tests rank, and the two right-hand columns rest on few points per "
                 r"cell."),
        label="tab:exp002_region_quantiles", col_spec="l" + "r" * len(cols),
        header=(r"\textbf{Strategy} & \textbf{25th} & \textbf{50th} & \textbf{75th} & "
                r"\textbf{90th} & \textbf{99th} & \textbf{Max}"),
        body="\n".join(body), resize=None)


def tab_diag(diag: pd.DataFrame) -> str:
    body = [(f"{r.dataset} & {r.n_test_per_fold} & {r.pct_rare:.2f} & {r.min_rare} & "
             f"{r.max_rare} & {r.folds_empty_rare} " + r"\\") for r in diag.itertuples()]
    return table_local(
        caption=(r"\textbf{How much rare region there is to score on.} Per dataset: test points "
                 r"per Monte Carlo fold, the share of them with $\phi(y) \ge 0.9$, the smallest "
                 r"and largest per-fold rare count, and the number of folds holding no rare case "
                 r"at all. Those folds are dropped from the dataset's paired vector, from both "
                 r"sides at once, so the pairing stays exact."),
        label="tab:exp002_region_diag", col_spec="lrrrrr",
        header=(r"\textbf{Dataset} & \textbf{Test points/fold} & \textbf{Rare (\%)} & "
                r"\textbf{Min rare} & \textbf{Max rare} & \textbf{Empty folds}"),
        body="\n".join(body), resize=None)


def tab_quadrature(gap: pd.DataFrame) -> str:
    bad = gap[gap.p95_rel > 0.05]
    macro("rgQuadTypical", f"{gap.median_rel.median()*100:.2f}")
    macro("rgQuadNBad", len(bad))
    macro("rgQuadBadList", ", ".join(bad.dataset))
    body = [(f"{r.dataset} & {r.median_rel*100:.2f} & {r.p95_rel*100:.1f} & {r.max_rel*100:.1f} "
             + r"\\") for r in gap.itertuples()]
    return table_local(
        caption=(r"\textbf{Diagnostic: the pipeline's SERA is a quadrature approximation of the "
                 r"value used here.} This analysis computes SERA as "
                 r"$\sum_i \phi(y_i) e_i^2$, the exact continuous-limit value, because only that "
                 r"form is additive over regions. \texttt{SERA Metric/sera\_metric.py} instead "
                 r"integrates $SER_t$ over 101 thresholds with the trapezoid rule. The columns "
                 r"give the relative gap between the two over every fold $\times$ workflow of "
                 r"each dataset. It stays a fraction of a percent on most datasets, but the "
                 r"trapezoid rule overshoots where a point of near-zero relevance carries a very "
                 r"large squared error, because the rule credits it with half a threshold step "
                 r"instead of $\phi \approx 0$. Nothing in this section uses the trapezoid form."),
        label="tab:exp002_region_quadrature", col_spec="lrrr",
        header=(r"\textbf{Dataset} & \textbf{Median gap (\%)} & \textbf{95th pct.\ (\%)} & "
                r"\textbf{Max (\%)}"),
        body="\n".join(body), resize=None)


# ----------------------------------------------------------------- figures ---

def fig_flip(wl_all: dict):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    x = np.arange(len(STRATS))
    for ax, (metric, lab) in zip(axes, METRICS.items()):
        for off, region, colour in ((-0.2, "rare", C_TEAL), (0.2, "normal", C_GREY)):
            sub = wl_all[metric][wl_all[metric].region == region].groupby("strategy")[
                ["win", "loss", "tie"]].sum()
            share = 100 * sub.win / sub.sum(axis=1)
            ax.bar(x + off, share.reindex(STRATS), width=0.4, color=colour,
                   label=f"{region} region")
        ax.axhline(50, color="black", lw=1, ls="--", alpha=0.6)
        ax.set_xticks(x); ax.set_xticklabels(STRATS, rotation=60, ha="right", fontsize=10)
        ax.set_title(lab); ax.set_ylim(0, 100)
    axes[0].set_ylabel("Strategy wins (\\% of comparisons)")
    axes[0].legend(loc="upper left", fontsize=11)
    save(fig, str(FIGURES / "fig_region_flip.png"))


def fig_bayes(b: pd.DataFrame, metric: str, region: str):
    g = b[b.region == region].groupby("strategy")[
        ["p_baseline", "p_rope", "p_strategy"]].mean().reindex(STRATS)
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.barh(y, g.p_strategy, color=C_TEAL, label="Strategy wins", edgecolor="white", linewidth=0.5)
    ax.barh(y, g.p_rope, left=g.p_strategy, color=C_GOLD, label="Draw (ROPE)",
            edgecolor="white", linewidth=0.5)
    ax.barh(y, g.p_baseline, left=g.p_strategy + g.p_rope, color=C_GREY,
            label="Baseline wins", edgecolor="white", linewidth=0.5)
    ax.set_yticks(y); ax.set_yticklabels(g.index); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xlabel("Posterior probability")
    ax.set_title(f"{METRICS[metric]}, {region} region")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    ax.grid(axis="y", visible=False)
    save(fig, str(FIGURES / f"fig_region_bayes_{metric}_{region}.png"))


# -------------------------------------------------------------------- main ---

def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA.glob("DS??.csv"))
    if not files:
        print("no region metrics on disk; run run_region_analysis.py first")
        return 1
    fm = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    diag = pd.read_csv(DATA / "diag.csv")
    quant = pd.read_csv(DATA / "quantiles.csv")

    wl_all, bayes_all, eff_all = {}, {}, {}
    for metric in METRICS:
        print(f"  {metric}: wilcoxon, effect sizes, bayes", flush=True)
        wl_all[metric] = wl_table(fm, metric)
        eff_all[metric] = effect_cells(fm, metric)
        bayes_all[metric] = bayes_table(fm, metric)
        for region in ORDER:
            (TABLES / f"wl_{metric}_{region}.tex").write_text(tab_wl(wl_all[metric], metric, region))
            fig_bayes(bayes_all[metric], metric, region)

    (TABLES / "flip.tex").write_text(tab_flip(wl_all))
    (TABLES / "effect.tex").write_text(tab_effect(eff_all))
    (TABLES / "bayes.tex").write_text(tab_bayes(bayes_all))
    (TABLES / "quantiles.tex").write_text(tab_quantiles(quant))
    (TABLES / "diag.tex").write_text(tab_diag(diag))
    gap_csv = DATA / "sera_quadrature_gap.csv"
    if gap_csv.exists():
        (TABLES / "quadrature.tex").write_text(tab_quadrature(pd.read_csv(gap_csv)))
    fig_flip(wl_all)

    for metric in METRICS:
        for region in ORDER:
            w = wl_all[metric].query("region == @region")
            macro(f"rgWinPct{metric}{region}", f"{100 * w.win.sum() / w[['win','loss','tie']].sum().sum():.1f}")
            per_s = w.groupby("strategy").win.sum()
            macro(f"rgBest{metric}{region}", per_s.idxmax())
            macro(f"rgBest{metric}{region}N", int(per_s.max()))
            b = bayes_all[metric].query("region == @region")
            macro(f"rgBayes{metric}{region}Mean", f"{b.p_strategy.mean():.2f}")
    r = wl_all["mae"].query("region == 'rare'").groupby("strategy").win.sum()
    for name, group in (("Under", ["UNDERB", "UNDERT", "UNDERTPhi"]),
                        ("Over", ["OVERB", "OVERT", "OVERTPhi"]),
                        ("Smote", ["SMOTEB", "SMOTET", "SMOTETPhi"])):
        macro(f"rgMaeRare{name}", int(r[group].sum()))
    macro("rgNDatasets", diag.dataset.nunique())
    macro("rgRopePct", f"{100*(np.exp(ROPE)-1):.0f}")
    macro("rgEmptyFolds", int(diag.folds_empty_rare.sum()))
    macro("rgEmptyFoldDs", ", ".join(diag[diag.folds_empty_rare > 0].dataset))
    macro("rgMedPctRare", f"{diag.pct_rare.median():.1f}")
    (TABLES / "region_numbers.tex").write_text(
        "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(_MACROS.items())) + "\n")
    print(f"wrote {len(_MACROS)} macros, {len(list(TABLES.glob('*.tex')))} tables, "
          f"{len(list(FIGURES.glob('*.pdf')))} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
