#!/usr/bin/env python3
"""Paper-fidelity audit: for every table in Moniz, Branco & Torgo (2017)
that publishes a checkable number, compute this project's own equivalent
from Results Data/raw_iterations_by_dataset_v2/*.csv and set it beside the
paper's published value.

This is a comparison against the CURRENT run (pre- optimal-parameter-fix,
i.e. still using the #EXAMPLE PARAMETRIZATION placeholder in
src/adapted/Exps.R at the time this was generated -- the fix has been
landed in Exps.R but the cluster rerun that would produce corrected data
has not happened yet). Re-running this generator after that rerun is what
will show whether the parameter fix closed the gap; see CLAUDE.md
2026-09-03 and CHANGES.md Section 10.

Output: one .tex + one .txt per comparison, in
        Results (Clean)/Evaluation Metrics/Latex Tables/Paper Comparison/
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tables_common import FAMILY_LABELS, LATEX_DIR

from paired_comparisons import ALPHA, DATA_DIR, dataset_ids, wl_vs_reference, _load_all

# Reuse the causal module's dataset->source clustering rather than restating it
# -- it is the single place that mapping is maintained and tested.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Causal"))
from causal_effects import SOURCE_MAP
from paper_reference import (
    FAMILIES, PCT_RARE, STRATEGY_CODE_MAP,
    TABLE3, TABLE4, TABLE4_BASE, TABLE5, TABLE6_SVM_F1, TOTAL_DATASETS_PAPER,
)

OUT_DIR = LATEX_DIR / "Paper Comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# Counted from whatever raw per-dataset CSVs are actually present, not
# hardcoded -- this run may have anywhere from 20 to 24 datasets depending
# on how far the Apuana rerun has gotten (see CLAUDE.md 2026-09-04).
N_OURS = len(list(DATA_DIR.glob("DS??_*.csv")))


def _fmt_cell(ours, paper, n_ours=N_OURS, n_paper=TOTAL_DATASETS_PAPER):
    ow, osw, ol, osl, _ = ours
    pw, psw, pl, psl = paper
    return (f"{ow}({osw})/{ol}({osl}) [{100*ow/n_ours:.0f}%W]  vs.  "
            f"paper {pw}({psw})/{pl}({psl}) [{100*pw/n_paper:.0f}%W]")


def _latex_escape(s: str) -> str:
    for ch in "&%#_":
        s = s.replace(ch, "\\" + ch)
    return s


# Report text width (letterpaper, 1in margins) and the measured advance width of
# one `newtxtt` character as a fraction of the font size -- calibrated against a
# real overfull-hbox report from the report build, not guessed.  The size ladder
# is largest-first; the first size whose widest line still fits is used, so a
# 153-column dump lands on \tiny and a 90-column one stays readable.
_TEXTWIDTH_PT = 469.0
_TT_EM_FRACTION = 0.60
_SIZE_LADDER = [(10.0, r"\small"), (9.0, r"\footnotesize"),
                (8.0, r"\scriptsize"), (5.0, r"\tiny")]


def _fitting_size(lines: list[str]) -> str:
    widest = max((len(ln) for ln in lines), default=0)
    for pt, cmd in _SIZE_LADDER:
        if widest * pt * _TT_EM_FRACTION <= _TEXTWIDTH_PT:
            return cmd
    return _SIZE_LADDER[-1][1]


def _write(name: str, title: str, lines: list[str]):
    txt = "\n".join(lines) + "\n"
    (OUT_DIR / f"{name}.txt").write_text(txt)

    # NOT a float: these bodies run to 150 verbatim lines, which no float can
    # hold.  `\captionof` (caption package) still gives a numbered, referencable
    # caption, and the verbatim body is then free to break across pages.
    tex = (
        r"\begingroup" "\n"
        rf"\captionof{{table}}{{{_latex_escape(title)}}}" "\n"
        rf"\label{{tab:exp002_cmp_{name}}}" "\n"
        f"{_fitting_size(lines)}\n"
        r"\begin{verbatim}" "\n" + "\n".join(lines) + "\n" r"\end{verbatim}" "\n"
        r"\endgroup" "\n"
    )
    (OUT_DIR / f"{name}.tex").write_text(tex)
    print(f"Wrote {OUT_DIR / (name + '.tex')} and .txt")


# ---------------------------------------------------------------------------
# cmp_table3: resampling vs. baseline, per family
# ---------------------------------------------------------------------------
def gen_table3():
    lines = [f"Table 3 equivalent -- resampling vs. baseline, ours ({N_OURS} datasets) vs. paper ({TOTAL_DATASETS_PAPER})  [alpha={ALPHA}]", ""]
    for code, suffix in STRATEGY_CODE_MAP.items():
        if code not in TABLE3:
            continue
        lines.append(f"{code}:")
        for fam in FAMILIES:
            ours = wl_vs_reference(fam, suffix, None)
            paper = TABLE3[code][fam]
            lines.append(f"  {FAMILY_LABELS[fam]:24s} {_fmt_cell(ours, paper)}")
        lines.append("")
    _write("cmp_table3", "Table 3 replication check: resampling vs. baseline", lines)


# ---------------------------------------------------------------------------
# cmp_table4: biased variants vs. their biased base
# ---------------------------------------------------------------------------
def gen_table4():
    lines = [f"Table 4 equivalent -- biased variant vs. its own biased base, ours vs. paper  [alpha={ALPHA}]", ""]
    for variant, base_code in TABLE4_BASE.items():
        variant_suffix = STRATEGY_CODE_MAP[variant]
        base_suffix = STRATEGY_CODE_MAP[base_code]
        lines.append(f"{variant} vs. {base_code}:")
        for fam in FAMILIES:
            ours = wl_vs_reference(fam, variant_suffix, base_suffix)
            paper = TABLE4[variant][fam]
            lines.append(f"  {FAMILY_LABELS[fam]:24s} {_fmt_cell(ours, paper)}")
        lines.append("")
    _write("cmp_table4", "Table 4 replication check: biased variants vs. biased base", lines)


# ---------------------------------------------------------------------------
# cmp_table5: every strategy vs. ARIMA and BDES
# ---------------------------------------------------------------------------
def gen_table5():
    lines = [f"Table 5 equivalent -- every strategy vs. ARIMA / BDES, ours vs. paper  [alpha={ALPHA}]",
             "Never checked before this audit.", ""]
    for fam in FAMILIES:
        lines.append(f"{FAMILY_LABELS[fam]}:")
        for code, suffix in STRATEGY_CODE_MAP.items():
            if code not in TABLE5[fam]:
                continue
            # ARIMA/BDES are separate top-level workflows (mc.arima, mc.BDES),
            # not mc.<family> -- compute directly against each.
            for ref_name, ref_wf_full in (("ARIMA", "arima"), ("BDES", "BDES")):
                ours = _wl_vs_fixed_workflow(fam, suffix, ref_wf_full)
                paper = TABLE5[fam][code][ref_name]
                lines.append(f"  {code:10s} vs {ref_name:6s} {_fmt_cell(ours, paper)}")
        lines.append("")
    _write("cmp_table5", "Table 5 replication check: every strategy vs. ARIMA/BDES", lines)


def _wl_vs_fixed_workflow(family: str, target_suffix: str, ref_workflow_family: str):
    """ARIMA/BDES are their own top-level workflows (mc.arima, mc.BDES), not
    mc.<family>_ARIMA -- same for every model family, so compare each
    family's resampled workflow against the single shared mc.arima/mc.BDES
    reference rather than a per-family one."""
    from paired_comparisons import compute_wl
    df = _load_all()
    target_wf = f"mc.{family}_{target_suffix}"
    ref_wf = f"mc.{ref_workflow_family}"
    win = sigwin = loss = sigloss = tie = 0
    for ds_id, ds_df in df.groupby("dataset_id"):
        ref_vec = ds_df[ds_df.workflow == ref_wf].sort_values("iteration")["F1"].to_numpy()
        tgt_vec = ds_df[ds_df.workflow == target_wf].sort_values("iteration")["F1"].to_numpy()
        n = min(len(ref_vec), len(tgt_vec))
        if n == 0:
            continue
        w, sw, l, sl, t = compute_wl(ref_vec[:n], tgt_vec[:n])
        win += w; sigwin += sw; loss += l; sigloss += sl; tie += t
    return (win, sigwin, loss, sigloss, tie)


# ---------------------------------------------------------------------------
# cmp_table6: absolute SVM F1 on DS4/DS10/DS12
# ---------------------------------------------------------------------------
def gen_table6():
    lines = [
        "Table 6 equivalent -- absolute SVM F1_phi, DS4/DS10/DS12, ours vs. paper.",
        "CAVEAT: paper's Table 6 jointly optimizes cost/gamma AND resampling",
        "percentages per dataset, using 10 MC reps (not 50) -- anchors magnitude,",
        "not an exact target. Our numbers use the single per-dataset Table 7",
        "cost/gamma (not further tuned) and default balanced resampling, 50 reps.",
        "",
    ]
    df = _load_all()
    for code, values in TABLE6_SVM_F1.items():
        wf = "mc.svm" if code == "svm" else f"mc.svm_{STRATEGY_CODE_MAP[code]}"
        row = [f"  {code:10s}"]
        for ds in ("DS04", "DS10", "DS12"):
            sub = df[(df.dataset_id == ds) & (df.workflow == wf)]
            ours_mean = sub["F1"].mean() if len(sub) else float("nan")
            paper_val = values[ds]
            row.append(f"{ds}: ours={ours_mean:.3f} paper={paper_val:.3f}")
        lines.append("  ".join(row))
    _write("cmp_table6", "Table 6 replication check: absolute SVM F1 on DS4/DS10/DS12", lines)


# ---------------------------------------------------------------------------
# cmp_precision_recall: paper's "gains come mostly from precision" claim
# ---------------------------------------------------------------------------
def gen_precision_recall():
    lines = [
        "Precision/recall split -- paper claims F1 gains come mostly from",
        "higher precision, not recall (Sec 5.1). Delta = resampled mean - baseline mean,",
        f"averaged over all 9 strategies and {N_OURS} datasets, per family.",
        "",
        "Also flagged: baseline recall stays ~0.4-0.6 even in datasets where",
        "baseline precision is exactly 0.00 -- worth inspecting independently",
        "of the hyperparameter-placeholder finding (Eq. 2's recall denominator",
        "does not obviously predict this).",
        "",
    ]
    df = _load_all()
    for fam in FAMILIES:
        fam_df = df[df.workflow.str.startswith(f"mc.{fam}")]
        base = fam_df[fam_df.workflow == f"mc.{fam}"]
        base_prec, base_rec = base.prec.mean(), base.rec.mean()
        resampled = fam_df[fam_df.workflow != f"mc.{fam}"]
        d_prec = resampled.prec.mean() - base_prec
        d_rec = resampled.rec.mean() - base_rec
        zero_prec_ds = base.groupby("dataset_id").prec.mean()
        n_zero = int((zero_prec_ds < 0.01).sum())
        lines.append(
            f"  {FAMILY_LABELS[fam]:24s} baseline prec={base_prec:.3f} rec={base_rec:.3f} | "
            f"mean resampled delta: prec={d_prec:+.3f} rec={d_rec:+.3f} | "
            f"baseline prec~0 in {n_zero}/{len(zero_prec_ds)} datasets"
        )
    _write("cmp_precision_recall", "Precision/recall delta check vs. paper Sec 5.1 claim", lines)


# ---------------------------------------------------------------------------
# cmp_pct_rare: Table 1 %Rare regression guard
# ---------------------------------------------------------------------------
def gen_pct_rare():
    lines = ["%Rare (Table 1) regression guard -- see SERA Metric/test_sera_metric.py",
             "for the primary phi-fix regression test; this repeats the same check",
             "from the paper-comparison audit's own data path.", ""]
    lines.append(f"{'Dataset':8s} {'paper %Rare':>12s}")
    for ds_id in sorted(PCT_RARE):
        lines.append(f"  {ds_id:6s} {PCT_RARE[ds_id]:>10.1f}%")
    lines.append("")
    lines.append("(Computed phi %Rare vs. these values: see test_sera_metric.py's")
    lines.append("test_phi_matches_paper_pct_rare, MAE 0.99pp post-fix -- not recomputed")
    lines.append("here to avoid a second phi implementation; this file is the reference")
    lines.append("values only.)")
    _write("cmp_pct_rare", "Table 1 %Rare reference values (see SERA Metric for the check)", lines)


# ---------------------------------------------------------------------------
# cmp_sensitivity: leave-one-dataset-out / leave-one-source-out diagnostic
# ---------------------------------------------------------------------------
# The three strategies Table 3 compares directly against the baseline; these
# are the rows where our SVM result still disagrees with the paper.
_BASELINE_STRATEGIES = ("U_B", "O_B", "SM_B")


def _win_pct(family: str, suffix: str, exclude: frozenset[str] | None) -> float:
    """Win share over the datasets actually compared, so holding one out
    rescales the denominator instead of silently deflating the percentage."""
    win, _, loss, _, tie = wl_vs_reference(family, suffix, None, exclude=exclude)
    n = win + loss + tie
    return 100.0 * win / n if n else float("nan")


def gen_sensitivity():
    ids = dataset_ids()
    lines = [
        f"Leave-one-out sensitivity of the Table 3 baseline comparison  [alpha={ALPHA}, N={len(ids)}]",
        "",
        "DIAGNOSTIC ONLY -- this is NOT a jackknife and implies no standard error.",
        "The datasets are source-clustered (7 sources over 24 datasets: e.g. DS14-20",
        "are seven indices from one exchange), so leave-one-DATASET-out values are not",
        "independent and their spread must not be read as a variance estimate. Dropping",
        "one dataset while its siblings remain damps the apparent effect, so 'little",
        "movement' is weak evidence. The leave-one-SOURCE-out block below is the clean",
        "cut; the causal module's jackknife (Eval_Metrics Code/Causal) uses that unit",
        "for exactly this reason.",
        "",
        "Question this answers: is our residual SVM disagreement with the paper spread",
        "across datasets (systemic -- something is still wrong in the pipeline), or",
        "carried by one or two (local -- a dataset-specific discrepancy)?",
        "",
    ]

    for fam in FAMILIES:
        lines.append(f"{FAMILY_LABELS[fam]}:")
        for code in _BASELINE_STRATEGIES:
            suffix = STRATEGY_CODE_MAP[code]
            full = _win_pct(fam, suffix, None)
            loo = {ds: _win_pct(fam, suffix, frozenset({ds})) for ds in ids}
            worst = max(loo, key=lambda d: abs(loo[d] - full))
            paper_w, _, _, _ = TABLE3[code][fam]
            lines.append(
                f"  {code:7s} full={full:5.1f}%W  (paper {100*paper_w/TOTAL_DATASETS_PAPER:5.1f}%W)  "
                f"LOO range [{min(loo.values()):5.1f}, {max(loo.values()):5.1f}]  "
                f"most influential: {worst} -> {loo[worst]:5.1f}%W ({loo[worst]-full:+.1f}pp)"
            )
        lines.append("")

    lines += ["", "Leave-one-SOURCE-out (the defensible cut -- whole correlated group removed):", ""]
    sources = sorted({SOURCE_MAP[ds] for ds in ids})
    for fam in FAMILIES:
        lines.append(f"{FAMILY_LABELS[fam]}:")
        for code in _BASELINE_STRATEGIES:
            suffix = STRATEGY_CODE_MAP[code]
            full = _win_pct(fam, suffix, None)
            cells = []
            for src in sources:
                drop = frozenset(d for d in ids if SOURCE_MAP[d] == src)
                cells.append(f"{src}={_win_pct(fam, suffix, drop):.0f}")
            lines.append(f"  {code:7s} full={full:5.1f}%W  " + "  ".join(cells))
        lines.append("")

    lines += ["", "Per-dataset detail, SVM only (the family whose Table 3 row still disagrees):", ""]
    for code in _BASELINE_STRATEGIES:
        suffix = STRATEGY_CODE_MAP[code]
        full = _win_pct("svm", suffix, None)
        lines.append(f"  {code} (full {full:.1f}%W):")
        for ds in ids:
            w = _win_pct("svm", suffix, frozenset({ds}))
            lines.append(f"    drop {ds}: {w:5.1f}%W ({w-full:+.1f}pp)  [{SOURCE_MAP[ds]}]")
        lines.append("")

    _write("cmp_sensitivity", "Leave-one-out sensitivity of the Table 3 baseline comparison", lines)


if __name__ == "__main__":
    gen_table3()
    gen_table4()
    gen_table5()
    gen_table6()
    gen_precision_recall()
    gen_pct_rare()
    gen_sensitivity()
    print("Done.")
