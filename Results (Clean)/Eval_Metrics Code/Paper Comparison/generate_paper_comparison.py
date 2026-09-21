#!/usr/bin/env python3
"""Paper-fidelity audit: for every table in Moniz, Branco & Torgo (2017)
that publishes a checkable number, compute this project's own equivalent
from Results Data/raw_iterations_by_dataset_v2/*.csv and set it beside the
paper's published value.

Output: one .tex (booktabs table, matching the F1/RMSE/SERA table style)
and one .txt (plain-text dump, unchanged) per comparison, in
        Results (Clean)/Evaluation Metrics/Latex Tables/Paper Comparison/
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tables_common import FAMILY_LABELS, LATEX_DIR, table_star

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


def _wl_cell(wl, n) -> str:
    """One Win/Loss/sig cell: 'W(sigW)/L(sigL) [P%W]'. Shared by every
    table below so ours/paper cells render identically side by side."""
    w, sw, l, sl, _ = wl
    return rf"{w}({sw})/{l}({sl}) [{100*w/n:.0f}\%W]" if n else "--"


def _tex(code: str) -> str:
    """Strategy codes (U_B, O_TPhi, ...) contain a literal underscore --
    escape it for LaTeX text mode (\\texttt{U_B} is a compile error)."""
    return code.replace("_", r"\_")


def _txt_cell(ours, paper, n_ours=N_OURS, n_paper=TOTAL_DATASETS_PAPER):
    ow, osw, ol, osl, _ = ours
    pw, psw, pl, psl = paper
    return (f"{ow}({osw})/{ol}({osl}) [{100*ow/n_ours:.0f}%W]  vs.  "
            f"paper {pw}({psw})/{pl}({psl}) [{100*pw/n_paper:.0f}%W]")


def _write_txt(name: str, lines: list[str]):
    (OUT_DIR / f"{name}.txt").write_text("\n".join(lines) + "\n")


def _write_tex(name: str, tex: str):
    (OUT_DIR / f"{name}.tex").write_text(tex)
    print(f"Wrote {OUT_DIR / (name + '.tex')} and .txt")


# ---------------------------------------------------------------------------
# cmp_table3: resampling vs. baseline, per family
# ---------------------------------------------------------------------------
def gen_table3():
    lines = [f"Table 3 equivalent -- resampling vs. baseline, ours ({N_OURS} datasets) vs. paper ({TOTAL_DATASETS_PAPER})  [alpha={ALPHA}]", ""]
    codes = [c for c in STRATEGY_CODE_MAP if c in TABLE3]
    cells = {}  # (code, fam) -> (ours_wl, paper_wl)
    for code in codes:
        suffix = STRATEGY_CODE_MAP[code]
        lines.append(f"{code}:")
        for fam in FAMILIES:
            ours = wl_vs_reference(fam, suffix, None)
            paper = TABLE3[code][fam]
            cells[(code, fam)] = (ours, paper)
            lines.append(f"  {FAMILY_LABELS[fam]:24s} {_txt_cell(ours, paper)}")
        lines.append("")
    _write_txt("cmp_table3", lines)

    header = (r"\textbf{Family} & " + " & ".join(
        rf"\multicolumn{{2}}{{c}}{{\textbf{{{_tex(code)}}}}}" for code in codes))
    subheader = " & " + " & ".join(r"Ours & Paper" for _ in codes)
    rows = []
    for fam in FAMILIES:
        cols = []
        for code in codes:
            ours, paper = cells[(code, fam)]
            pw, psw, pl, psl = paper
            cols.append(_wl_cell(ours, N_OURS))
            cols.append(_wl_cell((pw, psw, pl, psl, 0), TOTAL_DATASETS_PAPER))
        rows.append(f"{FAMILY_LABELS[fam]} & " + " & ".join(cols) + r" \\")
    body = subheader + r" \\" + "\n" + "\n".join(rows)
    tex = table_star(
        caption=r"\textbf{Table 3 replication check --- resampling vs.\ baseline}, "
                rf"Win(sigWin)/Loss(sigLoss) [\%W] per family, ours ({N_OURS} datasets) "
                rf"vs.\ the paper ({TOTAL_DATASETS_PAPER}), $\alpha={ALPHA}$.",
        label="tab:exp002_cmp_cmp_table3",
        col_spec="l" + "cc" * len(codes),
        header=header,
        body=body,
    )
    _write_tex("cmp_table3", tex)


# ---------------------------------------------------------------------------
# cmp_table4: biased variants vs. their biased base
# ---------------------------------------------------------------------------
def gen_table4():
    lines = [f"Table 4 equivalent -- biased variant vs. its own biased base, ours vs. paper  [alpha={ALPHA}]", ""]
    cells = {}  # (variant, fam) -> (ours_wl, paper_wl)
    for variant, base_code in TABLE4_BASE.items():
        variant_suffix = STRATEGY_CODE_MAP[variant]
        base_suffix = STRATEGY_CODE_MAP[base_code]
        lines.append(f"{variant} vs. {base_code}:")
        for fam in FAMILIES:
            ours = wl_vs_reference(fam, variant_suffix, base_suffix)
            paper = TABLE4[variant][fam]
            cells[(variant, fam)] = (ours, paper)
            lines.append(f"  {FAMILY_LABELS[fam]:24s} {_txt_cell(ours, paper)}")
        lines.append("")
    _write_txt("cmp_table4", lines)

    # One table per resampling family (Under/Over/SMOTE), each with its two
    # biased variants (T, TPhi) side by side -- keeps every table the same
    # width as Table 3's rather than one 13-column table.
    groups = [("Undersampling", ["U_T", "U_TPhi"], "U_B"),
              ("Oversampling", ["O_T", "O_TPhi"], "O_B"),
              ("SMOTE", ["SM_T", "SM_TPhi"], "SM_B")]
    group_tex = []
    for group_name, variants, base_code in groups:
        header = (r"\textbf{Family} & " + " & ".join(
            rf"\multicolumn{{2}}{{c}}{{\textbf{{{_tex(v)}}} vs.\ {_tex(base_code)}}}" for v in variants))
        subheader = " & " + " & ".join(r"Ours & Paper" for _ in variants)
        rows = []
        for fam in FAMILIES:
            cols = []
            for v in variants:
                ours, paper = cells[(v, fam)]
                pw, psw, pl, psl = paper
                cols.append(_wl_cell(ours, N_OURS))
                cols.append(_wl_cell((pw, psw, pl, psl, 0), TOTAL_DATASETS_PAPER))
            rows.append(f"{FAMILY_LABELS[fam]} & " + " & ".join(cols) + r" \\")
        body = subheader + r" \\" + "\n" + "\n".join(rows)
        tex = table_star(
            caption=rf"\textbf{{Table 4 replication check --- {group_name}}}, "
                    r"biased variant vs.\ its own biased base, Win(sigWin)/Loss(sigLoss) "
                    rf"[\%W], ours vs.\ paper, $\alpha={ALPHA}$.",
            label=f"tab:exp002_cmp_cmp_table4_{group_name.lower()}",
            col_spec="l" + "cc" * len(variants),
            header=header,
            body=body,
            resize=None,
        )
        if group_name == groups[0][0]:
            # Backward-compat: the report's prose \ref's the whole Table 4
            # group under one bare label -- carry it on the first sub-table.
            tex = tex.replace(
                f"\\label{{tab:exp002_cmp_cmp_table4_{group_name.lower()}}}",
                f"\\label{{tab:exp002_cmp_cmp_table4_{group_name.lower()}}}\n"
                r"\label{tab:exp002_cmp_cmp_table4}",
            )
        group_tex.append(tex)

    # A single combined .tex (all three groups stacked) keeps one \input site
    # working for callers that expect one file, matching the Wins_Loss
    # f1_sera_table_best_counts.tex convention of stacked table* blocks.
    combined = "\n\n".join(group_tex)
    _write_tex("cmp_table4", combined)


# ---------------------------------------------------------------------------
# cmp_table5: every strategy vs. ARIMA and BDES
# ---------------------------------------------------------------------------
def gen_table5():
    lines = [f"Table 5 equivalent -- every strategy vs. ARIMA / BDES, ours vs. paper  [alpha={ALPHA}]",
             "Never checked before this audit.", ""]
    cells = {}  # (fam, code, ref) -> (ours_wl, paper_wl)
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
                cells[(fam, code, ref_name)] = (ours, paper)
                lines.append(f"  {code:10s} vs {ref_name:6s} {_txt_cell(ours, paper)}")
        lines.append("")
    _write_txt("cmp_table5", lines)

    codes = [c for c in STRATEGY_CODE_MAP if c in TABLE5["lm"]]
    header = r"\textbf{Strategy} & \multicolumn{2}{c}{\textbf{vs.\ ARIMA}} & \multicolumn{2}{c}{\textbf{vs.\ BDES}}"
    subheader = r" & Ours & Paper & Ours & Paper"
    per_family_tex = []
    for fam in FAMILIES:
        rows = []
        for code in codes:
            cols = []
            for ref in ("ARIMA", "BDES"):
                ours, paper = cells[(fam, code, ref)]
                pw, psw, pl, psl = paper
                cols.append(_wl_cell(ours, N_OURS))
                cols.append(_wl_cell((pw, psw, pl, psl, 0), TOTAL_DATASETS_PAPER))
            rows.append(rf"\texttt{{{_tex(code)}}} & " + " & ".join(cols) + r" \\")
        body = subheader + r" \\" + "\n" + "\n".join(rows)
        tex = table_star(
            caption=rf"\textbf{{Table 5 replication check --- {FAMILY_LABELS[fam]}}}, "
                    r"every strategy vs.\ \texttt{ARIMA}/\texttt{BDES}, "
                    rf"Win(sigWin)/Loss(sigLoss) [\%W], ours vs.\ paper, $\alpha={ALPHA}$.",
            label=f"tab:exp002_cmp_cmp_table5_{fam}",
            col_spec="lcccc",
            header=header,
            body=body,
            resize=None,
        )
        if fam == FAMILIES[0]:
            # Backward-compat: the report's prose \ref's the whole Table 5
            # group under one bare label -- carry it on the first sub-table.
            tex = tex.replace(
                f"\\label{{tab:exp002_cmp_cmp_table5_{fam}}}",
                f"\\label{{tab:exp002_cmp_cmp_table5_{fam}}}\n"
                r"\label{tab:exp002_cmp_cmp_table5}",
            )
        per_family_tex.append(tex)
    _write_tex("cmp_table5", "\n\n".join(per_family_tex))


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
    row_vals = {}  # code -> {ds: (ours, paper)}
    for code, values in TABLE6_SVM_F1.items():
        wf = "mc.svm" if code == "svm" else f"mc.svm_{STRATEGY_CODE_MAP[code]}"
        row = [f"  {code:10s}"]
        row_vals[code] = {}
        for ds in ("DS04", "DS10", "DS12"):
            sub = df[(df.dataset_id == ds) & (df.workflow == wf)]
            ours_mean = sub["F1"].mean() if len(sub) else float("nan")
            paper_val = values[ds]
            row_vals[code][ds] = (ours_mean, paper_val)
            row.append(f"{ds}: ours={ours_mean:.3f} paper={paper_val:.3f}")
        lines.append("  ".join(row))
    _write_txt("cmp_table6", lines)

    header = (r"\textbf{Strategy} & \multicolumn{2}{c}{\textbf{DS04}} & "
              r"\multicolumn{2}{c}{\textbf{DS10}} & \multicolumn{2}{c}{\textbf{DS12}}")
    subheader = r" & Ours & Paper & Ours & Paper & Ours & Paper"
    rows = []
    for code, vals in row_vals.items():
        cols = []
        for ds in ("DS04", "DS10", "DS12"):
            ours_mean, paper_val = vals[ds]
            cols.append(f"{ours_mean:.3f}")
            cols.append(f"{paper_val:.3f}")
        rows.append(rf"\texttt{{{_tex(code)}}} & " + " & ".join(cols) + r" \\")
    body = subheader + r" \\" + "\n" + "\n".join(rows)
    tex = table_star(
        caption=r"\textbf{Table 6 replication check --- absolute SVM $F_1^\phi$} on "
                r"DS04/DS10/DS12, ours vs.\ paper (paper jointly optimizes "
                r"hyperparameters \emph{and} resampling percentages over 10 MC reps; "
                r"anchors magnitude, not an exact target).",
        label="tab:exp002_cmp_cmp_table6",
        col_spec="lcccccc",
        header=header,
        body=body,
        resize=None,
    )
    _write_tex("cmp_table6", tex)


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
    rows = []
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
        rows.append((FAMILY_LABELS[fam], base_prec, base_rec, d_prec, d_rec, n_zero, len(zero_prec_ds)))
    _write_txt("cmp_precision_recall", lines)

    header = (r"\textbf{Family} & \textbf{Baseline Prec} & \textbf{Baseline Rec} & "
              r"\textbf{$\Delta$Prec} & \textbf{$\Delta$Rec} & \textbf{Baseline $\approx$0 (of 24)}")
    body_rows = []
    for fam_label, base_prec, base_rec, d_prec, d_rec, n_zero, n_total in rows:
        body_rows.append(
            rf"{fam_label} & {base_prec:.3f} & {base_rec:.3f} & "
            rf"{d_prec:+.3f} & {d_rec:+.3f} & {n_zero}/{n_total} \\"
        )
    tex = table_star(
        caption=r"\textbf{Precision/recall delta vs.\ baseline, by family} --- checks "
                r"the paper's claim (Sec.\ 5.1) that $\Fphi$ gains come chiefly from "
                r"precision. Delta = resampled mean $-$ baseline mean, averaged over "
                rf"all 9 strategies and {N_OURS} datasets.",
        label="tab:exp002_cmp_cmp_precision_recall",
        col_spec="lccccc",
        header=header,
        body="\n".join(body_rows),
        resize=None,
    )
    _write_tex("cmp_precision_recall", tex)


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
    _write_txt("cmp_pct_rare", lines)

    # Two 12-row columns side by side (Dataset | %Rare, twice) rather than one
    # long 24-row column -- same space-saving convention as the datasets table.
    ids = sorted(PCT_RARE)
    left, right = ids[:12], ids[12:]
    header = r"\textbf{Dataset} & \textbf{Paper \%Rare} & \textbf{Dataset} & \textbf{Paper \%Rare}"
    rows = []
    for l, r in zip(left, right):
        rows.append(rf"{l} & {PCT_RARE[l]:.1f}\% & {r} & {PCT_RARE[r]:.1f}\% \\")
    tex = table_star(
        caption=r"\textbf{Table 1 \%Rare reference values} (paper) --- regression guard "
                r"for the relevance function $\phi$; see SERA Metric's own test for the "
                r"reconstructed-$\phi$ comparison (MAE 0.99pp).",
        label="tab:exp002_cmp_cmp_pct_rare",
        col_spec="lclc",
        header=header,
        body="\n".join(rows),
        resize=None,
    )
    _write_tex("cmp_pct_rare", tex)


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

    loo_rows = []  # (family, code, full, paper_pct, lo, hi, worst_ds, worst_val)
    for fam in FAMILIES:
        lines.append(f"{FAMILY_LABELS[fam]}:")
        for code in _BASELINE_STRATEGIES:
            suffix = STRATEGY_CODE_MAP[code]
            full = _win_pct(fam, suffix, None)
            loo = {ds: _win_pct(fam, suffix, frozenset({ds})) for ds in ids}
            worst = max(loo, key=lambda d: abs(loo[d] - full))
            paper_w, _, _, _ = TABLE3[code][fam]
            paper_pct = 100 * paper_w / TOTAL_DATASETS_PAPER
            lines.append(
                f"  {code:7s} full={full:5.1f}%W  (paper {paper_pct:5.1f}%W)  "
                f"LOO range [{min(loo.values()):5.1f}, {max(loo.values()):5.1f}]  "
                f"most influential: {worst} -> {loo[worst]:5.1f}%W ({loo[worst]-full:+.1f}pp)"
            )
            loo_rows.append((FAMILY_LABELS[fam], code, full, paper_pct,
                              min(loo.values()), max(loo.values()), worst, loo[worst] - full))
        lines.append("")

    lines += ["", "Leave-one-SOURCE-out (the defensible cut -- whole correlated group removed):", ""]
    sources = sorted({SOURCE_MAP[ds] for ds in ids})
    source_rows = []  # (family, code, full, {source: pct})
    for fam in FAMILIES:
        lines.append(f"{FAMILY_LABELS[fam]}:")
        for code in _BASELINE_STRATEGIES:
            suffix = STRATEGY_CODE_MAP[code]
            full = _win_pct(fam, suffix, None)
            cells = []
            src_vals = {}
            for src in sources:
                drop = frozenset(d for d in ids if SOURCE_MAP[d] == src)
                v = _win_pct(fam, suffix, drop)
                src_vals[src] = v
                cells.append(f"{src}={v:.0f}")
            lines.append(f"  {code:7s} full={full:5.1f}%W  " + "  ".join(cells))
            source_rows.append((FAMILY_LABELS[fam], code, full, src_vals))
        lines.append("")

    lines += ["", "Per-dataset detail, SVM only (the family whose Table 3 row still disagrees):", ""]
    svm_full = {code: _win_pct("svm", STRATEGY_CODE_MAP[code], None) for code in _BASELINE_STRATEGIES}
    svm_detail = {ds: {} for ds in ids}  # ds -> {code: (pct, shift)}
    for code in _BASELINE_STRATEGIES:
        suffix = STRATEGY_CODE_MAP[code]
        full = svm_full[code]
        lines.append(f"  {code} (full {full:.1f}%W):")
        for ds in ids:
            w = _win_pct("svm", suffix, frozenset({ds}))
            svm_detail[ds][code] = (w, w - full)
            lines.append(f"    drop {ds}: {w:5.1f}%W ({w-full:+.1f}pp)  [{SOURCE_MAP[ds]}]")
        lines.append("")
    _write_txt("cmp_sensitivity", lines)

    # (a) Leave-one-dataset-out summary
    header_a = (r"\textbf{Family} & \textbf{Strategy} & \textbf{Full \%W} & "
                r"\textbf{Paper \%W} & \textbf{LOO Range} & \textbf{Most Influential}")
    rows_a = []
    for fam_label, code, full, paper_pct, lo, hi, worst, shift in loo_rows:
        rows_a.append(
            rf"{fam_label} & \texttt{{{_tex(code)}}} & {full:.1f} & {paper_pct:.1f} & "
            rf"[{lo:.1f}, {hi:.1f}] & {worst} ({shift:+.1f}pp) \\"
        )
    tex_a = table_star(
        caption=r"\textbf{Leave-one-dataset-out sensitivity of Table 3's baseline "
                r"comparison} --- diagnostic only, datasets are source-correlated so "
                r"this is not a variance estimate (see leave-one-source-out below for "
                r"the defensible cut).",
        label="tab:exp002_cmp_cmp_sensitivity_loo",
        col_spec="llcccc",
        header=header_a,
        body="\n".join(rows_a),
    )
    # Backward-compat: the report's prose \ref's the whole sensitivity
    # analysis under one bare label -- carry it on this first sub-table.
    tex_a = tex_a.replace(
        r"\label{tab:exp002_cmp_cmp_sensitivity_loo}",
        r"\label{tab:exp002_cmp_cmp_sensitivity_loo}" "\n"
        r"\label{tab:exp002_cmp_cmp_sensitivity}",
    )

    # (b) Leave-one-source-out
    header_b = (r"\textbf{Family} & \textbf{Strategy} & \textbf{Full \%W} & " +
                " & ".join(rf"\textbf{{{s.replace('_', ' ').title()}}}" for s in sources))
    rows_b = []
    for fam_label, code, full, src_vals in source_rows:
        cells = " & ".join(f"{src_vals[s]:.0f}" for s in sources)
        rows_b.append(rf"{fam_label} & \texttt{{{_tex(code)}}} & {full:.1f} & {cells} \\")
    tex_b = table_star(
        caption=r"\textbf{Leave-one-source-out sensitivity of Table 3's baseline "
                r"comparison} --- \%W with each of the 7 correlated dataset sources "
                r"dropped in turn; the clean cut (independent draws), unlike the "
                r"per-dataset table above.",
        label="tab:exp002_cmp_cmp_sensitivity_source",
        col_spec="llc" + "c" * len(sources),
    header=header_b,
        body="\n".join(rows_b),
    )

    # (c) Per-dataset SVM-only detail, one row per dataset, 3 strategies as columns
    header_c = (r"\textbf{Dataset} & \textbf{Source} & " + " & ".join(
        rf"\textbf{{{_tex(code)}}} \%W (shift)" for code in _BASELINE_STRATEGIES))
    rows_c = []
    for ds in ids:
        cells = []
        for code in _BASELINE_STRATEGIES:
            w, shift = svm_detail[ds][code]
            cells.append(f"{w:.1f} ({shift:+.1f}pp)")
        rows_c.append(f"{ds} & {SOURCE_MAP[ds].replace('_', ' ').title()} & " + " & ".join(cells) + r" \\")
    tex_c = table_star(
        caption=r"\textbf{Per-dataset detail, SVM only} --- \%W and shift from the "
                r"full-sample value when that single dataset is excluded, for the "
                r"three baseline-comparison strategies (the family whose Table 3 row "
                r"is fragile, per Section~\ref{sec:paper-comparison}).",
        label="tab:exp002_cmp_cmp_sensitivity_svm",
        col_spec="llccc",
        header=header_c,
        body="\n".join(rows_c),
    )

    _write_tex("cmp_sensitivity", tex_a + "\n\n" + tex_b + "\n\n" + tex_c)


if __name__ == "__main__":
    gen_table3()
    gen_table4()
    gen_table5()
    gen_table6()
    gen_precision_recall()
    gen_pct_rare()
    gen_sensitivity()
    print("Done.")
