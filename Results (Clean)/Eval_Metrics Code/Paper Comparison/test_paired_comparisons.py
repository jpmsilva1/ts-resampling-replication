"""TDD seams: compute_wl() (the paired-comparison classifier) and the
paper_reference.py transcription's internal consistency -- written before
paired_comparisons.py exists, so the first run below is red.

compute_wl() must reproduce src/adapted/PairedComparisons.R's compute_WL()
semantics exactly (Win if median(variant) > median(baseline), sigWin if
also p < alpha via paired Wilcoxon; Loss/sigLoss symmetric; Tie if medians
are identical) -- that R script is the paper's own methodology reference,
never run against real data because its input Rdata doesn't exist, so this
is the Python port's independent-oracle check.

Expected win/loss/tie labels below are worked by hand from each dataset's
construction (20/20 same-signed paired diffs, or all-identical vectors),
not from calling compute_wl() itself -- an all-same-sign paired difference
with n=20 gives an exact two-sided sign-test-style Wilcoxon p bounded by
2/2**20 =~ 1.9e-6, far under alpha=0.05, so the sig-vs-not classification
is derivable without running the code.
"""
import re
from pathlib import Path

import numpy as np

from paired_comparisons import compute_wl, dataset_ids, wl_vs_reference
from paper_reference import MC_SPLIT, OPT_PARAMS_TABLE7, PARAM_GRIDS, TABLE3, TABLE4, TABLE5

EXPS_R = Path(__file__).resolve().parents[3] / "src/adapted/Exps.R"


def test_compute_wl_win_sigwin():
    baseline = np.zeros(20)
    variant = np.ones(20)
    win, sigwin, loss, sigloss, tie = compute_wl(baseline, variant)
    assert (win, sigwin, loss, sigloss, tie) == (1, 1, 0, 0, 0)


def test_compute_wl_loss_sigloss():
    baseline = np.ones(20)
    variant = np.zeros(20)
    win, sigwin, loss, sigloss, tie = compute_wl(baseline, variant)
    assert (win, sigwin, loss, sigloss, tie) == (0, 0, 1, 1, 0)


def test_compute_wl_tie_when_identical():
    v = np.arange(20, dtype=float)
    win, sigwin, loss, sigloss, tie = compute_wl(v, v.copy())
    assert (win, sigwin, loss, sigloss, tie) == (0, 0, 0, 0, 1)


def test_compute_wl_signs_from_median_not_mean():
    """The sign comes from MEDIANS, not means.

    WLdef() in src/original/R_Code/PairedComparisons.R reads column 2 of
    performanceEstimation's WilcoxonSignedRank.test array, which that
    package fills with

        DiffMedScores = median(baseline) - median(variant)

    (verified by deparsing performanceEstimation::pairedComparisons) and
    counts a Win when it is negative. The means are carried on a *separate*
    t.test array the paper never uses, so a mean-signed port silently
    disagrees with the paper on every right-skewed cell -- exactly the
    shape utility-based F1 has, where most folds sit at uba's DELTA floor
    and a handful score high.

    Constructed so the two statistics disagree: variant wins on mean
    (one huge fold) but loses on median (it is below baseline on 40 of 50).
    """
    baseline = np.full(50, 0.50)
    variant = np.full(50, 0.40)
    variant[:10] = 0.60
    variant[0] = 20.0
    assert variant.mean() > baseline.mean()
    assert np.median(variant) < np.median(baseline)
    win, sigwin, loss, sigloss, tie = compute_wl(baseline, variant)
    assert (win, loss, tie) == (0, 1, 0), "Win/Loss must be signed from the median"


def test_compute_wl_tie_on_equal_medians_not_equal_means():
    """WLdef()'s Tie branch fires on DiffMedScores == 0 exactly. Two
    vectors with the same median but different means are a Tie, matching
    the R original; a mean-signed port would call it a Win or a Loss."""
    baseline = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    variant = np.array([0.0, 0.0, 0.3, 0.9, 0.9])
    assert np.median(variant) == np.median(baseline)
    assert variant.mean() != baseline.mean()
    assert compute_wl(baseline, variant) == (0, 0, 0, 0, 1)


def test_compute_wl_aggregates_across_datasets():
    """Two datasets: one clear win, one clear loss -- matches
    PairedComparisons.R's compute_WL(), which sums per-task classifications
    into one WL row rather than averaging p-values."""
    datasets = [
        (np.zeros(20), np.ones(20)),   # win, significant
        (np.ones(20), np.zeros(20)),   # loss, significant
    ]
    win = sigwin = loss = sigloss = tie = 0
    for baseline, variant in datasets:
        w, sw, l, sl, t = compute_wl(baseline, variant)
        win += w; sigwin += sw; loss += l; sigloss += sl; tie += t
    assert (win, sigwin, loss, sigloss, tie) == (1, 1, 1, 1, 0)


def test_wl_vs_reference_exclude_drops_exactly_one_dataset():
    """The leave-one-dataset-out seam: holding out one dataset must remove
    exactly that dataset's own Win/Loss classification from the total and
    change nothing else. Each dataset contributes exactly one of
    win/loss/tie (compute_wl's postcondition), so the full-set win+loss+tie
    must exceed the held-out total by exactly 1."""
    ids = dataset_ids()
    full = wl_vs_reference("svm", "UNDERB", None)
    held = wl_vs_reference("svm", "UNDERB", None, exclude=frozenset({ids[0]}))
    assert sum(full[i] for i in (0, 2, 4)) - sum(held[i] for i in (0, 2, 4)) == 1
    # sigWin/sigLoss can only shrink, never grow, when a dataset is removed.
    assert held[1] <= full[1] and held[3] <= full[3]


def test_dataset_ids_are_present_and_well_formed():
    ids = dataset_ids()
    assert 20 <= len(ids) <= 24
    assert all(re.fullmatch(r"DS\d{2}", i) for i in ids)


def test_table3_rows_sum_to_24():
    """Arithmetic check on the Table 3 transcription: wins+losses must sum
    to the paper's 24 datasets for every strategy/family cell -- catches a
    typo in the numbers copied out of the PDF, independent of any
    replication logic."""
    for strategy, families in TABLE3.items():
        for family, (w, sw, l, sl) in families.items():
            assert w + l == 24, (strategy, family, w, l)
            assert sw <= w and sl <= l, (strategy, family)


def test_table4_rows_sum_to_24():
    """Two cells in the paper's own Table 4 (p.174) sum to 23, not 24 --
    RPART for SM_T (8(1)/15(11)) and SM_TPhi (6(4)/17(9)), confirmed against
    the raw pdftotext extraction directly (not a transcription slip here;
    the paper itself under-reports one dataset in each, presumably an
    unlabeled tie). Every other cell sums to exactly 24."""
    known_23 = {("SM_T", "rpart"), ("SM_TPhi", "rpart")}
    for variant, families in TABLE4.items():
        for family, (w, sw, l, sl) in families.items():
            expected = 23 if (variant, family) in known_23 else 24
            assert w + l == expected, (variant, family, w, l)


def test_table5_rows_sum_to_24():
    for family, strategies in TABLE5.items():
        for strategy, refs in strategies.items():
            for ref_name, (w, sw, l, sl) in refs.items():
                assert w + l == 24, (family, strategy, ref_name, w, l)


def test_opt_params_table7_within_paper_grids():
    """Every DS1-DS20 parameter transcribed from Annex 1 Table 7 must fall
    inside the paper's own published search grids (Annex 1 text) -- the
    same guard applied to the R-side OPT_PARAMS table landed in Exps.R."""
    assert len(OPT_PARAMS_TABLE7) == 24
    for ds_id, params in OPT_PARAMS_TABLE7.items():
        for key, value in params.items():
            assert value in PARAM_GRIDS[key], (ds_id, key, value)


def _parse_table7_pdf_extract(path):
    """Pull the DS-row lines out of the raw pdftotext extract of Table 7
    (Annex 1, p.180) -- e.g. 'DS4    150 0.01      10 1        0.001 7
    750    10       0.1' -- ignoring the Table 8 columns that share the
    same lines in the two-column PDF layout (Table 8's per-dataset rows
    start with 'DS<n>' alone on a line, never 'DS<n> <9 numbers>', so the
    9-number pattern below only matches Table 7 rows)."""
    out = {}
    pattern = re.compile(
        r"^DS(\d+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)(?!\d)",
        re.MULTILINE,
    )
    for m in pattern.finditer(Path(path).read_text()):
        ds_id = f"DS{int(m.group(1)):02d}"
        out[ds_id] = dict(
            svm_cost=float(m.group(2)), svm_gamma=float(m.group(3)),
            mars_nk=float(m.group(4)), mars_degree=float(m.group(5)),
            mars_thresh=float(m.group(6)), rf_mtry=float(m.group(7)),
            rf_ntree=float(m.group(8)), rpart_minsplit=float(m.group(9)),
            rpart_cp=float(m.group(10)),
        )
    return out


def test_opt_params_table7_matches_pdf_extract():
    """OPT_PARAMS_TABLE7 (paper_reference.py) checked against the raw PDF
    text directly, not against src/adapted/Exps.R's own copy of the same
    table -- closes the circularity flagged in DIVERGENCE_ANALYSIS.md
    section 6, where the old guard only compared two by-hand
    transcriptions to each other and would pass even if both were wrong
    in the same way."""
    extract_path = Path(__file__).parent / "table7_pdf_extract.txt"
    parsed = _parse_table7_pdf_extract(extract_path)
    assert len(parsed) == 24, len(parsed)
    for ds_id, params in OPT_PARAMS_TABLE7.items():
        for key, value in params.items():
            assert value == parsed[ds_id][key], (ds_id, key, value, parsed[ds_id][key])


def _r_numeric_vectors(path):
    """Pull `name=c(1,2,...)` / `NAME <- c(...)` numeric vectors out of Exps.R,
    expanding rep(x, n). Enough to read OPT_PARAMS/MC_SZ_* without an R
    interpreter -- these are plain literal tables, not computed."""
    import re
    src = Path(path).read_text()
    out = {}
    # One vector per line in Exps.R, so match to end-of-line rather than
    # trying to balance the parens of a nested rep(...).
    for name, body in re.findall(r"(\w+)\s*(?:=|<-)\s*c\((.*)\)", src):
        vals = []
        num = r"\d*\.?\d+"
        for tok in re.findall(rf"rep\(\s*({num})\s*,\s*(\d+)\s*\)|(-?{num})", body):
            if tok[0]:
                vals += [float(tok[0])] * int(tok[1])
            elif tok[2]:
                vals.append(float(tok[2]))
        out.setdefault(name, vals)
    return out


def test_r_opt_params_and_mc_split_match_paper_reference():
    """The R driver carries its own copy of Table 7 and of the paper's
    per-dataset Monte Carlo splits. Nothing keeps the two transcriptions in
    sync, so assert they agree -- this is the check that would have caught
    DS21-DS24 being added to one table and not the other."""
    v = _r_numeric_vectors(EXPS_R)
    ds_ids = [f"DS{i:02d}" for i in range(1, 25)]

    for key in ("svm_cost", "svm_gamma", "mars_nk", "mars_degree", "mars_thresh",
                "rf_mtry", "rf_ntree", "rpart_minsplit", "rpart_cp"):
        assert len(v[key]) == 24, (key, len(v[key]))
        for ds_id, got in zip(ds_ids, v[key]):
            assert got == OPT_PARAMS_TABLE7[ds_id][key], (ds_id, key, got)

    for ds_id, train, test in zip(ds_ids, v["MC_SZ_TRAIN"], v["MC_SZ_TEST"]):
        assert (train, test) == MC_SPLIT[ds_id], (ds_id, train, test)


if __name__ == "__main__":
    test_compute_wl_win_sigwin()
    test_compute_wl_loss_sigloss()
    test_compute_wl_tie_when_identical()
    test_compute_wl_signs_from_median_not_mean()
    test_compute_wl_tie_on_equal_medians_not_equal_means()
    test_compute_wl_aggregates_across_datasets()
    test_table3_rows_sum_to_24()
    test_table4_rows_sum_to_24()
    test_table5_rows_sum_to_24()
    test_opt_params_table7_within_paper_grids()
    test_opt_params_table7_matches_pdf_extract()
    test_r_opt_params_and_mc_split_match_paper_reference()
    print("All paired-comparison tests passed.")
