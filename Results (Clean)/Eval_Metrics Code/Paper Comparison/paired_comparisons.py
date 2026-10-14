"""Python port of src/adapted/PairedComparisons.R's compute_WL(): the
paper's own paired-comparison methodology (paired Wilcoxon signed-rank,
Win/sigWin/Loss/sigLoss/Tie classification per dataset, summed across
datasets), applied here to this project's real
Results Data/raw_iterations_by_dataset_v2/*.csv instead of the R script's
never-populated results/merged_results.Rdata input.

Kept Python per this project's established convention (CLAUDE.md
2026-08-26, "Language: Python over R").
"""
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tables_common import RESULTS_DATA

DATA_DIR = RESULTS_DATA / "raw_iterations_by_dataset_v2"
ALPHA = 0.05


def wilcoxon_p(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sided paired Wilcoxon p-value computed the way R's
    wilcox.test(x, y, paired=TRUE) does it (the test performanceEstimation
    calls): zero differences dropped; the exact distribution only when n < 50
    with no ties and no zeros; otherwise the normal approximation with tie and
    continuity corrections. scipy's default uses the exact distribution at
    n = 50, which flipped one paper-Table-4 cell (DS14, SVM O_TPhi vs O_B:
    R p = 0.05004, scipy 0.0494)."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    zeroes = bool(np.any(d == 0))
    d = d[d != 0]
    n = d.size
    if n == 0:
        return 1.0
    r = rankdata(np.abs(d))
    _, counts = np.unique(r, return_counts=True)
    if n < 50 and counts.max() == 1 and not zeroes:
        return float(wilcoxon(d, method="exact").pvalue)
    z = r[d > 0].sum() - n * (n + 1) / 4
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24 - (counts**3 - counts).sum() / 48)
    if sigma == 0:
        return 1.0
    z = (z - np.sign(z) * 0.5) / sigma
    return float(min(1.0, 2 * min(norm.cdf(z), norm.sf(z))))


def compute_wl(baseline: np.ndarray, variant: np.ndarray, alpha: float = ALPHA) -> tuple[int, int, int, int, int]:
    """One dataset's paired classification against a reference workflow.
    Returns (win, sig_win, loss, sig_loss, tie) -- exactly one of these
    five is 1, the rest 0, matching compute_WL()'s per-(task, workflow)
    cell in the R original.

    The sign is the MEDIAN difference, not the mean. WLdef() reads column 2
    of performanceEstimation's WilcoxonSignedRank.test array, and that
    package fills it with median(baseline) - median(variant); the means
    live on a separate t.test array the paper never uses. Tie is exact
    median equality, matching WLdef()'s `== 0` branch.
    """
    diff = float(np.median(variant) - np.median(baseline))
    if diff == 0.0:
        return (0, 0, 0, 0, 1)
    p = wilcoxon_p(variant, baseline)
    if diff > 0:
        return (1, int(p < alpha), 0, 0, 0)
    return (0, 0, 1, int(p < alpha), 0)


@lru_cache(maxsize=1)
def _load_all() -> pd.DataFrame:
    # Cached: the leave-one-dataset-out sweep calls this a few hundred times
    # and the CSVs never change within a run. Callers must not mutate.
    files = sorted(DATA_DIR.glob("DS??_*.csv"))
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def dataset_ids() -> list[str]:
    """The dataset ids actually present on disk -- 20 to 24 depending on how
    far the Apuana rerun has gotten (CLAUDE.md 2026-09-04)."""
    return sorted(_load_all()["dataset_id"].unique())


def wl_vs_reference(family: str, target_suffix: str | None, ref_suffix: str | None, metric: str = "F1",
                    exclude: frozenset[str] | None = None) -> tuple[int, int, int, int, int]:
    """Aggregate compute_wl() across every dataset present for one
    (family, target strategy) vs. (family, reference strategy) pair.
    suffix=None means the bare baseline workflow (mc.<family>).

    `exclude` holds out dataset ids, for the leave-one-dataset-out
    sensitivity diagnostic in generate_paper_comparison.gen_sensitivity()."""
    df = _load_all()
    target_wf = f"mc.{family}" if target_suffix is None else f"mc.{family}_{target_suffix}"
    ref_wf = f"mc.{family}" if ref_suffix is None else f"mc.{family}_{ref_suffix}"

    win = sigwin = loss = sigloss = tie = 0
    for ds_id, ds_df in df.groupby("dataset_id"):
        if exclude and ds_id in exclude:
            continue
        ref_vec = ds_df[ds_df.workflow == ref_wf].sort_values("iteration")[metric].to_numpy()
        tgt_vec = ds_df[ds_df.workflow == target_wf].sort_values("iteration")[metric].to_numpy()
        n = min(len(ref_vec), len(tgt_vec))
        if n == 0:
            continue
        w, sw, l, sl, t = compute_wl(ref_vec[:n], tgt_vec[:n])
        win += w; sigwin += sw; loss += l; sigloss += sl; tie += t
    return (win, sigwin, loss, sigloss, tie)
