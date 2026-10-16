"""The two significance perspectives on the region-stratified metrics.

Frequentist: the paper's own paired Wilcoxon win/loss, reusing
Paper Comparison/paired_comparisons.py unchanged. That module already
reproduces R's wilcox.test(paired=TRUE) and the median-sign
Win/sigWin/Loss/sigLoss/Tie rule, verified against R on all 3,240 of the
Table 3 to 5 comparisons. One test per (dataset, family, strategy, region,
metric) over the Monte Carlo folds, then counted across datasets.

Bayesian: the signed-rank test of Benavoli et al. (2017), reusing
Bayes Signed-Rank/bayes_stats.py. Each dataset contributes one paired value,
its mean over folds, compared on the log ratio with a region of practical
equivalence of plus or minus 5 percent. This is the convention the project's
existing Bayes charts already use for the unbounded, lower-is-better metrics.

The two answer different questions. The Wilcoxon test asks whether a
difference exists; the Bayesian test asks whether it is large enough to
matter. Section 'Where the error falls' in the report shows a case where they
disagree, which is why both are carried.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parent, _HERE.parent / "Paper Comparison", _HERE.parent / "Bayes Signed-Rank"):
    sys.path.insert(0, str(_p))

from paired_comparisons import compute_wl, wilcoxon_p  # noqa: E402,F401
from bayes_stats import bayes_signedrank_probs, log_ratio_diffs  # noqa: E402

ROPE = float(np.log(1.05))
WL_FIELDS = ["win", "sigwin", "loss", "sigloss", "tie"]
CELL_KEYS = ["dataset", "family", "strategy", "region"]


def paired_vectors(fm: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """One cell's per-fold (baseline, strategy) values, with empty-region folds
    dropped. Both sides are scored on the same points, so a dropped fold leaves
    the pairing exact."""
    g = fm[fm["n"] > 0].sort_values("iteration")
    return g[f"{metric}_base"].to_numpy(), g[f"{metric}_strat"].to_numpy()


def wl_table(fold_metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Wins, losses and ties summed across datasets, per (family, strategy,
    region).

    These metrics are lower-is-better, while compute_wl treats a positive
    difference as a win for the variant, so both vectors are negated.
    """
    rows = []
    for (fam, strat, region), g in fold_metrics.groupby(["family", "strategy", "region"], sort=False):
        acc = dict.fromkeys(WL_FIELDS, 0)
        n_ds = n_dropped = 0
        for _ds, cell in g.groupby("dataset", sort=False):
            base, strat_v = paired_vectors(cell, metric)
            n_dropped += int((cell["n"] == 0).sum())
            if base.size == 0:
                continue
            n_ds += 1
            for k, v in zip(WL_FIELDS, compute_wl(-base, -strat_v)):
                acc[k] += v
        rows.append({"family": fam, "strategy": strat, "region": region,
                     "n_datasets": n_ds, "folds_dropped": n_dropped, **acc})
    return pd.DataFrame(rows)


def effect_cells(fold_metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per (dataset, family, strategy, region): the percentage change in the
    metric against the baseline, positive meaning the strategy has the lower
    error. One row per comparison, so a caller can take the median over any
    slice rather than a median of medians.

    The log ratio is used for the same reason the Bayesian test uses it: these
    metrics span orders of magnitude across datasets, so raw differences
    cannot be pooled.
    """
    rows = []
    for (ds, fam, strat, region), cell in fold_metrics.groupby(CELL_KEYS, sort=False):
        base, strat_v = paired_vectors(cell, metric)
        if base.size == 0:
            continue
        lr = float(log_ratio_diffs(strategy=strat_v.mean(), baseline=base.mean()))
        rows.append({"dataset": ds, "family": fam, "strategy": strat, "region": region,
                     "log_ratio": lr, "pct": 100.0 * (np.exp(lr) - 1.0)})
    return pd.DataFrame(rows)


def bayes_table(fold_metrics: pd.DataFrame, metric: str, seed: int = 0) -> pd.DataFrame:
    """Posterior probabilities that the baseline wins, that the two are
    practically tied, and that the strategy wins, per (family, strategy,
    region)."""
    rows = []
    for (fam, strat, region), g in fold_metrics.groupby(["family", "strategy", "region"], sort=False):
        base, strat_v = [], []
        for _ds, cell in g.groupby("dataset", sort=False):
            b, s = paired_vectors(cell, metric)
            if b.size == 0:
                continue
            base.append(b.mean()); strat_v.append(s.mean())
        base, strat_v = np.asarray(base), np.asarray(strat_v)
        d = log_ratio_diffs(strategy=strat_v, baseline=base)
        p_base, p_rope, p_strat = bayes_signedrank_probs(d, rope=ROPE, seed=seed)
        rows.append({"family": fam, "strategy": strat, "region": region, "n_datasets": len(base),
                     "p_baseline": p_base, "p_rope": p_rope, "p_strategy": p_strat,
                     "median_log_ratio": float(np.median(d))})
    return pd.DataFrame(rows)


__all__ = ["ROPE", "WL_FIELDS", "CELL_KEYS", "paired_vectors", "wl_table", "effect_cells",
           "bayes_table", "compute_wl", "wilcoxon_p"]
