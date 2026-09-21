#!/usr/bin/env python3
"""Causal estimands for the resampling-strategy experiment: the design is a
fully-crossed factorial (20 datasets x 5 model families x 10 strategies), so
identification is by construction -- strategy is assigned, not selected. The
dataset/family blocking below buys precision, not confounding control. See
docs/causal-plans/2026-08-28-resampling-causal-effect/plan.md for the full
design rationale (estimands, threats, why randomization inference replaces
cluster-robust SEs).

Unit of observation matches generate_regression.py's panel: one row per
(dataset_id, model_family, strategy), already averaged across the 50 Monte
Carlo iterations upstream.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "CD Metrics"))
from cd_stats import average_ranks

BLOCK_COLS = ("dataset_id", "model_family")
STRATEGY_COL = "strategy"
BASELINE = "baseline"

# Dataset -> source, from data/paper_datasets_csv/manifest.csv (see plan's
# "source-clustering" threat: the 24 datasets are really 7 independent sources).
# DS21-24 added 2026-09-04 with the DS21-24 unblock; note porto_water (a water
# utility's half-hourly meters) is a different source from porto_weather.
SOURCE_MAP = {
    **{f"DS{i:02d}": "bike_daily" for i in range(1, 5)},
    **{f"DS{i:02d}": "bike_hourly" for i in range(5, 9)},
    "DS09": "icelandic_river",
    **{f"DS{i:02d}": "porto_weather" for i in range(10, 14)},
    **{f"DS{i:02d}": "istanbul_stock" for i in range(14, 21)},
    **{f"DS{i:02d}": "australian_electricity" for i in range(21, 23)},
    **{f"DS{i:02d}": "porto_water" for i in range(23, 25)},
}


def paired_diffs(panel: pd.DataFrame, dv: str, block_cols=BLOCK_COLS,
                  strategy_col: str = STRATEGY_COL, baseline: str = BASELINE) -> pd.DataFrame:
    """Long frame: one row per (block, non-baseline strategy) with the paired
    difference `value[strategy] - value[baseline]` for that block."""
    wide = panel.pivot_table(index=list(block_cols), columns=strategy_col, values=dv)
    base = wide[baseline]
    diffs = wide.drop(columns=[baseline]).sub(base, axis=0)
    return diffs.reset_index().melt(id_vars=list(block_cols), var_name=strategy_col, value_name="diff")


def tau_k(panel: pd.DataFrame, dv: str, block_cols=BLOCK_COLS,
          strategy_col: str = STRATEGY_COL, baseline: str = BASELINE) -> pd.Series:
    """Point estimate: mean paired difference vs. baseline across blocks, per
    strategy. Equals the two-way fixed-effects OLS coefficient exactly under
    this design's full balance (every block has exactly one baseline and one
    treated observation per strategy)."""
    diffs = paired_diffs(panel, dv, block_cols, strategy_col, baseline)
    return diffs.groupby(strategy_col)["diff"].mean()


def permute_within_blocks(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independently permute each row of `values` (one row per block). This
    is the exchangeability step the sharp null licenses: within a block, the
    experimenter assigned strategies to that block's units, so their labels
    are interchangeable. Permuting across rows instead would mix outcomes
    from different blocks and is NOT licensed by the design."""
    return np.array([rng.permutation(row) for row in values])


def randomization_inference(panel: pd.DataFrame, dv: str, block_cols=BLOCK_COLS,
                             strategy_col: str = STRATEGY_COL, baseline: str = BASELINE,
                             n_perm: int = 10000, seed: int = 0) -> pd.DataFrame:
    """Exact inference for tau_k under the sharp null of no effect for any
    unit: strategy labels are exchangeable *within* each block (dataset,
    family), since the experimenter assigned every strategy to every block.
    Permuting within blocks (not across the whole panel) preserves the
    factorial design and sidesteps both the small-cluster-count problem and
    the dataset-dummy SE degeneracy that afflict the cluster-robust OLS fit.

    Two-sided p-value: fraction of permutation draws with |tau_permuted| >=
    |tau_observed|, plus-one-smoothed to avoid a zero p-value from finite
    Monte Carlo draws (Davison & Hinkley 1997, 4.2.2)."""
    rng = np.random.default_rng(seed)
    observed = tau_k(panel, dv, block_cols, strategy_col, baseline)

    wide = panel.pivot_table(index=list(block_cols), columns=strategy_col, values=dv)
    strategies = [s for s in wide.columns if s != baseline]
    values = wide[[baseline] + strategies].to_numpy()  # shape (n_blocks, 1 + k)
    n_blocks, n_cols = values.shape

    null_tau = np.empty((n_perm, len(strategies)))
    for p in range(n_perm):
        permuted = permute_within_blocks(values, rng)
        base_col = permuted[:, 0]
        null_tau[p] = (permuted[:, 1:] - base_col[:, None]).mean(axis=0)

    obs = observed[strategies].to_numpy()
    p_values = (np.abs(null_tau) >= np.abs(obs)).sum(axis=0) + 1
    p_values = p_values / (n_perm + 1)

    return pd.DataFrame({"tau": obs, "p_value": p_values}, index=strategies)


def win_probability(panel: pd.DataFrame, value_col: str, block_cols=BLOCK_COLS,
                     strategy_col: str = STRATEGY_COL, baseline: str = BASELINE,
                     higher_is_better: bool = True) -> pd.Series:
    """Distribution-free secondary estimand for F1 (see plan: raw F1's mean
    is partly a floor artifact when baseline F1 approx 0 on most datasets).
    P(strategy beats baseline on a randomly drawn block), independent of the
    outcome's scale or floor."""
    diffs = paired_diffs(panel, value_col, block_cols, strategy_col, baseline)
    wins = diffs["diff"] > 0 if higher_is_better else diffs["diff"] < 0
    return wins.groupby(diffs[strategy_col]).mean()


def rank_shift(panel: pd.DataFrame, value_col: str, block_cols=BLOCK_COLS,
               strategy_col: str = STRATEGY_COL, higher_is_better: bool = True) -> pd.Series:
    """Average rank per strategy across all blocks (dataset x family treated
    jointly as blocks, same convention the CD diagrams use) -- reuses
    cd_stats.average_ranks rather than re-deriving ranking math."""
    wide = panel.pivot_table(index=list(block_cols), columns=strategy_col, values=value_col)
    return average_ranks(wide, higher_is_better=higher_is_better)


def leave_one_source_out(panel: pd.DataFrame, dv: str, source_map: dict,
                          dataset_col: str = "dataset_id", block_cols=BLOCK_COLS,
                          strategy_col: str = STRATEGY_COL, baseline: str = BASELINE) -> pd.DataFrame:
    """Jackknife across dataset SOURCES, not individual datasets (see plan:
    the 20 datasets are really 5 sources -- 7 Istanbul indices, 2 bike-sharing
    granularities, one weather station, one river -- so a per-dataset
    jackknife would understate how correlated the "20 draws" really are).
    Returns one row per dropped source, tau_k recomputed on the rest."""
    sources = panel[dataset_col].map(source_map)
    # An unmapped dataset would sit in every fold and never be jackknifed out
    # -- silent, and exactly how DS21-24 would have slipped through.
    if sources.isna().any():
        missing = sorted(panel.loc[sources.isna(), dataset_col].unique())
        raise KeyError(f"datasets missing from source_map: {missing}")
    rows = {}
    for source in sorted(sources.unique()):
        subset = panel[sources != source]
        rows[source] = tau_k(subset, dv, block_cols, strategy_col, baseline)
    return pd.DataFrame(rows).T


def holm_adjust(p_values: pd.Series) -> pd.Series:
    """Holm step-down multiplicity adjustment (Holm 1979), family-wise error
    controlled without Bonferroni's over-conservatism. Used for the pooled
    tau_k tests (plan: 9 strategies x 2 metrics is a confirmatory family);
    the per-family 9x5x2 grid is reported as exploratory, unadjusted."""
    order = p_values.sort_values().index
    m = len(p_values)
    raw_sorted = p_values.loc[order].to_numpy()
    multipliers = np.arange(m, 0, -1)
    stepped = raw_sorted * multipliers
    adjusted_sorted = np.clip(np.maximum.accumulate(stepped), 0, 1)
    return pd.Series(adjusted_sorted, index=order).reindex(p_values.index)


def bootstrap_ci(panel: pd.DataFrame, dv: str, block_cols=BLOCK_COLS,
                  strategy_col: str = STRATEGY_COL, baseline: str = BASELINE,
                  n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Nonparametric percentile CI for tau_k's magnitude, complementing
    randomization_inference's p-value (which answers "is there an effect,"
    not "how big"). Resamples BLOCKS with replacement (the actual unit of
    replication in this design), not individual strategy observations."""
    rng = np.random.default_rng(seed)
    observed = tau_k(panel, dv, block_cols, strategy_col, baseline)

    wide = panel.pivot_table(index=list(block_cols), columns=strategy_col, values=dv)
    strategies = [s for s in wide.columns if s != baseline]
    values = wide[[baseline] + strategies].to_numpy()
    n_blocks = values.shape[0]

    boot_tau = np.empty((n_boot, len(strategies)))
    for b in range(n_boot):
        sample = values[rng.integers(0, n_blocks, size=n_blocks)]
        boot_tau[b] = (sample[:, 1:] - sample[:, [0]]).mean(axis=0)

    lo = np.percentile(boot_tau, 100 * alpha / 2, axis=0)
    hi = np.percentile(boot_tau, 100 * (1 - alpha / 2), axis=0)

    return pd.DataFrame({"tau": observed[strategies].to_numpy(), "ci_lo": lo, "ci_hi": hi},
                         index=strategies)


if __name__ == "__main__":
    pass
