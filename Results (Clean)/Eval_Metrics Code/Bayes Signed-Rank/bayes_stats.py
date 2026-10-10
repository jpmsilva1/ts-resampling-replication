"""Bayesian signed-rank test (Benavoli, Corani, Demsar & Zaffalon, 2017),
via baycomp.SignedRankTest. Compares a strategy against a baseline via paired
per-dataset differences, returning posterior probabilities that each side
wins or that the two are practically equivalent (within a ROPE around 0)."""
import baycomp
import numpy as np


def bayes_signedrank_probs(diffs: np.ndarray, rope: float, n_samples: int = 50_000,
                            seed: int | None = None) -> tuple[float, float, float]:
    """diffs: paired differences (strategy - baseline), positive = strategy
    better. Returns (p_baseline_wins, p_rope, p_strategy_wins)."""
    diffs = np.asarray(diffs, dtype=float)
    # baycomp returns (P(x > y), P(rope), P(x < y)); x = diffs, y = 0.
    p_strategy, p_rope, p_baseline = baycomp.SignedRankTest.probs(
        diffs, np.zeros_like(diffs), rope=rope, nsamples=n_samples, random_state=seed)
    return float(p_baseline), float(p_rope), float(p_strategy)


def log_ratio_diffs(strategy: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """For unbounded, scale-heterogeneous metrics (RMSE/SERA): use the log
    ratio instead of the raw difference so a ROPE is meaningful across
    datasets of very different magnitude. Lower-is-better metrics, so a
    POSITIVE log ratio log(baseline) - log(strategy) means the strategy wins."""
    return np.log(np.asarray(baseline, dtype=float)) - np.log(np.asarray(strategy, dtype=float))
