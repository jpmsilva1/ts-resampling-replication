"""Bayesian signed-rank test (Benavoli, Corani, Demsar & Zaffalon, 2017),
the same 'Zbis' procedure baycomp.SignedRankTest implements. Compares a
strategy against a baseline via paired per-dataset differences, returning
posterior probabilities that each side wins or that the two are practically
equivalent (within a ROPE around 0)."""
import numpy as np


def bayes_signedrank_probs(diffs: np.ndarray, rope: float, n_samples: int = 50_000,
                            seed: int | None = None) -> tuple[float, float, float]:
    """diffs: paired differences (strategy - baseline), positive = strategy
    better. Returns (p_baseline_wins, p_rope, p_strategy_wins)."""
    diffs = np.asarray(diffs, dtype=float)
    z = np.concatenate(([0.0], diffs))  # prior pseudo-observation at 0
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(len(z)), size=n_samples)
    sample_means = weights @ z
    p_strategy = float(np.mean(sample_means > rope))
    p_baseline = float(np.mean(sample_means < -rope))
    p_rope = 1.0 - p_strategy - p_baseline
    return p_baseline, p_rope, p_strategy


def log_ratio_diffs(strategy: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """For unbounded, scale-heterogeneous metrics (RMSE/SERA): use the log
    ratio instead of the raw difference so a ROPE is meaningful across
    datasets of very different magnitude. Lower-is-better metrics, so a
    NEGATIVE log ratio (strategy < baseline) means the strategy wins."""
    return np.log(np.asarray(baseline, dtype=float)) - np.log(np.asarray(strategy, dtype=float))
