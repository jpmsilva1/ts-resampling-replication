import numpy as np
from bayes_stats import bayes_signedrank_probs, log_ratio_diffs

# All differences strongly positive -> strategy should win almost always.
p_b, p_r, p_s = bayes_signedrank_probs(np.full(20, 5.0), rope=0.01, seed=0)
assert p_s > 0.99 and p_b < 0.01, (p_b, p_r, p_s)

# All differences exactly 0 -> should be a draw almost always.
p_b, p_r, p_s = bayes_signedrank_probs(np.zeros(20), rope=0.01, seed=0)
assert p_r > 0.99, (p_b, p_r, p_s)

# Symmetric differences around 0 -> roughly balanced wins, small draw region.
diffs = np.array([5.0, -5.0] * 10)
p_b, p_r, p_s = bayes_signedrank_probs(diffs, rope=0.01, seed=0)
assert abs(p_b - p_s) < 0.1, (p_b, p_r, p_s)

# log_ratio_diffs: strategy beats baseline (lower is better) -> positive diff.
d = log_ratio_diffs(strategy=np.array([1.0]), baseline=np.array([2.0]))
assert d[0] > 0, d

print("All bayes_stats checks passed.")
