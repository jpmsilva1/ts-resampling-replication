import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

# Demsar (2006), Table 5: q_alpha critical values for the Nemenyi post-hoc
# test (studentized range statistic / sqrt(2)), indexed by number of methods
# k. The same constants every CD-diagram implementation hardcodes (Orange,
# scmamp, scikit-posthocs) -- there's no closed-form alternative in common use.
Q_ALPHA_005 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
    9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354,
    15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544,
}
Q_ALPHA_010 = {
    2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693, 8: 2.780,
    9: 2.855, 10: 2.920, 11: 2.978, 12: 3.030, 13: 3.077, 14: 3.120,
    15: 3.159, 16: 3.196, 17: 3.230, 18: 3.261, 19: 3.291, 20: 3.319,
}


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference: CD = q_alpha * sqrt(k(k+1) / (6N)),
    k = number of methods compared, n = number of datasets (blocks)."""
    table = {0.05: Q_ALPHA_005, 0.10: Q_ALPHA_010}[alpha]
    q_alpha = table[k]
    return q_alpha * np.sqrt(k * (k + 1) / (6 * n))


def friedman_p(matrix: pd.DataFrame) -> float:
    """Friedman omnibus test p-value. Demsar's procedure gates the Nemenyi
    post-hoc on this: if the omnibus null (all methods equivalent) isn't
    rejected, the pairwise CD bars aren't licensed and the diagram
    shouldn't be read as showing real differences."""
    return friedmanchisquare(*[matrix[c].values for c in matrix.columns]).pvalue


def average_ranks(matrix: pd.DataFrame, higher_is_better: bool) -> pd.Series:
    """matrix: rows = datasets, columns = methods, cells = that method's
    score on that dataset. Returns the average rank per method across
    datasets (rank 1 = best on that dataset), ties handled by averaging."""
    ranks = matrix.rank(axis=1, ascending=not higher_is_better)
    return ranks.mean(axis=0)
