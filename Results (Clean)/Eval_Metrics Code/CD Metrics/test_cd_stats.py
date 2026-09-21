"""TDD: the Nemenyi critical-difference formula and average-rank computation,
written before the implementation. CD values cross-checked against Demšar
(2006) Table 5's published q_alpha constants (the standard table used by
every CD-diagram implementation -- Orange, scmamp, scikit-posthocs).
"""
import numpy as np
import pandas as pd

from cd_stats import average_ranks, friedman_p, nemenyi_cd


def test_cd_matches_demsar_2006_k10_n20():
    # Demsar 2006, alpha=0.05: q_alpha(k=10) = 3.164
    # CD = q_alpha * sqrt(k(k+1) / (6N))
    k, n = 10, 20
    expected = 3.164 * np.sqrt(k * (k + 1) / (6 * n))
    assert np.isclose(nemenyi_cd(k, n, alpha=0.05), expected, rtol=1e-3)


def test_cd_matches_demsar_2006_k2_n20():
    # k=2 is the pairwise Wilcoxon-equivalent case, q_alpha(k=2) = 1.960
    k, n = 2, 20
    expected = 1.960 * np.sqrt(k * (k + 1) / (6 * n))
    assert np.isclose(nemenyi_cd(k, n, alpha=0.05), expected, rtol=1e-3)


def test_average_ranks_higher_is_better():
    # 2 datasets x 3 methods; higher value = better = rank 1
    df = pd.DataFrame({"A": [3.0, 1.0], "B": [2.0, 2.0], "C": [1.0, 3.0]})
    ranks = average_ranks(df, higher_is_better=True)
    # ds1: A=3(best,rank1) B=2(rank2) C=1(rank3); ds2: C=3(rank1) B=2(rank2) A=1(rank3)
    # avg: A=(1+3)/2=2.0, B=(2+2)/2=2.0, C=(3+1)/2=2.0 -- all tied
    assert np.isclose(ranks["A"], 2.0)
    assert np.isclose(ranks["B"], 2.0)
    assert np.isclose(ranks["C"], 2.0)


def test_average_ranks_lower_is_better():
    # For an error metric (RMSE/SERA), the smallest value should get rank 1.
    df = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]})
    ranks = average_ranks(df, higher_is_better=False)
    assert ranks["A"] == 1.0
    assert ranks["B"] == 2.0
    assert ranks["C"] == 3.0


def test_friedman_p_rejects_when_one_method_always_wins():
    # A always beats B beats C on every block -> the omnibus null (all
    # methods equivalent) must be rejected.
    df = pd.DataFrame({"A": [1.0] * 10, "B": [2.0] * 10, "C": [3.0] * 10})
    assert friedman_p(df) < 0.05


def test_friedman_p_does_not_reject_when_ranks_are_symmetric():
    # Each method wins on exactly one third of the blocks -> average ranks
    # are identical, so the omnibus test must NOT reject.
    df = pd.DataFrame({
        "A": [1.0, 2.0, 3.0] * 4,
        "B": [2.0, 3.0, 1.0] * 4,
        "C": [3.0, 1.0, 2.0] * 4,
    })
    assert friedman_p(df) > 0.05


if __name__ == "__main__":
    test_cd_matches_demsar_2006_k10_n20()
    test_cd_matches_demsar_2006_k2_n20()
    test_average_ranks_higher_is_better()
    test_average_ranks_lower_is_better()
    test_friedman_p_rejects_when_one_method_always_wins()
    test_friedman_p_does_not_reject_when_ranks_are_symmetric()
    print("All CD stats tests passed.")
