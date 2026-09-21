"""TDD seam 2: rmse() must match R's own eval.stats() formula
(sqrt(mean((trues-preds)^2))) exactly -- written before the implementation.
"""
import numpy as np

from rmse_metric import rmse


def test_matches_r_formula_on_known_fold():
    # A small fixed fold: R's rmse = sqrt(mean((trues-preds)^2)).
    trues = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    preds = np.array([1.5, 2.5, 2.0, 4.5, 4.0])
    expected = np.sqrt(np.mean((trues - preds) ** 2))  # == R's sqrt(mean((trues-preds)^2))
    assert rmse(trues, preds) == expected


def test_zero_for_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0


if __name__ == "__main__":
    test_matches_r_formula_on_known_fold()
    test_zero_for_perfect_predictions()
    print("All RMSE tests passed.")
