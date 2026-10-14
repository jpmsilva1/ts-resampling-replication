"""The H.2 interval must resample datasets, not the 5 family cells inside them."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_bias_ts_baselines import paired_effect  # noqa: E402


def test_bootstrap_resamples_datasets_not_cells():
    # two datasets, every family cell equal within a dataset: 0 and 1
    diff = pd.DataFrame([[0.0] * 5, [1.0] * 5], index=["DS01", "DS02"])
    m, lo, hi = paired_effect(diff, n_boot=4000, seed=0)
    assert m == 0.5
    # dataset bootstrap can only draw both-0 or both-1, so the CI spans [0, 1];
    # a cell bootstrap over 10 values would give roughly [0.2, 0.8]
    assert lo == 0.0 and hi == 1.0


if __name__ == "__main__":
    test_bootstrap_resamples_datasets_not_cells()
    print("ok")
