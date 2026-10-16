"""Plain-assert tests for region_metrics.py / region_stats.py (hand-built
fixtures, no CSV I/O)."""
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from region_metrics import error_quantiles, rare_diagnostics, region_fold_metrics
from region_stats import compute_wl, paired_vectors, wl_table


def _cell(n=200, seed=0, phi=None):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "iteration": np.repeat([1, 2], n // 2),
        "phi_y": rng.uniform(size=n) if phi is None else phi,
        "e": rng.normal(size=n), "e_base": rng.normal(size=n),
    })
    df["e2"] = df["e"] ** 2
    df["e2_base"] = df["e_base"] ** 2
    return df


def test_region_metrics_match_direct_computation():
    df = _cell()
    out = region_fold_metrics(df, 0.9)
    m = (df["iteration"] == 2) & (df["phi_y"] >= 0.9)
    r = out[(out.iteration == 2) & (out.region == "rare")].iloc[0]
    assert r.n == m.sum()
    assert np.isclose(r.mae_strat, df.loc[m, "e"].abs().mean())
    assert np.isclose(r.rmse_base, np.sqrt(df.loc[m, "e2_base"].mean()))
    assert np.isclose(r.sera_strat, (df.loc[m, "phi_y"] * df.loc[m, "e2"]).mean())


def test_regions_partition_the_fold_and_sums_add_back():
    df = _cell()
    out = region_fold_metrics(df, 0.9)
    for it in (1, 2):
        g = out[out.iteration == it]
        f = df[df["iteration"] == it]
        assert g["n"].sum() == len(f)
        assert np.isclose(g["sse_strat"].sum(), f["e2"].sum())
        assert np.isclose(g["sera_sum_base"].sum(), (f["phi_y"] * f["e2_base"]).sum())


def test_region_is_defined_by_y_true_so_both_sides_score_the_same_points():
    """Wrecking the strategy's predictions must not move a single point
    between regions -- that is what makes the pairing exact."""
    df = _cell()
    before = region_fold_metrics(df, 0.9)[["iteration", "region", "n"]]
    df2 = df.copy()
    df2["e"] = df2["e"] + 1000.0
    df2["e2"] = df2["e"] ** 2
    after = region_fold_metrics(df2, 0.9)[["iteration", "region", "n"]]
    assert before.equals(after)


def test_empty_region_gives_n_zero_and_is_dropped_not_nan_propagated():
    phi = np.where(np.arange(200) < 100, 0.1, 0.1)  # fold 1 and 2: no rare at all
    phi[150:] = 0.95                                 # ... except half of fold 2
    df = _cell(phi=phi)
    out = region_fold_metrics(df, 0.9)
    rare1 = out[(out.iteration == 1) & (out.region == "rare")].iloc[0]
    assert rare1.n == 0 and np.isnan(rare1.mae_strat)
    base, strat = paired_vectors(out[out.region == "rare"], "mae")
    assert base.size == 1 and not np.isnan(base).any() and not np.isnan(strat).any()


def test_rmse_and_mse_give_the_same_win_loss():
    rng = np.random.default_rng(7)
    base, strat = rng.uniform(1, 5, 50), rng.uniform(1, 5, 50)
    # compute_wl is called on negated values (lower-is-better -> higher-is-better)
    assert compute_wl(-base, -strat) == compute_wl(-(base**2), -(strat**2))


def test_wl_table_flips_sign_so_a_lower_error_counts_as_a_win():
    fm = []
    for ds in ("DS01", "DS02"):
        for it in range(1, 21):
            fm.append({"dataset": ds, "family": "lm", "strategy": "OVERB", "region": "rare",
                       "iteration": it, "n": 5,
                       "mae_base": 2.0 + 0.01 * it, "mae_strat": 1.0 + 0.01 * it,
                       "rmse_base": 1.0, "rmse_strat": 1.0, "sera_base": 1.0, "sera_strat": 1.0})
    t = wl_table(pd.DataFrame(fm), "mae").iloc[0]
    assert t.win == 2 and t.sigwin == 2 and t.loss == 0  # strategy halves the error everywhere


def test_error_quantiles_are_ratios_and_skip_thin_cells():
    df = _cell(n=400, phi=np.where(np.arange(400) < 200, 0.1, 0.95))
    df["e"] = df["e_base"] * 0.5                      # strategy halves every error
    q = error_quantiles(df, 0.9)
    assert all(np.isclose(v, 0.5) for v in q.values())
    assert error_quantiles(_cell(n=40, phi=np.full(40, 0.1)), 0.9) == {}


def test_rare_diagnostics_counts_empty_folds():
    phi = np.full(200, 0.1)
    phi[150:] = 0.95
    d = rare_diagnostics(_cell(phi=phi), 0.9)
    assert d["n_folds"] == 2 and d["folds_empty_rare"] == 1
    assert d["min_rare"] == 0 and d["max_rare"] == 50
    assert np.isclose(d["pct_rare"], 25.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
