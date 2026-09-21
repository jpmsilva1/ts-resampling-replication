"""TDD: causal estimands for the resampling-strategy experiment (blocked
factorial design -- see docs/causal-plans/2026-08-28-resampling-causal-effect/
plan.md for the design rationale). Plain asserts, matching CD Metrics' style.

Seams under test (all public functions in causal_effects.py):
  - tau_k: point estimate, mean paired difference vs. baseline across blocks
  - randomization_inference: exact p-values via within-block permutation
  - win_probability / rank_shift: distribution-free secondary estimand
  - leave_one_source_out: jackknife across the 5 dataset sources
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from causal_effects import (tau_k, randomization_inference, permute_within_blocks,
                             win_probability, rank_shift, leave_one_source_out, holm_adjust,
                             bootstrap_ci)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "CD Metrics"))
from cd_stats import average_ranks


def _toy_panel():
    # 2 blocks (dataset x family), 3 strategies: baseline, A, B.
    # Block 1: baseline=1.0, A=1.5 (+0.5), B=0.8 (-0.2)
    # Block 2: baseline=2.0, A=2.5 (+0.5), B=1.7 (-0.3)
    rows = []
    for ds, base, a, b in [("DS01", 1.0, 1.5, 0.8), ("DS02", 2.0, 2.5, 1.7)]:
        rows += [
            {"dataset_id": ds, "model_family": "lm", "strategy": "baseline", "value": base},
            {"dataset_id": ds, "model_family": "lm", "strategy": "A", "value": a},
            {"dataset_id": ds, "model_family": "lm", "strategy": "B", "value": b},
        ]
    return pd.DataFrame(rows)


def test_tau_k_is_mean_paired_difference_vs_baseline():
    panel = _toy_panel()
    tau = tau_k(panel, dv="value")
    assert np.isclose(tau["A"], 0.5)
    assert np.isclose(tau["B"], (-0.2 + -0.3) / 2)


def _no_effect_panel(n_blocks=100, n_strategies=9, seed=0):
    # Every strategy (incl. baseline) drawn from the SAME distribution per
    # block -> the sharp null is exactly true, no strategy has a real effect.
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_blocks):
        block_level = rng.normal(0, 5)  # block-level noise, irrelevant to the null
        for s in ["baseline"] + [f"S{k}" for k in range(n_strategies)]:
            rows.append({"dataset_id": f"DS{b:03d}", "model_family": "lm",
                         "strategy": s, "value": block_level + rng.normal(0, 1)})
    return pd.DataFrame(rows)


def _large_effect_panel(n_blocks=20, seed=0):
    # A has a large, consistent +5 effect vs. baseline in every block.
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_blocks):
        base = rng.normal(0, 1)
        rows.append({"dataset_id": f"DS{b:03d}", "model_family": "lm", "strategy": "baseline", "value": base})
        rows.append({"dataset_id": f"DS{b:03d}", "model_family": "lm", "strategy": "A", "value": base + 5 + rng.normal(0, 0.1)})
    return pd.DataFrame(rows)


def test_ri_p_values_roughly_uniform_under_sharp_null():
    panel = _no_effect_panel()
    result = randomization_inference(panel, dv="value", n_perm=2000, seed=1)
    # No strategy has a real effect, so p-values should be spread out, not
    # clustered near 0. With 9 strategies at alpha=0.05 we'd expect ~0-1
    # false positives -- assert well under half falsely "significant".
    n_significant = (result["p_value"] < 0.05).sum()
    assert n_significant <= 2, f"too many false positives under the null: {n_significant}/9"
    assert result["p_value"].mean() > 0.2


def test_ri_p_value_near_zero_for_large_true_effect():
    panel = _large_effect_panel()
    result = randomization_inference(panel, dv="value", n_perm=2000, seed=1)
    assert result.loc["A", "p_value"] < 0.01
    assert np.isclose(result.loc["A", "tau"], 5.0, atol=0.5)


def test_permutation_stays_within_blocks_not_across():
    # Each row is one block's (baseline, S0, S1, ...) values. Permuting WITHIN
    # a block only reorders that row -> row sums are invariant. A shuffle
    # across the whole flattened array (the bug this guards against) would
    # move values between blocks and break this invariant with near
    # certainty on non-constant rows.
    rng = np.random.default_rng(0)
    values = rng.normal(size=(50, 10))  # 50 blocks x 10 strategies, all distinct
    permuted = permute_within_blocks(values, rng)
    assert permuted.shape == values.shape
    assert np.allclose(permuted.sum(axis=1), values.sum(axis=1)), \
        "row sums changed -- permutation leaked across block boundaries"
    # Actually permuted (not a no-op) for at least most rows.
    n_identical_rows = (permuted == values).all(axis=1).sum()
    assert n_identical_rows < values.shape[0] // 2


def _three_block_fixture():
    # 3 blocks, baseline + A + B. Hand-worked win counts for A vs baseline
    # (higher is better, F1-style): block1 A=0.6>base=0.5 WIN; block2
    # A=0.3<base=0.4 LOSS; block3 A=0.9>base=0.1 WIN -> win_prob(A)=2/3.
    rows = []
    for ds, base, a, b in [("DS01", 0.5, 0.6, 0.55), ("DS02", 0.4, 0.3, 0.2), ("DS03", 0.1, 0.9, 0.05)]:
        rows += [
            {"dataset_id": ds, "model_family": "lm", "strategy": "baseline", "value": base},
            {"dataset_id": ds, "model_family": "lm", "strategy": "A", "value": a},
            {"dataset_id": ds, "model_family": "lm", "strategy": "B", "value": b},
        ]
    return pd.DataFrame(rows)


def test_win_probability_matches_hand_count():
    panel = _three_block_fixture()
    wp = win_probability(panel, value_col="value", higher_is_better=True)
    assert np.isclose(wp["A"], 2 / 3)
    # B: block1 0.55<0.5? no wait 0.55>0.5 WIN; block2 0.2<0.4 LOSS; block3 0.05<0.1 LOSS -> 1/3
    assert np.isclose(wp["B"], 1 / 3)


def test_rank_shift_matches_cd_stats_average_ranks():
    panel = _three_block_fixture()
    shift = rank_shift(panel, value_col="value", higher_is_better=True)
    wide = panel.pivot_table(index=["dataset_id", "model_family"], columns="strategy", values="value")
    expected = average_ranks(wide, higher_is_better=True)
    for s in ["baseline", "A", "B"]:
        assert np.isclose(shift[s], expected[s])


def test_leave_one_source_out_excludes_only_that_sources_datasets():
    # 2 sources x 2 datasets each; source SA has a real +1 effect, SB has 0.
    rows = []
    for ds, effect in [("A1", 1.0), ("A2", 1.0)]:
        rows += [{"dataset_id": ds, "model_family": "lm", "strategy": "baseline", "value": 0.0},
                 {"dataset_id": ds, "model_family": "lm", "strategy": "X", "value": effect}]
    for ds, effect in [("B1", 0.0), ("B2", 0.0)]:
        rows += [{"dataset_id": ds, "model_family": "lm", "strategy": "baseline", "value": 0.0},
                 {"dataset_id": ds, "model_family": "lm", "strategy": "X", "value": effect}]
    panel = pd.DataFrame(rows)
    source_map = {"A1": "SA", "A2": "SA", "B1": "SB", "B2": "SB"}

    jk = leave_one_source_out(panel, dv="value", source_map=source_map)
    # Dropping SA leaves only SB's zero-effect blocks -> tau(X) should be 0.
    assert np.isclose(jk.loc["SA", "X"], 0.0)
    # Dropping SB leaves only SA's +1-effect blocks -> tau(X) should be 1.0.
    assert np.isclose(jk.loc["SB", "X"], 1.0)


def test_holm_adjust_matches_hand_worked_example():
    # Standard textbook Holm example, m=4: raw p = 0.01, 0.02, 0.03, 0.04
    # step 1: 4*0.01=0.04; step2: 3*0.02=0.06; step3: 2*0.03=0.06 (max w/ prev, tie);
    # step4: 1*0.04=0.04 -> cummax enforces monotonicity -> 0.06.
    raw = pd.Series({"s1": 0.01, "s2": 0.02, "s3": 0.03, "s4": 0.04})
    adj = holm_adjust(raw)
    assert np.isclose(adj["s1"], 0.04)
    assert np.isclose(adj["s2"], 0.06)
    assert np.isclose(adj["s3"], 0.06)
    assert np.isclose(adj["s4"], 0.06)


def test_bootstrap_ci_contains_true_effect_and_excludes_zero():
    panel = _large_effect_panel(n_blocks=30)
    ci = bootstrap_ci(panel, dv="value", n_boot=2000, seed=2)
    assert ci.loc["A", "ci_lo"] < ci.loc["A", "tau"] < ci.loc["A", "ci_hi"]
    # True effect is +5 with tiny noise -- a 95% CI over 30 blocks should
    # clear zero by a wide margin.
    assert ci.loc["A", "ci_lo"] > 1.0


def test_bootstrap_ci_is_wide_under_the_null():
    panel = _no_effect_panel(n_blocks=100)
    ci = bootstrap_ci(panel, dv="value", n_boot=2000, seed=2)
    # No true effect anywhere -- every strategy's CI should contain 0.
    n_excluding_zero = ((ci["ci_lo"] > 0) | (ci["ci_hi"] < 0)).sum()
    assert n_excluding_zero <= 1


if __name__ == "__main__":
    test_tau_k_is_mean_paired_difference_vs_baseline()
    test_ri_p_values_roughly_uniform_under_sharp_null()
    test_ri_p_value_near_zero_for_large_true_effect()
    test_permutation_stays_within_blocks_not_across()
    test_win_probability_matches_hand_count()
    test_rank_shift_matches_cd_stats_average_ranks()
    test_leave_one_source_out_excludes_only_that_sources_datasets()
    test_holm_adjust_matches_hand_worked_example()
    test_bootstrap_ci_contains_true_effect_and_excludes_zero()
    test_bootstrap_ci_is_wide_under_the_null()
    print("All causal_effects checks passed.")
