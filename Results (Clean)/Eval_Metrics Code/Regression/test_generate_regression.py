"""Shape guards on the regression panel.

Rewritten 2026-09-04: these were module-level asserts hardcoding 20 datasets,
which broke test collection outright once DS21 landed (1050 != 20*5*10). The
panel must be *complete* -- one row per (dataset, family, strategy) with no
gaps -- but the dataset count is whatever is on disk, so it is derived rather
than asserted against a constant.
"""
from generate_regression import build_panel

FAMILIES = 5
STRATEGIES = 10  # baseline + 9 resampling strategies


def test_panel_is_complete_and_well_formed():
    for metric in ("F1", "SERA"):
        panel = build_panel(metric)
        assert set(panel.columns) == {"dataset_id", "model_family", "strategy", "value"}

        n_datasets = panel["dataset_id"].nunique()
        assert 20 <= n_datasets <= 24, f"{metric}: unexpected dataset count {n_datasets}"
        # Fully crossed design: every dataset must carry every family x strategy
        # cell, so a partial extraction shows up here rather than silently
        # skewing a mean downstream.
        assert len(panel) == n_datasets * FAMILIES * STRATEGIES, (
            f"{metric}: panel has {len(panel)} rows, expected "
            f"{n_datasets}x{FAMILIES}x{STRATEGIES}={n_datasets*FAMILIES*STRATEGIES}"
        )
        assert panel["model_family"].nunique() == FAMILIES
        assert panel["strategy"].nunique() == STRATEGIES
        assert panel["value"].notna().all(), f"{metric}: NaN values in panel"
        assert panel.duplicated(["dataset_id", "model_family", "strategy"]).sum() == 0
