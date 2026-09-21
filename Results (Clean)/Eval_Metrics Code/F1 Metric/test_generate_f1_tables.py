"""TDD: build_table() now takes `agg` as a parameter (was a module-level
global) so it can be tested against a small synthetic aggregate instead of
the real 20-dataset F1 CSVs.
"""
import pandas as pd

from generate_f1_tables import build_table
from tables_common import DS_LABELS, STRATEGIES


def _synthetic_agg(best_strategy: str) -> pd.DataFrame:
    rows = [{"dataset_id": ds_id, "model_family": "lm", "strategy": strategy,
              "mean": 0.9 if strategy == best_strategy else 0.5, "sd": 0.01}
            for ds_id in DS_LABELS for strategy in STRATEGIES]
    return pd.DataFrame(rows)


def test_bold_lands_on_max_per_dataset_row():
    agg = _synthetic_agg("OVERT")
    tex = build_table(agg, "lm", "Linear Model (LM)")
    # every dataset row plus the Overall Average row bolds the winning strategy
    assert tex.count(r"\textbf{0.9000}") == len(DS_LABELS) + 1


def test_overall_average_row_present():
    agg = _synthetic_agg("SMOTEB")
    tex = build_table(agg, "lm", "Linear Model (LM)")
    assert r"\textbf{Overall Average}" in tex


def test_missing_strategy_data_does_not_silently_render_nan():
    agg = _synthetic_agg("baseline")
    tex = build_table(agg, "lm", "Linear Model (LM)")
    assert "nan" not in tex.lower()


if __name__ == "__main__":
    test_bold_lands_on_max_per_dataset_row()
    test_overall_average_row_present()
    test_missing_strategy_data_does_not_silently_render_nan()
    print("All F1 table-generator tests passed.")
