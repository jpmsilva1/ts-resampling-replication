"""TDD seam 4: the RMSE table generator, run end-to-end against a tiny
synthetic prediction-CSV fixture (same folder layout the real rerun's CSV
extraction produces: <out>/<DSxx>/<workflow>_predictions.csv). Written
before the implementation.
"""
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from generate_rmse_tables import build_family_table, load_predictions


def make_fixture(tmp_dir: Path):
    ds_dir = tmp_dir / "DS01"
    ds_dir.mkdir(parents=True)
    # mc.lm baseline: errors of 1 and 1 -> RMSE = 1
    pd.DataFrame({
        "iteration": [1, 1], "obs_index": [1, 2],
        "y_true": [10.0, 20.0], "y_pred": [11.0, 21.0],
    }).to_csv(ds_dir / "mc.lm_predictions.csv", index=False)
    # mc.lm_OVERB: errors of 0 -> RMSE = 0 (should be bolded, lower=better)
    pd.DataFrame({
        "iteration": [1, 1], "obs_index": [1, 2],
        "y_true": [10.0, 20.0], "y_pred": [10.0, 20.0],
    }).to_csv(ds_dir / "mc.lm_OVERB_predictions.csv", index=False)


def test_load_predictions_reads_csv():
    tmp = Path(tempfile.mkdtemp())
    try:
        make_fixture(tmp)
        df = load_predictions(tmp / "DS01" / "mc.lm_predictions.csv")
        assert list(df["y_true"]) == [10.0, 20.0]
    finally:
        shutil.rmtree(tmp)


def test_bold_marks_lowest_rmse_not_highest():
    tmp = Path(tempfile.mkdtemp())
    try:
        make_fixture(tmp)
        tex = build_family_table(
            data_dir=tmp, family="lm",
            ds_labels={"DS01": "DS01 (Test)"},
            strategies=["baseline", "OVERB"],
        )
        # OVERB has RMSE 0.0000, baseline has RMSE 1.0000 -> OVERB must be bold.
        assert r"\textbf{0.0000}" in tex
        assert r"\textbf{1.0000}" not in tex
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_load_predictions_reads_csv()
    test_bold_marks_lowest_rmse_not_highest()
    print("All RMSE table-generator tests passed.")
