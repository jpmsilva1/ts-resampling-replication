"""TDD seam 4: the SERA table generator, run end-to-end against a tiny
synthetic prediction + phi_ctrl CSV fixture (same folder layout the real
rerun's CSV extraction produces). Written before the implementation.
"""
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from generate_sera_tables import build_family_table


def make_fixture(tmp_dir: Path):
    ds_dir = tmp_dir / "DS01"
    ds_dir.mkdir(parents=True)
    # Flat phi=1 control points (two points spanning the value range) for
    # both workflows, so SERA reduces to plain sum-of-squared-errors --
    # keeps the fixture's expected numbers easy to hand-verify.
    ctrl = pd.DataFrame({
        "iteration": [1, 1], "point_index": [1, 2],
        "ctrl_x": [10.0, 20.0], "ctrl_phi": [1.0, 1.0], "ctrl_deriv": [0.0, 0.0],
    })

    # mc.lm baseline: errors of 1 and 1 -> sum sq errors = 2 -> SERA = 2
    pd.DataFrame({
        "iteration": [1, 1], "obs_index": [1, 2],
        "y_true": [10.0, 20.0], "y_pred": [11.0, 21.0],
    }).to_csv(ds_dir / "mc.lm_predictions.csv", index=False)
    ctrl.to_csv(ds_dir / "mc.lm_phi_ctrl.csv", index=False)

    # mc.lm_OVERB: perfect predictions -> SERA = 0 (should be bolded)
    pd.DataFrame({
        "iteration": [1, 1], "obs_index": [1, 2],
        "y_true": [10.0, 20.0], "y_pred": [10.0, 20.0],
    }).to_csv(ds_dir / "mc.lm_OVERB_predictions.csv", index=False)
    ctrl.to_csv(ds_dir / "mc.lm_OVERB_phi_ctrl.csv", index=False)


def test_bold_marks_lowest_sera_not_highest():
    tmp = Path(tempfile.mkdtemp())
    try:
        make_fixture(tmp)
        tex = build_family_table(
            data_dir=tmp, family="lm",
            ds_labels={"DS01": "DS01 (Test)"},
            strategies=["baseline", "OVERB"],
        )
        assert r"\textbf{0.0000}" in tex
        assert r"\textbf{2.0000}" not in tex
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_bold_marks_lowest_sera_not_highest()
    print("All SERA table-generator tests passed.")
