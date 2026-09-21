"""TDD seams: sera() must reduce exactly to sum of squared errors when
relevance phi(y)=1 uniformly (the paper's own stated special case) --
written before the implementation. ctrl_pts uses the same flat
(value, phi, derivative)-triple format R's uba::phi.control()$control.pts
captures, so real per-fold R output can be passed straight through.

test_uniform_relevance_equals_sum_squared_errors and
test_zero_for_perfect_predictions both use degenerate 2-point flat-phi=1
control sets, where PchipInterpolator and CubicHermiteSpline are
mathematically identical -- this is exactly why they never caught the
PCHIP-vs-Hermite bug (see test_phi_matches_paper_pct_rare below, and
docs/adr/0002-hermite-not-pchip.md in the sibling tsresample-kit repo).
"""
import json
import os

import numpy as np

from sera_metric import _phi, sera

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "phi_pct_rare_oracle.json"
)
DATASET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "paper_datasets_csv"
)


def test_uniform_relevance_equals_sum_squared_errors():
    y_true = np.array([1.0, 5.0, 10.0, 20.0, 100.0])
    y_pred = np.array([2.0, 4.0, 12.0, 18.0, 90.0])
    # A flat phi=1 line across the full value range -> phi(y)=1 for all y.
    ctrl_pts = [y_true.min(), 1, 0, y_true.max(), 1, 0]

    expected = np.sum((y_pred - y_true) ** 2)
    result = sera(y_true, y_pred, ctrl_pts, step=0.01)
    assert np.isclose(result, expected, rtol=1e-3), (result, expected)


def test_zero_for_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0])
    ctrl_pts = [1, 1, 0, 3, 1, 0]
    assert sera(y, y, ctrl_pts) == 0.0


def test_phi_matches_paper_pct_rare():
    """Independent-oracle regression test (TDD slice 1). Expected values
    come from Moniz, Branco & Torgo (2017) Table 1's %Rare column via a
    vendored fixture (fixtures/phi_pct_rare_oracle.json) -- never from
    this implementation's own output, which would be tautological. Control
    points are the r_oracle_median_control_x variant (R uba::phi.control,
    extremes method); phi/derivative at each of the 3 knots are always
    [1, 0, 1] / [0, 0, 0] by construction of that method.

    Before the fix (PchipInterpolator, ignoring ctrl_deriv): MAE ~2.99pp.
    After the fix (CubicHermiteSpline with recorded derivatives): MAE
    ~0.86pp, matching ADR-0002's independently-computed evidence table.
    """
    import csv

    with open(FIXTURE_PATH) as f:
        oracle = json.load(f)["datasets"]

    errors = []
    for dsid, entry in sorted(oracle.items()):
        lo, med, hi = entry["r_oracle_median_control_x"]
        ctrl_pts = [lo, 1, 0, med, 0, 0, hi, 1, 0]

        ds_path = None
        for fname in os.listdir(DATASET_DIR):
            if fname.startswith(dsid + "_"):
                ds_path = os.path.join(DATASET_DIR, fname)
                break
        assert ds_path is not None, f"no dataset file found for {dsid}"

        with open(ds_path, newline="") as f:
            reader = csv.DictReader(f)
            y = np.array(
                [float(row["target"]) for row in reader if row["target"] != "NA"]
            )

        pct_rare = 100.0 * np.mean(_phi(y, ctrl_pts) > 0.9)
        errors.append(abs(pct_rare - entry["paper_pct_rare"]))

    mae = np.mean(errors)
    assert mae <= 1.5, f"phi %Rare MAE vs. paper Table 1 too high: {mae:.3f}pp"


def test_phi_is_flat_one_outside_control_range():
    """Extension, not extrapolation: phi(y)=1 strictly outside [lo, hi] --
    the other half of the PCHIP defect (PCHIP extrapolates the cubic then
    clips; the recorded R phi is flat by construction)."""
    ctrl_pts = [-10, 1, 0, 0, 0, 0, 10, 1, 0]
    y = np.array([-1000.0, -10.0001, 10.0001, 1000.0])
    assert np.all(_phi(y, ctrl_pts) == 1.0)


if __name__ == "__main__":
    test_uniform_relevance_equals_sum_squared_errors()
    test_zero_for_perfect_predictions()
    test_phi_matches_paper_pct_rare()
    test_phi_is_flat_one_outside_control_range()
    print("All SERA tests passed.")
