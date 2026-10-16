#!/usr/bin/env python3
"""Cross-check the region split against the pipeline's own published numbers,
and measure one discrepancy it exposes.

1. Recombining the two regions' per-fold sums of squared error must give back
   the whole-fold value behind raw_rmse_by_dataset/. This exercises the whole
   chain (relevance reconstruction, per-point error frame, fold pairing)
   against numbers the project already trusts. Asserted below.

2. SERA is computed here as sum(phi * e^2), the exact continuous-limit value,
   because only that form is additive over regions. The pipeline's SERA
   Metric instead integrates SER_t over 101 thresholds with the trapezoid
   rule. The gap between the two is measured per dataset and written out for
   the report's diagnostics appendix. It is quadrature error, not a
   disagreement about phi.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paths import ROOT  # noqa: E402

DATA = ROOT / "Results (Clean)/Results Data/raw_region_by_dataset"
RMSE_DIR = ROOT / "Results (Clean)/Results Data/raw_rmse_by_dataset"
SERA_DIR = ROOT / "Results (Clean)/Results Data/raw_sera_by_dataset"
TOL = 1e-12


def main() -> int:
    worst_rmse, n_cmp, sera_rows = 0.0, 0, []
    ids = sorted(p.stem for p in DATA.glob("DS??.csv"))
    if not ids:
        print("no region metrics on disk; run run_region_analysis.py first")
        return 1
    for ds in ids:
        fm = pd.read_csv(DATA / f"{ds}.csv")
        n_test = fm.groupby(["family", "strategy", "iteration"])["n"].sum().iloc[0]
        tot = (fm.groupby(["family", "strategy", "iteration"])
                 [["sse_strat", "sera_sum_strat"]].sum().reset_index())
        tot["rmse_mine"] = np.sqrt(tot["sse_strat"] / n_test)
        for ref_dir, col, mine in ((RMSE_DIR, "rmse", "rmse_mine"),
                                   (SERA_DIR, "sera", "sera_sum_strat")):
            ref = pd.read_csv(ref_dir / f"{ds}.csv")
            m = tot.merge(ref[["model_family", "strategy", "iteration", col]],
                          left_on=["family", "strategy", "iteration"],
                          right_on=["model_family", "strategy", "iteration"], how="inner")
            assert len(m) == len(tot), f"{ds}: {col} rows did not line up"
            rel = np.abs(m[mine] - m[col]) / np.maximum(np.abs(m[col]), 1e-300)
            if col == "rmse":
                worst_rmse = max(worst_rmse, float(rel.max())); n_cmp += len(m)
            else:
                sera_rows.append({"dataset": ds, "median_rel": float(rel.median()),
                                  "p95_rel": float(rel.quantile(0.95)),
                                  "max_rel": float(rel.max())})
        print(f"{ds}: ok", flush=True)

    print(f"\nRMSE: regions recombine to the published value, worst relative error "
          f"{worst_rmse:.2e} over {n_cmp} fold comparisons")
    assert worst_rmse < TOL, "regional SSE does not add back to the published RMSE"

    gap = pd.DataFrame(sera_rows)
    gap.to_csv(DATA / "sera_quadrature_gap.csv", index=False)
    bad = gap[gap.p95_rel > 0.05]
    print(f"SERA: trapezoid vs exact, median gap {gap.median_rel.median()*100:.2f}% "
          f"typical; {len(bad)} dataset(s) with a p95 above 5%"
          + (f" ({', '.join(bad.dataset)})" if len(bad) else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
