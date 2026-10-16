"""Relevance-region decomposition: score the same predictions on the rare
points and on the normal points separately.

Every test point is labelled by the relevance of its TRUE value,

    rare   = phi(y) >= 0.9        normal = phi(y) < 0.9

and MAE, RMSE and SERA are computed inside each region, per Monte Carlo fold,
for both the strategy and its own un-resampled baseline. The method follows
Leitao et al.'s Table V: hold the predictions fixed, change only the set of
instances being scored, then re-run the paper's own paired comparison.

Three properties the analysis rests on, all covered by test_region_metrics.py:

1. The label is a function of y alone, never of a prediction, so a fold's rare
   set is identical for a strategy and its baseline. The pairing is exact and
   no strategy can move a point between regions to flatter itself.
2. Sums and means give the same paired test, since within a fold they differ
   by the constant n_region on both sides. Means are reported because they are
   comparable across datasets; the SERA sums are what add back to the whole
   fold.
3. MSE and RMSE give the same paired test, because the square root is
   monotone. Only RMSE is reported.

F_phi is absent by construction. Utility-based precision and recall are
properties of a set, not of a point: restricted to the rare region, recall is
unchanged but precision loses the normal points that were falsely predicted
rare, and restricted to the normal region there are no true-rare cases at all,
so recall is 0/0. Leitao et al. report only MAE and MSE for the same reason.

phi is reconstructed from the per-fold control points the R pipeline recorded,
through the same sera_metric._phi the SERA tables use, so the relevance
function here is the one the rest of the project already applies.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "SERA Metric"))
sys.path.insert(0, str(_HERE.parent))

from sera_metric import _phi  # noqa: E402

T_R = 0.9          # rare threshold: Exps.R's util.control(event.thr=0.9) and the resamplers' thr.rel
REGIONS = ["rare", "normal"]
METRICS = {"mae": "MAE", "rmse": "RMSE", "sera": "SERA"}
KEYS = ["iteration", "obs_index"]


# ------------------------------------------------------------------ loading ---

def build_fold_cache(ctrl: pd.DataFrame) -> dict:
    """The relevance control points of each Monte Carlo fold, keyed by
    iteration. Built once per (dataset, family) and reused for every strategy:
    phi.control() is fit on the pre-resampling training target, so a family's
    baseline and all nine of its strategies share one control-point set per
    fold."""
    return {it: g[["ctrl_x", "ctrl_phi", "ctrl_deriv"]].to_numpy()
            for it, g in ctrl.groupby("iteration")}


def phi_of(values: np.ndarray, iterations: np.ndarray, cache: dict) -> np.ndarray:
    out = np.empty(len(values), dtype=float)
    for it in np.unique(iterations):
        idx = np.flatnonzero(iterations == it)
        out[idx] = _phi(values[idx], cache[it])
    return out


def load_baseline(data_dir: Path, ds: str, family: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    stem = f"mc.{family}"
    return (pd.read_csv(data_dir / ds / f"{stem}_predictions.csv"),
            pd.read_csv(data_dir / ds / f"{stem}_phi_ctrl.csv"))


def load_cell(data_dir: Path, ds: str, family: str, strategy: str,
              base_preds: pd.DataFrame, cache: dict) -> pd.DataFrame:
    """One row per test point: the strategy's error, its baseline's error on
    the same point, and phi(y). Monte Carlo folds are identical across a
    family's workflows, so the merge must keep every row; asserted, not
    assumed."""
    strat = pd.read_csv(data_dir / ds / f"mc.{family}_{strategy}_predictions.csv")
    merged = strat.merge(base_preds[KEYS + ["y_pred"]].rename(columns={"y_pred": "y_pred_base"}),
                         on=KEYS, how="inner")
    assert len(merged) == len(strat), f"{ds}/{family}/{strategy}: folds differ from the baseline"
    merged["phi_y"] = phi_of(merged["y_true"].to_numpy(), merged["iteration"].to_numpy(), cache)
    merged["e"] = merged["y_pred"] - merged["y_true"]
    merged["e_base"] = merged["y_pred_base"] - merged["y_true"]
    merged["e2"] = merged["e"] ** 2
    merged["e2_base"] = merged["e_base"] ** 2
    return merged


# ------------------------------------------------------------------ metrics ---

def region_fold_metrics(cell: pd.DataFrame, phi_thr: float = T_R) -> pd.DataFrame:
    """One row per (iteration, region): the point count, and the baseline and
    strategy MAE, RMSE and SERA inside that region. A fold with an empty
    region gets n = 0 and NaN metrics; callers drop it from both sides at once
    (region_stats.paired_vectors)."""
    rare = (cell["phi_y"] >= phi_thr).to_numpy()
    it_vals, inv = np.unique(cell["iteration"].to_numpy(), return_inverse=True)
    phi = cell["phi_y"].to_numpy()
    cols = {
        "sae_base": np.abs(cell["e_base"].to_numpy()), "sae_strat": np.abs(cell["e"].to_numpy()),
        "sse_base": cell["e2_base"].to_numpy(), "sse_strat": cell["e2"].to_numpy(),
        "sera_base": phi * cell["e2_base"].to_numpy(), "sera_strat": phi * cell["e2"].to_numpy(),
    }
    out = []
    for name, mask in (("rare", rare), ("normal", ~rare)):
        w = mask.astype(float)
        tot = {k: np.bincount(inv, weights=w * v, minlength=len(it_vals)) for k, v in cols.items()}
        n = np.bincount(inv, weights=w, minlength=len(it_vals))
        safe_n = np.where(n > 0, n, np.nan)
        out.append(pd.DataFrame({
            "iteration": it_vals, "region": name, "n": n.astype(int),
            "mae_base": tot["sae_base"] / safe_n, "mae_strat": tot["sae_strat"] / safe_n,
            "rmse_base": np.sqrt(tot["sse_base"] / safe_n),
            "rmse_strat": np.sqrt(tot["sse_strat"] / safe_n),
            "sera_base": tot["sera_base"] / safe_n, "sera_strat": tot["sera_strat"] / safe_n,
            # retained for verify_region.py's additivity check against the published RMSE
            "sse_base": tot["sse_base"], "sse_strat": tot["sse_strat"],
            "sera_sum_base": tot["sera_base"], "sera_sum_strat": tot["sera_strat"],
        }))
    return pd.concat(out, ignore_index=True)


def error_quantiles(cell: pd.DataFrame, phi_thr: float = T_R,
                    qs=(25, 50, 75, 90, 99)) -> dict:
    """Quantiles of the absolute error inside the rare region, strategy over
    baseline. This separates two things a single average cannot: whether a
    strategy improves the typical rare case or the worst one. Pooled over the
    cell's folds, so it describes the error distribution rather than anything
    the fold-level tests rank."""
    r = cell[cell["phi_y"] >= phi_thr]
    if len(r) < 50:
        return {}
    a, b = r["e"].abs().to_numpy(), r["e_base"].abs().to_numpy()
    out = {f"q{q}": float(np.quantile(a, q / 100) / np.quantile(b, q / 100)) for q in qs}
    out["max"] = float(a.max() / b.max())
    return out


def rare_diagnostics(cell: pd.DataFrame, phi_thr: float = T_R) -> dict:
    """How much rare region there is to score on, per dataset."""
    rare = cell["phi_y"] >= phi_thr
    per_fold = rare.groupby(cell["iteration"]).sum()
    return {
        "pct_rare": 100.0 * float(rare.mean()), "n_folds": int(per_fold.size),
        "folds_empty_rare": int((per_fold == 0).sum()),
        "min_rare": int(per_fold.min()), "max_rare": int(per_fold.max()),
        "n_test_per_fold": int(len(cell) // per_fold.size),
    }


__all__ = ["T_R", "REGIONS", "METRICS", "KEYS", "build_fold_cache", "phi_of", "load_baseline",
           "load_cell", "region_fold_metrics", "error_quantiles", "rare_diagnostics"]
