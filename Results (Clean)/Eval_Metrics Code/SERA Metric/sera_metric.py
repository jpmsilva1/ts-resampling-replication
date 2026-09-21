import numpy as np
from scipy.interpolate import CubicHermiteSpline


def _phi(y, ctrl_pts):
    """Reconstructs R's uba::phi() relevance function from phi.control()'s
    control points. ctrl_pts is a flat (value, phi, derivative)-triple
    sequence -- the exact shape R's phi.control()$control.pts captures.
    Uses the recorded derivative column directly (CubicHermiteSpline),
    not PchipInterpolator, which discards it and instead fits its own
    nonzero derivatives at the outer knots -- verified against Moniz,
    Branco & Torgo (2017) Table 1 %Rare (see test_phi_matches_paper_pct_rare
    and docs/adr/0002-hermite-not-pchip.md in the sibling tsresample-kit
    repo). Outside [lo, hi], phi is extended flat at 1.0, not extrapolated."""
    pts = np.asarray(ctrl_pts, dtype=float).reshape(-1, 3)
    lo, hi = pts[0, 0], pts[-1, 0]
    spline = CubicHermiteSpline(x=pts[:, 0], y=pts[:, 1], dydx=pts[:, 2])
    inside = np.clip(spline(np.clip(y, lo, hi)), 0.0, 1.0)
    return np.where((y <= lo) | (y >= hi), 1.0, inside)


def sera(y_true, y_pred, ctrl_pts, step=0.01):
    """Squared Error-Relevance Area (Ribeiro & Moniz 2020): area under the
    SER_t curve, SER_t = sum of squared errors for cases with phi(y)>=t,
    integrated over t via the trapezoidal rule."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    phi_true = _phi(y_true, ctrl_pts)
    squared_errors = (y_pred - y_true) ** 2

    thresholds = np.arange(0, 1 + step, step)
    ser_t = np.array([squared_errors[phi_true >= t].sum() for t in thresholds])
    return np.trapezoid(ser_t, thresholds)
