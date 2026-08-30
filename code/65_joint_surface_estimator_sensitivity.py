"""
how much of the multiplicative surface's candidate-count exponent `b`
is a property of the data, and how much of the ESTIMATOR used to fit it?

WHY THIS SCRIPT EXISTS. `code/57` fits the multiplicative form
gap = exp(a + b lnK + c z0) by Gauss-Newton least squares ON THE RAW GAPS. That
is the right choice for the paper's purpose: it minimizes squared error in the
same units as the additive model, so M5's R^2 = 0.976 is commensurable with
M3's 0.688 at equal parameter count. The paper then reports b's stability by
deleting each K column in turn (b = 0.09 to 0.21) and calls it "the least stable
coefficient in the fit."

An independent review pointed out that this understates the instability, because
the same functional form fitted by the OTHER standard estimator -- ordinary
least squares in log space, i.e. regressing ln(gap) on (1, lnK, z0) -- lands
outside that entire range. Log-space OLS and raw-space NLS fit the same curve
family but minimize different loss functions (relative vs absolute error), and
with cell means spanning two orders of magnitude (+0.00012 to +0.01100) they
weight the grid very differently: log-space OLS gives the tiny high-AUROC cells
the same leverage as the large low-AUROC ones.

WHAT THIS REPORTS. Both estimators, on all 20 cells and under every
single-K-column deletion, so the paper can state plainly which coefficient is
robust to what:

  * b  (candidate-count exponent)  -- moves MORE under estimator choice than
        under column deletion. This is a fitted-parameter stability caveat.
  * c  (operating-point coefficient) -- checked under both estimators and all
        deletions, since the paper's §5.3-5.4 claims lean on it, not on b.

This is a disclosure, not a correction: no number in the paper is wrong, and
raw-space NLS remains the reported fit for the stated reason. What changes is
that the quoted 0.09-0.21 range is now labelled as the column-deletion range
under one estimator rather than as the range of b.

Input: results/joint_severity_surface.json (cell means only -- no re-simulation).
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "results" / "joint_severity_surface.json"
OUT_PATH = ROOT / "results" / "joint_surface_estimator_sensitivity.json"


def r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_nls(lnK, z0, y):
    """gap = exp(a + b lnK + c z0), Gauss-Newton on raw gaps. Verbatim logic
    from code/57.fit_loglinear (re-implemented here so this script is
    self-contained; agreement with the shipped coefficients is asserted)."""
    pos = y > 1e-6
    X = np.column_stack([np.ones_like(lnK), lnK, z0])
    b, *_ = np.linalg.lstsq(X[pos], np.log(y[pos]), rcond=None)
    for _ in range(200):
        f = np.exp(X @ b)
        J = f[:, None] * X
        try:
            step, *_ = np.linalg.lstsq(J, y - f, rcond=None)
        except np.linalg.LinAlgError:
            break
        b = b + step
        if np.max(np.abs(step)) < 1e-12:
            break
    return b, r2(y, np.exp(X @ b))


def fit_log_ols(lnK, z0, y):
    """The identical functional form fitted by ordinary least squares on
    ln(gap): minimizes RELATIVE rather than absolute error. Only strictly
    positive cells can enter (ln is undefined at 0); on this grid all 20 are
    positive."""
    pos = y > 0
    X = np.column_stack([np.ones_like(lnK), lnK, z0])
    b, *_ = np.linalg.lstsq(X[pos], np.log(y[pos]), rcond=None)
    yhat = np.exp(X @ b)
    return b, r2(y, yhat), r2(np.log(y[pos]), (X[pos] @ b)), int(pos.sum())


def main():
    d = json.load(open(IN_PATH))
    cells = d["cells"]
    keys = list(cells.keys())
    K = np.array([cells[k]["K"] for k in keys], dtype=float)
    a0 = np.array([cells[k]["target_auroc"] for k in keys], dtype=float)
    y = np.array([cells[k]["gap_mean"] for k in keys], dtype=float)
    from scipy.stats import norm
    lnK, z0 = np.log(K), norm.ppf(a0)
    assert len(y) == 20 and (y > 0).all()

    # Guard: our NLS re-implementation must reproduce code/57's shipped fit.
    b_nls_all, r2_nls_all = fit_nls(lnK, z0, y)
    shipped = d["fits"]["M5_loglinear"]["coefficients"]
    drift = float(np.max(np.abs(np.array(shipped) - b_nls_all)))
    assert drift < 1e-8, f"NLS re-implementation drifted from code/57 by {drift:.2e}"
    print(f"NLS re-implementation reproduces code/57's M5 coefficients (max |d|={drift:.1e})\n")

    b_ols_all, r2_ols_raw, r2_ols_log, n_pos = fit_log_ols(lnK, z0, y)

    rows = {}
    for label, mask in ([("all_20_cells", np.ones(len(y), bool))] +
                        [(f"drop_K{int(k)}", K != k) for k in sorted(set(K))]):
        bn, rn = fit_nls(lnK[mask], z0[mask], y[mask])
        bo, ro_raw, ro_log, _ = fit_log_ols(lnK[mask], z0[mask], y[mask])
        rows[label] = {
            "n_cells": int(mask.sum()),
            "nls_b": float(bn[1]), "nls_c": float(bn[2]), "nls_r2_raw": float(rn),
            "log_ols_b": float(bo[1]), "log_ols_c": float(bo[2]),
            "log_ols_r2_raw": float(ro_raw), "log_ols_r2_logspace": float(ro_log),
        }
        print(f"  {label:14s} n={int(mask.sum()):2d}   "
              f"NLS b={bn[1]:+.4f} c={bn[2]:+.4f} (R2raw={rn:.3f})   "
              f"logOLS b={bo[1]:+.4f} c={bo[2]:+.4f} (R2raw={ro_raw:.3f})")

    nls_b = [v["nls_b"] for v in rows.values()]
    ols_b = [v["log_ols_b"] for v in rows.values()]
    nls_c = [v["nls_c"] for v in rows.values()]
    ols_c = [v["log_ols_c"] for v in rows.values()]
    all_b, all_c = nls_b + ols_b, nls_c + ols_c

    # sensitivity attribution, measured on the all-cells fit
    est_shift_b = abs(rows["all_20_cells"]["log_ols_b"] - rows["all_20_cells"]["nls_b"])
    del_span_b = max(nls_b) - min(nls_b)
    est_shift_c = abs(rows["all_20_cells"]["log_ols_c"] - rows["all_20_cells"]["nls_c"])
    del_span_c = max(nls_c) - min(nls_c)

    out = {
        "n_cells": 20,
        "by_subset": rows,
        "summary": {
            "b_nls_all_cells": rows["all_20_cells"]["nls_b"],
            "b_log_ols_all_cells": rows["all_20_cells"]["log_ols_b"],
            "b_nls_column_deletion_range": [float(min(nls_b[1:])), float(max(nls_b[1:]))],
            "b_log_ols_column_deletion_range": [float(min(ols_b[1:])), float(max(ols_b[1:]))],
            "b_range_over_both_estimators_and_all_deletions": [float(min(all_b)), float(max(all_b))],
            "b_estimator_shift_all_cells": float(est_shift_b),
            "b_column_deletion_span_within_nls": float(del_span_b),
            "b_estimator_shift_exceeds_deletion_span": bool(est_shift_b > del_span_b),
            "b_log_ols_outside_nls_deletion_range": bool(
                rows["all_20_cells"]["log_ols_b"] > max(nls_b[1:])
                or rows["all_20_cells"]["log_ols_b"] < min(nls_b[1:])),
            "c_nls_all_cells": rows["all_20_cells"]["nls_c"],
            "c_log_ols_all_cells": rows["all_20_cells"]["log_ols_c"],
            "c_range_over_both_estimators_and_all_deletions": [float(min(all_c)), float(max(all_c))],
            "c_estimator_shift_all_cells": float(est_shift_c),
            "c_column_deletion_span_within_nls": float(del_span_c),
            "c_relative_span_over_both_estimators": float(
                (max(all_c) - min(all_c)) / abs(np.mean(all_c))),
            "b_relative_span_over_both_estimators": float(
                (max(all_b) - min(all_b)) / abs(np.mean(all_b))),
        },
        "reading": (
            "The candidate-count exponent b moves further under a change of ESTIMATOR "
            "(raw-space Gauss-Newton NLS vs ordinary log-space OLS, same functional form, "
            "same 20 cells) than it does under deleting any single K column, and the "
            "log-OLS value falls outside the column-deletion range the paper quotes. The "
            "operating-point coefficient c is by contrast stable under both estimators and "
            "all deletions. This is a fitted-parameter stability caveat on b, not a "
            "validity failure of the surface: the two estimators minimize absolute and "
            "relative error respectively, and on a grid whose cell means span roughly two "
            "orders of magnitude they weight it very differently. Raw-space NLS remains the "
            "reported fit because its R^2 is then commensurable with the additive model's "
            "at equal parameter count. b is not a quantity this paper's conclusions rest "
            "on; c is, and c survives."),
    }

    print(f"\nb: NLS(all)={out['summary']['b_nls_all_cells']:+.4f}  "
          f"logOLS(all)={out['summary']['b_log_ols_all_cells']:+.4f}  "
          f"estimator shift={est_shift_b:.4f} vs NLS column-deletion span={del_span_b:.4f}")
    print(f"c: NLS(all)={out['summary']['c_nls_all_cells']:+.4f}  "
          f"logOLS(all)={out['summary']['c_log_ols_all_cells']:+.4f}  "
          f"full range over both estimators + all deletions = "
          f"[{min(all_c):+.4f}, {max(all_c):+.4f}] "
          f"({100 * out['summary']['c_relative_span_over_both_estimators']:.1f}% of its mean, "
          f"against {100 * out['summary']['b_relative_span_over_both_estimators']:.1f}% for b)")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
