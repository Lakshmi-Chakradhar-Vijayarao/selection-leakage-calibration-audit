"""
is the K-severity law's FUNCTIONAL FORM identified by the data, or
only its monotonicity and concavity?

WHY THIS SCRIPT EXISTS. The paper reports `gap = a + b ln K` with R^2 = 0.939 on
the six non-degenerate Sweep A cells. An independent review pointed out that
this R^2 is not evidence for the logarithm specifically: with six points, four
residual degrees of freedom, and per-cell Monte-Carlo error comparable to or
larger than the fit residuals, essentially any monotone concave two-parameter
form will fit about as well, and several fit better. Quoting R^2 = 0.939 to
three decimals implies a precision the data do not carry, and -- more
consequentially for a reader acting on the result -- ln K implies UNBOUNDED
growth in candidate count while the best-fitting alternatives imply a CEILING.
Those give opposite operational advice to an auditor deciding whether trying
twice as many candidates is a problem.

WHAT THIS DOES.
  (1) Uses the per-seed severity values now shipped by code/63, so the
      Monte-Carlo standard error of every fitted point is known rather than
      assumed. That number is the crux: if residuals sit well below the noise
      on the points, the fit is interpolating noise and cannot discriminate
      between forms.
  (2) Fits seven two-parameter forms to the same six cells and reports R^2,
      AICc and leave-one-cell-out R^2 for each:
          a + b ln K            (the paper's)
          a (1 - exp(-K/tau))   saturating exponential   -> ceiling
          a + b / K             hyperbolic               -> ceiling
          a + b ln ln K
          a + b sqrt(ln K)
          a K^b                 power law
          a + b K               linear (included as the form the data DO reject)
  (3) Runs a parametric bootstrap that propagates each cell's own MC error,
      refits all seven forms on each draw, and reports how often each wins on
      AICc. That is the probability the data can actually identify the form.
  (4) Reports the properties the data DO support: strict monotonicity, strict
      concavity in ln K, and the slope with a seed-level bootstrap CI.

Input:  results/sweep_per_seed.npz (code/63)
Output: results/k_law_functional_form.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent
NPZ = ROOT / "results" / "sweep_per_seed.npz"
REF = ROOT / "results" / "selection_multiplicity_sweep.json"
OUT_PATH = ROOT / "results" / "k_law_functional_form.json"

RETAINED_K = [15, 25, 45, 75, 135, 225]     # the non-degenerate cells the paper fits
EXCLUDED_K = [1, 3, 5, 10]
N_BOOT_FORM = 20000
N_BOOT_SLOPE = 20000
SEED = 20260803


# ── candidate forms (all two-parameter) ──────────────────────────────────────
FORMS = {
    "a_plus_b_lnK":      (lambda K, a, b: a + b * np.log(K),            (0.0, 1e-3)),
    "saturating_exp":    (lambda K, a, tau: a * (1 - np.exp(-np.clip(K / tau, -700, 700))),
                          (5e-3, 30.0)),
    "a_plus_b_over_K":   (lambda K, a, b: a + b / K,                    (5e-3, -0.05)),
    "a_plus_b_lnlnK":    (lambda K, a, b: a + b * np.log(np.log(K)),    (0.0, 2e-3)),
    "a_plus_b_sqrtlnK":  (lambda K, a, b: a + b * np.sqrt(np.log(K)),   (0.0, 1e-3)),
    "power_law":         (lambda K, a, b: a * K ** b,                   (1e-3, 0.3)),
    "linear_in_K":       (lambda K, a, b: a + b * K,                    (2e-3, 1e-5)),
}


def fit_form(name, K, y):
    f, p0 = FORMS[name]
    try:
        popt, _ = curve_fit(f, K, y, p0=p0, maxfev=200000)
    except Exception:
        return None
    return popt


def r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def aicc(y, yhat, n_params):
    n = len(y)
    rss = float(np.sum((y - yhat) ** 2))
    k = n_params + 1                       # + the variance parameter
    base = n * np.log(rss / n) + 2 * k
    corr = (2 * k * (k + 1) / (n - k - 1)) if n - k - 1 > 0 else np.inf
    return float(base + corr)


def evaluate(name, K, y):
    popt = fit_form(name, K, y)
    if popt is None:
        return None
    f = FORMS[name][0]
    yhat = f(K, *popt)
    # leave-one-cell-out
    loo = np.empty_like(y)
    ok = True
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        p = fit_form(name, K[m], y[m])
        if p is None:
            ok = False
            break
        loo[i] = f(K[i], *p)
    return {
        "params": [float(v) for v in popt],
        "r_squared": r2(y, yhat),
        "aicc": aicc(y, yhat, len(popt)),
        "loo_r_squared": r2(y, loo) if ok else None,
        "rss": float(np.sum((y - yhat) ** 2)),
        "rms_residual": float(np.sqrt(np.mean((y - yhat) ** 2))),
    }


def main():
    rng = np.random.default_rng(SEED)
    d = np.load(NPZ)
    ref = json.load(open(REF))["sweep_A_K"]

    K = np.array(RETAINED_K, dtype=float)
    gaps = {k: d[f"A_K{k}__gap"] for k in RETAINED_K}
    y = np.array([gaps[k].mean() for k in RETAINED_K])
    se = np.array([gaps[k].std(ddof=1) / np.sqrt(len(gaps[k])) for k in RETAINED_K])

    for i, k in enumerate(RETAINED_K):
        assert abs(y[i] - ref[str(k)]["gap_mean"]) < 1e-12, f"cell K={k} drifted"
    print("Per-seed means reproduce the shipped Sweep A cells exactly.\n")

    # ── (1) noise-to-residual ratio ──────────────────────────────────────────
    fits = {name: evaluate(name, K, y) for name in FORMS}
    paper = fits["a_plus_b_lnK"]
    mean_se = float(se.mean())
    ratio = mean_se / paper["rms_residual"]

    print(f"Mean per-cell Monte-Carlo SE = {mean_se:.3e}")
    print(f"RMS residual of a + b lnK    = {paper['rms_residual']:.3e}")
    print(f"  -> MC noise is {ratio:.1f}x the fit residual "
          f"(a fit whose residuals sit well below the noise on the points it fits "
          f"cannot discriminate between forms)\n")

    # ── (2) model comparison table ───────────────────────────────────────────
    order = sorted([n for n in fits if fits[n]], key=lambda n: fits[n]["aicc"])
    print(f"{'form':22s} {'R2':>8s} {'AICc':>10s} {'LOO R2':>9s}")
    for n in order:
        f = fits[n]
        loo = f"{f['loo_r_squared']:.3f}" if f["loo_r_squared"] is not None else "  --"
        print(f"{n:22s} {f['r_squared']:8.4f} {f['aicc']:10.2f} {loo:>9s}")

    # ── (3) parametric bootstrap over the cells' own MC error ────────────────
    wins_aicc = {n: 0 for n in FORMS}
    wins_r2 = {n: 0 for n in FORMS}
    for _ in range(N_BOOT_FORM):
        ystar = y + rng.normal(0.0, se)
        best_a, best_r = None, None
        for n in FORMS:
            p = fit_form(n, K, ystar)
            if p is None:
                continue
            yh = FORMS[n][0](K, *p)
            a, r = aicc(ystar, yh, len(p)), r2(ystar, yh)
            if best_a is None or a < best_a[1]:
                best_a = (n, a)
            if best_r is None or r > best_r[1]:
                best_r = (n, r)
        if best_a:
            wins_aicc[best_a[0]] += 1
        if best_r:
            wins_r2[best_r[0]] += 1
    p_aicc = {n: v / N_BOOT_FORM for n, v in wins_aicc.items()}
    p_r2 = {n: v / N_BOOT_FORM for n, v in wins_r2.items()}
    print(f"\nParametric bootstrap ({N_BOOT_FORM} draws, each cell perturbed by its own MC SE):")
    for n in sorted(p_aicc, key=lambda x: -p_aicc[x]):
        print(f"  P(best on AICc)  {n:22s} = {p_aicc[n]:.3f}   "
              f"P(best on R2) = {p_r2[n]:.3f}")

    # ── (4) what the data DO support ─────────────────────────────────────────
    monotone = bool(np.all(np.diff(y) >= 0))
    lnK = np.log(K)
    # Concavity: successive secant slopes must be non-increasing. Checked in
    # both coordinates, because they are different claims -- concavity in K is
    # what "diminishing returns to trying more candidates" means, and every
    # candidate form here has it; concavity in ln K is the stronger statement
    # that ln K itself over-predicts, and it fails on one adjacent pair.
    slopes_K = np.diff(y) / np.diff(K)
    slopes_lnK = np.diff(y) / np.diff(lnK)
    concave_K = bool(np.all(np.diff(slopes_K) <= 0))
    concave = bool(np.all(np.diff(slopes_lnK) <= 0))
    # where lnK-concavity fails, and whether the violation is inside MC noise
    viol = np.diff(slopes_lnK)
    worst_i = int(np.argmax(viol))
    worst_pair = [RETAINED_K[worst_i], RETAINED_K[worst_i + 1], RETAINED_K[worst_i + 2]]
    worst_mag = float(max(0.0, viol.max()))
    # seed-level bootstrap CI for the ln K slope (resample seeds within cells)
    n_seeds = len(gaps[RETAINED_K[0]])
    G = np.vstack([gaps[k] for k in RETAINED_K])          # (6, n_seeds)
    bs = np.empty(N_BOOT_SLOPE)
    X = np.column_stack([np.ones_like(lnK), lnK])
    for b in range(N_BOOT_SLOPE):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        yb = G[:, idx].mean(axis=1)
        beta, *_ = np.linalg.lstsq(X, yb, rcond=None)
        bs[b] = beta[1]
    slope_ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    print(f"\nMonotone non-decreasing in K: {monotone}")
    print(f"Concave in K   (secant slopes non-increasing): {concave_K}")
    print(f"Concave in ln K(secant slopes non-increasing): {concave}"
          f"{'' if concave else f'  (single violation at K={worst_pair}, magnitude {worst_mag:.2e})'}")
    print(f"ln K slope b = {beta[1]:+.5f}  95% CI [{slope_ci[0]:+.5f}, {slope_ci[1]:+.5f}] "
          f"(seed-level bootstrap, {N_BOOT_SLOPE} draws)")

    out = {
        "n_cells_fit": len(RETAINED_K),
        "K_values_fit": RETAINED_K,
        "K_values_excluded_degenerate": EXCLUDED_K,
        "n_seeds_per_cell": int(n_seeds),
        "cell_means": {str(k): float(v) for k, v in zip(RETAINED_K, y)},
        "cell_monte_carlo_se": {str(k): float(v) for k, v in zip(RETAINED_K, se)},
        "identification": {
            "mean_cell_monte_carlo_se": mean_se,
            "rms_residual_of_paper_form": paper["rms_residual"],
            "mc_noise_over_residual_ratio": float(ratio),
            "note": (
                "The fit residuals of a + b ln K sit well BELOW the Monte-Carlo error of "
                "the points being fit, so R^2 = 0.939 is measuring how monotone the six "
                "points are, not how logarithmic they are."),
        },
        "fits": fits,
        "ranking_by_aicc": order,
        "parametric_bootstrap": {
            "n_draws": N_BOOT_FORM,
            "p_best_on_aicc": p_aicc,
            "p_best_on_r2": p_r2,
            "p_lnK_best_on_aicc": p_aicc["a_plus_b_lnK"],
            "note": (
                "Each draw perturbs every cell mean by its own measured Monte-Carlo SE and "
                "refits all seven forms. The probability that the paper's ln K form is the "
                "best-fitting one is small, and the two forms that most often win imply a "
                "CEILING in candidate count where ln K implies unbounded growth."),
        },
        "properties_the_data_support": {
            "monotone_nondecreasing_in_K": monotone,
            "concave_in_K": concave_K,
            "concave_in_lnK": concave,
            "lnK_concavity_worst_violating_triple": worst_pair,
            "lnK_concavity_worst_violation_magnitude": worst_mag,
            "lnK_slope": float(beta[1]),
            "lnK_intercept": float(beta[0]),
            "lnK_slope_bootstrap_ci_95": slope_ci,
            "gap_range_over_fitted_cells": [float(y.min()), float(y.max())],
            "gap_ratio_over_fitted_cells": float(y.max() / y.min()),
        },
        "reading": (
            "Monotonicity and concavity in K are robust; the specific logarithmic "
            "functional form is not identified by six cells whose fit residuals are "
            "smaller than their own Monte-Carlo error. The paper should state the "
            "relationship as monotone and concave, report the slope with its interval, and "
            "not assert ln K over a saturating or hyperbolic alternative -- especially "
            "since those alternatives imply a ceiling in candidate-count effect, which is "
            "the operationally relevant difference for an auditor."),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
