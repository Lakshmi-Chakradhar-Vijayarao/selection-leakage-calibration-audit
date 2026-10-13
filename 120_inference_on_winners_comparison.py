"""
Inference on winners vs. the direct held-out estimate.  [Limitations; Sec. 4.5]

The paper concedes that "where exact conditional selective inference is
available it is stronger than what we propose, and we do not compare against
it empirically." This script makes the comparison, against the method built
for exactly this problem:

    Andrews, I., Kitagawa, T., & McCloskey, A. (2024). Inference on Winners.
    The Quarterly Journal of Economics, 139(1), 305-358.

Their setting is ours. Candidate values X ~ N(mu, Sigma) are estimated, the
winner j = argmax X is selected, and the target is mu_j, the winner's true
value. They give a conditional interval (exact coverage given that j won) and
a hybrid interval (conditional on j winning AND mu_j lying in a level-beta
projection interval) that stays short when the winner barely wins. Both use
the selection data alone.

The audit has something those methods are designed to do without: a held-out
estimate of the selected candidate, ho_j = mu_j + independent noise. The
question is what AKM's machinery buys an auditor who has it.

SETUP (one replicate = one audit, matching 113's notation):
    q_l ~ N(0, tau^2)                     true quality of candidate l
    cv  = q + e_cv,  e_cv ~ N(0, s^2 R)   selection estimates, correlation R
    ho  = q + e_ho,  e_ho ~ N(0, s^2 R)   held-out estimates, independent of cv
    j   = argmax cv;  target q_j

R is independent, equicorrelated (0.75) or banded AR(1) (0.957 adjacent), the
last two being the main harness's measured mean and adjacent correlations;
Sigma = s^2 R is treated as known, as AKM assume.

ESTIMATORS of q_j, each with a nominal 95% interval:
  naive        cv_j +/- 1.96 s                    (ignores selection)
  AKM-cond     conditional median-unbiased estimate and equal-tailed interval
  AKM-hybrid   hybrid estimate and interval, beta = alpha/10 as AKM recommend
  direct       ho_j +/- 1.96 s                    (uses the held-out replicate)

The direct estimator uses a second, equal-sized sample. That is the point, not
an unfairness: an audit always has one, and AKM exists for settings that do
not.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "results" / "inference_on_winners_comparison.json"

ALPHA, BETA = 0.05, 0.005
S = 0.03
N_AUDITS = 1000
K_GRID = [8, 32]
TAU_GRID = [0.0, 0.010, 0.030]
CORRS = ["independent", "equicorr_0.75", "banded_ar1_0.957"]
SEED = 20261011
Z = stats.norm.ppf(1 - ALPHA / 2)


def corr_matrix(kind, K):
    if kind == "independent":
        return np.eye(K)
    if kind == "equicorr_0.75":
        return 0.75 * np.ones((K, K)) + 0.25 * np.eye(K)
    if kind == "banded_ar1_0.957":
        i = np.arange(K)
        return 0.957 ** np.abs(i[:, None] - i[None, :])
    raise ValueError(kind)


def tn_cdf(x, mu, sd, lo, hi):
    """CDF at x of N(mu, sd^2) truncated to [lo, hi], computed stably."""
    a, b = (lo - mu) / sd, (hi - mu) / sd
    return float(stats.truncnorm.cdf(x, a, b, loc=mu, scale=sd))


def solve_mu(f, target, lo, hi):
    """Find mu with f(mu) = target for f decreasing in mu, on [lo, hi]."""
    flo, fhi = f(lo) - target, f(hi) - target
    if flo < 0:
        return lo
    if fhi > 0:
        return hi
    return brentq(lambda m: f(m) - target, lo, hi, xtol=1e-7)


def truncation(x, Sigma, j):
    """AKM / Lee et al. truncation set for X_j given the winner is j and the
    statistic orthogonal to X_j: X_i = Z_i + c_i X_j with c_i = S_ij / S_jj."""
    c = Sigma[:, j] / Sigma[j, j]
    Zv = x - c * x[j]
    lo, hi = -np.inf, np.inf
    for i in range(len(x)):
        if i == j:
            continue
        if c[i] < 1 - 1e-12:
            lo = max(lo, Zv[i] / (1 - c[i]))
        elif c[i] > 1 + 1e-12:
            hi = min(hi, Zv[i] / (1 - c[i]))
    return lo, hi


def projection_c(R, rng, n=200_000):
    """c_beta: (1-beta) quantile of max_i |xi_i| / sd_i, xi ~ N(0, R)."""
    L = np.linalg.cholesky(R + 1e-12 * np.eye(len(R)))
    xi = rng.standard_normal((n, len(R))) @ L.T
    return float(np.quantile(np.abs(xi).max(axis=1), 1 - BETA))


def akm(x, Sigma, j, c_beta):
    sd = np.sqrt(Sigma[j, j]); xj = x[j]
    lo, hi = truncation(x, Sigma, j)
    span = 30 * sd
    # conditional: equal-tailed and median-unbiased
    F = lambda m: tn_cdf(xj, m, sd, lo, hi)
    cond = (solve_mu(F, 1 - ALPHA / 2, xj - span, xj + span),
            solve_mu(F, ALPHA / 2, xj - span, xj + span))
    cond_est = solve_mu(F, 0.5, xj - span, xj + span)
    # hybrid: also condition on mu_j in the level-beta projection interval
    p_lo, p_hi = xj - c_beta * sd, xj + c_beta * sd
    FH = lambda m: tn_cdf(xj, m, sd, max(lo, m - c_beta * sd), min(hi, m + c_beta * sd))
    tail = (ALPHA - BETA) / (2 * (1 - BETA))
    hyb = (solve_mu(FH, 1 - tail, p_lo, p_hi), solve_mu(FH, tail, p_lo, p_hi))
    hyb_est = solve_mu(FH, 0.5, p_lo, p_hi)
    return cond_est, cond, hyb_est, hyb


def main():
    rng = np.random.default_rng(SEED)
    out = {"alpha": ALPHA, "beta": BETA, "s": S, "n_audits": N_AUDITS, "seed": SEED,
           "reference": "Andrews, Kitagawa & McCloskey (2024), QJE 139(1):305-358",
           "cells": {}}
    print(f"{'corr':<18} {'K':>3} {'tau':>6} | {'estimator':<11} {'cover':>6} "
          f"{'width':>7} {'bias':>9} {'RMSE':>8}")
    for kind in CORRS:
        for K in K_GRID:
            R = corr_matrix(kind, K); Sigma = S ** 2 * R
            Lc = np.linalg.cholesky(R + 1e-12 * np.eye(K))
            c_beta = projection_c(R, rng)
            for tau in TAU_GRID:
                rec = {e: {"cover": [], "width": [], "err": []}
                       for e in ("naive", "akm_conditional", "akm_hybrid", "direct")}
                for _ in range(N_AUDITS):
                    q = rng.normal(0, tau, K) if tau > 0 else np.zeros(K)
                    cv = q + S * (Lc @ rng.standard_normal(K))
                    ho = q + S * (Lc @ rng.standard_normal(K))
                    j = int(cv.argmax()); t = q[j]
                    ce, (cl, ch), he, (hl, hh) = akm(cv, Sigma, j, c_beta)
                    for e, est, lo, hi in (
                            ("naive", cv[j], cv[j] - Z * S, cv[j] + Z * S),
                            ("akm_conditional", ce, cl, ch),
                            ("akm_hybrid", he, hl, hh),
                            ("direct", ho[j], ho[j] - Z * S, ho[j] + Z * S)):
                        rec[e]["cover"].append(lo <= t <= hi)
                        rec[e]["width"].append(hi - lo)
                        rec[e]["err"].append(est - t)
                cell = {"corr": kind, "K": K, "tau": tau, "c_beta": c_beta, "estimators": {}}
                for e, r in rec.items():
                    err = np.array(r["err"])
                    cell["estimators"][e] = {
                        "coverage": float(np.mean(r["cover"])),
                        "median_width": float(np.median(r["width"])),
                        "mean_width": float(np.mean(r["width"])),
                        "bias": float(err.mean()), "median_bias": float(np.median(err)),
                        "rmse": float(np.sqrt((err ** 2).mean()))}
                    v = cell["estimators"][e]
                    print(f"{kind:<18} {K:>3} {tau:>6.3f} | {e:<11.11} {v['coverage']:>6.3f} "
                          f"{v['median_width']:>7.4f} {v['bias']:>+9.5f} {v['rmse']:>8.5f}", flush=True)
                out["cells"][f"{kind}|K={K}|tau={tau}"] = cell
    cells = out["cells"].values()
    def rng_of(e, k):
        v = [c["estimators"][e][k] for c in cells]
        return [min(v), max(v)]
    out["summary"] = {e: {"coverage_range": rng_of(e, "coverage"),
                          "median_width_range": rng_of(e, "median_width"),
                          "abs_bias_max": max(abs(c["estimators"][e]["bias"]) for c in cells)}
                      for e in ("naive", "akm_conditional", "akm_hybrid", "direct")}
    print("\nSummary:", json.dumps(out["summary"], indent=1))
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
