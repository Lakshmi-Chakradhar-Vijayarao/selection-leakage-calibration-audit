"""
A fair interval for the shrinkage estimator.  [Referee K3.]

113 gave every estimator the same interval: the within-audit standard
deviation of the estimator across replicates, turned into a t interval. That
is the right interval for the direct estimator and the WRONG one for Tweedie
shrinkage, which deliberately trades variance for bias. Its replicate-to-
replicate spread does not represent its error, so the t interval is
artificially narrow and its coverage collapses. Reporting that collapse as a
property of the method, as 113 did, compares our honest interval against a
naive interval we constructed for the competitor.

This script gives the shrinkage estimator the interval an empirical-Bayes
practitioner would actually use -- a parametric bootstrap through the fitted
model -- and re-reports coverage on that basis, so the comparison is between
each estimator at its best.

Procedure per audit: estimate mu and tau^2 from the observed cv by method of
moments, then resample N_BOOT synthetic candidate sets from the fitted model
q ~ N(mu_hat, tau2_hat), cv = q + N(0, sigma^2), recompute the shrinkage
estimate on each, and take a percentile interval. Coverage is then checked
against that audit's realised theta, exactly as for the direct estimator.

Fewer audits than 113 because the bootstrap is nested; enough for coverage to
be resolved to about +/- 0.02.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "results" / "fair_interval_comparison.json"

N_AUDITS = 600
N_REPS = 50
N_BOOT = 400
K_GRID = [8, 32, 128]
TAU_GRID = [0.0, 0.010, 0.030]
SIGMA = 0.03
ALPHA = 0.05
SEED = 20261001


def tweedie_estimate(cv):
    """Shrinkage estimate of the optimism at the argmax, from cv alone.
    cv has shape (..., K); returns (..., ) estimates."""
    lstar = cv.argmax(axis=-1)
    take = np.take_along_axis(cv, lstar[..., None], axis=-1)[..., 0]
    mu = cv.mean(axis=-1)
    tau2 = np.clip(cv.var(axis=-1, ddof=1) - SIGMA ** 2, 0.0, None)
    shrink = tau2 / (tau2 + SIGMA ** 2)
    q_hat = mu + shrink * (take - mu)
    return take - q_hat, mu, tau2


def main():
    rng = np.random.default_rng(SEED)
    out = {"n_audits": N_AUDITS, "n_reps": N_REPS, "n_boot": N_BOOT,
           "sigma": SIGMA, "alpha": ALPHA, "seed": SEED, "results": {}}

    print("=" * 96)
    print("COVERAGE WITH A FAIR INTERVAL FOR THE SHRINKAGE ESTIMATOR")
    print(f"{N_AUDITS} audits x {N_REPS} replicates, {N_BOOT} parametric "
          f"bootstrap draws per audit")
    print("=" * 96)
    print(f"\n{'K':>5} {'tau':>7} | {'direct (t)':>12} {'Tweedie (t)':>13} "
          f"{'Tweedie (boot)':>16} | {'Tweedie bias':>13}")
    print("-" * 82)

    for K in K_GRID:
        for tau in TAU_GRID:
            shape = (N_AUDITS, N_REPS, K)
            q = (rng.normal(0, tau, size=(N_AUDITS, 1, K)) if tau > 0
                 else np.zeros((N_AUDITS, 1, K)))
            q = np.broadcast_to(q, shape)
            cv = q + rng.normal(0, SIGMA, size=shape)
            ho = q + rng.normal(0, SIGMA, size=shape)

            lstar = cv.argmax(axis=2)
            ix = np.ogrid[:N_AUDITS, :N_REPS]
            theta = (cv[ix[0], ix[1], lstar] - q[ix[0], ix[1], lstar]).mean(axis=1)

            direct = (cv[ix[0], ix[1], lstar] - ho[ix[0], ix[1], lstar])
            tw, mu_hat, tau2_hat = tweedie_estimate(cv)

            crit = stats.t.ppf(1 - ALPHA / 2, N_REPS - 1)
            def t_cover(est):
                m = est.mean(axis=1); se = est.std(axis=1, ddof=1) / np.sqrt(N_REPS)
                return float(np.mean((m - crit * se <= theta) & (theta <= m + crit * se)))

            cov_direct_t = t_cover(direct)
            cov_tw_t = t_cover(tw)

            # parametric bootstrap through the fitted model, per audit
            mu_a = mu_hat.mean(axis=1)                      # (audits,)
            tau2_a = tau2_hat.mean(axis=1)
            boot = np.empty((N_AUDITS, N_BOOT))
            for b in range(N_BOOT):
                qb = rng.normal(mu_a[:, None, None],
                                np.sqrt(tau2_a)[:, None, None], size=shape)
                cvb = qb + rng.normal(0, SIGMA, size=shape)
                est_b, _, _ = tweedie_estimate(cvb)
                boot[:, b] = est_b.mean(axis=1)
            lo = np.quantile(boot, ALPHA / 2, axis=1)
            hi = np.quantile(boot, 1 - ALPHA / 2, axis=1)
            # centre the bootstrap distribution on the observed estimate
            shift = tw.mean(axis=1) - boot.mean(axis=1)
            cov_tw_boot = float(np.mean((lo + shift <= theta) & (theta <= hi + shift)))

            bias_tw = float((tw.mean(axis=1) - theta).mean())
            out["results"][f"K={K},tau={tau}"] = {
                "K": K, "tau": tau,
                "coverage_direct_t": cov_direct_t,
                "coverage_tweedie_t_naive": cov_tw_t,
                "coverage_tweedie_parametric_bootstrap": cov_tw_boot,
                "bias_tweedie": bias_tw,
            }
            print(f"{K:>5} {tau:>7.3f} | {cov_direct_t:>12.3f} {cov_tw_t:>13.3f} "
                  f"{cov_tw_boot:>16.3f} | {bias_tw:>+13.5f}")
        print("-" * 82)

    vals = list(out["results"].values())
    out["summary"] = {
        "direct_t_coverage_range": [min(v["coverage_direct_t"] for v in vals),
                                    max(v["coverage_direct_t"] for v in vals)],
        "tweedie_naive_range": [min(v["coverage_tweedie_t_naive"] for v in vals),
                                max(v["coverage_tweedie_t_naive"] for v in vals)],
        "tweedie_bootstrap_range": [
            min(v["coverage_tweedie_parametric_bootstrap"] for v in vals),
            max(v["coverage_tweedie_parametric_bootstrap"] for v in vals)],
        "tweedie_bias_range": [min(v["bias_tweedie"] for v in vals),
                               max(v["bias_tweedie"] for v in vals)],
    }
    s = out["summary"]
    print(f"\ndirect, t interval           {s['direct_t_coverage_range'][0]:.3f}"
          f"-{s['direct_t_coverage_range'][1]:.3f}")
    print(f"Tweedie, naive t interval    {s['tweedie_naive_range'][0]:.3f}"
          f"-{s['tweedie_naive_range'][1]:.3f}   (the unfair comparison)")
    print(f"Tweedie, parametric bootstrap{s['tweedie_bootstrap_range'][0]:.3f}"
          f"-{s['tweedie_bootstrap_range'][1]:.3f}   (the fair one)")
    print(f"Tweedie bias                 {s['tweedie_bias_range'][0]:+.5f}"
          f" to {s['tweedie_bias_range'][1]:+.5f}")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
