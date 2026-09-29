"""
Bias, variance and coverage of four estimators of selection-induced optimism,
including a winner's-curse correction.  [Referee K2(b) and M1(b).]

THE QUESTION. An auditor reports cv_{l*} for the candidate chosen by
argmax. How much of that number will not transfer? With q_l the true quality
of candidate l, the quantity they want is

    theta = cv_{l*} - q_{l*}                (optimism actually incurred)

which is knowable in simulation and not in practice. We compare four ways of
estimating it, including the two the referee named -- the naive statistic the
audit literature uses, and a winner's-curse correction from the genetics
literature -- and report bias, SD, RMSE and CI coverage for each.

THE ESTIMATORS.

  E1  Delta_sel = A - B, the selection-specific component, as audits report
      it. Included because it is what the literature uses, not because it
      targets theta.

  E2  gap_{l*} = cv_{l*} - ho_{l*}. The direct estimator. Held-out data played
      no part in selecting l*, so E[ho_{l*} | l*] = q_{l*} and this is
      unbiased for theta with no correction of any kind.

  E3  Tweedie / empirical-Bayes shrinkage, using cv ONLY (Efron, 2011). With
      cv_l ~ N(q_l, sigma^2) and q_l ~ N(mu, tau^2), the posterior mean
      shrinks cv toward the mean; tau^2 is estimated by method of moments from
      the spread of cv. The optimism estimate is cv_{l*} - qhat_{l*}. This is
      the winner's-curse correction, and its point is that it needs no
      held-out data at all.

  E4  Conditional-MLE-style correction, also cv only: a one-parameter
      profile correction that subtracts the expected selection bias of a
      maximum of K exchangeable normals, E[max] - mu = sigma * e_K, with e_K
      obtained by simulation. Included because it is the other standard
      family and is cruder than E3.

WHAT THIS ESTABLISHES. It answers the referee's question directly: what does
winner's-curse correction give here that the naive statistic does not, and
what does it cost? The expected answer is that E3/E4 recover theta without
held-out data, where the naive statistic does not estimate theta at all; and
that when held-out data IS available -- which is the audit setting -- E2 is
unbiased, better covered and simpler than either correction.

Fully vectorised; seconds to run.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "results" / "estimator_comparison.json"

N_AUDITS = 4000
N_REPS = 50
K_GRID = [8, 32, 128]
TAU_GRID = [0.0, 0.010, 0.030]
SIGMA = 0.03
ALPHA = 0.05
SEED = 20260930


def expected_max_shift(K, n=200000, rng=None):
    """E[max of K standard normals]: the selection bias of an argmax in
    sigma units, used by E4."""
    rng = rng or np.random.default_rng(0)
    return float(rng.standard_normal((n, K)).max(axis=1).mean())


def run(K, tau, rng, e_K):
    shape = (N_AUDITS, N_REPS, K)
    q = (rng.normal(0.0, tau, size=(N_AUDITS, 1, K)) if tau > 0
         else np.zeros((N_AUDITS, 1, K)))
    q = np.broadcast_to(q, shape)
    cv = q + rng.normal(0.0, SIGMA, size=shape)
    ho = q + rng.normal(0.0, SIGMA, size=shape)

    lstar = cv.argmax(axis=2)
    ix = np.ogrid[:N_AUDITS, :N_REPS]
    cv_s = cv[ix[0], ix[1], lstar]
    ho_s = ho[ix[0], ix[1], lstar]
    q_s = q[ix[0], ix[1], lstar]

    theta = cv_s - q_s                                    # the target

    E1 = (cv_s - cv.mean(axis=2)) - (ho_s - ho.mean(axis=2))   # Delta_sel
    E2 = cv_s - ho_s                                            # direct

    # E3: Tweedie / empirical Bayes, cv only. tau^2 by method of moments.
    var_cv = cv.var(axis=2, ddof=1)
    tau2 = np.clip(var_cv - SIGMA ** 2, 0.0, None)
    shrink = tau2 / (tau2 + SIGMA ** 2)
    mu_hat = cv.mean(axis=2)
    q_hat = mu_hat + shrink * (cv_s - mu_hat)
    E3 = cv_s - q_hat

    # E4: subtract the expected argmax shift of K exchangeable normals.
    E4 = np.full_like(E2, SIGMA * e_K)

    out = {}
    crit = stats.t.ppf(1 - ALPHA / 2, N_REPS - 1)
    for name, est in [("E1_delta_sel", E1), ("E2_direct_gap", E2),
                      ("E3_tweedie_cv_only", E3), ("E4_argmax_shift_cv_only", E4)]:
        # one audit = mean over its replicates; bias is against that audit's
        # own realised theta, so this is estimation error, not model error.
        est_m, th_m = est.mean(axis=1), theta.mean(axis=1)
        err = est_m - th_m
        sd_within = est.std(axis=1, ddof=1) / np.sqrt(N_REPS)
        lo, hi = est_m - crit * sd_within, est_m + crit * sd_within
        cover = np.mean((lo <= th_m) & (th_m <= hi)) if name != "E4_argmax_shift_cv_only" \
            else float("nan")   # E4 is deterministic within an audit: no interval
        out[name] = {
            "bias": float(err.mean()),
            "sd": float(est_m.std(ddof=1)),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "ci_coverage": float(cover) if cover == cover else None,
            "uses_heldout": name == "E2_direct_gap",
        }
    out["mean_theta"] = float(theta.mean())
    return out


def main():
    rng = np.random.default_rng(SEED)
    eK = {K: expected_max_shift(K, rng=np.random.default_rng(SEED + K)) for K in K_GRID}
    res = {"n_audits": N_AUDITS, "n_reps": N_REPS, "sigma": SIGMA, "alpha": ALPHA,
           "seed": SEED, "expected_max_shift": eK, "results": {}}

    print("=" * 108)
    print("ESTIMATING THE OPTIMISM ACTUALLY INCURRED, theta = cv_{l*} - q_{l*}")
    print(f"{N_AUDITS} audits x {N_REPS} replicates per cell; sigma={SIGMA}")
    print("=" * 108)

    for K in K_GRID:
        for tau in TAU_GRID:
            r = run(K, tau, rng, eK[K])
            res["results"][f"K={K},tau={tau}"] = {"K": K, "tau": tau, **r}
            print(f"\nK={K}, tau={tau}   (true mean theta = {r['mean_theta']:+.5f})")
            print(f"  {'estimator':<26} {'bias':>10} {'SD':>9} {'RMSE':>9} "
                  f"{'cover':>8}  held-out?")
            for name in ("E1_delta_sel", "E2_direct_gap", "E3_tweedie_cv_only",
                         "E4_argmax_shift_cv_only"):
                e = r[name]
                cov = f"{e['ci_coverage']:.3f}" if e["ci_coverage"] is not None else "   n/a"
                print(f"  {name:<26} {e['bias']:>+10.5f} {e['sd']:>9.5f} "
                      f"{e['rmse']:>9.5f} {cov:>8}  "
                      f"{'yes' if e['uses_heldout'] else 'no'}")

    # headline summary at the null and at a realistic alternative
    key = f"K=32,tau=0.01"
    k = res["results"][key]
    res["summary"] = {
        "at_K32_tau0.01": {n: k[n] for n in
                           ("E1_delta_sel", "E2_direct_gap", "E3_tweedie_cv_only",
                            "E4_argmax_shift_cv_only")},
        "direct_is_unbiased": abs(k["E2_direct_gap"]["bias"]) < 0.001,
        "direct_covers": abs(k["E2_direct_gap"]["ci_coverage"] - 0.95) < 0.02,
        "delta_sel_is_biased_for_theta": abs(k["E1_delta_sel"]["bias"]) > 0.005,
        "tweedie_beats_delta_sel_on_rmse":
            k["E3_tweedie_cv_only"]["rmse"] < k["E1_delta_sel"]["rmse"],
    }
    with open(OUT_PATH, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
