"""
Size and power of the selection-optimism test as practised, and of the
corrected test.  [Referee K1, K3, M1.]

THE PROBLEM. Leakage audits report the selection-specific component

    Delta_sel = gap_{l*} - mean_l(gap_l),   gap_l = cv_l - ho_l,
    l* = argmax_l cv_l

and declare a selection effect when a bootstrap interval for Delta_sel
excludes zero. Delta_sel is a perfectly good unbiased estimator -- the error is
the null it is tested against. Because l* maximises cv and gap contains +cv,

    Delta_sel = A - B,   A = cv_{l*} - mean(cv),   B = ho_{l*} - mean(ho),

and under a null in which candidates carry NO transferable quality we have
E[B] = 0 but E[A] > 0. The null value of Delta_sel is therefore A, not 0.
Testing against 0 tests a hypothesis that is false whenever argmax is applied
to anything noisy, so the test rejects almost regardless of the data.

THE CORRECTION. A is computable from the same quantities the audit already
has, so the corrected statistic is Delta_sel - A = -B. Equivalently: test B.
B is built only from held-out values, which played no part in choosing l*, so
its null value is genuinely zero. No permutation, no bootstrap of a
re-parameterisation, no conditional-inference machinery: a one-sample t-test
on B over the audit's own replicates.

WHAT THIS SCRIPT SHOWS. A simulation with known ground truth, in which
candidate quality is controlled by tau (tau = 0 is the no-transferable-signal
null):

    q_l  ~ N(0, tau^2)            true held-out quality of candidate l
    cv_l = q_l + N(0, s_cv^2)     selection-fold estimate
    ho_l = q_l + N(0, s_ho^2)     held-out estimate

We report, over many independent audits, how often each test declares a
selection effect. At tau = 0 that rejection rate is the type I error; above it
it is power. The practised test should be badly oversized; the corrected test
should sit at nominal level and still have power when transferable quality is
real.

Fully vectorised; runs in seconds. No dependence on the paper's data -- this
is the estimator's behaviour, not this harness's.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "results" / "estimator_size_power.json"

N_AUDITS = 2000      # independent audits per configuration
N_REPS = 50          # replicates within an audit (matches the paper's 50 splits)
ALPHA = 0.05
SEED = 20260929

TAU_GRID = [0.0, 0.005, 0.010, 0.020, 0.040]
K_GRID = [8, 32, 128]
S_CV, S_HO = 0.03, 0.03


def run(K, tau, rng):
    """Simulate N_AUDITS audits of K candidates; return rejection rates."""
    shape = (N_AUDITS, N_REPS, K)
    q = rng.normal(0.0, tau, size=(N_AUDITS, 1, K)) if tau > 0 else np.zeros((N_AUDITS, 1, K))
    cv = q + rng.normal(0.0, S_CV, size=shape)
    ho = q + rng.normal(0.0, S_HO, size=shape)

    lstar = cv.argmax(axis=2)                                    # (audits, reps)
    ix = np.ogrid[:N_AUDITS, :N_REPS]
    A = cv[ix[0], ix[1], lstar] - cv.mean(axis=2)                # winner's curse
    B = ho[ix[0], ix[1], lstar] - ho.mean(axis=2)                # transferred quality
    delta = A - B

    def reject(x):
        """Two-sided t-test at ALPHA over the audit's own replicates,
        which is what a bootstrap interval excluding zero amounts to here."""
        m, sd = x.mean(axis=1), x.std(axis=1, ddof=1)
        t = m / (sd / np.sqrt(N_REPS))
        crit = stats.t.ppf(1 - ALPHA / 2, N_REPS - 1)
        return np.abs(t) > crit

    # the practised test declares a selection effect when Delta_sel differs
    # from zero; the corrected test asks whether B does.
    return {
        "reject_rate_practised_delta_vs_zero": float(reject(delta).mean()),
        "reject_rate_corrected_B_vs_zero": float(reject(B).mean()),
        "mean_delta": float(delta.mean()),
        "mean_A": float(A.mean()),
        "mean_B": float(B.mean()),
        # Delta_sel's null value is A, so this is the residual the corrected
        # statistic targets; it must be ~0 at tau = 0.
        "mean_delta_minus_A": float((delta - A).mean()),
    }


def main():
    rng = np.random.default_rng(SEED)
    out = {"n_audits": N_AUDITS, "n_reps": N_REPS, "alpha": ALPHA, "seed": SEED,
           "s_cv": S_CV, "s_ho": S_HO, "tau_grid": TAU_GRID, "k_grid": K_GRID,
           "results": {}}

    print("=" * 100)
    print("HOW OFTEN DOES EACH TEST DECLARE A SELECTION EFFECT?")
    print(f"{N_AUDITS} independent audits per cell, {N_REPS} replicates each, "
          f"alpha={ALPHA}")
    print("tau = 0 is the no-transferable-signal null: the column there IS the "
          "type I error.")
    print("=" * 100)
    print(f"\n{'K':>5} {'tau':>7} | {'practised':>10} {'corrected':>10} | "
          f"{'E[Delta]':>9} {'E[A]':>9} {'E[B]':>9}")
    print("-" * 78)

    for K in K_GRID:
        for tau in TAU_GRID:
            r = run(K, tau, rng)
            out["results"][f"K={K},tau={tau}"] = {"K": K, "tau": tau, **r}
            flag = "  <-- type I error" if tau == 0 else ""
            print(f"{K:>5} {tau:>7.3f} | "
                  f"{r['reject_rate_practised_delta_vs_zero']:>10.3f} "
                  f"{r['reject_rate_corrected_B_vs_zero']:>10.3f} | "
                  f"{r['mean_delta']:>9.5f} {r['mean_A']:>9.5f} "
                  f"{r['mean_B']:>9.5f}{flag}")
        print("-" * 78)

    nulls = [v for v in out["results"].values() if v["tau"] == 0.0]
    out["summary"] = {
        "practised_type_I_error_range": [
            min(v["reject_rate_practised_delta_vs_zero"] for v in nulls),
            max(v["reject_rate_practised_delta_vs_zero"] for v in nulls)],
        "corrected_type_I_error_range": [
            min(v["reject_rate_corrected_B_vs_zero"] for v in nulls),
            max(v["reject_rate_corrected_B_vs_zero"] for v in nulls)],
        "corrected_is_nominal": all(
            abs(v["reject_rate_corrected_B_vs_zero"] - ALPHA) < 0.02 for v in nulls),
    }
    s = out["summary"]
    print(f"\nType I error at tau=0, across all K:")
    print(f"  as practised (Delta_sel vs 0): "
          f"{s['practised_type_I_error_range'][0]:.3f} to "
          f"{s['practised_type_I_error_range'][1]:.3f}")
    print(f"  corrected (B vs 0):            "
          f"{s['corrected_type_I_error_range'][0]:.3f} to "
          f"{s['corrected_type_I_error_range'][1]:.3f}   "
          f"(nominal {ALPHA})")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
