"""
Does the size/power result survive its own assumptions?  [Limitations, Sec. 4.4]

111 shows that testing Delta_sel against zero rejects in essentially every
audit whether or not selection recovered anything, and that testing B instead
holds its level. Its simulation makes two assumptions the paper's Limitations
section concedes:

  (1) candidate noise is Gaussian with equal variance on both arms, and
      independent across candidates;
  (2) the 50 replicates within an audit are INDEPENDENT draws.

This script relaxes each.

PART A -- the noise model. Same construction as 111 (q_l ~ N(0, tau^2),
cv_l = q_l + e_cv, ho_l = q_l + e_ho), with the noise replaced by heavy-tailed
(t, 3 df), skewed (centred chi-square, 3 df), unequal-variance (held-out arm
twice and half as noisy as the selection arm), heterogeneous across candidates
(per-candidate sd drawn once per audit from U[0.015, 0.06]), equicorrelated
across candidates (rho = 0.75, the harness's measured mean correlation), and
banded across candidates (AR(1), adjacent rho = 0.957, the harness's measured
adjacent-layer correlation). Every variant keeps the marginal sd at 0.03 where
a single sd is meaningful, so only the shape changes.

PART B -- the replicates. In a real audit the 50 replicates are 50 random
re-splits of ONE finite sample, so they share data and are not independent. A
t-test over them then answers a split-conditional question -- is B positive on
average over re-splits of THIS sample? -- and a layer that happens to be good
on this particular sample is good on both halves of every split, so B can be
positive split-conditionally with no population-level difference between
layers at all. Part B simulates that structure directly:

    N = 700 items, base rate 0.765 (the harness's), K = 32 candidates;
    item score z_il = mu_l * y_i + e_il, e ~ N(0,1) with AR(1) correlation
    0.957 across candidates; mu_l = mu_0 + tau * u_l, u_l ~ N(0,1), with mu_0
    set so a typical candidate has population AUROC 0.80;
    50 stratified 400/300 re-splits of that one sample; cv_l = AUROC on the
    400, ho_l = AUROC on the 300; l* = argmax cv.

At tau = 0 no candidate is better in the population, so any test that
declares "the selected candidate transfers" is making a false population
claim. We report the rejection rate of the B test as practised (naive t over
re-splits) and with the Nadeau & Bengio (2003) correction for re-split
dependence, which inflates the variance of a mean over J random splits from
s^2/J to (1/J + n_test/n_train) s^2.

Simplification, stated: the selection criterion here is the AUROC of a fixed
score on the 400, not a probe re-fitted under 5-fold CV. That removes fitting
noise, which can only make re-split dependence weaker, not stronger.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "results" / "size_power_robustness.json"

ALPHA = 0.05
N_REPS = 50
SEED = 20261010

# ── Part A ──────────────────────────────────────────────────────────────────
N_AUDITS_A = 2000
K_GRID = [8, 32, 128]
TAU_GRID_A = [0.0, 0.010, 0.040]
S = 0.03


def _ar1(rng, shape, rho):
    """Gaussian noise, unit variance, AR(1) across the last axis."""
    z = rng.standard_normal(shape)
    out = np.empty(shape)
    out[..., 0] = z[..., 0]
    c = np.sqrt(1 - rho ** 2)
    for k in range(1, shape[-1]):
        out[..., k] = rho * out[..., k - 1] + c * z[..., k]
    return out


def noise(rng, kind, shape, arm):
    """Noise of the given kind with sd S (except where the kind changes sd)."""
    if kind == "gaussian":
        return S * rng.standard_normal(shape)
    if kind == "t3":
        return S * rng.standard_t(3, size=shape) / np.sqrt(3.0)
    if kind == "skewed_chi2_3":
        return S * (rng.chisquare(3, size=shape) - 3.0) / np.sqrt(6.0)
    if kind == "ho_sd_2x":
        return (S if arm == "cv" else 2 * S) * rng.standard_normal(shape)
    if kind == "ho_sd_half":
        return (S if arm == "cv" else S / 2) * rng.standard_normal(shape)
    if kind == "equicorr_0.75":
        rho = 0.75
        z0 = rng.standard_normal(shape[:-1] + (1,))
        return S * (np.sqrt(rho) * z0 + np.sqrt(1 - rho) * rng.standard_normal(shape))
    if kind == "banded_ar1_0.957":
        return S * _ar1(rng, shape, 0.957)
    raise ValueError(kind)


KINDS = ["gaussian", "t3", "skewed_chi2_3", "ho_sd_2x", "ho_sd_half",
         "heterogeneous_sd", "equicorr_0.75", "banded_ar1_0.957"]


def t_reject(x, var_factor=None):
    """Two-sided one-sample t-test over the last-but-one axis (replicates).
    var_factor replaces 1/J by the Nadeau-Bengio factor when given."""
    J = x.shape[1]
    m, sd = x.mean(axis=1), x.std(axis=1, ddof=1)
    f = (1.0 / J) if var_factor is None else var_factor
    t = m / np.sqrt(f * sd ** 2)
    return np.abs(t) > stats.t.ppf(1 - ALPHA / 2, J - 1)


def part_a(rng):
    out = {}
    for kind in KINDS:
        for K in K_GRID:
            for tau in TAU_GRID_A:
                shape = (N_AUDITS_A, N_REPS, K)
                q = (rng.normal(0, tau, (N_AUDITS_A, 1, K)) if tau > 0
                     else np.zeros((N_AUDITS_A, 1, K)))
                if kind == "heterogeneous_sd":
                    sd = rng.uniform(0.015, 0.06, (N_AUDITS_A, 1, K))
                    cv = q + sd * rng.standard_normal(shape)
                    ho = q + sd * rng.standard_normal(shape)
                else:
                    cv = q + noise(rng, kind, shape, "cv")
                    ho = q + noise(rng, kind, shape, "ho")
                ls = cv.argmax(axis=2)
                ix = np.ogrid[:N_AUDITS_A, :N_REPS]
                A = cv[ix[0], ix[1], ls] - cv.mean(axis=2)
                B = ho[ix[0], ix[1], ls] - ho.mean(axis=2)
                out[f"{kind}|K={K}|tau={tau}"] = {
                    "kind": kind, "K": K, "tau": tau,
                    "reject_practised_delta_vs_zero": float(t_reject(A - B).mean()),
                    "reject_corrected_B_vs_zero": float(t_reject(B).mean()),
                }
    return out


# ── Part B ──────────────────────────────────────────────────────────────────
N_AUDITS_B = 1000
N_ITEMS, N_SEL, BASE_RATE, K_B = 700, 400, 0.765, 32
MU0 = np.sqrt(2) * stats.norm.ppf(0.80)          # population AUROC 0.80
TAU_GRID_B = [0.0, 0.05, 0.10, 0.20]


def auroc_cols(z, y):
    """AUROC of every column of z against labels y (continuous scores)."""
    r = z.argsort(axis=0).argsort(axis=0) + 1.0
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def strat_split(rng, y, n_sel):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    k_pos = int(round(n_sel * len(pos) / len(y)))
    sel = np.concatenate([rng.choice(pos, k_pos, replace=False),
                          rng.choice(neg, n_sel - k_pos, replace=False)])
    mask = np.zeros(len(y), bool); mask[sel] = True
    return mask


def part_b(rng):
    nb = 1.0 / N_REPS + (N_ITEMS - N_SEL) / N_SEL
    out = {"nadeau_bengio_variance_factor": nb,
           "naive_variance_factor": 1.0 / N_REPS,
           "t_shrinkage": float(np.sqrt((1.0 / N_REPS) / nb)), "cells": {}}
    for tau in TAU_GRID_B:
        rej_naive = rej_nb = rej_delta = 0
        pop_gain = []
        for _ in range(N_AUDITS_B):
            y = (rng.random(N_ITEMS) < BASE_RATE).astype(int)
            mu = MU0 + tau * rng.standard_normal(K_B)
            z = mu[None, :] * y[:, None] + _ar1(rng, (N_ITEMS, K_B), 0.957)
            pop_auc = stats.norm.cdf(mu / np.sqrt(2))
            B = np.empty(N_REPS); D = np.empty(N_REPS)
            for r in range(N_REPS):
                m = strat_split(rng, y, N_SEL)
                cv = auroc_cols(z[m], y[m]); ho = auroc_cols(z[~m], y[~m])
                l = int(cv.argmax())
                B[r] = ho[l] - ho.mean()
                D[r] = (cv[l] - cv.mean()) - B[r]
                pop_gain.append(pop_auc[l] - pop_auc.mean())
            Bm = B[None, :]
            rej_naive += bool(t_reject(Bm)[0])
            rej_nb += bool(t_reject(Bm, nb)[0])
            rej_delta += bool(t_reject(D[None, :])[0])
        out["cells"][f"tau={tau}"] = {
            "tau": tau,
            "population_auroc_gain_of_selected_mean": float(np.mean(pop_gain)),
            "reject_B_naive_t": rej_naive / N_AUDITS_B,
            "reject_B_nadeau_bengio": rej_nb / N_AUDITS_B,
            "reject_delta_sel_vs_zero_naive": rej_delta / N_AUDITS_B,
        }
        c = out["cells"][f"tau={tau}"]
        print(f"  tau={tau:<5} pop gain of selected {c['population_auroc_gain_of_selected_mean']:+.4f} | "
              f"B naive {c['reject_B_naive_t']:.3f}  B Nadeau-Bengio {c['reject_B_nadeau_bengio']:.3f} | "
              f"Delta_sel vs 0 {c['reject_delta_sel_vs_zero_naive']:.3f}", flush=True)
    return out


def main():
    rng = np.random.default_rng(SEED)
    print("PART A -- noise model (independent replicates)")
    a = part_a(rng)
    print(f"{'kind':<18} {'K':>4} {'tau':>6} | {'practised':>9} {'corrected':>9}")
    for v in a.values():
        print(f"{v['kind']:<18} {v['K']:>4} {v['tau']:>6.3f} | "
              f"{v['reject_practised_delta_vs_zero']:>9.3f} {v['reject_corrected_B_vs_zero']:>9.3f}")
    null = [v for v in a.values() if v["tau"] == 0.0]
    summ_a = {
        "corrected_type_I_error_range": [min(v["reject_corrected_B_vs_zero"] for v in null),
                                         max(v["reject_corrected_B_vs_zero"] for v in null)],
        "practised_rejection_at_tau0_range": [min(v["reject_practised_delta_vs_zero"] for v in null),
                                              max(v["reject_practised_delta_vs_zero"] for v in null)],
    }
    print("Part A summary:", summ_a)

    print("\nPART B -- one finite sample, 50 overlapping re-splits")
    b = part_b(rng)
    with open(OUT_PATH, "w") as f:
        json.dump({"seed": SEED, "alpha": ALPHA, "n_reps": N_REPS,
                   "part_a": {"n_audits": N_AUDITS_A, "results": a, "summary": summ_a},
                   "part_b": {"n_audits": N_AUDITS_B, "n_items": N_ITEMS, "n_sel": N_SEL,
                              "base_rate": BASE_RATE, "K": K_B, **b}}, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
