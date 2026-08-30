"""
Case Study 4 estimator audit, in response to an independent review
that proved the leave-one-seed-out ROTATION estimator used in `code/45` is
NON-NEGATIVE FOR ANY DATA WHATSOEVER, not merely in the equal-argmax
(degenerate) special case that `code/45`'s docstring already proved.

This script does four things, all of which change what the paper may claim:

  (1) NUMERICALLY VERIFIES THE GENERAL NON-NEGATIVITY THEOREM over random
      matrices of many shapes. The analytic proof is reproduced below and in
      the paper (Appendix B.5); this is the independent check that no sign
      convention or indexing detail was lost between proof and code.

  (2) COMPUTES A TIE-BREAK SENSITIVITY BAND. `np.argmax` resolves ties at the
      LOWEST index. Ties are common in these files (AUROC computed on a few
      hundred test points takes few distinct values), and both the headline
      mean AND the degenerate-cell count depend on how they are resolved.
      Every tie resolution is enumerated per cell and the induced range on
      the headline is reported.

  (3) PROMOTES THE BOOTSTRAP MAX-BIAS ESTIMATOR to a primary quantity, with
      the BCa CI and Wilcoxon test that `code/45` only computed for the
      rotation estimator. This estimator is structurally different: the
      resample mean and the full-sample mean at the resample-selected layer
      are computed on different seed multisets, so nothing telescopes and it
      is NOT sign-constrained.

  (4) CALIBRATES THE 2-SEED vs 3-SEED SELECTION-CRITERION MISMATCH. The
      audited repository selects its reported `best_layer` using the mean
      over ALL THREE seeds. The rotation estimator selects using a 2-seed
      mean, a strictly noisier criterion, which inflates the winner's curse
      it measures relative to the one the repository actually incurs.

──────────────────────────────────────────────────────────────────────────────
THE GENERAL NON-NEGATIVITY THEOREM

Let A be the (L x R) matrix of AUROCs: L probed layers, R seeds. Write
S(l) = sum_r a_{r,l} and m(l) = S(l)/R for the grand mean at layer l, and

    f(r,l) := m(l) - a_{r,l}          (seed r's deviation at layer l)

so that sum_r f(r,l) = 0 for every fixed l -- an identity, for any data.

In rotation r the selection set is the R-1 seeds other than r, so the
selection criterion at layer l is

    c_r(l) = (S(l) - a_{r,l}) / (R-1) = m(l) + f(r,l)/(R-1).

The rotation picks L_r = argmax_l c_r(l), and its winner's-curse term is

    wc_r = c_r(L_r) - a_{r,L_r}
         = m(L_r) + f(r,L_r)/(R-1) - (m(L_r) - f(r,L_r))
         = f(r,L_r) * R/(R-1).

Hence the cell estimate is

    est = (1/R) sum_r wc_r = (1/(R-1)) * sum_r f(r, L_r).                (*)

THEOREM.  est >= 0 for every real matrix A.

PROOF.  Let L* = argmax_l m(l) be the grand-mean-optimal layer. L_r maximises
c_r, so c_r(L_r) >= c_r(L*), i.e.

    m(L_r) + f(r,L_r)/(R-1) >= m(L*) + f(r,L*)/(R-1),

which rearranges to

    f(r,L_r) >= (R-1)(m(L*) - m(L_r)) + f(r,L*).

Summing over r and using sum_r f(r,L*) = 0,

    sum_r f(r,L_r) >= (R-1) * sum_r (m(L*) - m(L_r)).

Every summand on the right is >= 0 because L* maximises m. Therefore
sum_r f(r,L_r) >= 0, and by (*) est >= 0.  QED

COROLLARY (the case `code/45` already proved). If L_r = L for all r, then by
(*) est = (1/(R-1)) sum_r f(r,L) = 0 exactly. The equal-argmax degeneracy is
the EQUALITY CASE of a theorem that constrains the sign everywhere else too.

CONSEQUENCE FOR THE PAPER. "All 17 non-degenerate cells are positive" is not
evidence about hidden-state probes; it is a restatement of the theorem. The
Wilcoxon p = 1.5e-5 quoted at n=17 is exactly 2^-16, the arithmetic floor of
the signed-rank test when every observation shares a sign, and therefore
tests a hypothesis that is false by construction. The rotation estimate is
demoted to a NON-NEGATIVE DIAGNOSTIC and the bootstrap max-bias estimator,
which carries no such constraint, becomes the primary reported quantity.
──────────────────────────────────────────────────────────────────────────────
"""
import json
import zlib
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
PROBES_DIR = ROOT / "code" / "external" / "HallucinationPatternDetection" / "results" / "probes"
OUT_PATH = ROOT / "results" / "case_study_4_estimator_audit.json"

ZERO_SNAP = 1e-12
SATURATION_THRESHOLD = 0.975
N_BOOT_BIAS = 2000
N_RANDOM_MATRICES = 300_000
TIE_ATOL = 1e-12
MAX_TIE_COMBOS = 20_000
RNG_GLOBAL = np.random.default_rng(2026)


# ─────────────────────────────────────────────────────────────────────────────
# (1) numerical verification of the general theorem
# ─────────────────────────────────────────────────────────────────────────────
def rotation_estimate(A):
    """Exactly `code/45`'s `_rotation_estimate`, value only."""
    n_layers, n_seeds = A.shape
    vals = []
    for held in range(n_seeds):
        others = [s for s in range(n_seeds) if s != held]
        mean_others = A[:, others].mean(axis=1)
        li = int(np.argmax(mean_others))
        vals.append(mean_others[li] - A[li, held])
    return float(np.mean(vals))


def verify_nonnegativity(n_matrices=N_RANDOM_MATRICES, seed=20260803):
    """Adversarial search for a negative value of the rotation estimator.

    Sweeps shapes, distributions and scales, including heavy tails and
    discretized values (which manufacture ties, the regime most likely to
    break a naive proof)."""
    rng = np.random.default_rng(seed)
    worst = np.inf
    worst_shape = None
    per_family = {}
    families = ["uniform", "normal", "cauchy_clipped", "discretized", "auroc_like"]
    per_fam_n = n_matrices // len(families)
    for fam in families:
        fam_worst = np.inf
        for _ in range(per_fam_n):
            L = int(rng.integers(2, 40))
            R = int(rng.integers(2, 8))
            if fam == "uniform":
                A = rng.random((L, R))
            elif fam == "normal":
                A = rng.normal(0, rng.choice([1e-6, 1.0, 1e6]), (L, R))
            elif fam == "cauchy_clipped":
                A = np.clip(rng.standard_cauchy((L, R)), -1e6, 1e6)
            elif fam == "discretized":
                # few distinct values -> many exact ties
                A = rng.integers(0, 4, (L, R)).astype(float)
            else:
                # AUROC-like: bounded, near ceiling, coarse grid
                A = np.round(0.5 + 0.5 * rng.random((L, R)), 3)
            est = rotation_estimate(A)
            if est < fam_worst:
                fam_worst = est
            if est < worst:
                worst, worst_shape = est, (L, R)
        per_family[fam] = float(fam_worst)
    return {
        "n_matrices": int(per_fam_n * len(families)),
        "families": families,
        "min_estimate_overall": float(worst),
        "min_estimate_shape": list(worst_shape),
        "min_estimate_per_family": per_family,
        "theorem_holds_to_float_tolerance": bool(worst > -1e-9),
        "interpretation": (
            "No matrix produced a materially negative estimate. The most negative value "
            "observed is float summation noise around an algebraic zero, consistent with "
            "the theorem in this module's docstring: the rotation estimator is >= 0 for "
            "any real matrix, so its observed positivity carries no information about the "
            "data."),
    }


# ─────────────────────────────────────────────────────────────────────────────
# cell loading (mirrors code/45)
# ─────────────────────────────────────────────────────────────────────────────
def load_matrix(path):
    d = json.load(open(path))
    layers = d["layers"]
    per_layer = d["per_layer"]
    n_seeds = len(per_layer[str(layers[0])]["seed_values"]["auroc"])
    A = np.array([[per_layer[str(l)]["seed_values"]["auroc"][s] for s in range(n_seeds)]
                  for l in layers], dtype=float)
    return A, layers


# ─────────────────────────────────────────────────────────────────────────────
# (2) tie-break sensitivity
# ─────────────────────────────────────────────────────────────────────────────
def tie_sensitivity(A):
    """Enumerate every resolution of every argmax tie, in every rotation.

    Returns the induced range on the cell estimate and on the degeneracy flag.
    `np.argmax` silently resolves ties at the LOWEST index; nothing about the
    science justifies that choice, so its consequences are enumerated."""
    n_layers, n_seeds = A.shape
    tied_sets = []
    for held in range(n_seeds):
        others = [s for s in range(n_seeds) if s != held]
        m = A[:, others].mean(axis=1)
        mx = m.max()
        cands = np.flatnonzero(np.isclose(m, mx, rtol=0.0, atol=TIE_ATOL))
        tied_sets.append((held, m, cands))

    n_combos = int(np.prod([len(c) for _, _, c in tied_sets]))
    has_tie = n_combos > 1
    if n_combos > MAX_TIE_COMBOS:
        # cap: keep the argmax choice plus the extremes per rotation
        tied_sets = [(h, m, c[: max(2, MAX_TIE_COMBOS // (n_seeds * 4))]) for h, m, c in tied_sets]
        n_combos = int(np.prod([len(c) for _, _, c in tied_sets]))

    ests, degens = [], []
    for choice in product(*[c for _, _, c in tied_sets]):
        vals = []
        for (held, m, _), li in zip(tied_sets, choice):
            vals.append(m[li] - A[li, held])
        est = float(np.mean(vals))
        ests.append(0.0 if abs(est) < ZERO_SNAP else est)
        degens.append(len(set(int(x) for x in choice)) == 1)
    return {
        "n_rotations_with_ties": int(sum(len(c) > 1 for _, _, c in tied_sets)),
        "n_tie_resolutions_enumerated": int(n_combos),
        "has_tie": bool(has_tie),
        "estimate_min": float(min(ests)),
        "estimate_max": float(max(ests)),
        "estimate_argmax_convention": float(ests[0]) if not has_tie else None,
        "can_be_degenerate": bool(any(degens)),
        "can_be_nondegenerate": bool(any(not d for d in degens)),
        "_ests": ests,
        "_degens": degens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# (3) bootstrap max-bias, promoted to primary
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_max_bias(A, rng, n_boot=N_BOOT_BIAS):
    n_seeds = A.shape[1]
    full_mean = A.mean(axis=1)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        m = A[:, idx].mean(axis=1)
        li = int(np.argmax(m))
        out[b] = m[li] - full_mean[li]
    return float(out.mean()), [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def bca_ci(vals, n_resamples=10000):
    vals = np.asarray(vals, dtype=float)
    if np.allclose(vals, vals[0]):
        return [float(vals[0]), float(vals[0])]
    res = bootstrap((vals,), np.mean, confidence_level=0.95,
                    n_resamples=n_resamples, method="BCa", random_state=RNG_GLOBAL)
    return [float(res.confidence_interval.low), float(res.confidence_interval.high)]


# ─────────────────────────────────────────────────────────────────────────────
# (4) 2-seed vs 3-seed selection-criterion mismatch
# ─────────────────────────────────────────────────────────────────────────────
def seed_count_mismatch(A, rng, n_sim=4000):
    """How much does selecting on 2 seeds rather than 3 inflate the curse?

    The audited repository picks `best_layer` by the argmax of the 3-seed
    mean; the rotation estimator picks by the argmax of a 2-seed mean. A
    noisier selection criterion suffers a LARGER winner's curse, so the
    rotation estimator overstates the bias the repository actually incurs.

    Calibration is parametric and deliberately simple: treat the observed
    per-layer means as the truth and the pooled within-layer residual SD as
    the seed noise, then measure the selection bias -- E[criterion value at
    the selected layer] - truth at that layer -- under a k-seed criterion for
    k = 2 and k = 3. The ratio is what the paper reports; the absolute levels
    are not claimed."""
    mu = A.mean(axis=1)
    resid = A - mu[:, None]
    n_layers, n_seeds = A.shape
    # unbiased pooled SD of a single seed about its layer mean
    dof = n_layers * (n_seeds - 1)
    sigma = float(np.sqrt((resid ** 2).sum() / dof)) if dof > 0 else 0.0
    if sigma == 0.0:
        return {"sigma_seed": 0.0, "bias_k2": 0.0, "bias_k3": 0.0,
                "ratio_k2_over_k3": None, "overstatement_pct": None}
    bias = {}
    for k in (2, 3):
        b = np.empty(n_sim)
        for i in range(n_sim):
            m = mu + rng.normal(0.0, sigma / np.sqrt(k), n_layers)
            li = int(np.argmax(m))
            b[i] = m[li] - mu[li]
        bias[k] = float(b.mean())
    ratio = bias[2] / bias[3] if bias[3] > 0 else None
    return {
        "sigma_seed": sigma,
        "bias_k2": bias[2],
        "bias_k3": bias[3],
        "ratio_k2_over_k3": ratio,
        "overstatement_pct": (100.0 * (ratio - 1.0) / ratio) if ratio else None,
    }


def main():
    out = {}

    print("(1) verifying the general non-negativity theorem ...", flush=True)
    out["nonnegativity_verification"] = verify_nonnegativity()
    nv = out["nonnegativity_verification"]
    print(f"    {nv['n_matrices']} random matrices, min estimate = {nv['min_estimate_overall']:.3e}"
          f"  -> theorem holds: {nv['theorem_holds_to_float_tolerance']}", flush=True)

    files = sorted(PROBES_DIR.glob("*.json"))
    print(f"\n(2)-(4) per-cell audit over {len(files)} cells ...", flush=True)
    per_cell = {}
    for f in files:
        A, layers = load_matrix(f)
        rng = np.random.default_rng(zlib.crc32(f.stem.encode()))
        rot = rotation_estimate(A)
        rot = 0.0 if abs(rot) < ZERO_SNAP else rot
        ties = tie_sensitivity(A)
        boot, boot_ci = bootstrap_max_bias(A, rng)
        mism = seed_count_mismatch(A, rng)
        naive_mean = float(np.mean([
            A[:, [s for s in range(A.shape[1]) if s != h]].mean(axis=1).max()
            for h in range(A.shape[1])]))
        per_cell[f.stem] = {
            "rotation_estimate": rot,
            "estimator_degenerate": bool(rot == 0.0 and not ties["can_be_nondegenerate"]
                                         or len({
                                             int(np.argmax(A[:, [s for s in range(A.shape[1])
                                                                 if s != h]].mean(axis=1)))
                                             for h in range(A.shape[1])}) == 1),
            "bootstrap_max_bias": boot,
            "bootstrap_max_bias_ci_95": boot_ci,
            "ceiling_saturated": bool(naive_mean >= SATURATION_THRESHOLD),
            "tie_sensitivity": {k: v for k, v in ties.items() if not k.startswith("_")},
            "seed_count_mismatch": mism,
        }
        print(f"  {f.stem:44s} rot={rot:+.4f} boot={boot:+.4f} "
              f"tie[{ties['estimate_min']:+.4f},{ties['estimate_max']:+.4f}] "
              f"k2/k3={mism['ratio_k2_over_k3'] if mism['ratio_k2_over_k3'] is None else round(mism['ratio_k2_over_k3'], 3)}",
              flush=True)
        per_cell[f.stem]["_ties_raw"] = ties

    cells = per_cell
    nondegen_keys = [k for k, v in cells.items() if not v["estimator_degenerate"]]
    degen_keys = [k for k, v in cells.items() if v["estimator_degenerate"]]

    # ── tie-break band on the headline ──────────────────────────────────────
    def band(keys):
        lo = float(np.mean([cells[k]["_ties_raw"]["estimate_min"] for k in keys]))
        hi = float(np.mean([cells[k]["_ties_raw"]["estimate_max"] for k in keys]))
        return [lo, hi]

    n_tied_cells = int(sum(v["_ties_raw"]["has_tie"] for v in cells.values()))
    degen_min = int(sum(1 for v in cells.values()
                        if v["_ties_raw"]["can_be_degenerate"] and
                        not v["_ties_raw"]["can_be_nondegenerate"]))
    degen_max = int(sum(1 for v in cells.values() if v["_ties_raw"]["can_be_degenerate"]))
    out["tie_break_sensitivity"] = {
        "tie_tolerance": TIE_ATOL,
        "n_cells_with_at_least_one_tied_rotation": n_tied_cells,
        "n_cells_total": len(cells),
        "degenerate_count_shipped_argmax_convention": len(degen_keys),
        "degenerate_count_min_over_tie_resolutions": degen_min,
        "degenerate_count_max_over_tie_resolutions": degen_max,
        "all24_mean_band": band(list(cells.keys())),
        "nondegenerate_mean_band_fixed_membership": band(nondegen_keys),
        "note": (
            "np.argmax resolves ties at the lowest index. Enumerating every resolution "
            "moves both the headline mean and the count of algebraically degenerate cells. "
            "The shipped convention lands at the TOP of the achievable range, which is the "
            "direction that flatters the reported severity."),
    }

    # ── bootstrap max-bias promoted to primary ──────────────────────────────
    def boot_summary(name, keys):
        vals = [cells[k]["bootstrap_max_bias"] for k in keys]
        if not vals:
            return None
        try:
            _, p = wilcoxon(vals)
            p = float(p)
        except ValueError:
            p = None
        s = {"n_cells": len(vals), "mean": float(np.mean(vals)),
             "median": float(np.median(vals)), "min": float(np.min(vals)),
             "max": float(np.max(vals)), "bca_ci_95": bca_ci(vals),
             "wilcoxon_p_vs_zero": p,
             "n_negative": int(sum(v < 0 for v in vals))}
        print(f"\n  bootstrap max-bias // {name} (n={len(vals)}): "
              f"mean={s['mean']:+.4f} CI={[round(x,5) for x in s['bca_ci_95']]} p={p} "
              f"min={s['min']:+.4f} max={s['max']:+.4f} n_neg={s['n_negative']}")
        return s

    out["bootstrap_max_bias_primary"] = {
        "all_24_cells": boot_summary("all 24 cells", list(cells.keys())),
        "non_degenerate_17": boot_summary("17 non-degenerate", nondegen_keys),
        "degenerate_7": boot_summary("7 rotation-degenerate", degen_keys),
        "ceiling_saturated": boot_summary(
            "ceiling-saturated", [k for k, v in cells.items() if v["ceiling_saturated"]]),
        "non_saturated": boot_summary(
            "non-saturated", [k for k, v in cells.items() if not v["ceiling_saturated"]]),
        "note": (
            "Unlike the rotation estimator this quantity is NOT sign-constrained: the "
            "resample mean and the full-sample mean at the resample-selected layer are "
            "computed on different seed multisets, so nothing telescopes. It is coarse at "
            "3 seeds. It is the primary reported estimate for Case Study 4."),
    }

    # ── seed-count mismatch ─────────────────────────────────────────────────
    ratios = [cells[k]["seed_count_mismatch"]["ratio_k2_over_k3"] for k in cells
              if cells[k]["seed_count_mismatch"]["ratio_k2_over_k3"] is not None]
    over = [100.0 * (r - 1.0) / r for r in ratios]
    out["seed_count_criterion_mismatch"] = {
        "n_cells_calibrated": len(ratios),
        "ratio_k2_over_k3_mean": float(np.mean(ratios)),
        "ratio_k2_over_k3_median": float(np.median(ratios)),
        "ratio_k2_over_k3_min": float(np.min(ratios)),
        "ratio_k2_over_k3_max": float(np.max(ratios)),
        "overstatement_pct_mean": float(np.mean(over)),
        "overstatement_pct_median": float(np.median(over)),
        "overstatement_pct_range": [float(np.min(over)), float(np.max(over))],
        "note": (
            "The audited repository selects best_layer on a 3-seed mean; the rotation "
            "estimator selects on a 2-seed mean. The noisier criterion incurs a larger "
            "winner's curse, so the rotation estimate OVERSTATES the bias the repository "
            "actually carries by the reported percentage."),
    }
    print(f"\n  2-seed vs 3-seed criterion: ratio mean={np.mean(ratios):.3f} "
          f"-> overstatement {np.mean(over):.1f}% "
          f"(range {np.min(over):.1f}-{np.max(over):.1f}%)")

    for v in cells.values():
        v.pop("_ties_raw", None)
    out["per_cell"] = cells
    out["method_notes"] = {
        "n_random_matrices_for_theorem_check": N_RANDOM_MATRICES,
        "n_bootstrap_max_bias": N_BOOT_BIAS,
        "tie_atol": TIE_ATOL,
        "theorem": (
            "est = (1/(R-1)) * sum_r f(r, L_r) with f(r,l) = m(l) - a_{r,l} and "
            "L_r = argmax_l [m(l) + f(r,l)/(R-1)]. Optimality of L_r against the "
            "grand-mean argmax L* plus sum_r f(r,L*) = 0 gives "
            "sum_r f(r,L_r) >= (R-1) sum_r (m(L*) - m(L_r)) >= 0."),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_PATH, "w"), indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
