"""
Mechanism 2, CONSTRUCTION B: a second, genuinely different
synthetic harness for the same underlying phenomenon code/97's Mechanism 2
measures (CV-argmax layer selection, GUARDIAN), built to directly test
whether code/97's fitted per-mechanism intercept for Mechanism 2 is a
property of the MECHANISM or an artifact of the one harness that measured it.

WHY THIS SCRIPT EXISTS. §5.9 (code/97) fits a per-mechanism-intercept model
across all five leakage mechanisms at a shared (K, AUROC_0) grid and finds it
beats a pooled, mechanism-agnostic model decisively (R^2 0.826 -> 0.997,
nested F(3,42)=799.7, p=1.1e-16). Three independent reviewers across two
rounds flagged the same gap in that result: Mechanism 2's harness (K
independent isotropic-Gaussian feature blocks, each an i.i.d. draw calibrated
to a shared target AUROC_0) was built solely for that section, has never been
independently reconstructed a second way, and has been through none of the
adversarial correction rounds this paper's other four harnesses have. Its
fitted intercept (a_2 = -3.389, the second-most-extreme of the four) could
reflect a real property of "CV-argmax selection among exchangeable
candidates" as a mechanism, or it could just reflect this ONE harness's own
modeling choices. This script settles that as far as one clean, well-motivated
alternative construction can: it changes exactly one thing about how the K
candidates relate to each other, reruns the identical measurement at the
identical grid, and reports whether the fitted severity level moves.

WHAT CHANGES, AND WHY IT IS A GENUINE CONSTRUCTION CHANGE, NOT A COSMETIC ONE.
code/97's Mechanism 2 draws each candidate "layer" k's feature block as an
INDEPENDENT isotropic Gaussian, X_k = class_sep/2 * sign + N(0, I_64), with
zero correlation between any two layers' feature draws: Cov(X_k, X_k') = 0
for k != k'. Real transformer layers are not like this -- adjacent layers of
one residual stream carry heavily overlapping content, so K real candidate
"layers" sit much closer to "one shared signal plus a smaller idiosyncratic
increment per layer" than to "K independent draws." This script replaces the
independence assumption with a one-factor latent model:

    X_k = class_sep/2 * sign + sqrt(RHO) * L + sqrt(1 - RHO) * E_k

where L ~ N(0, I_64) is drawn ONCE per seed and shared by all K candidate
layers, and E_k ~ N(0, I_64) is idiosyncratic noise, independent across k and
independent of L. RHO in [0, 1) is the fraction of each layer's unit noise
variance carried by the shared factor; RHO = 0 recovers construction A
exactly (code/97). RHO = 0.5 is used here -- substantial shared structure,
representative of the kind of redundancy adjacent hidden layers of one
residual stream show, without collapsing the K candidates to being identical
(RHO -> 1, which would make every layer's CV/held-out draw the same up to
noise realized after fitting) or independent (RHO = 0, construction A). This
is a disclosed, one-parameter choice, not a search over RHO for the answer we
wanted -- RHO = 0.5 was fixed before this script was run.

Because Var(X_k) = RHO + (1 - RHO) = 1 for every k regardless of RHO, EACH
LAYER'S OWN MARGINAL distribution -- and therefore its calibration to the
target AUROC_0 via this paper's Phi(sqrt(J/2)) binormal identity -- is
IDENTICAL to construction A's, cell for cell. Only the CROSS-layer covariance
changes (Cov(X_k, X_k') = RHO for k != k', vs. 0 in construction A). This is
exactly the property that governs how much independent "luck" an
argmax-over-K selection rule can exploit: correlated candidates are, in
effect, fewer independent draws than K, so a smaller winner's-curse-style
selection bias is the a priori expectation under this construction relative
to construction A's fully independent candidates -- a testable prediction of
the construction, reported rather than assumed.

EVERYTHING ELSE IS HELD FIXED, IMPORTED FROM code/97 RATHER THAN RETYPED, so
that construction A and B differ in exactly the one respect described above:
the grid (K in {5,10,20,40,80} x AUROC_0 in {0.70,0.80,0.95}), N_SEEDS_M2=100,
FEAT_DIM_M2=64, N_SAMPLES_M2=700, TEST_SIZE_M2=0.20, N_INNER_FOLDS_M2=5, the
LogisticRegression estimator, the LEAKY non-nested argmax rule, the
gap_l = cv_l - ho_l / Delta_sel = gap_{l*} - mean_l(gap_l) severity
definition, the mechanical (layer-permutation) null, and the log-linear
NLS-fit machinery used for the per-mechanism-intercept comparison.

PRE-REGISTERED STABILITY RULE (fixed before this script was run, evaluated
after). Construction B's own fitted log-linear intercept a_B (gap = exp(a_B +
beta ln K + c z0), fit to construction B's 15 cells alone, identical form and
fitting procedure to code/97's per-mechanism intercepts) is compared against
construction A's Mechanism-2 intercept, a_2 = -3.389 (code/97,
results/matched_head_to_head_five_mechanisms.json). We call the intercept
STABLE if |a_B - a_2| < ln(10) ~= 2.303 (i.e., construction B's fitted
severity level sits within one order of magnitude of construction A's) AND
the two constructions' 15 matched per-cell gaps are positively and
significantly correlated (Pearson r > 0, p < 0.05) across the shared grid.
We call it UNSTABLE if either condition fails. This is the honest reading
either way: STABLE supports "mechanism identity is a real, harness-independent
property this comparison is entitled to name"; UNSTABLE supports "the
maturity/construction confound the reviewers named is real, and §5.9's
causal language must stay hedged regardless of what the four-mechanism model
comparison's AICc/LOO-R^2 show."

A second, complementary check: the four-mechanism pooled model comparison is
refit with Mechanism 2's construction-A cells REPLACED by construction B's
(Mechanisms 1, 3, 4 untouched, taken verbatim from code/97's own output), to
see whether the qualitative conclusion (mechanism-aware model wins; wide
intercept spread) survives the substitution, and by how much the fitted
Mechanism-2 intercept itself moves once it is estimated under a different
harness for the identical mechanism.

Output: results/mechanism2_construction_b_correlated_latent.json
"""
import importlib.util
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm, pearsonr

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
OUT_PATH = ROOT / "results" / "mechanism2_construction_b_correlated_latent.json"
H2H_PATH = ROOT / "results" / "matched_head_to_head_five_mechanisms.json"

# ── import code/97 verbatim for every shared piece of machinery ─────────────
_spec = importlib.util.spec_from_file_location(
    "s97_m99", CODE / "97_matched_head_to_head_five_mechanisms.py")
s97 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s97)

K_GRID = s97.K_GRID
AUROC0_GRID = s97.AUROC0_GRID
N_SEEDS_M2 = s97.N_SEEDS_M2          # 100
FEAT_DIM_M2 = s97.FEAT_DIM_M2        # 64
N_SAMPLES_M2 = s97.N_SAMPLES_M2      # 700
TEST_SIZE_M2 = s97.TEST_SIZE_M2      # 0.20
N_INNER_FOLDS_M2 = s97.N_INNER_FOLDS_M2  # 5

RHO = 0.5   # disclosed, fixed before the run; RHO=0 would recover construction A


def _m2b_one_seed(seed, K, auroc0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.default_rng(seed)
    class_sep = s97.realized_class_sep(auroc0, FEAT_DIM_M2)
    y = np.array([0, 1] * (N_SAMPLES_M2 // 2))
    rng.shuffle(y)
    sign = np.where(y == 1, 1.0, -1.0).reshape(-1, 1)

    idx = rng.permutation(N_SAMPLES_M2)
    n_ho = int(round(N_SAMPLES_M2 * TEST_SIZE_M2))
    ho_idx, sel_idx = idx[:n_ho], idx[n_ho:]
    y_sel, y_ho = y[sel_idx], y[ho_idx]

    skf = StratifiedKFold(n_splits=N_INNER_FOLDS_M2, shuffle=True,
                          random_state=seed + 900000)
    splits = list(skf.split(np.zeros(len(y_sel)), y_sel))

    # ONE shared latent factor per seed, common to every candidate "layer" k --
    # the one change from construction A's fully independent per-layer draws.
    L = rng.standard_normal((N_SAMPLES_M2, FEAT_DIM_M2))

    cv = np.empty(K)
    ho = np.empty(K)
    for k in range(K):
        Ek = rng.standard_normal((N_SAMPLES_M2, FEAT_DIM_M2))
        X = (class_sep / 2.0 * sign + np.sqrt(RHO) * L
             + np.sqrt(1.0 - RHO) * Ek)
        X_sel, X_ho = X[sel_idx], X[ho_idx]
        fold_aucs = []
        for tr, te in splits:
            clf = LogisticRegression(max_iter=1000).fit(X_sel[tr], y_sel[tr])
            fold_aucs.append(roc_auc_score(y_sel[te], clf.predict_proba(X_sel[te])[:, 1]))
        cv[k] = np.mean(fold_aucs)
        clf_full = LogisticRegression(max_iter=1000).fit(X_sel, y_sel)
        ho[k] = roc_auc_score(y_ho, clf_full.predict_proba(X_ho)[:, 1])
    return cv, ho


def _m2b_cell_job(args):
    K, a0, n_seeds = args
    CV = np.empty((n_seeds, K))
    HO = np.empty((n_seeds, K))
    for s in range(n_seeds):
        CV[s], HO[s] = _m2b_one_seed(s, K, a0)
    l_star = CV.argmax(axis=1)
    rows = np.arange(n_seeds)
    gaps = CV - HO
    delta_per_seed = gaps[rows, l_star] - gaps.mean(axis=1)
    A_per_seed = CV[rows, l_star] - CV.mean(axis=1)
    B_per_seed = HO[rows, l_star] - HO.mean(axis=1)

    # empirical induced correlation of the CV columns across candidates --
    # the diagnostic that confirms the construction actually differs from A's.
    if K > 1:
        C = np.corrcoef(CV.T)
        off_diag = C[~np.eye(K, dtype=bool)]
        mean_pairwise_corr_cv = float(np.mean(off_diag))
    else:
        mean_pairwise_corr_cv = float("nan")

    rng = np.random.default_rng(800000 + K * 1000 + int(round(a0 * 1000)))
    n_draws = 20000
    null_draws = np.empty(n_draws)
    for d in range(n_draws):
        perm = rng.permuted(np.tile(np.arange(K), (n_seeds, 1)), axis=1)
        HOp = np.take_along_axis(HO, perm, axis=1)
        g = CV - HOp
        null_draws[d] = (g[rows, l_star] - g.mean(axis=1)).mean()

    delta_obs = float(delta_per_seed.mean())
    return {
        "K": K, "auroc0": a0, "n_seeds": n_seeds,
        "gap_mean": delta_obs,
        "gap_sem": float(delta_per_seed.std(ddof=1) / np.sqrt(n_seeds)),
        "gap_bca_ci_95": s97.bca_ci(delta_per_seed, seed=K * 41 + int(a0 * 1000)),
        "wilcoxon_p_vs_zero": s97.wilcoxon_p(delta_per_seed),
        "A_winners_curse_on_selection_criterion": float(A_per_seed.mean()),
        "B_transferred_layer_quality": float(B_per_seed.mean()),
        "mean_pairwise_corr_cv_across_candidates": mean_pairwise_corr_cv,
        "mechanical_null_mean": float(null_draws.mean()),
        "mechanical_null_sd": float(null_draws.std(ddof=1)),
        "mechanical_null_ci_95": [float(np.percentile(null_draws, 2.5)),
                                  float(np.percentile(null_draws, 97.5))],
        "observed_vs_mechanical_null": float(delta_obs - float(null_draws.mean())),
        "observed_below_null_ci": bool(delta_obs < np.percentile(null_draws, 2.5)),
        "observed_above_null_ci": bool(delta_obs > np.percentile(null_draws, 97.5)),
    }


def run_construction_b():
    t0 = time.time()
    jobs = [(K, a0, N_SEEDS_M2) for a0 in AUROC0_GRID for K in K_GRID]
    n_proc = max(1, min(6, (os.cpu_count() or 2) - 2))
    print(f"=== Mechanism 2, construction B (RHO={RHO} shared-latent-factor "
          f"correlated candidates), n_seeds={N_SEEDS_M2}, {n_proc} procs ===",
          flush=True)
    cells = {}
    with mp.Pool(n_proc) as pool:
        for r in pool.imap_unordered(_m2b_cell_job, jobs):
            key = f"{r['K']}|{r['auroc0']}"
            cells[key] = r
            print(f"  K={r['K']:3d} AUROC0={r['auroc0']}: gap={r['gap_mean']:+.5f} "
                  f"mean_pairwise_corr_cv={r['mean_pairwise_corr_cv_across_candidates']:+.3f} "
                  f"mech_null={r['mechanical_null_mean']:+.5f} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
    return {
        "design": (
            f"CONSTRUCTION B (independent second reconstruction of Mechanism 2, "
            f"see module docstring). K candidate 'layers' share ONE latent factor "
            f"L per seed (Cov(X_k,X_k')=RHO={RHO} for k!=k', vs. 0 in construction "
            f"A), while each layer's own marginal distribution -- and its "
            f"calibration to the target AUROC_0 -- is unchanged from construction "
            f"A. Same LEAKY/severity definition, same grid, same N_SEEDS, same "
            f"estimator (LogisticRegression), same mechanical null."),
        "rho": RHO,
        "n_seeds": N_SEEDS_M2, "feat_dim": FEAT_DIM_M2, "n_samples": N_SAMPLES_M2,
        "cells": cells,
        "runtime_seconds": time.time() - t0,
    }


def main():
    t0 = time.time()
    b = run_construction_b()

    h2h = json.load(open(H2H_PATH))
    a_cells = h2h["mechanism_2"]["cells"]
    a_intercept_2 = h2h["pooled_auroc_scale_model_comparison"][
        "model_b_mechanism_aware"]["per_mechanism_intercept_a_m"]["2"]

    # ── own-fit log-linear model on construction B's 15 cells alone ─────────
    keys = [f"{K}|{a0}" for a0 in AUROC0_GRID for K in K_GRID]
    lnK = np.array([np.log(b["cells"][k]["K"]) for k in keys])
    z0 = np.array([s97.z0_of(b["cells"][k]["auroc0"]) for k in keys])
    y_b = np.array([b["cells"][k]["gap_mean"] for k in keys])
    floor = 1e-6
    n_floored_b = int((y_b <= 0).sum())
    y_b_floored = np.maximum(y_b, floor)
    X_own = s97._design_pooled(lnK, z0)
    coef_own, yhat_own = s97._nls_fit(X_own, y_b_floored, n_intercepts=1)
    r2_own = s97._r2(y_b_floored, yhat_own)
    a_B_own = float(coef_own[0])

    # ── per-cell A vs B comparison ───────────────────────────────────────────
    y_a = np.array([a_cells[k]["gap_mean"] for k in keys])
    pearson_r, pearson_p = pearsonr(y_a, y_b)
    ratio = np.where(np.abs(y_a) > 1e-9, y_b / y_a, np.nan)
    per_cell = {
        k: {"K": b["cells"][k]["K"], "auroc0": b["cells"][k]["auroc0"],
            "gap_construction_A": float(a_cells[k]["gap_mean"]),
            "gap_construction_B": float(b["cells"][k]["gap_mean"]),
            "ratio_B_over_A": float(ratio[i]) if np.isfinite(ratio[i]) else None,
            "mean_pairwise_corr_cv_B": b["cells"][k]["mean_pairwise_corr_cv_across_candidates"]}
        for i, k in enumerate(keys)
    }

    # ── stability rule, pre-registered in the module docstring ─────────────
    intercept_delta = abs(a_B_own - a_intercept_2)
    intercept_within_order_of_magnitude = intercept_delta < np.log(10)
    correlated_and_significant = bool(pearson_r > 0 and pearson_p < 0.05)
    stable = bool(intercept_within_order_of_magnitude and correlated_and_significant)

    # ── substituted 4-mechanism pooled model: swap construction B in for M2 ──
    pooled_cells_swapped = []
    for mech, key_prefix in ((1, "1|"), (3, None), (4, None)):
        pass  # placeholder, built explicitly below for clarity
    pooled_cells_swapped = []
    for k, c in h2h["mechanism_1"]["cells"].items():
        pooled_cells_swapped.append({"mechanism": 1, "K": c["K"], "auroc0": c["auroc0"],
                                     "gap": max(c["gap_mean"], floor)})
    for k, c in h2h["mechanism_3"]["cells"].items():
        pooled_cells_swapped.append({"mechanism": 3, "K": c["K"], "auroc0": c["auroc0"],
                                     "gap": max(c["gap_mean"], floor)})
    for k, c in h2h["mechanism_4"]["cells"].items():
        pooled_cells_swapped.append({"mechanism": 4, "K": c["K"], "auroc0": c["auroc0"],
                                     "gap": max(c["gap_mean"], floor)})
    for k in keys:
        c = b["cells"][k]
        pooled_cells_swapped.append({"mechanism": 2, "K": c["K"], "auroc0": c["auroc0"],
                                     "gap": max(c["gap_mean"], floor)})
    comparison_swapped = s97.fit_model_comparison(pooled_cells_swapped)

    a_intercept_2_swapped = comparison_swapped["model_b_mechanism_aware"][
        "per_mechanism_intercept_a_m"]["2"]

    out = {
        "framing": (
            "Independent second construction of Mechanism 2 (code/97's newly "
            "built harness), changing exactly one thing -- cross-candidate "
            "correlation via a shared latent factor, RHO=0.5, vs. construction "
            "A's fully independent candidates -- to test whether construction "
            "A's fitted per-mechanism intercept for Mechanism 2 (a_2=-3.389) is "
            "a property of the mechanism or an artifact of that one harness. "
            "Reported as measured, in whichever direction it came out."),
        "grid": {"K_values": K_GRID, "auroc0_values": AUROC0_GRID},
        "construction_a_reference": {
            "source": "results/matched_head_to_head_five_mechanisms.json (code/97)",
            "per_mechanism_intercept_a_2": a_intercept_2,
        },
        "construction_b": b,
        "construction_b_own_fit": {
            "form": "gap = exp(a_B + beta lnK + c z0), fit to construction B's 15 cells alone",
            "a_B": a_B_own,
            "beta": float(coef_own[1]), "c": float(coef_own[2]),
            "r_squared": r2_own,
            "n_cells_floored_to_positive": n_floored_b, "floor_value": floor,
        },
        "per_cell_a_vs_b": per_cell,
        "a_vs_b_correlation": {
            "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
            "n_cells": len(keys),
        },
        "stability_assessment": {
            "rule": (
                "Pre-registered in this script's module docstring, fixed before "
                "the run. STABLE requires BOTH: (1) |a_B - a_2| < ln(10)~=2.303 "
                "(construction B's own fitted intercept within one order of "
                "magnitude of construction A's Mechanism-2 intercept), and (2) "
                "the two constructions' 15 matched per-cell gaps correlate "
                "positively and significantly (Pearson r>0, p<0.05)."),
            "a_B_own_fit": a_B_own,
            "a_2_construction_A": a_intercept_2,
            "intercept_delta_log_scale": float(intercept_delta),
            "intercept_within_order_of_magnitude": bool(intercept_within_order_of_magnitude),
            "per_cell_correlation_positive_and_significant": correlated_and_significant,
            "verdict": "STABLE" if stable else "UNSTABLE",
        },
        "pooled_model_comparison_with_construction_b_substituted_for_mechanism_2": {
            "note": (
                "Mechanisms 1, 3, 4 cells are taken verbatim from code/97's "
                "output (untouched); Mechanism 2's 15 cells are construction "
                "B's, replacing construction A's. Refit with the identical "
                "per-mechanism-intercept model code/97 uses."),
            "comparison": comparison_swapped,
            "mechanism_2_intercept_original_construction_a": a_intercept_2,
            "mechanism_2_intercept_construction_b_substituted": a_intercept_2_swapped,
            "mechanism_2_intercept_shift": float(a_intercept_2_swapped - a_intercept_2),
        },
        "limitations": [
            "One correlation structure (RHO=0.5, one-factor latent model) is "
            "tested; this is not a sweep over RHO and does not characterize how "
            "severity depends on RHO continuously. RHO=0.5 was fixed before this "
            "script was run, chosen as a disclosed, representative middle value "
            "between construction A's RHO=0 and full collapse at RHO->1, not "
            "selected after seeing the result.",
            "This still leaves the estimator (LogisticRegression), the sample "
            "sizes, and the severity definition identical to construction A's; "
            "it isolates the cross-candidate-correlation axis specifically and "
            "does not test estimator-family sensitivity (a genuinely different "
            "possible third construction, not attempted here, per this script's "
            "brief to change one thing at a time).",
            "Both constructions remain synthetic reconstructions of Mechanism 2, "
            "not a measurement on GUARDIAN's real pipeline; this comparison "
            "bounds what a controlled-harness robustness check can say about "
            "construction-sensitivity, not whether either harness matches real "
            "hidden-state layer selection quantitatively.",
        ],
        "runtime_seconds": time.time() - t0,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")
    print(f"a_B (construction B own fit) = {a_B_own:+.4f}  vs  "
          f"a_2 (construction A) = {a_intercept_2:+.4f}  "
          f"(delta={intercept_delta:.4f}, threshold ln(10)={np.log(10):.4f})")
    print(f"Pearson r (per-cell A vs B) = {pearson_r:+.4f}, p={pearson_p:.4g}")
    print(f"STABILITY VERDICT: {'STABLE' if stable else 'UNSTABLE'}")
    print(f"Mechanism-2 intercept under substitution: {a_intercept_2_swapped:+.4f} "
          f"(shift from construction A: {a_intercept_2_swapped - a_intercept_2:+.4f})")
    print(f"Total runtime: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
