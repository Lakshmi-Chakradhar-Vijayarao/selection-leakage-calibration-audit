"""
Case Study 2 (GUARDIAN): the MECHANICAL NULL for Delta_sel, and the
A/B decomposition it makes available.

WHY THIS SCRIPT EXISTS. The paper already concedes (SS4.2, Appendix B.2) that
E[Delta_sel] > 0 under a pure-noise null, because
l* = argmax_l cv_l while gap_l = cv_l - ho_l shares that same +cv_l term. Having
conceded that, it then reported Delta_sel = +0.0255 against a Wilcoxon test of
H0: Delta_sel = 0 -- a hypothesis the paper itself says is false a priori. An
independent review made the obvious point: the reference the measurement needs
is not zero, it is the value the same estimator returns when the layer ordering
carries no transferable information.

THE NULL. Delta_sel is a function of two per-layer vectors, cv (the CV AUROC on
the selection pool) and ho (the held-out AUROC). The mechanism under test is
whether the layer that wins on cv is also a genuinely better layer on ho. The
null that removes exactly that -- and nothing else -- permutes which layer's ho
value is paired with which layer's cv value, leaving both marginal
distributions, the argmax rule, and the number of layers untouched:

    l*      = argmax_l cv_l                       (unchanged)
    gap_l   = cv_l - ho_{pi(l)}                   (pi a random layer permutation)
    Delta   = gap_{l*} - mean_l gap_l

Drawn 20,000 times, averaging over all 50 randomized reps within each draw, so
the null is a null for the reported rep-mean.

THE DECOMPOSITION THE NULL EXPOSES. Because E_pi[ho_{pi(l)}] = mean_l ho_l, the
null's expectation is exactly

    A = mean_reps ( cv_{l*} - mean_l cv_l )       "winner's curse on the
                                                   selection criterion"

and the observed statistic is exactly

    Delta_sel = A - B,   B = mean_reps ( ho_{l*} - mean_l ho_l )
                                                  "transferred layer quality"

B is how much better than an average layer the CV-selected layer actually is on
data that played no part in selecting it. B > 0 means CV-argmax layer selection
carries real, transferable signal, and B/A is the fraction of the raw winner's
curse that signal cancels. This is a strictly more informative reading of the
same data than a test against zero, and it is favourable to the finding rather
than damaging: the reported +0.0255 is the RESIDUAL that survives after real
transferred signal has already offset part of the curse.

Input: results/case_study_2_probe_scores.npz (the shipped 1 MB derived
artifact -- no 171 MB hidden-state cache required). The per-layer cv/ho AUROCs
are recomputed exactly as code/51 does, and the resulting Delta_sel is asserted
against the committed JSON before any null is drawn.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "results" / "case_study_2_probe_scores.npz"
REF_JSON = ROOT / "results" / "case_study_2_layer_decomposition.json"
OUT_PATH = ROOT / "results" / "case_study_2_selection_null.json"

N_DRAWS = 20000
SEED = 20260803
TOL = 1e-5  # the artifact stores scores as float32; see code/51


def per_layer_cv_ho(a, variant, n_layers):
    from sklearn.metrics import roc_auc_score
    y_sel = a[f"{variant}__y_sel"]
    y_ho = a[f"{variant}__y_ho"]
    fold_id = a[f"{variant}__fold_id"]
    cv_scores = a[f"{variant}__cv_scores"]
    ho_scores = a[f"{variant}__ho_scores"]
    folds = sorted(set(fold_id.tolist()))
    cv = np.array([
        np.mean([roc_auc_score(y_sel[fold_id == f], cv_scores[l][fold_id == f])
                 for f in folds]) for l in range(n_layers)])
    ho = np.array([roc_auc_score(y_ho, ho_scores[l]) for l in range(n_layers)])
    return cv, ho


def main():
    a = np.load(ARTIFACT)
    n_layers = int(a["n_layers"])
    n_reps = int(a["n_reps"])
    ref = json.load(open(REF_JSON))["randomized_stratified_splits"]

    CV = np.empty((n_reps, n_layers))
    HO = np.empty((n_reps, n_layers))
    for r in range(n_reps):
        CV[r], HO[r] = per_layer_cv_ho(a, f"rand{r}", n_layers)
    l_star = CV.argmax(axis=1)
    rows = np.arange(n_reps)

    gaps = CV - HO
    delta_per_rep = gaps[rows, l_star] - gaps.mean(axis=1)
    delta_obs = float(delta_per_rep.mean())

    # Guard: the replayed statistic must match the committed one before any
    # null is drawn from it.
    d = abs(delta_obs - ref["selection_specific_component_mean"])
    assert d <= TOL, (f"replayed Delta_sel {delta_obs:+.6f} does not match committed "
                      f"{ref['selection_specific_component_mean']:+.6f} (|d|={d:.2e})")
    assert list(l_star) == list(ref["selected_layer_per_rep"]), "selected layers differ"
    print(f"Replayed Delta_sel = {delta_obs:+.6f} "
          f"(committed {ref['selection_specific_component_mean']:+.6f}, |d|={d:.1e})\n")

    # ── A / B decomposition ──────────────────────────────────────────────────
    A_per_rep = CV[rows, l_star] - CV.mean(axis=1)
    B_per_rep = HO[rows, l_star] - HO.mean(axis=1)
    A, B = float(A_per_rep.mean()), float(B_per_rep.mean())
    tB, pB = ttest_1samp(B_per_rep, 0.0)
    tA, pA = ttest_1samp(A_per_rep, 0.0)
    assert abs((A - B) - delta_obs) < 1e-12, "A - B must equal Delta_sel exactly"

    # ── layer-permutation null ───────────────────────────────────────────────
    rng = np.random.default_rng(SEED)
    null = np.empty(N_DRAWS)
    for d_i in range(N_DRAWS):
        # independent layer permutation per rep, per draw
        perm = rng.permuted(np.tile(np.arange(n_layers), (n_reps, 1)), axis=1)
        HOp = np.take_along_axis(HO, perm, axis=1)
        g = CV - HOp
        null[d_i] = (g[rows, l_star] - g.mean(axis=1)).mean()

    null_mean = float(null.mean())
    null_sd = float(null.std(ddof=1))
    lo, hi = float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))
    p_le = float((null <= delta_obs).mean())
    p_ge = float((null >= delta_obs).mean())

    out = {
        "observed_delta_sel": delta_obs,
        "n_reps": n_reps,
        "n_layers": n_layers,
        "n_null_draws": N_DRAWS,
        "layer_permutation_null": {
            "mean": null_mean,
            "sd": null_sd,
            "ci_95": [lo, hi],
            "p_null_le_observed": p_le,
            "p_null_ge_observed": p_ge,
            "observed_below_entire_ci": bool(delta_obs < lo),
            "description": (
                "Permutes which layer's held-out AUROC is paired with which layer's CV "
                "AUROC, 20,000 draws, independently per rep per draw. Leaves both marginal "
                "vectors, the argmax rule and the layer count untouched; removes only the "
                "cv-to-ho layer correspondence, i.e. exactly the transferable component."),
        },
        "decomposition": {
            "A_winners_curse_on_selection_criterion": A,
            "A_sd_over_reps": float(A_per_rep.std(ddof=1)),
            "A_t": float(tA), "A_p": float(pA),
            "B_transferred_layer_quality": B,
            "B_sd_over_reps": float(B_per_rep.std(ddof=1)),
            "B_t": float(tB), "B_p": float(pB),
            "B_n_negative_reps": int((B_per_rep < 0).sum()),
            "B_over_A_fraction_offset": float(B / A),
            "identity_check_A_minus_B_equals_delta": float(A - B - delta_obs),
            "A_equals_null_mean_abs_diff": float(abs(A - null_mean)),
            "description": (
                "Delta_sel = A - B exactly. A = cv at the selected layer minus mean cv "
                "(the winner's curse on the selection criterion; also the analytic mean of "
                "the permutation null). B = ho at the selected layer minus mean ho (how "
                "much genuinely better the selected layer is on data that played no part "
                "in selecting it). B/A is the fraction of the raw curse that real, "
                "transferred layer quality cancels."),
        },
        "reading": (
            "Delta_sel = +0.0255 is NOT above its mechanical null -- it is below the "
            "null's entire 95% interval. The correct statement is not 'selection optimism "
            "is significantly positive' (an unsurprising claim against a null the paper "
            "itself calls false) but 'CV-argmax layer selection recovers a genuinely "
            "better-than-average layer, and that transferred quality offsets "
            f"{100 * B / A:.1f}% of the raw winner's curse, leaving +{delta_obs:.4f} as "
            "the residual optimism a practitioner still pays.'"),
    }

    print(f"A (winner's curse on cv)        = {A:+.6f}  (SD {A_per_rep.std(ddof=1):.4f}, t={tA:.2f}, p={pA:.3g})")
    print(f"B (transferred layer quality)   = {B:+.6f}  (SD {B_per_rep.std(ddof=1):.4f}, t={tB:.2f}, p={pB:.3g}, "
          f"{int((B_per_rep < 0).sum())}/{n_reps} negative)")
    print(f"Delta_sel = A - B               = {delta_obs:+.6f}")
    print(f"B offsets {100 * B / A:.1f}% of A\n")
    print(f"Permutation null: mean={null_mean:+.6f}  SD={null_sd:.6f}  95% [{lo:+.6f}, {hi:+.6f}]")
    print(f"  |A - null mean| = {abs(A - null_mean):.2e}  (they must agree; A is the null's analytic mean)")
    print(f"  P(null <= observed) = {p_le:.4f}   observed below the whole interval: {delta_obs < lo}")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
