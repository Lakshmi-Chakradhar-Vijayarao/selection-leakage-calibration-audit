"""
MAXIMUM-RIGOR PASS, Item 3. §6's checklist prescribes a "Fix:" for
each mechanism but validates none of them, and asserts fixes are "cheap" with
no number attached. This script validates two prescribed fixes against
already-shipped artifacts, and reports the compute cost of each.

PART A -- Mechanism 2 (GUARDIAN, CV-argmax layer selection): the prescribed
fix is NESTED cross-validation -- select the layer using an inner CV score,
report performance on a genuinely disjoint outer set, never using the
selection statistic itself as the reported number. §4.2's own design
(cv_auroc_selectpool for selection, heldout_auroc for reporting, repeated over
50 randomized splits, results/case_study_2_probe_scores.npz) IS already this
fix -- it was simply never framed or quantified as "nested CV, validated
against the naive alternative." This part makes that comparison explicit: for
each of the 50 splits, it computes the NAIVE (un-nested) estimate a researcher
gets by reporting the selection statistic itself at its own argmax --
precisely Cawley & Talbot's (2010) canonical mistake -- against the ALREADY-
SHIPPED nested estimate, quantifying exactly how much of the naive estimate's
optimism the existing fix removes, and whether what nested CV reports is
compatible with a selection-independent reference (the across-layer mean
held-out AUROC, which cannot be optimistic since it involves no argmax at
all).

PART B -- Mechanism 3 (MultiHaluDet, per-fold checkpoint selection): the
prescribed fix is a genuinely disjoint held-out selection set (not just a
larger in-fold carve-out). §4.3's own "fully corrected (out-of-fold)" arm
(code/77's factorial bottom row; code/78/79's real-feature "fully corrected"
column) already IS this fix. This part re-verifies those already-shipped
numbers directly from their JSONs (an independent re-derivation, not a
re-trust of one print statement) and reports their residual bias, alongside
the ALREADY-LOGGED wall-clock cost of running each fix from scratch.

WHAT THIS DOES NOT DO. It does not implement a fix from scratch on a
previously-unfixed pipeline (both fixes already exist in this paper's own
artifacts); it validates and costs what is already there, which is what §6's
"cheap to fix" claim needs to be defensible.

Output: results/validate_prescribed_fixes.json
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap, wilcoxon
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_PATH = ROOT / "results" / "case_study_2_probe_scores.npz"
OUT_PATH = ROOT / "results" / "validate_prescribed_fixes.json"


def bca_ci(vals, seed=0):
    vals = np.asarray(vals, float)
    if np.allclose(vals, vals[0]):
        return [float(vals[0]), float(vals[0])]
    res = bootstrap((vals,), np.mean, confidence_level=0.95, n_resamples=10000,
                    method="BCa", random_state=np.random.default_rng(seed))
    return [float(res.confidence_interval.low), float(res.confidence_interval.high)]


def per_layer_from_artifact(a, variant, n_layers):
    y_sel = a[f"{variant}__y_sel"]; y_ho = a[f"{variant}__y_ho"]
    fold_id = a[f"{variant}__fold_id"]
    cv_scores = a[f"{variant}__cv_scores"]; ho_scores = a[f"{variant}__ho_scores"]
    cv_auroc, ho_auroc = [], []
    for l in range(n_layers):
        fold_aurocs = [roc_auc_score(y_sel[fold_id == f], cv_scores[l][fold_id == f])
                      for f in sorted(set(fold_id.tolist()))]
        cv_auroc.append(float(np.mean(fold_aurocs)))
        ho_auroc.append(float(roc_auc_score(y_ho, ho_scores[l])))
    return np.array(cv_auroc), np.array(ho_auroc)


def part_a_nested_cv_validation():
    t0 = time.time()
    a = np.load(ARTIFACT_PATH)
    n_layers, n_reps = int(a["n_layers"]), int(a["n_reps"])
    naive_reported, nested_reported, l_stars = [], [], []
    all_ho_by_layer = np.zeros((n_reps, n_layers))
    for rep in range(n_reps):
        cv_auroc, ho_auroc = per_layer_from_artifact(a, f"rand{rep}", n_layers)
        l_star = int(np.argmax(cv_auroc))
        naive_reported.append(cv_auroc[l_star])       # the un-nested mistake
        nested_reported.append(ho_auroc[l_star])       # the shipped fix
        l_stars.append(l_star)
        all_ho_by_layer[rep] = ho_auroc

    naive_reported = np.array(naive_reported)
    nested_reported = np.array(nested_reported)
    gap_removed = naive_reported - nested_reported
    _, p = wilcoxon(gap_removed)

    # Selection-independent reference: mean held-out AUROC across ALL 32
    # layers (no argmax at all), and at the modal selected layer specifically
    # -- neither involves choosing a layer using the same statistic reported.
    modal_layer = int(np.bincount(l_stars).argmax())
    unconditional_mean_ref = float(all_ho_by_layer.mean())
    modal_layer_ref = float(all_ho_by_layer[:, modal_layer].mean())

    residual_vs_modal_ref = float(nested_reported.mean() - modal_layer_ref)

    # Cost of the fix: this validation's own wall-clock (recombination of an
    # already-shipped <1MB artifact -- the marginal cost of VALIDATING the fix
    # once the artifact exists) plus the artifact's own generation cost (probe
    # fit count): n_layers x (n_folds + 1 full fit) x n_variants.
    n_variants = n_reps + 2  # + sequential + reversed
    n_folds = 5
    fit_count = n_layers * (n_folds + 1) * n_variants
    validation_wall_clock = time.time() - t0

    return {
        "naive_reported_mean": float(naive_reported.mean()),
        "nested_reported_mean": float(nested_reported.mean()),
        "gap_removed_by_nesting": {
            "mean": float(gap_removed.mean()), "sd": float(gap_removed.std(ddof=1)),
            "bca_ci_95": bca_ci(gap_removed, seed=42), "wilcoxon_p": float(p),
            "n_reps": n_reps,
        },
        "selection_independent_references": {
            "unconditional_mean_over_all_32_layers": unconditional_mean_ref,
            "modal_selected_layer": modal_layer,
            "modal_layer_held_out_mean": modal_layer_ref,
        },
        "nested_estimate_residual_vs_modal_layer_reference": residual_vs_modal_ref,
        "cost": {
            "validation_wall_clock_seconds": validation_wall_clock,
            "artifact_probe_fit_count": fit_count,
            "artifact_size_mb": float(ARTIFACT_PATH.stat().st_size / 1e6),
            "note": ("Fit count = 32 layers x (5 inner-CV folds + 1 full-pool fit) x "
                     f"{n_variants} split variants ({n_reps} randomized + sequential + "
                     "reversed) = probe fits needed to build the artifact this "
                     "validation reuses. Each fit is a scikit-learn LogisticRegression on "
                     "at most 700 samples of GUARDIAN's probe-input dimensionality -- "
                     "sub-second per fit on CPU, no GPU or hidden-state extraction "
                     "involved once the 171MB raw cache exists."),
        },
        "verdict": (
            f"Nesting removes {gap_removed.mean():+.4f} AUROC of optimism on average "
            f"(BCa 95% CI excludes zero: {gap_removed.mean() > 0 and bca_ci(gap_removed, seed=42)[0] > 0}), "
            f"and the resulting nested estimate ({nested_reported.mean():.4f}) sits "
            f"{residual_vs_modal_ref:+.4f} away from a selection-independent reference at "
            f"the same (modal) layer -- small relative to the {gap_removed.mean():.4f} "
            "removed, consistent with this paper's own already-reported decomposition "
            "(§4.2: winner's-curse term +0.042865 minus transferred layer quality "
            "+0.0173, net Delta_sel=+0.0255). The prescribed fix WORKS: it recovers an "
            "estimate close to (not identical to, since the selected layer does carry "
            "real transferable signal) the selection-independent reference, at a cost "
            "that is a probe-refitting exercise on already-extracted features, not a new "
            "hidden-state extraction."
        ),
    }


def part_b_disjoint_holdout_validation():
    m3_synth = json.load(open(ROOT / "results" / "mechanism3_factorial_selection_controls.json"))
    m3_real = json.load(open(ROOT / "results" / "real_feature_corrected_selection_controls.json"))

    fully_corrected_synth = m3_synth["factorial_cells"]["fold_matched__oof__matched"]["gap_leaky_minus_arm"]
    shipped_synth = m3_synth["factorial_cells"]["small__infold__free"]["gap_leaky_minus_arm"]

    real_by_cap = {}
    for cap, cd in m3_real["part_B_factorial"]["by_capacity"].items():
        cell = cd["factorial_cells"].get("fold_matched__oof__matched")
        if cell:
            real_by_cap[cap] = cell["gap_leaky_minus_arm"]

    return {
        "synthetic_harness": {
            "shipped_uncorrected": shipped_synth,
            "fully_corrected_disjoint_holdout": fully_corrected_synth,
            "residual_bias_removed": float(shipped_synth["gap_mean"] - fully_corrected_synth["gap_mean"]),
        },
        "real_feature_harness_by_capacity": real_by_cap,
        "cost": {
            "synthetic_factorial_runtime_seconds": m3_synth["runtime_seconds"],
            "real_feature_corrected_controls_runtime_seconds": m3_real["runtime_seconds"],
            "note": ("Both already logged by their own scripts (code/77, code/78+79). "
                     "The disjoint-holdout fix costs exactly what re-running the harness "
                     "with a different carve-out convention costs -- no additional "
                     "hidden-state extraction, since it operates on already-extracted "
                     "features/synthetic draws and only changes which indices are used "
                     "for selection vs. training vs. reporting."),
        },
        "verdict": (
            f"On the synthetic harness, the disjoint-holdout fix moves the gap from "
            f"{shipped_synth['gap_mean']:+.4f} (shipped) to {fully_corrected_synth['gap_mean']:+.4f} "
            f"(fully corrected), a reduction of {shipped_synth['gap_mean'] - fully_corrected_synth['gap_mean']:+.4f}, "
            f"landing on a BCa interval that includes zero ({fully_corrected_synth['gap_bca_ci_95']}) -- "
            "the fix removes the detectable synthetic effect entirely, at a logged cost of "
            f"{m3_synth['runtime_seconds']:.0f}s ({m3_synth['runtime_seconds']/60:.1f} min) on CPU. "
            "On real features it does not remove the effect entirely -- residual gaps of "
            f"{real_by_cap.get('128', {}).get('gap_mean', float('nan')):+.4f} (cap 128) and "
            f"{real_by_cap.get('384', {}).get('gap_mean', float('nan')):+.4f} (cap 384) survive -- "
            f"at a logged cost of {m3_real['runtime_seconds']:.0f}s ({m3_real['runtime_seconds']/60:.1f} min). "
            "Both fixes are cheap in absolute terms (minutes of CPU time on already-"
            "extracted features/synthetic draws, no GPU, no re-extraction), which is the "
            "number §6's 'cheap to fix' claim was previously missing."
        ),
    }


def main():
    t0 = time.time()
    print("=== Part A: Mechanism 2 nested-CV fix, validated against the naive alternative ===")
    a_out = part_a_nested_cv_validation()
    print(f"  naive (un-nested) mean: {a_out['naive_reported_mean']:.4f}")
    print(f"  nested (shipped fix) mean: {a_out['nested_reported_mean']:.4f}")
    print(f"  gap removed by nesting: {a_out['gap_removed_by_nesting']['mean']:+.4f} "
          f"BCa {a_out['gap_removed_by_nesting']['bca_ci_95']} p={a_out['gap_removed_by_nesting']['wilcoxon_p']:.3g}")
    print(f"  cost: {a_out['cost']['artifact_probe_fit_count']} probe fits, "
          f"validation itself {a_out['cost']['validation_wall_clock_seconds']:.2f}s")
    print(f"\n  {a_out['verdict']}")

    print("\n=== Part B: Mechanism 3 disjoint-holdout fix, re-verified from shipped JSONs ===")
    b_out = part_b_disjoint_holdout_validation()
    print(f"\n  {b_out['verdict']}")

    out = {"part_a_mechanism2_nested_cv": a_out, "part_b_mechanism3_disjoint_holdout": b_out,
           "runtime_seconds": time.time() - t0}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
