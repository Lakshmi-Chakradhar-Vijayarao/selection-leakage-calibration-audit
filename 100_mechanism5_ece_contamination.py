"""
UncertaiNLP addition: quantifying a real, code-verified ECE
contamination in the audited MultiHaluDet repository itself, found by
tracing Mechanism 5 (test-set-driven threshold selection) into its own
uncertainty-metric code.

THE BUG, VERIFIED DIRECTLY IN THE VENDORED REPOSITORY.
`MultiHaluDet/src/utils/metrics.py::compute_uncertainty_metrics(probs, labels, preds)`
computes ECE with `confidence = max(probs, 1-probs)` (correctly
threshold-independent) but `correct = (preds == labels)` (thresholded).
`MultiHaluDet/run_pipeline.py` lines 139-141 call:
    thresholds = find_best_thresholds(probs, y_test)   # Mechanism 5's leak
    uncertainty = compute_uncertainty_metrics(probs, y_test,
                      (probs >= thresholds['youden']).astype(int))
so `preds` -- and therefore the reported ECE's "correct" term -- is computed
at a threshold selected by optimizing directly against the test labels.
This is Mechanism 5 (already in this paper's taxonomy) contaminating a
calibration metric, in the audited pipeline's own code, not a bridge
experiment constructed on our own cached data.

EVIDENTIARY CATEGORY, STATED EXPLICITLY (do not conflate with
Sec. 6's Mechanisms 1/2 findings). Those are LEAKY/CLEAN_MATCHED
comparisons WE construct on OUR OWN cached scores. This is a bug we FOUND,
verified directly in an externally published, pinned-commit repository's
own reported metric -- a categorically different, stronger form of
evidence, and its cause (threshold selected on test labels) is structurally
unrelated to Mechanisms 1/2's near-in-sample-separation cause. We quantify
its SIZE using this paper's own already-validated synthetic harness (same
isotropic-Gaussian generator, same Youden-threshold selection code, ported
verbatim from code/46), since re-running MultiHaluDet's actual 7B-scale
pipeline is outside this paper's compute budget -- exactly the same
methodological choice code/46 already makes for Mechanism 5's F1/accuracy
gaps.

SINGLE-REPOSITORY CAVEAT. The second Mechanism-5 repository this paper
already audits (HallucinationPatternDetection) does not report ECE at all,
so this finding is verified in one of the two audited Mechanism-5
repositories, not both -- stated here and in the paper text, not left to a
footnote.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap, norm, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "mechanism5_ece_contamination.json"

N_SAMPLES = 700
TEST_SIZE = 0.20
VAL_SIZE_OF_TRAIN = 0.20
N_SEEDS = 200
TARGET_AUROCS = [0.70, 0.80, 0.90, 0.95, 0.985]
FEAT_DIM = 64
RNG_GLOBAL = np.random.default_rng(2026)


def make_synthetic_data(seed, target_auroc, n_samples=None):
    """Verbatim from code/46: per-dimension mean difference = class_sep,
    matching code/02d's mean_pos=CLASS_SEP/2, mean_neg=-CLASS_SEP/2
    convention."""
    n_samples = N_SAMPLES if n_samples is None else n_samples
    rng = np.random.default_rng(seed)
    j_target = 2 * (norm.ppf(target_auroc)) ** 2
    class_sep = np.sqrt(j_target / FEAT_DIM)
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos
    X_pos = class_sep / 2 + rng.standard_normal((n_pos, FEAT_DIM))
    X_neg = -class_sep / 2 + rng.standard_normal((n_neg, FEAT_DIM))
    X = np.vstack([X_pos, X_neg]).astype(np.float64)
    y = np.array([1] * n_pos + [0] * n_neg)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def find_best_thresholds_youden(probs, labels):
    """Verbatim port of MultiHaluDet/src/utils/metrics.py::find_best_thresholds,
    Youden component only (the component that feeds compute_uncertainty_metrics)."""
    fpr, tpr, thresholds_roc = roc_curve(labels, probs)
    youden_j = tpr - fpr
    return thresholds_roc[np.argmax(youden_j)]


def ece_multihaludet_formula(probs, labels, preds, n_bins=10):
    """Verbatim port of MultiHaluDet/src/utils/metrics.py::compute_uncertainty_metrics's
    ECE term: confidence is threshold-independent, but 'correct' uses the
    THRESHOLDED prediction -- this is the contamination vector."""
    confidence = np.maximum(probs, 1 - probs)
    correct = (preds == labels)
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        in_bin = (confidence >= bin_boundaries[i]) & (confidence < bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            avg_confidence = confidence[in_bin].mean()
            avg_accuracy = correct[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    return float(ece)


def run_one_seed(seed, target_auroc):
    X, y = make_synthetic_data(seed, target_auroc)
    n = len(y)
    n_test = int(n * TEST_SIZE)
    n_val = int((n - n_test) * VAL_SIZE_OF_TRAIN)
    rng = np.random.default_rng(seed + 50000)
    idx = rng.permutation(n)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    clf = LogisticRegression(max_iter=2000).fit(X[train_idx], y[train_idx])
    probs_val = clf.predict_proba(X[val_idx])[:, 1]
    probs_test = clf.predict_proba(X[test_idx])[:, 1]
    y_test = y[test_idx]

    # LEAKY: exactly MultiHaluDet's own run_pipeline.py lines 139-141
    leaky_thresh = find_best_thresholds_youden(probs_test, y_test)
    leaky_preds = (probs_test >= leaky_thresh).astype(int)
    leaky_ece = ece_multihaludet_formula(probs_test, y_test, leaky_preds)

    # HONEST: threshold selected on an independent validation split
    honest_thresh = find_best_thresholds_youden(probs_val, y[val_idx])
    honest_preds = (probs_test >= honest_thresh).astype(int)
    honest_ece = ece_multihaludet_formula(probs_test, y_test, honest_preds)

    return leaky_ece, honest_ece


def bca_ci(a, b, n_resamples=10000):
    diff = np.asarray(a) - np.asarray(b)
    res = bootstrap((diff,), np.mean, confidence_level=0.95, n_resamples=n_resamples,
                     method="BCa", random_state=RNG_GLOBAL)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def main():
    out = {"description": "ECE gap under MultiHaluDet's own compute_uncertainty_metrics "
                           "formula, LEAKY (Youden threshold selected on test labels, "
                           "verbatim MultiHaluDet protocol) vs HONEST (Youden threshold "
                           "selected on an independent validation split).",
           "cells": {}}
    all_leaky, all_honest = [], []
    for target_auroc in TARGET_AUROCS:
        leaky_eces, honest_eces = [], []
        for seed in range(N_SEEDS):
            le, he = run_one_seed(seed, target_auroc)
            leaky_eces.append(le)
            honest_eces.append(he)
        gap = np.array(leaky_eces) - np.array(honest_eces)
        _, p = wilcoxon(leaky_eces, honest_eces)
        ci = bca_ci(leaky_eces, honest_eces)
        out["cells"][str(target_auroc)] = {
            "leaky_ece_mean": float(np.mean(leaky_eces)), "honest_ece_mean": float(np.mean(honest_eces)),
            "ece_gap_mean": float(gap.mean()), "ece_gap_bca_95ci": list(ci), "wilcoxon_p": float(p),
        }
        all_leaky.extend(leaky_eces)
        all_honest.extend(honest_eces)
        print(f"AUROC target={target_auroc}: LEAKY ECE={np.mean(leaky_eces):.4f}  "
              f"HONEST ECE={np.mean(honest_eces):.4f}  gap={gap.mean():+.4f}  "
              f"BCa 95% {ci}  p={p:.3g}")

    pooled_gap_mean = float(np.mean(np.array(all_leaky) - np.array(all_honest)))
    out["pooled_across_5_operating_points"] = {
        "n_total_seeds": len(all_leaky), "pooled_ece_gap_mean": pooled_gap_mean,
    }
    print(f"\nPooled across all 5 operating points: mean ECE gap = {pooled_gap_mean:+.4f}")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
