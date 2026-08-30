"""
Mechanism 5 re-measured at the threshold rule the audited pipeline
ACTUALLY REPORTS AT.

WHY THIS SCRIPT EXISTS (an independent review found the mismatch).
`MultiHaluDet/run_pipeline.py:139-140` does this:

    thresholds = find_best_thresholds(probs, y_test)      # sweeps 81 F1
                                                          # candidates AND Youden
    metrics    = evaluate_all(probs, y_test, thresholds['youden'])

i.e. `find_best_thresholds` COMPUTES an F1-argmax threshold, but the object that
is actually handed to `evaluate_all` -- and therefore the threshold at which
every reported threshold-dependent metric (F1 included) is scored -- is
`thresholds['youden']`. code/46 measured the F1 gap at `leaky_thresh["f1"]`,
the 81-point F1-argmax. That is a threshold the audited repository computes and
then discards.

Two consequences, both of which this script quantifies:

 1. MAGNITUDE. The repo-faithful (Youden) F1 gap is LARGER than the F1-argmax
    gap at every operating point, by up to ~2.4x at the low-AUROC end. code/46's
    numbers therefore UNDERSTATE Mechanism 5 as the audited pipeline actually
    reports it.

 2. NON-NEGATIVITY. At the F1-argmax rule the gap is algebraically
    non-negative: LEAKY's threshold is the argmax over the very 81-point grid
    the test F1 is then scored on, and HONEST's threshold is a member of that
    same grid, so LEAKY >= HONEST identically (0/200 negative reps, and a
    Wilcoxon p that encodes only n and the tie structure). At the Youden rule
    that argument does NOT apply: the Youden threshold maximizes tpr-fpr on the
    ROC curve, which is a different objective from F1 and is drawn from
    `roc_curve`'s own threshold set rather than the 81-point grid. LEAKY can and
    does lose to HONEST on test F1. This script reports the negative-rep count
    per cell, which is what makes the Youden measurement a genuine empirical
    result with a meaningful p-value rather than an arithmetic identity.

Everything else -- the isotropic-Gaussian calibration, the verbatim port of
`find_best_thresholds`, the LEAKY/HONEST protocols, the seeds, the split
fractions, N_SAMPLES=700 -- is IDENTICAL to code/46, so the F1-argmax column
this script emits reproduces code/46's shipped `f1_gap_mean` exactly. That
equality is asserted at the end of the run, so the two scripts cannot drift.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "m46", Path(__file__).resolve().parent / "46_mechanism5_threshold_selection.py")
m46 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m46)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "mechanism5_youden_threshold.json"
REF_PATH = ROOT / "results" / "mechanism5_threshold_selection.json"

N_SEEDS = 200
TARGET_AUROCS = [0.70, 0.80, 0.90, 0.95, 0.985]
RNG_GLOBAL = np.random.default_rng(2026)


def run_one_seed(seed, target_auroc, n_samples=None):
    """Identical data/split/fit path to code/46.run_one_seed; scores F1 at BOTH
    threshold conventions instead of only at the F1-argmax one."""
    X, y = m46.make_synthetic_data(seed, target_auroc, n_samples)
    n = len(y)
    n_test = int(n * m46.TEST_SIZE)
    n_val = int((n - n_test) * m46.VAL_SIZE_OF_TRAIN)
    rng = np.random.default_rng(seed + 50000)
    idx = rng.permutation(n)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    clf = LogisticRegression(max_iter=2000).fit(X[train_idx], y[train_idx])
    probs_val = clf.predict_proba(X[val_idx])[:, 1]
    probs_test = clf.predict_proba(X[test_idx])[:, 1]
    yte = y[test_idx]

    leaky_thresh = m46.find_best_thresholds(probs_test, yte)
    honest_thresh = m46.find_best_thresholds(probs_val, y[val_idx])

    def f1_at(t):
        return f1_score(yte, (probs_test >= t).astype(int), zero_division=0)

    return {
        # repo-faithful rule: every threshold-dependent metric at thresholds['youden']
        "leaky_f1_youden": f1_at(leaky_thresh["youden"]),
        "honest_f1_youden": f1_at(honest_thresh["youden"]),
        # code/46's rule, kept so the two scripts can be checked against each other
        "leaky_f1_argmax": f1_at(leaky_thresh["f1"]),
        "honest_f1_argmax": f1_at(honest_thresh["f1"]),
        "leaky_acc": accuracy_score(yte, (probs_test >= leaky_thresh["youden"]).astype(int)),
        "honest_acc": accuracy_score(yte, (probs_test >= honest_thresh["youden"]).astype(int)),
    }


def bca_ci(diff, n_resamples=10000):
    res = bootstrap((np.asarray(diff),), np.mean, confidence_level=0.95,
                    n_resamples=n_resamples, method="BCa", random_state=RNG_GLOBAL)
    return [float(res.confidence_interval.low), float(res.confidence_interval.high)]


def main():
    out = {"cells": {}}
    for target in TARGET_AUROCS:
        rows = [run_one_seed(s, target) for s in range(N_SEEDS)]
        g_y = np.array([r["leaky_f1_youden"] - r["honest_f1_youden"] for r in rows])
        g_a = np.array([r["leaky_f1_argmax"] - r["honest_f1_argmax"] for r in rows])
        g_acc = np.array([r["leaky_acc"] - r["honest_acc"] for r in rows])

        _, p_y = wilcoxon(g_y)
        cell = {
            "f1_gap_youden_mean": float(g_y.mean()),
            "f1_gap_youden_sd": float(g_y.std(ddof=1)),
            "f1_gap_youden_bca_ci_95": bca_ci(g_y),
            "f1_gap_youden_wilcoxon_p": float(p_y),
            "f1_gap_youden_n_negative": int((g_y < 0).sum()),
            "f1_gap_youden_n_zero": int((g_y == 0).sum()),
            "f1_gap_youden_n_positive": int((g_y > 0).sum()),
            "f1_gap_f1argmax_mean": float(g_a.mean()),
            "f1_gap_f1argmax_n_negative": int((g_a < 0).sum()),
            "f1_gap_f1argmax_n_zero": int((g_a == 0).sum()),
            "acc_gap_mean": float(g_acc.mean()),
            "acc_gap_n_negative": int((g_acc < 0).sum()),
            "ratio_youden_over_f1argmax": float(g_y.mean() / g_a.mean()),
            "leaky_f1_youden_mean": float(np.mean([r["leaky_f1_youden"] for r in rows])),
            "honest_f1_youden_mean": float(np.mean([r["honest_f1_youden"] for r in rows])),
        }
        out["cells"][str(target)] = cell
        print(f"AUROC0={target}: F1@Youden={cell['f1_gap_youden_mean']:+.4f} "
              f"(p={p_y:.3g}, {cell['f1_gap_youden_n_negative']}/{N_SEEDS} negative)  "
              f"F1@argmax={cell['f1_gap_f1argmax_mean']:+.4f} "
              f"({cell['f1_gap_f1argmax_n_negative']}/{N_SEEDS} negative)  "
              f"ratio={cell['ratio_youden_over_f1argmax']:.2f}x", flush=True)

    ys = [out["cells"][str(t)]["f1_gap_youden_mean"] for t in TARGET_AUROCS]
    as_ = [out["cells"][str(t)]["f1_gap_f1argmax_mean"] for t in TARGET_AUROCS]
    neg = [out["cells"][str(t)]["f1_gap_youden_n_negative"] for t in TARGET_AUROCS]
    out["summary"] = {
        "n_seeds": N_SEEDS,
        "n_samples": m46.N_SAMPLES,
        "n_test": int(m46.N_SAMPLES * m46.TEST_SIZE),
        "f1_gap_youden_range": [float(min(ys)), float(max(ys))],
        "f1_gap_f1argmax_range": [float(min(as_)), float(max(as_))],
        "max_ratio_youden_over_f1argmax": float(max(y / a for y, a in zip(ys, as_))),
        "youden_negative_reps_range": [int(min(neg)), int(max(neg))],
        "f1argmax_total_negative_reps": int(sum(
            out["cells"][str(t)]["f1_gap_f1argmax_n_negative"] for t in TARGET_AUROCS)),
        "max_youden_wilcoxon_p": float(max(
            out["cells"][str(t)]["f1_gap_youden_wilcoxon_p"] for t in TARGET_AUROCS)),
        "note": (
            "The Youden column is the repo-faithful measurement: "
            "MultiHaluDet/run_pipeline.py:140 passes thresholds['youden'] to evaluate_all, "
            "so every threshold-dependent metric it reports -- F1 included -- is scored at "
            "the Youden threshold, not at the 81-point F1-argmax threshold "
            "find_best_thresholds also computes. The F1-argmax column reproduces code/46 "
            "exactly and is retained only to show the two conventions diverge."),
    }

    # ── sample-size sensitivity, at the repo-faithful threshold rule ─────────
    # code/46 measured this at the F1-argmax rule too; redone here at Youden so
    # SS4.5's "a pipeline reporting at ten times that size should expect
    # roughly a third of it" is stated in the same units as the headline.
    ss = {}
    for n_samples in [700, 1750, 3500, 10000]:
        rows = [run_one_seed(s, 0.985, n_samples) for s in range(N_SEEDS)]
        g_y = np.array([r["leaky_f1_youden"] - r["honest_f1_youden"] for r in rows])
        g_a = np.array([r["leaky_f1_argmax"] - r["honest_f1_argmax"] for r in rows])
        _, p_y = wilcoxon(g_y)
        n_test = int(n_samples * m46.TEST_SIZE)
        ss[str(n_samples)] = {
            "n_samples": n_samples, "n_test": n_test,
            "n_val": int((n_samples - n_test) * m46.VAL_SIZE_OF_TRAIN),
            "f1_gap_youden_mean": float(g_y.mean()),
            "f1_gap_youden_bca_ci_95": bca_ci(g_y),
            "f1_gap_youden_wilcoxon_p": float(p_y),
            "f1_gap_youden_n_negative": int((g_y < 0).sum()),
            "f1_gap_f1argmax_mean": float(g_a.mean()),
        }
        print(f"  N={n_samples:6d} (n_test={n_test}): F1@Youden={g_y.mean():+.4f} "
              f"p={p_y:.3g} ({int((g_y < 0).sum())}/{N_SEEDS} negative)  "
              f"F1@argmax={g_a.mean():+.4f}", flush=True)
    out["sample_size_sensitivity_at_youden"] = {
        "target_auroc": 0.985, "n_seeds": N_SEEDS, "by_n_samples": ss,
        "shrinkage_factor_700_to_10000": float(
            ss["700"]["f1_gap_youden_mean"] / ss["10000"]["f1_gap_youden_mean"]),
        "note": (
            "Same sweep as code/46's sample_size_sensitivity, measured at the threshold "
            "rule MultiHaluDet actually reports at. Holds the operating point at 0.985 and "
            "scales N_SAMPLES, which scales n_test (LEAKY's selection set) and n_val "
            "(HONEST's) together."),
    }

    # The F1-argmax column must reproduce code/46's shipped cells exactly.
    ref = json.load(open(REF_PATH))["capacities"]
    mismatches = []
    for t in TARGET_AUROCS:
        a = out["cells"][str(t)]["f1_gap_f1argmax_mean"]
        b = ref[str(t)]["f1_gap_mean"]
        if abs(a - b) > 1e-12:
            mismatches.append((t, a, b))
    out["summary"]["reproduces_code46_f1argmax_column"] = not mismatches
    assert not mismatches, f"F1-argmax column drifted from code/46: {mismatches}"
    print("\nF1-argmax column reproduces code/46's shipped cells exactly (all 5).")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
