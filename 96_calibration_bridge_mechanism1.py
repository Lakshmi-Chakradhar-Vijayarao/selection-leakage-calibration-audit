"""
UncertaiNLP calibration bridge, Mechanism 1 real-feature analog.

Mirrors code/94's design exactly (same cached real Mistral-7B/HaluEval
features, same LogisticRegression convention, same fold-assignment-resampling
interval), but reports Brier score and 10-bin ECE using predict_proba
(actual probabilities), not decision_function (unbounded logits used only
for the AUROC sanity check), since calibration metrics require [0,1] scores.

LEAKY: probe fit on all 400 samples, scored on those same 400 (in-sample).
CLEAN_MATCHED: 5-fold out-of-fold refit, scored only on held-out folds.
Reported over the same 40 fold-assignment seeds code/94 uses, with a BCa 95%
interval on the gap (matching code/44's bca_ci convention throughout the
paper).
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
FEATS_PATH = ROOT / "results" / "real_features_mistral7b_halueval.npz"
OUT_PATH = ROOT / "results" / "calibration_bridge_mechanism1.json"

N_FOLDS = 5
N_BOOT_SEEDS = 40
MAX_ITER = 500
N_BINS = 10


def load_features():
    d = np.load(FEATS_PATH)
    X_seq, X_glob, y = d["X_seq"], d["X_glob"], d["y"]
    X = np.hstack([X_seq.reshape(X_seq.shape[0], -1), X_glob]).astype(np.float64)
    return X, y


def ece_binned(scores, labels, n_bins=N_BINS):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(scores)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (scores >= lo) & (scores <= hi) if i == n_bins - 1 else (scores >= lo) & (scores < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(labels[mask].mean() - scores[mask].mean())
    return float(ece)


def one_fold_seed(X, y, fold_seed):
    Xs_leaky = StandardScaler().fit_transform(X)
    clf_full = LogisticRegression(max_iter=MAX_ITER, C=1.0).fit(Xs_leaky, y)
    p_leaky = clf_full.predict_proba(Xs_leaky)[:, 1]
    s_leaky = clf_full.decision_function(Xs_leaky)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=fold_seed)
    p_oof = np.zeros(len(y))
    s_oof = np.zeros(len(y))
    for tr_idx, te_idx in skf.split(X, y):
        scaler = StandardScaler().fit(X[tr_idx])
        Xtr, Xte = scaler.transform(X[tr_idx]), scaler.transform(X[te_idx])
        clf = LogisticRegression(max_iter=MAX_ITER, C=1.0).fit(Xtr, y[tr_idx])
        p_oof[te_idx] = clf.predict_proba(Xte)[:, 1]
        s_oof[te_idx] = clf.decision_function(Xte)

    return {
        "auroc_leaky": roc_auc_score(y, s_leaky), "auroc_clean": roc_auc_score(y, s_oof),
        "brier_leaky": brier_score_loss(y, p_leaky), "brier_clean": brier_score_loss(y, p_oof),
        "ece_leaky": ece_binned(p_leaky, y), "ece_clean": ece_binned(p_oof, y),
    }


def bca_ci(a, b):
    diff = np.asarray(a) - np.asarray(b)
    res = bootstrap((diff,), np.mean, confidence_level=0.95, n_resamples=5000, method="BCa",
                     random_state=np.random.default_rng(12345))
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def main():
    t0 = time.time()
    X, y = load_features()
    print(f"Loaded real features: X={X.shape}, hall_rate={1 - y.mean():.3f}")

    rows = [one_fold_seed(X, y, seed) for seed in range(N_BOOT_SEEDS)]
    auroc_leaky = np.array([r["auroc_leaky"] for r in rows])
    auroc_clean = np.array([r["auroc_clean"] for r in rows])
    brier_leaky = np.array([r["brier_leaky"] for r in rows])
    brier_clean = np.array([r["brier_clean"] for r in rows])
    ece_leaky = np.array([r["ece_leaky"] for r in rows])
    ece_clean = np.array([r["ece_clean"] for r in rows])

    auroc_gap = auroc_leaky - auroc_clean
    brier_gap = brier_clean - brier_leaky  # positive = LEAKY looks better-calibrated than honest
    ece_gap = ece_clean - ece_leaky

    a_lo, a_hi = bca_ci(auroc_leaky, auroc_clean)
    b_lo, b_hi = bca_ci(brier_clean, brier_leaky)
    e_lo, e_hi = bca_ci(ece_clean, ece_leaky)

    out = {
        "n_fold_seeds": N_BOOT_SEEDS,
        "mechanism": "Mechanism 1 real-feature analog -- calibration bridge",
        "auroc_leaky_mean": float(auroc_leaky.mean()), "auroc_clean_mean": float(auroc_clean.mean()),
        "auroc_gap_mean": float(auroc_gap.mean()), "auroc_gap_bca_95ci": [a_lo, a_hi],
        "brier_leaky_mean": float(brier_leaky.mean()), "brier_clean_mean": float(brier_clean.mean()),
        "brier_gap_mean": float(brier_gap.mean()), "brier_gap_bca_95ci": [b_lo, b_hi],
        "ece_leaky_mean": float(ece_leaky.mean()), "ece_clean_mean": float(ece_clean.mean()),
        "ece_gap_mean": float(ece_gap.mean()), "ece_gap_bca_95ci": [e_lo, e_hi],
        "runtime_seconds": time.time() - t0,
    }
    print(f"\nAUROC gap (sanity vs code/94's +0.0111): {auroc_gap.mean():+.4f}  BCa 95% [{a_lo:+.4f},{a_hi:+.4f}]")
    print(f"Brier gap (clean - leaky): {brier_gap.mean():+.4f}  BCa 95% [{b_lo:+.4f},{b_hi:+.4f}]  "
          f"(LEAKY={brier_leaky.mean():.4f}, CLEAN={brier_clean.mean():.4f})")
    print(f"ECE gap (clean - leaky):   {ece_gap.mean():+.4f}  BCa 95% [{e_lo:+.4f},{e_hi:+.4f}]  "
          f"(LEAKY={ece_leaky.mean():.4f}, CLEAN={ece_clean.mean():.4f})")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
