"""
UncertaiNLP robustness check, directly responding to independent
reviewer feedback: is Mechanism 1's calibration-corruption / selective-
prediction finding (code/96, code/97) an artifact of near-interpolation
overfitting (d=414, n=400, in-sample AUROC=1.0000 at the default C=1.0), or
does it reflect the selection/reuse mechanism itself?

METHOD. Repeats code/96's calibration-bridge and code/97's selective-
prediction-consequence analyses at C=1.0 (primary, as originally reported)
and C=0.01 (a robustness point two orders of magnitude more regularized,
where in-sample AUROC drops from 1.0000 to ~0.997 -- no longer exact
interpolation), both with the same 40 fold-assignment seeds and BCa 95%
intervals as code/96/97.

This is the same kind of robustness check the paper already runs for the
AUROC gap alone (code/94's C-sweep, "not a regularization artifact"),
extended here to the calibration and selective-prediction metrics that
code/94 never touched.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "mechanism1_regularization_robustness.json"
N_BOOT_SEEDS = 40
MAX_ITER = 2000
N_BINS = 10
C_VALUES = [1.0, 0.01]


def load_features():
    d = np.load(ROOT / "results" / "real_features_mistral7b_halueval.npz")
    X_seq, X_glob, y = d["X_seq"], d["X_glob"], d["y"]
    X = np.hstack([X_seq.reshape(X_seq.shape[0], -1), X_glob]).astype(np.float64)
    return X, y


def ece_binned(scores, labels, n_bins=N_BINS):
    scores, labels = np.asarray(scores, float), np.asarray(labels, float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(scores)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (scores >= lo) & (scores <= hi) if i == n_bins - 1 else (scores >= lo) & (scores < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(labels[mask].mean() - scores[mask].mean())
    return float(ece)


def risk_at_coverage(scores, labels, cov):
    conf = np.abs(scores - 0.5) * 2.0
    pred = (scores >= 0.5).astype(int)
    correct = (pred == labels).astype(int)
    n = len(scores)
    k = max(1, int(round(cov * n)))
    order = np.argsort(-conf)
    tau = conf[order[k - 1]]
    kept = conf >= tau
    return tau, float(1.0 - correct[kept].mean()) if kept.sum() else float("nan")


def apply_threshold(scores, labels, tau):
    conf = np.abs(scores - 0.5) * 2.0
    pred = (scores >= 0.5).astype(int)
    correct = (pred == labels).astype(int)
    kept = conf >= tau
    return float(1.0 - correct[kept].mean()) if kept.sum() else float("nan")


def bca_ci(a, b):
    diff = np.asarray(a) - np.asarray(b)
    res = bootstrap((diff,), np.mean, confidence_level=0.95, n_resamples=5000, method="BCa",
                     random_state=np.random.default_rng(12345))
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def one_fold_seed(X, y, C, fold_seed):
    Xs_leaky = StandardScaler().fit_transform(X)
    clf_full = LogisticRegression(max_iter=MAX_ITER, C=C).fit(Xs_leaky, y)
    p_leaky = clf_full.predict_proba(Xs_leaky)[:, 1]
    auroc_leaky = roc_auc_score(y, p_leaky)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold_seed)
    p_oof = np.zeros(len(y))
    for tr_idx, te_idx in skf.split(X, y):
        scaler = StandardScaler().fit(X[tr_idx])
        Xtr, Xte = scaler.transform(X[tr_idx]), scaler.transform(X[te_idx])
        clf = LogisticRegression(max_iter=MAX_ITER, C=C).fit(Xtr, y[tr_idx])
        p_oof[te_idx] = clf.predict_proba(Xte)[:, 1]
    auroc_clean = roc_auc_score(y, p_oof)

    tau50, believed50 = risk_at_coverage(p_leaky, y, 0.50)
    actual50 = apply_threshold(p_oof, y, tau50)

    return {
        "auroc_leaky": auroc_leaky, "auroc_clean": auroc_clean,
        "brier_leaky": brier_score_loss(y, p_leaky), "brier_clean": brier_score_loss(y, p_oof),
        "ece_leaky": ece_binned(p_leaky, y), "ece_clean": ece_binned(p_oof, y),
        "believed_risk_50": believed50, "actual_risk_50": actual50,
    }


def main():
    X, y = load_features()
    out = {}
    for C in C_VALUES:
        rows = [one_fold_seed(X, y, C, seed) for seed in range(N_BOOT_SEEDS)]
        auroc_leaky = np.array([r["auroc_leaky"] for r in rows])
        auroc_clean = np.array([r["auroc_clean"] for r in rows])
        brier_leaky = np.array([r["brier_leaky"] for r in rows])
        brier_clean = np.array([r["brier_clean"] for r in rows])
        ece_leaky = np.array([r["ece_leaky"] for r in rows])
        ece_clean = np.array([r["ece_clean"] for r in rows])
        believed = np.array([r["believed_risk_50"] for r in rows])
        actual = np.array([r["actual_risk_50"] for r in rows])

        brier_gap = brier_clean - brier_leaky
        ece_gap = ece_clean - ece_leaky
        violation = actual - believed
        b_lo, b_hi = bca_ci(brier_clean, brier_leaky)
        e_lo, e_hi = bca_ci(ece_clean, ece_leaky)
        v_lo, v_hi = bca_ci(actual, believed)

        entry = {
            "C": C,
            "in_sample_auroc_mean": float(auroc_leaky.mean()),
            "oof_auroc_mean": float(auroc_clean.mean()),
            "brier_leaky_mean": float(brier_leaky.mean()), "brier_clean_mean": float(brier_clean.mean()),
            "brier_gap_mean": float(brier_gap.mean()), "brier_gap_bca_95ci": [b_lo, b_hi],
            "ece_leaky_mean": float(ece_leaky.mean()), "ece_clean_mean": float(ece_clean.mean()),
            "ece_gap_mean": float(ece_gap.mean()), "ece_gap_bca_95ci": [e_lo, e_hi],
            "believed_risk_50_mean": float(believed.mean()),
            "actual_risk_50_mean": float(actual.mean()),
            "risk_violation_50_mean": float(violation.mean()), "risk_violation_50_bca_95ci": [v_lo, v_hi],
        }
        out[f"C_{C}"] = entry
        print(f"C={C}: in-sample AUROC={entry['in_sample_auroc_mean']:.4f}  "
              f"Brier gap={entry['brier_gap_mean']:+.4f} BCa[{b_lo:+.4f},{b_hi:+.4f}]  "
              f"ECE gap={entry['ece_gap_mean']:+.4f} BCa[{e_lo:+.4f},{e_hi:+.4f}]  "
              f"believed_risk@50%={entry['believed_risk_50_mean']:.4f}  "
              f"violation={entry['risk_violation_50_mean']:+.4f} BCa[{v_lo:+.4f},{v_hi:+.4f}]")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
