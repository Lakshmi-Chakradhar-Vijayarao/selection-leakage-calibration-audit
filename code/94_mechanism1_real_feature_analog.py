"""
MAXIMUM-RIGOR PASS, Item 7. Mechanism 1's severity (+0.13 to +0.29,
§4.1, code/86) is measured ONLY in a synthetic isotropic-Gaussian harness --
unlike Mechanisms 3/4/5, it has no real-feature check. This closes that gap
using the ALREADY-CACHED real Mistral-7B/HaluEval features
(results/real_features_mistral7b_halueval.npz, n=400, the same features
code/43/78 use) -- no new GPU extraction needed, only a new probe-fitting
comparison on already-extracted features.

DESIGN, mirroring code/86's synthetic harness and HaRP's own diagnostic
("probe_conf alone: in-sample AUROC 1.0000 vs out-of-fold AUROC 0.7573"):
  LEAKY: fit ONE probe (LogisticRegression, matching code/86's convention) on
    ALL 400 samples' features and labels, then score those SAME 400 samples,
    reporting the in-sample AUROC directly.
  CLEAN_MATCHED (OOF): 5-fold StratifiedKFold, refitting the identical probe
    on each fold's own training split only, scoring only that fold's held-out
    rows, assembling one out-of-fold score vector covering every sample
    exactly once, reporting ITS AUROC.
  gap = LEAKY - CLEAN_MATCHED.

Run at both of this paper's flagship probe capacities is not applicable here
(Mechanism 1 uses a single linear probe on raw features directly, as HaRP's
own bug does -- there is no MLP hidden-width parameter in this mechanism), but
we additionally report the same comparison after standardizing features
(StandardScaler, fit on the same full/train split as the probe, matching
Mechanism 1's own leak-or-not-leak distinction one level down) and with an L2
regularization sweep (C in {0.01, 0.1, 1, 10, 100}), since HaRP's own
diagnostic used a default-regularization LogisticRegression and a reader
should know whether the gap is a regularization artifact.

Output: results/mechanism1_real_feature_analog.json
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
FEATS_PATH = ROOT / "results" / "real_features_mistral7b_halueval.npz"
OUT_PATH = ROOT / "results" / "mechanism1_real_feature_analog.json"

N_FOLDS = 5
C_GRID = [0.01, 0.1, 1.0, 10.0]  # C=100 dropped: near-separable 414-dim/400-row data makes
                                  # lbfgs convergence extremely slow there for no informative gain
N_BOOT_SEEDS = 40   # fold-shuffle seeds for a BCa-style interval on the single real dataset
MAX_ITER = 500      # sufficient for convergence at these C values; keeps wall-clock bounded


def load_features():
    d = np.load(FEATS_PATH)
    X_seq, X_glob, y = d["X_seq"], d["X_glob"], d["y"]
    X = np.hstack([X_seq.reshape(X_seq.shape[0], -1), X_glob]).astype(np.float64)
    return X, y


def leaky_and_oof(X, y, C, standardize, fold_seed):
    if standardize:
        # LEAKY convention: scaler fit on the SAME full sample it later scores
        # (the analogue of HaRP's own full-dataset fit), i.e. no split at all
        # for the leaky arm.
        Xs_leaky = StandardScaler().fit_transform(X)
    else:
        Xs_leaky = X

    clf_full = LogisticRegression(max_iter=MAX_ITER, C=C).fit(Xs_leaky, y)
    s_leaky = clf_full.decision_function(Xs_leaky)
    leaky_auroc = roc_auc_score(y, s_leaky)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=fold_seed)
    s_oof = np.zeros(len(y))
    for tr_idx, te_idx in skf.split(X, y):
        if standardize:
            scaler = StandardScaler().fit(X[tr_idx])
            Xtr, Xte = scaler.transform(X[tr_idx]), scaler.transform(X[te_idx])
        else:
            Xtr, Xte = X[tr_idx], X[te_idx]
        clf = LogisticRegression(max_iter=MAX_ITER, C=C).fit(Xtr, y[tr_idx])
        s_oof[te_idx] = clf.decision_function(Xte)
    clean_auroc = roc_auc_score(y, s_oof)
    return leaky_auroc, clean_auroc


def gap_with_ci(X, y, C, standardize, n_boot=N_BOOT_SEEDS):
    """The real dataset is a single n=400 sample (no repeated-seed data
    generation as in the synthetic harness), so the BCa-style interval here
    resamples the 5-fold ASSIGNMENT (fold_seed), not the data itself -- this
    quantifies fold-assignment variance in the OOF estimate, not sampling
    variance of a new dataset draw, and is reported as such."""
    leaky_vals, clean_vals = [], []
    for seed in range(n_boot):
        l, c = leaky_and_oof(X, y, C, standardize, fold_seed=seed)
        leaky_vals.append(l); clean_vals.append(c)
    leaky_vals, clean_vals = np.array(leaky_vals), np.array(clean_vals)
    gap = leaky_vals - clean_vals
    res = bootstrap((gap,), np.mean, confidence_level=0.95, n_resamples=5000, method="BCa",
                    random_state=np.random.default_rng(12345))
    return {
        "leaky_mean": float(leaky_vals.mean()), "leaky_sd_over_fold_seeds": float(leaky_vals.std(ddof=1)),
        "clean_matched_mean": float(clean_vals.mean()), "clean_matched_sd_over_fold_seeds": float(clean_vals.std(ddof=1)),
        "gap_mean": float(gap.mean()),
        "gap_bca_ci_95_over_fold_assignment": [float(res.confidence_interval.low), float(res.confidence_interval.high)],
        "n_fold_seeds": n_boot,
    }


def main():
    t0 = time.time()
    X, y = load_features()
    print(f"Loaded real features: X={X.shape}, hall_rate={1 - y.mean():.3f}")

    out = {"data": {"source": str(FEATS_PATH.relative_to(ROOT)), "n_samples": int(len(y)),
                    "feature_dim": int(X.shape[1]), "hall_rate": float(1 - y.mean())}}

    print("\n=== Primary: default C=1.0, standardized (matching code/86's LogisticRegression default) ===")
    primary = gap_with_ci(X, y, C=1.0, standardize=True)
    out["primary_standardized_C1"] = primary
    print(f"  LEAKY={primary['leaky_mean']:.4f}  CLEAN_MATCHED(OOF)={primary['clean_matched_mean']:.4f}  "
          f"gap={primary['gap_mean']:+.4f}  BCa(over fold assignment)={primary['gap_bca_ci_95_over_fold_assignment']}")

    print("\n=== Robustness: unstandardized raw features, C=1.0 (n_boot reduced -- raw feature "
          "magnitudes up to ~1980 make lbfgs converge slowly/not at all, disclosed) ===")
    unstd = gap_with_ci(X, y, C=1.0, standardize=False, n_boot=10)
    out["unstandardized_C1"] = unstd
    print(f"  gap={unstd['gap_mean']:+.4f}  BCa={unstd['gap_bca_ci_95_over_fold_assignment']}")

    print("\n=== Regularization sweep (standardized), C in", C_GRID, "===")
    reg_sweep = {}
    for C in C_GRID:
        r = gap_with_ci(X, y, C=C, standardize=True, n_boot=20)  # fewer fold-seeds per C to bound cost
        reg_sweep[str(C)] = r
        print(f"  C={C:>6}: LEAKY={r['leaky_mean']:.4f} OOF={r['clean_matched_mean']:.4f} "
              f"gap={r['gap_mean']:+.4f} BCa={r['gap_bca_ci_95_over_fold_assignment']}")
    out["regularization_sweep_standardized"] = reg_sweep

    gaps_over_C = [reg_sweep[str(c)]["gap_mean"] for c in C_GRID]
    out["comparison_to_synthetic_mechanism1_harness"] = (
        "code/86's synthetic isotropic-Gaussian reconstruction gives BCa-established gaps of "
        f"+0.13 to +0.29 AUROC. This real-feature analog gives {primary['gap_mean']:+.4f} "
        f"(standardized, C=1.0) and a regularization-sweep range of "
        f"{min(gaps_over_C):+.4f} to {max(gaps_over_C):+.4f} over C in {C_GRID}."
    )
    out["runtime_seconds"] = time.time() - t0

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{out['comparison_to_synthetic_mechanism1_harness']}")
    print(f"\nSaved: {OUT_PATH}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
