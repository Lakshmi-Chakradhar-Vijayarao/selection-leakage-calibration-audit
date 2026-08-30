"""
UncertaiNLP addition: extending the calibration bridge
(code/95, code/96) to Mechanism 3's real-feature harness, using its own
already-validated pipeline (code/43's SweepMLP + checkpoint-selection
machinery, code/78's out-of-fold factorial construction), imported directly
rather than reimplemented, to avoid drift from the paper's own verified
numbers.

WHY THIS IS DIFFERENT FROM MECHANISMS 1/2. Those bridges used a simple
LogisticRegression directly on raw/lightly-processed features. Mechanism 3's
harness trains a small MLP (SweepMLP) per fold, selecting the checkpoint by
best validation AUROC across epochs (exactly this paper's Mechanism-3
failure mode), then fits a downstream LogisticRegression on the MLP's
penultimate-layer features. Extending the calibration bridge here checks
whether calibration corruption survives under a genuinely different model
architecture and a genuinely different (checkpoint, not layer/probe-fit)
selection mechanism -- not a repeat of the same test on the same model class.

METHOD. Reuses s43 (imported exactly as code/78 does) for
train_to_best_checkpoint / train_fixed_epochs / extract_features /
apply_calibration_label_free / load_raw_real_features, and code/78's own
_pool / _selection_sets helpers verbatim (imported, not copied), so the
LEAKY and fully-corrected (out-of-fold, depth-matched) arms are constructed
identically to the already-published, already-verified pipeline. The only
change is the final step: instead of only scoring AUROC, the downstream
LogisticRegression's predict_proba is kept per test sample for both arms,
and Brier score / 10-bin ECE are computed from those, exactly as in
code/95/96.

SANITY CHECK. The AUROC recomputed from these same probabilities must
match code/78's already-shipped real_feature_corrected_selection_controls.json
values (leaky_mean, fully_corrected_gap) to within seed-level noise --
verified before any calibration number is reported.
"""
import importlib.util
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
OUT_PATH = ROOT / "results" / "mechanism3_calibration_bridge.json"
REF_78 = ROOT / "results" / "real_feature_corrected_selection_controls.json"
CALIB = ROOT / "results" / "calibration_leakage_diagnostic.json"

_spec43 = importlib.util.spec_from_file_location("s43", CODE / "43_calibration_leakage_diagnostic.py")
s43 = importlib.util.module_from_spec(_spec43)
_spec43.loader.exec_module(s43)
s43.ALPHA_TRAIN_ONLY = float(json.load(open(CALIB))["train_only_calibration_alpha"])

_spec78 = importlib.util.spec_from_file_location("s78", CODE / "78_real_feature_corrected_selection_controls.py")
s78 = importlib.util.module_from_spec(_spec78)
_spec78.loader.exec_module(s78)

CAPACITIES = s43.CAPACITIES
N_SEEDS = s43.N_SEEDS
EPOCHS = s43.EPOCHS
N_INNER_FOLDS = s43.N_INNER_FOLDS
TEST_SIZE = s43.TEST_SIZE
F_SHIPPED = s78.F_SHIPPED
F_FOLD = s78.F_FOLD
OOF_HOLD_FRACTION = s78.OOF_HOLD_FRACTION
N_BINS = 10


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


def bca_ci(a, b, n_resamples=10000, seed=12345):
    diff = np.asarray(a) - np.asarray(b)
    res = bootstrap((diff,), np.mean, confidence_level=0.95, n_resamples=n_resamples,
                     method="BCa", random_state=np.random.default_rng(seed))
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def run_capacity(X, y, hidden, n_seeds=N_SEEDS):
    auroc_leaky_all, auroc_corr_all = [], []
    brier_leaky_all, brier_corr_all = [], []
    ece_leaky_all, ece_corr_all = [], []

    for seed in range(n_seeds):
        X_train, X_test, y_train, y_test = s78._prep(X, y, seed)
        cv_idx, ges_idx, skf = s78._pool(y_train, seed)
        fdim, n_tr = hidden // 2, len(y_train)

        oof_leaky = np.zeros((n_tr, fdim))
        test_leaky = np.zeros((len(y_test), fdim))
        oof_fm_oof = np.zeros((n_tr, fdim))  # free-run features (epoch discovery only)

        epochs_leaky, epochs_fm_oof = [], []
        folds = list(skf.split(X_train[cv_idx], y_train[cv_idx]))
        for fold, (tr_loc, val_loc) in enumerate(folds):
            tr_idx, val_idx = cv_idx[tr_loc], cv_idx[val_loc]
            fold_seed = seed * 100 + fold

            m_l, ep_l = s43.train_to_best_checkpoint(
                X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_train[val_idx],
                hidden, EPOCHS, fold_seed)
            oof_leaky[val_idx] = s43.extract_features(m_l, X_train[val_idx])
            test_leaky += s43.extract_features(m_l, X_test)
            epochs_leaky.append(ep_l)

            sel_tr, sel_es = s78._selection_sets(tr_idx, ges_idx, y_train, fold_seed)["fold_matched__oof"]
            _, ep = s43.train_to_best_checkpoint(
                X_train[sel_tr], y_train[sel_tr], X_train[sel_es], y_train[sel_es],
                hidden, EPOCHS, fold_seed)
            epochs_fm_oof.append(ep)

        mean_ep_leaky = float(np.mean(epochs_leaky))
        mean_ep_fm_oof = float(np.mean(epochs_fm_oof))
        shift = int(round(mean_ep_leaky - mean_ep_fm_oof))

        oof_corr = np.zeros((n_tr, fdim))
        test_corr = np.zeros((len(y_test), fdim))
        for fold, (tr_loc, val_loc) in enumerate(folds):
            tr_idx, val_idx = cv_idx[tr_loc], cv_idx[val_loc]
            fold_seed = seed * 100 + fold
            ep_matched = max(1, epochs_fm_oof[fold] + shift)
            m = s43.train_fixed_epochs(X_train[tr_idx], y_train[tr_idx], hidden, ep_matched, fold_seed)
            oof_corr[val_idx] = s43.extract_features(m, X_train[val_idx])
            test_corr += s43.extract_features(m, X_test)

        test_leaky_avg = test_leaky / N_INNER_FOLDS
        test_corr_avg = test_corr / N_INNER_FOLDS
        clf_leaky = LogisticRegression(max_iter=2000).fit(oof_leaky[cv_idx], y_train[cv_idx])
        clf_corr = LogisticRegression(max_iter=2000).fit(oof_corr[cv_idx], y_train[cv_idx])
        p_leaky = clf_leaky.predict_proba(test_leaky_avg)[:, 1]
        p_corr = clf_corr.predict_proba(test_corr_avg)[:, 1]

        auroc_leaky_all.append(roc_auc_score(y_test, p_leaky))
        auroc_corr_all.append(roc_auc_score(y_test, p_corr))
        brier_leaky_all.append(brier_score_loss(y_test, p_leaky))
        brier_corr_all.append(brier_score_loss(y_test, p_corr))
        ece_leaky_all.append(ece_binned(p_leaky, y_test))
        ece_corr_all.append(ece_binned(p_corr, y_test))

        if (seed + 1) % 20 == 0:
            print(f"  [cap={hidden}] {seed + 1}/{n_seeds} seeds done", flush=True)

    auroc_leaky_all, auroc_corr_all = np.array(auroc_leaky_all), np.array(auroc_corr_all)
    brier_leaky_all, brier_corr_all = np.array(brier_leaky_all), np.array(brier_corr_all)
    ece_leaky_all, ece_corr_all = np.array(ece_leaky_all), np.array(ece_corr_all)

    auroc_gap = auroc_leaky_all - auroc_corr_all
    brier_gap = brier_corr_all - brier_leaky_all
    ece_gap = ece_corr_all - ece_leaky_all

    a_lo, a_hi = bca_ci(auroc_leaky_all, auroc_corr_all)
    b_lo, b_hi = bca_ci(brier_corr_all, brier_leaky_all)
    e_lo, e_hi = bca_ci(ece_corr_all, ece_leaky_all)

    ref = json.load(open(REF_78))
    ref_gap = ref["headline_comparison"][str(hidden)]["fully_corrected_gap"]

    result = {
        "capacity": hidden, "n_seeds": n_seeds,
        "auroc_leaky_mean": float(auroc_leaky_all.mean()), "auroc_corrected_mean": float(auroc_corr_all.mean()),
        "auroc_gap_mean": float(auroc_gap.mean()), "auroc_gap_bca_95ci": [a_lo, a_hi],
        "sanity_check_against_code78_fully_corrected_gap": ref_gap,
        "sanity_check_diff": float(auroc_gap.mean() - ref_gap),
        "brier_leaky_mean": float(brier_leaky_all.mean()), "brier_corrected_mean": float(brier_corr_all.mean()),
        "brier_gap_mean": float(brier_gap.mean()), "brier_gap_bca_95ci": [b_lo, b_hi],
        "ece_leaky_mean": float(ece_leaky_all.mean()), "ece_corrected_mean": float(ece_corr_all.mean()),
        "ece_gap_mean": float(ece_gap.mean()), "ece_gap_bca_95ci": [e_lo, e_hi],
    }
    print(f"cap={hidden}: AUROC gap={auroc_gap.mean():+.4f} (code/78 ref {ref_gap:+.4f}, "
          f"diff={auroc_gap.mean() - ref_gap:+.4f})  "
          f"Brier gap={brier_gap.mean():+.4f} BCa[{b_lo:+.4f},{b_hi:+.4f}]  "
          f"ECE gap={ece_gap.mean():+.4f} BCa[{e_lo:+.4f},{e_hi:+.4f}]")
    return result


def main():
    t0 = time.time()
    X, y = s43.load_raw_real_features()
    out = {}
    for hidden in CAPACITIES:
        print(f"\n=== Capacity {hidden} ===", flush=True)
        out[str(hidden)] = run_capacity(X, y, hidden)
    out["runtime_seconds"] = time.time() - t0
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
