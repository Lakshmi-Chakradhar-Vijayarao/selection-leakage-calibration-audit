"""
how large is the TRAINING-BUDGET DEFICIT that Appendix A.3 discloses
as a confound of the adaptivity control (Mechanism 3, code/22)?

WHY THIS SCRIPT EXISTS. Appendix A.3 already states, correctly, that
CLEAN_MATCHED_ADAPTIVE is disadvantaged on two axes -- it trains on 85% of
LEAKY's per-fold data (the ES_HOLD_FRACTION=0.15 carve-out) and selects against
a 67-sample carve-out rather than LEAKY's 112-sample fold -- and that
LEAKY - CLEAN_MATCHED_ADAPTIVE (+0.0049 at capacity 128, +0.0059 at 384) should
therefore be read as an UPPER BOUND on the fold-reuse effect. What it does not
say is how big the first of those two confounds actually is, and §4.3's
main-text sentence carries no qualifier at all. An independent review flagged
the gap: without a magnitude, a reader cannot tell whether the upper bound is
tight or whether the confound could account for the whole effect.

THE ISOLATION. The budget deficit can be measured on its own, holding
everything else fixed. CLEAN_MATCHED trains on the full tr_idx for an epoch
count chosen on the disjoint tr2_idx/es_idx run. This script adds one arm,
CLEAN_MATCHED_85, which is byte-for-byte the same construction except that the
blind fixed-epoch retrain runs on tr2_idx (85% of tr_idx) instead of tr_idx:

    CLEAN_MATCHED    = train_fixed_epochs(tr_idx,  best_epoch)   [100% budget]
    CLEAN_MATCHED_85 = train_fixed_epochs(tr2_idx, best_epoch)   [ 85% budget]

Same selection rule, same selected epoch, same seeds, same folds, same
downstream OOF/test-feature pipeline. The difference between the two arms is
therefore attributable to training-set size alone, which is exactly the
confound A.3 names. The recomputed CLEAN_MATCHED arm is asserted against
code/22's shipped per-seed values before anything is reported, so the pairing
cannot silently drift.

Everything else -- the generator, SweepMLP, the split/fold/init seeds, the
capacities -- is imported from code/22 rather than copied.

Output: results/adaptivity_control_budget_deficit.json
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
OUT_PATH = ROOT / "results" / "adaptivity_control_budget_deficit.json"
REF_PATH = ROOT / "results" / "epoch_forcing_confound_control.json"

_spec = importlib.util.spec_from_file_location(
    "s22", CODE / "22_epoch_forcing_confound_control.py")
s22 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s22)

RNG_GLOBAL = np.random.default_rng(2026)


def run_one_seed(seed, hidden):
    """CLEAN_MATCHED and CLEAN_MATCHED_85 only, on code/22's exact split path."""
    X, y = s22.make_synthetic_data(seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=s22.TEST_SIZE, stratify=y, random_state=seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    skf = StratifiedKFold(n_splits=s22.N_INNER_FOLDS, shuffle=True, random_state=seed)
    feat_dim_out = hidden // 2
    n_tr = len(y_train)
    conds = ["clean_matched", "clean_matched_85"]
    oof = {k: np.zeros((n_tr, feat_dim_out)) for k in conds}
    test_feat = {k: np.zeros((len(y_test), feat_dim_out)) for k in conds}
    n_tr_full, n_tr_85 = [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        fold_seed = seed * 100 + fold
        tr2_idx, es_idx = train_test_split(
            tr_idx, test_size=s22.ES_HOLD_FRACTION, stratify=y_train[tr_idx],
            random_state=fold_seed)
        _, best_epoch = s22.train_to_best_checkpoint(
            X_train[tr2_idx], y_train[tr2_idx], X_train[es_idx], y_train[es_idx],
            hidden, s22.EPOCHS, fold_seed)

        m_full = s22.train_fixed_epochs(X_train[tr_idx], y_train[tr_idx],
                                        hidden, best_epoch, fold_seed)
        oof["clean_matched"][val_idx] = s22.extract_features(m_full, X_train[val_idx])
        test_feat["clean_matched"] += s22.extract_features(m_full, X_test)

        m_85 = s22.train_fixed_epochs(X_train[tr2_idx], y_train[tr2_idx],
                                      hidden, best_epoch, fold_seed)
        oof["clean_matched_85"][val_idx] = s22.extract_features(m_85, X_train[val_idx])
        test_feat["clean_matched_85"] += s22.extract_features(m_85, X_test)

        n_tr_full.append(len(tr_idx)); n_tr_85.append(len(tr2_idx))

    aucs = {}
    for k in oof:
        test_feat[k] /= s22.N_INNER_FOLDS
        clf = LogisticRegression(max_iter=2000).fit(oof[k], y_train)
        aucs[k] = roc_auc_score(y_test, clf.predict_proba(test_feat[k])[:, 1])
    return aucs, float(np.mean(n_tr_full)), float(np.mean(n_tr_85))


def main():
    t0 = time.time()
    ref = json.load(open(REF_PATH))["by_capacity"]
    out = {"by_capacity": {}}

    for hidden in s22.CAPACITIES:
        full, small = [], []
        n_full = n_85 = None
        for seed in range(s22.N_SEEDS):
            a, nf, n8 = run_one_seed(seed, hidden)
            full.append(a["clean_matched"]); small.append(a["clean_matched_85"])
            n_full, n_85 = nf, n8
            if (seed + 1) % 25 == 0:
                print(f"  cap={hidden} [{seed+1}/{s22.N_SEEDS}] "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
        full = np.array(full); small = np.array(small)

        shipped = np.array(ref[str(hidden)]["aucs"]["clean_matched"])
        drift = float(np.max(np.abs(full - shipped)))
        assert drift < 1e-12, (f"recomputed CLEAN_MATCHED drifted from code/22's shipped "
                               f"per-seed values by {drift:.2e} at capacity {hidden}")

        deficit = full - small
        _, p = wilcoxon(full, small)
        res = bootstrap((deficit,), np.mean, confidence_level=0.95, n_resamples=10000,
                        method="BCa", random_state=RNG_GLOBAL)
        leaky = np.array(ref[str(hidden)]["aucs"]["leaky"])
        cma = np.array(ref[str(hidden)]["aucs"]["clean_matched_adaptive"])
        fold_reuse = float(np.mean(leaky - cma))
        cm_gap = float(np.mean(leaky - full))

        out["by_capacity"][str(hidden)] = {
            "n_seeds": s22.N_SEEDS,
            "n_train_per_fold_full": n_full,
            "n_train_per_fold_85": n_85,
            "clean_matched_mean": float(full.mean()),
            "clean_matched_85_mean": float(small.mean()),
            "budget_deficit_mean": float(deficit.mean()),
            "budget_deficit_bca_ci_95": [float(res.confidence_interval.low),
                                         float(res.confidence_interval.high)],
            "budget_deficit_wilcoxon_p": float(p),
            "budget_deficit_n_positive": int((deficit > 0).sum()),
            "leaky_minus_clean_matched_adaptive": fold_reuse,
            "leaky_minus_clean_matched": cm_gap,
            "budget_deficit_over_fold_reuse_effect": float(deficit.mean() / fold_reuse),
            "budget_deficit_exceeds_fold_reuse_effect": bool(deficit.mean() > fold_reuse),
            "reproduces_code22_clean_matched_per_seed": True,
        }
        print(f"capacity {hidden}: budget deficit = {deficit.mean():+.4f} "
              f"(CI [{res.confidence_interval.low:+.4f}, {res.confidence_interval.high:+.4f}], "
              f"p={p:.4g}) vs LEAKY-CMA fold-reuse increment {fold_reuse:+.4f} "
              f"({deficit.mean()/fold_reuse:.2f}x)", flush=True)

    out["design"] = (
        "CLEAN_MATCHED_85 is CLEAN_MATCHED with the blind fixed-epoch retrain run on "
        "tr2_idx (85% of tr_idx) instead of tr_idx. Identical selection rule, identical "
        "selected epoch, identical seeds/folds/initialization, identical downstream OOF and "
        "test-feature pipeline. The only difference is training-set size, so CLEAN_MATCHED "
        "minus CLEAN_MATCHED_85 isolates the ~15% training-budget deficit that "
        "CLEAN_MATCHED_ADAPTIVE also pays.")
    out["reading"] = (
        "The adaptivity control's ~15% training-budget deficit is worth a measurable amount "
        "of AUROC on its own. Where it is comparable to or larger than the "
        "LEAKY-minus-CLEAN_MATCHED_ADAPTIVE increment, that increment cannot be read as a "
        "clean point estimate of fold-reuse-specific severity: it is an upper bound, which "
        "is what Appendix A.3 already said and what SS4.3 now says too. The confound biases "
        "toward the reported finding, not against it.")
    out["runtime_seconds"] = time.time() - t0

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
