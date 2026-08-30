"""
does the downstream-averaging suppression finding survive when it is
measured against a CORRECTED control?

WHY THIS SCRIPT EXISTS. `code/55` tests alternative explanation T2 --- that the
audited pipeline's cross-model feature averaging destroys most of the leakage
signal before the reported metric sees it --- and finds it supported: removing
the averaging step raises the LEAKY-minus-CLEAN_MATCHED gap by roughly
2.3-5.8x. The paper leans on that as one of the lines of evidence carrying
Mechanism 3.

A third independent review found that `code/55` cannot be that evidence in its
shipped form. Line 122 reads

    ES_HOLD_FRACTION = _M.ES_HOLD_FRACTION

importing the uncorrected 0.15 straight from `code/02d`, and the whole script is
a re-readout of `code/02d`'s own trained models. So its CLEAN_MATCHED arm is not
merely similar to the arm whose asymmetry `code/75` and `code/77` showed
accounts for the entire primary gap --- it IS that arm. Whatever `code/55`
measured, it measured against a control the paper no longer accepts.

WHAT THIS SCRIPT DOES. It repeats `code/55`'s comparison of readouts on
`code/77`'s corrected geometry, at capacity 128 and AUROC_0=0.80, n=100 seeds,
with three arms:

  LEAKY            -- selects on the fold it reports on.
  CM_SHIPPED       -- the uncorrected control: 56-sample in-fold carve-out, so
                      the selection run trains on 85% of tr_idx. This is
                      `code/55`'s arm, reconstructed on this geometry.
  CM_CORRECTED     -- the fully corrected control: a fold-matched out-of-fold
                      selection set (94 points, the size of val_idx) with the
                      selection run at 100% of tr_idx.

and three readouts, exactly `code/55`'s:

  (A) averaged      -- the shipped pipeline: mean of the K fold-models' test
                       features, then the meta-learner.
  (B) single_model  -- no cross-model averaging: the meta-learner scores each
                       fold-model's test features separately and the K AUROCs
                       are averaged after scoring.
  (C) oof_apparent  -- the meta-learner's in-sample AUROC on the OOF matrix,
                       which never passes through the averaging step. Biased
                       upward in absolute terms for every arm equally; used
                       only comparatively, exactly as `code/55` uses it.

THE QUESTION, PUT SHARPLY. `code/55`'s finding has two parts, and correcting the
control can affect them differently. The first part --- that averaging
attenuates whatever gap is there --- is a statement about the readout and could
survive. The second --- that the mechanism therefore "is not inherently benign
at its source" --- requires the un-averaged gap against a SOUND control to be
non-zero. If the corrected un-averaged gap is indistinguishable from zero, then
what the averaging was suppressing was the control asymmetry, not the leak, and
the T2 finding cannot carry the mechanism.

Everything is imported from `code/47` (via the same construction `code/77`
uses), so the operating point is identical to the primary estimate's.

Output: results/oof_averaging_control_corrected_arms.json
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
OUT_PATH = ROOT / "results" / "oof_averaging_control_corrected_arms.json"
REF_77 = ROOT / "results" / "mechanism3_factorial_selection_controls.json"

_spec = importlib.util.spec_from_file_location(
    "s47", CODE / "47_selection_multiplicity_sweep.py")
s47 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s47)

CAPACITY, N_SEEDS = 128, 100
EPOCHS = s47.DEFAULT_EPOCHS
N_SAMPLES = s47.DEFAULT_N_SAMPLES
TARGET_AUROC = s47.DEFAULT_TARGET_AUROC
K = s47.N_INNER_FOLDS
TEST_SIZE = s47.TEST_SIZE
F_SHIPPED = s47.ES_HOLD_FRACTION
OOF_HOLD_FRACTION = 1.0 / (K + 1)
ARMS = ["leaky", "cm_shipped", "cm_corrected"]


def run_seed(seed):
    data_seed, split_seed = seed, seed + 100000
    fold_seed_base, init_seed_base = seed + 200000, seed + 300000
    X, y = s47.make_synthetic_data(data_seed, N_SAMPLES, TARGET_AUROC)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=split_seed)
    sc = StandardScaler()
    X_train, X_test = sc.fit_transform(X_train), sc.transform(X_test)

    n_tr = len(y_train)
    cv_idx, ges_idx = train_test_split(
        np.arange(n_tr), test_size=OOF_HOLD_FRACTION, stratify=y_train,
        random_state=split_seed + 500000)
    cv_idx, ges_idx = np.sort(cv_idx), np.sort(ges_idx)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=fold_seed_base)
    fdim = CAPACITY // 2

    oof = {a: np.zeros((n_tr, fdim)) for a in ARMS}
    # Per-fold-model test features, kept separately so the averaging step can be
    # removed rather than approximated.
    per_model = {a: [] for a in ARMS}

    for fold, (tr_loc, val_loc) in enumerate(skf.split(X_train[cv_idx], y_train[cv_idx])):
        tr_idx, val_idx = cv_idx[tr_loc], cv_idx[val_loc]
        init_seed = init_seed_base * 100 + fold

        m_l, _, _ = s47.train_to_best_checkpoint(
            X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_train[val_idx],
            CAPACITY, EPOCHS, init_seed)
        oof["leaky"][val_idx] = s47.extract_features(m_l, X_train[val_idx])
        per_model["leaky"].append(s47.extract_features(m_l, X_test))

        # Uncorrected: in-fold carve-out, selection run on 85% of tr_idx.
        tr2_idx, es_idx = train_test_split(
            tr_idx, test_size=F_SHIPPED, stratify=y_train[tr_idx],
            random_state=init_seed)
        _, ep_s, _ = s47.train_to_best_checkpoint(
            X_train[tr2_idx], y_train[tr2_idx], X_train[es_idx], y_train[es_idx],
            CAPACITY, EPOCHS, init_seed)
        m_s = s47.train_fixed_epochs(X_train[tr_idx], y_train[tr_idx],
                                     CAPACITY, ep_s, init_seed)
        oof["cm_shipped"][val_idx] = s47.extract_features(m_s, X_train[val_idx])
        per_model["cm_shipped"].append(s47.extract_features(m_s, X_test))

        # Corrected: fold-matched out-of-fold selection set, 100% budget.
        _, ep_c, _ = s47.train_to_best_checkpoint(
            X_train[tr_idx], y_train[tr_idx], X_train[ges_idx], y_train[ges_idx],
            CAPACITY, EPOCHS, init_seed)
        m_c = s47.train_fixed_epochs(X_train[tr_idx], y_train[tr_idx],
                                     CAPACITY, ep_c, init_seed)
        oof["cm_corrected"][val_idx] = s47.extract_features(m_c, X_train[val_idx])
        per_model["cm_corrected"].append(s47.extract_features(m_c, X_test))

    out = {}
    for a in ARMS:
        clf = LogisticRegression(max_iter=2000).fit(oof[a][cv_idx], y_train[cv_idx])
        avg = np.mean(per_model[a], axis=0)
        out[f"{a}__averaged"] = float(roc_auc_score(
            y_test, clf.predict_proba(avg)[:, 1]))
        out[f"{a}__single_model"] = float(np.mean([
            roc_auc_score(y_test, clf.predict_proba(tf)[:, 1])
            for tf in per_model[a]]))
        out[f"{a}__oof_apparent"] = float(roc_auc_score(
            y_train[cv_idx], clf.predict_proba(oof[a][cv_idx])[:, 1]))
    return out


def gap_stats(a, b, boot_seed):
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return {"gap_mean": 0.0, "gap_bca_ci_95": [0.0, 0.0], "wilcoxon_p": 1.0,
                "n_positive": 0, "n_seeds": int(len(d))}
    _, p = wilcoxon(a, b)
    res = bootstrap((d,), np.mean, confidence_level=0.95, n_resamples=10000,
                    method="BCa", random_state=np.random.default_rng(boot_seed))
    return {"gap_mean": float(d.mean()),
            "gap_bca_ci_95": [float(res.confidence_interval.low),
                              float(res.confidence_interval.high)],
            "wilcoxon_p": float(p), "n_positive": int((d > 0).sum()),
            "n_seeds": int(len(d))}


def main():
    t0 = time.time()
    cols = {}
    for seed in range(N_SEEDS):
        for k, v in run_seed(seed).items():
            cols.setdefault(k, []).append(v)
        if (seed + 1) % 25 == 0:
            print(f"  {seed+1}/{N_SEEDS} elapsed={time.time()-t0:.0f}s", flush=True)
    arr = {k: np.array(v) for k, v in cols.items()}

    readouts = ["averaged", "single_model", "oof_apparent"]
    res = {}
    for i, r in enumerate(readouts):
        res[r] = {
            "leaky_mean": float(arr[f"leaky__{r}"].mean()),
            "cm_shipped_mean": float(arr[f"cm_shipped__{r}"].mean()),
            "cm_corrected_mean": float(arr[f"cm_corrected__{r}"].mean()),
            "gap_vs_shipped_control": gap_stats(
                arr[f"leaky__{r}"], arr[f"cm_shipped__{r}"], 20260803 + 10 * i),
            "gap_vs_corrected_control": gap_stats(
                arr[f"leaky__{r}"], arr[f"cm_corrected__{r}"], 20260803 + 10 * i + 5),
        }
    for r in readouts:
        for which in ("shipped", "corrected"):
            base = res["averaged"][f"gap_vs_{which}_control"]["gap_mean"]
            res[r][f"suppression_ratio_vs_averaged_{which}"] = (
                float(res[r][f"gap_vs_{which}_control"]["gap_mean"] / base)
                if abs(base) > 1e-12 else None)

    sm_corr = res["single_model"]["gap_vs_corrected_control"]
    survives = bool(sm_corr["gap_mean"] > 0 and sm_corr["gap_bca_ci_95"][0] > 0)

    out = {
        "config": {"capacity": CAPACITY, "n_seeds": N_SEEDS, "epochs": EPOCHS,
                   "n_samples": N_SAMPLES, "target_auroc": TARGET_AUROC,
                   "k_cv": K, "es_fraction_shipped": F_SHIPPED,
                   "oof_hold_fraction": OOF_HOLD_FRACTION,
                   "geometry_note": (
                       "code/77's shared out-of-fold geometry: 94 of the 560 "
                       "training samples are held out before folding for every "
                       "arm equally, so |val_idx|~93 and |tr_idx|~373. Absolute "
                       "AUROCs are therefore not comparable to code/55's, which "
                       "runs on the full pool; the readout CONTRASTS are."),
                   "what_was_wrong": (
                       "code/55:122 imports ES_HOLD_FRACTION from code/02d and "
                       "re-reads code/02d's own trained models, so its "
                       "CLEAN_MATCHED arm is the uncorrected arm whose asymmetry "
                       "code/77 shows accounts for the whole primary gap.")},
        "by_readout": res,
        "verdict_single_model_gap_survives_correction": survives,
        "reading": (
            "The first half of code/55's finding --- that removing the averaging "
            "step raises whatever gap is present --- is a property of the readout "
            "and is reported here against both controls. The second half, that the "
            "mechanism is therefore not benign at its source, requires the "
            "un-averaged gap against a SOUND control to be non-zero, which is what "
            "verdict_single_model_gap_survives_correction reports."),
        "runtime_seconds": time.time() - t0,
        "outputs": {"json": str(OUT_PATH.relative_to(ROOT)),
                    "script": str(Path(__file__).resolve().relative_to(ROOT))},
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 78)
    print(f"{'readout':<14} {'vs shipped control':>28} {'vs corrected control':>28}")
    for r in readouts:
        a = res[r]["gap_vs_shipped_control"]; b = res[r]["gap_vs_corrected_control"]
        print(f"{r:<14} {a['gap_mean']:+.4f} "
              f"[{a['gap_bca_ci_95'][0]:+.4f},{a['gap_bca_ci_95'][1]:+.4f}] "
              f"  {b['gap_mean']:+.4f} "
              f"[{b['gap_bca_ci_95'][0]:+.4f},{b['gap_bca_ci_95'][1]:+.4f}]")
    print(f"\nun-averaged gap survives the corrected control: {survives}")
    print("=" * 78)
    print(f"\nSaved: {OUT_PATH}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
