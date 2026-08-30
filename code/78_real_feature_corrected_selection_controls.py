"""
the corrected selection controls applied to the REAL-FEATURE harness.

WHY THIS SCRIPT EXISTS. The paper has been claiming, in its abstract, SS4.3, SS7
and its correction history, that the real-feature Mechanism-3 estimate is
"untouched by" the control asymmetry that dissolved the primary synthetic
estimate. That claim is false and a third independent review proved it by
grep: `code/43_calibration_leakage_diagnostic.py:79` sets

    ES_HOLD_FRACTION = 0.15

which is the SAME uncorrected constant `code/02d` uses, and `code/43:366` uses
it in the SAME in-fold `train_test_split(tr_idx, test_size=ES_HOLD_FRACTION...)`
construction. The real-feature line therefore carries both asymmetries the
previous round sized on the synthetic harness -- an undersized selection set and
an under-trained selection run -- and had never been tested under correction.

WHAT THIS SCRIPT DOES. It re-runs code/43's severity comparison under the
corrections, at BOTH shipped capacities, and reports the corrected numbers
alongside the originals rather than replacing them. code/43 itself is not
modified: its machinery (feature loading, the label-free calibration, SweepMLP,
train_to_best_checkpoint, train_fixed_epochs, the seeds, the epoch count, the
fold structure) is imported, so the two cannot drift.

  PART A -- the directly comparable correction. Same 400-sample pool, same
  outer split, same folds, same seeds; the only change is the in-fold carve-out
  fraction, swept over {0.15 (shipped), 0.25 (fold-matched)}. With n_train=320
  and K_CV=5, |val_idx| = 64 and |tr_idx| = 256, so the fraction that makes
  |es_idx| = |val_idx| is f = 1/(K_CV-1) = 0.25, exactly as on the synthetic
  harness. Because nothing but f moves, Part A's ES=0.25 gap is directly
  comparable to the shipped +0.0093 (capacity 128) / +0.0077 (384). The ES=0.15
  arm is asserted per seed against code/43's shipped arrays before anything is
  reported.

  PART B -- the fully corrected control, on the shared out-of-fold geometry.
  The 2 x 2 x 2 factorial of code/77 (selection-set size x selection-run budget
  x training depth) ported to this harness, so the real-feature line gets the
  same decomposition the synthetic line gets. All arms share one pool, one fold
  assignment and one LEAKY reference; the out-of-fold carve-out is present for
  every arm, so the in-fold/out-of-fold factor is not confounded with pool size.

  Part B's absolute AUROCs are not comparable to Part A's -- the carve-out costs
  every arm 1/(K+1) of the cross-validation pool -- but its gaps are internally
  matched, which is what a control has to be.

WHAT COUNTS AS THE ANSWER. If the corrected real-feature gap keeps its sign and
its BCa CI still excludes zero, Mechanism 3's real-feature evidence survives the
correction and the paper says so at the corrected magnitude. If it attenuates
the way the primary synthetic estimate did, that is the more important finding --
it would mean the confound is not an artifact of the synthetic generator but a
property of the control construction itself, and the paper reports it as a
finding in its own right rather than burying it.

Output: results/real_feature_corrected_selection_controls.json
"""
import importlib.util
import itertools
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
OUT_PATH = ROOT / "results" / "real_feature_corrected_selection_controls.json"
REF_43 = ROOT / "results" / "real_feature_test_train_only_calibrated.json"
CALIB = ROOT / "results" / "calibration_leakage_diagnostic.json"

_spec = importlib.util.spec_from_file_location(
    "s43", CODE / "43_calibration_leakage_diagnostic.py")
s43 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s43)
# code/43 sets this inside main() after its bisection; read the shipped value so
# this harness and code/43 cannot drift on calibration strength.
s43.ALPHA_TRAIN_ONLY = float(json.load(open(CALIB))["train_only_calibration_alpha"])

CAPACITIES = s43.CAPACITIES               # [128, 384]
N_SEEDS = s43.N_SEEDS                     # 100
EPOCHS = s43.EPOCHS                       # 45
N_INNER_FOLDS = s43.N_INNER_FOLDS         # 5
TEST_SIZE = s43.TEST_SIZE                 # 0.20

F_SHIPPED = s43.ES_HOLD_FRACTION          # 0.15
F_FOLD = 1.0 / (N_INNER_FOLDS - 1)        # 0.25
ES_FRACTIONS = [F_SHIPPED, F_FOLD]
OOF_HOLD_FRACTION = 1.0 / (N_INNER_FOLDS + 1)

SIZES = ["small", "fold_matched"]
LOCS = ["infold", "oof"]
DEPTHS = ["free", "matched"]
ARMS = [f"{s}__{l}__{d}" for s, l, d in itertools.product(SIZES, LOCS, DEPTHS)]
FREE_ARMS = [a for a in ARMS if a.endswith("__free")]
SEL_KEYS = [f"{s}__{l}" for s, l in itertools.product(SIZES, LOCS)]


def _prep(X, y, seed):
    """code/43's exact split-and-calibrate path, reproduced by import."""
    tr_i, te_i, y_train, y_test = train_test_split(
        np.arange(len(y)), y, test_size=TEST_SIZE, stratify=y, random_state=seed)
    Xc = s43.apply_calibration_label_free(
        X, y, s43.ALPHA_TRAIN_ONLY, tr_i, seed=seed).astype(np.float32)
    X_train, X_test = Xc[tr_i], Xc[te_i]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def _fit_and_score(oof, test_feat, rows, y_train, y_test):
    out = {}
    for k in oof:
        tf = test_feat[k] / N_INNER_FOLDS
        clf = LogisticRegression(max_iter=2000).fit(oof[k][rows], y_train[rows])
        out[k] = float(roc_auc_score(y_test, clf.predict_proba(tf)[:, 1]))
    return out


# ---------------------------------------------------------------------------
# PART A -- full pool, in-fold carve-out at f in {0.15, 0.25}
# ---------------------------------------------------------------------------
def partA_seed(X, y, seed, hidden):
    X_train, X_test, y_train, y_test = _prep(X, y, seed)
    skf = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=seed)
    fdim, n_tr = hidden // 2, len(y_train)
    keys = ["leaky"] + [f"cm_{f:.2f}" for f in ES_FRACTIONS]
    oof = {k: np.zeros((n_tr, fdim)) for k in keys}
    test_feat = {k: np.zeros((len(y_test), fdim)) for k in keys}
    geom = {f"cm_{f:.2f}": {"n_es": [], "n_tr2": [], "ep": []} for f in ES_FRACTIONS}
    n_val, n_trf, leaky_ep = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        fold_seed = seed * 100 + fold
        m_l, ep_l = s43.train_to_best_checkpoint(
            X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_train[val_idx],
            hidden, EPOCHS, fold_seed)
        oof["leaky"][val_idx] = s43.extract_features(m_l, X_train[val_idx])
        test_feat["leaky"] += s43.extract_features(m_l, X_test)
        n_val.append(len(val_idx)); n_trf.append(len(tr_idx)); leaky_ep.append(ep_l)

        for f in ES_FRACTIONS:
            k = f"cm_{f:.2f}"
            tr2_idx, es_idx = train_test_split(
                tr_idx, test_size=f, stratify=y_train[tr_idx], random_state=fold_seed)
            _, ep = s43.train_to_best_checkpoint(
                X_train[tr2_idx], y_train[tr2_idx], X_train[es_idx], y_train[es_idx],
                hidden, EPOCHS, fold_seed)
            m = s43.train_fixed_epochs(X_train[tr_idx], y_train[tr_idx],
                                       hidden, ep, fold_seed)
            oof[k][val_idx] = s43.extract_features(m, X_train[val_idx])
            test_feat[k] += s43.extract_features(m, X_test)
            geom[k]["n_es"].append(len(es_idx))
            geom[k]["n_tr2"].append(len(tr2_idx))
            geom[k]["ep"].append(ep)

    aucs = _fit_and_score(oof, test_feat, np.arange(n_tr), y_train, y_test)
    meta = {"n_val": float(np.mean(n_val)), "n_tr_fold": float(np.mean(n_trf)),
            "leaky_ep": float(np.mean(leaky_ep))}
    for k, g in geom.items():
        meta[k] = {kk: float(np.mean(vv)) for kk, vv in g.items()}
    return aucs, meta


# ---------------------------------------------------------------------------
# PART B -- shared out-of-fold geometry, 2 x 2 x 2 factorial
# ---------------------------------------------------------------------------
def _pool(y_train, seed):
    n_tr = len(y_train)
    cv_idx, ges_idx = train_test_split(
        np.arange(n_tr), test_size=OOF_HOLD_FRACTION, stratify=y_train,
        random_state=seed + 500000)
    skf = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=seed)
    return np.sort(cv_idx), np.sort(ges_idx), skf


def _selection_sets(tr_idx, ges_idx, y_train, fold_seed):
    out = {}
    for size, f in (("small", F_SHIPPED), ("fold_matched", F_FOLD)):
        tr2_idx, es_idx = train_test_split(
            tr_idx, test_size=f, stratify=y_train[tr_idx], random_state=fold_seed)
        out[f"{size}__infold"] = (tr2_idx, es_idx)
    n_small = len(out["small__infold"][1])
    small_ges, _ = train_test_split(
        ges_idx, train_size=n_small, stratify=y_train[ges_idx], random_state=fold_seed)
    out["small__oof"] = (tr_idx, np.sort(small_ges))
    out["fold_matched__oof"] = (tr_idx, ges_idx)
    return out


def partB_pass1(X, y, seed, hidden):
    X_train, X_test, y_train, y_test = _prep(X, y, seed)
    cv_idx, ges_idx, skf = _pool(y_train, seed)
    fdim, n_tr = hidden // 2, len(y_train)
    rng = np.random.default_rng(seed + 10000)
    keys = ["leaky", "placebo"] + FREE_ARMS
    oof = {k: np.zeros((n_tr, fdim)) for k in keys}
    test_feat = {k: np.zeros((len(y_test), fdim)) for k in keys}
    epochs = {k: [] for k in ["leaky"] + SEL_KEYS}
    sizes = {k: [] for k in SEL_KEYS}
    sel_tr_n = {k: [] for k in SEL_KEYS}
    n_val, n_trf, ident = [], [], []

    for fold, (tr_loc, val_loc) in enumerate(skf.split(X_train[cv_idx], y_train[cv_idx])):
        tr_idx, val_idx = cv_idx[tr_loc], cv_idx[val_loc]
        fold_seed = seed * 100 + fold
        n_val.append(len(val_idx)); n_trf.append(len(tr_idx))

        m_l, ep_l = s43.train_to_best_checkpoint(
            X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_train[val_idx],
            hidden, EPOCHS, fold_seed)
        oof["leaky"][val_idx] = s43.extract_features(m_l, X_train[val_idx])
        test_feat["leaky"] += s43.extract_features(m_l, X_test)
        epochs["leaky"].append(ep_l)

        for key, (sel_tr, sel_es) in _selection_sets(tr_idx, ges_idx, y_train, fold_seed).items():
            m_sel, ep = s43.train_to_best_checkpoint(
                X_train[sel_tr], y_train[sel_tr], X_train[sel_es], y_train[sel_es],
                hidden, EPOCHS, fold_seed)
            epochs[key].append(ep); sizes[key].append(len(sel_es))
            sel_tr_n[key].append(len(sel_tr))
            m_free = s43.train_fixed_epochs(X_train[tr_idx], y_train[tr_idx],
                                            hidden, ep, fold_seed)
            if key.endswith("__oof"):
                ident.append(float(_identical(m_sel, m_free)))
            a = f"{key}__free"
            oof[a][val_idx] = s43.extract_features(m_free, X_train[val_idx])
            test_feat[a] += s43.extract_features(m_free, X_test)

        y_perm = rng.permutation(y_train[val_idx])
        m_p, _ = s43.train_to_best_checkpoint(
            X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_perm,
            hidden, EPOCHS, fold_seed)
        oof["placebo"][val_idx] = s43.extract_features(m_p, X_train[val_idx])
        test_feat["placebo"] += s43.extract_features(m_p, X_test)

    aucs = _fit_and_score(oof, test_feat, cv_idx, y_train, y_test)
    return aucs, {"epochs": {k: [int(e) for e in v] for k, v in epochs.items()},
                  "n_val": float(np.mean(n_val)), "n_tr_fold": float(np.mean(n_trf)),
                  "n_ges": int(len(ges_idx)), "n_cv": int(len(cv_idx)),
                  "sel_sizes": {k: float(np.mean(v)) for k, v in sizes.items()},
                  "sel_train_sizes": {k: float(np.mean(v)) for k, v in sel_tr_n.items()},
                  "oof_retrain_identical": float(np.mean(ident))}


def _identical(m1, m2):
    import torch
    s1, s2 = m1.state_dict(), m2.state_dict()
    return all(torch.equal(s1[k], s2[k]) for k in s1)


def partB_pass2(X, y, seed, hidden, stored_epochs, shifts, const_depth):
    X_train, X_test, y_train, y_test = _prep(X, y, seed)
    cv_idx, ges_idx, skf = _pool(y_train, seed)
    fdim, n_tr = hidden // 2, len(y_train)
    keys = [f"{k}__matched" for k in SEL_KEYS] + ["const_depth"]
    oof = {k: np.zeros((n_tr, fdim)) for k in keys}
    test_feat = {k: np.zeros((len(y_test), fdim)) for k in keys}
    n_ident, n_checks = 0, 0

    for fold, (tr_loc, val_loc) in enumerate(skf.split(X_train[cv_idx], y_train[cv_idx])):
        tr_idx, val_idx = cv_idx[tr_loc], cv_idx[val_loc]
        fold_seed = seed * 100 + fold
        m_c = s43.train_fixed_epochs(X_train[tr_idx], y_train[tr_idx],
                                     hidden, const_depth, fold_seed)
        oof["const_depth"][val_idx] = s43.extract_features(m_c, X_train[val_idx])
        test_feat["const_depth"] += s43.extract_features(m_c, X_test)
        for key in SEL_KEYS:
            ep = max(1, stored_epochs[key][fold] + shifts[key])
            m = s43.train_fixed_epochs(X_train[tr_idx], y_train[tr_idx],
                                       hidden, ep, fold_seed)
            a = f"{key}__matched"
            oof[a][val_idx] = s43.extract_features(m, X_train[val_idx])
            test_feat[a] += s43.extract_features(m, X_test)
            if ep == const_depth:
                n_checks += 1
                n_ident += int(_identical(m, m_c))
    return _fit_and_score(oof, test_feat, cv_idx, y_train, y_test), (n_ident, n_checks)


def gap_stats(a, b, boot_seed):
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return {"gap_mean": 0.0, "gap_bca_ci_95": [0.0, 0.0], "wilcoxon_p": 1.0,
                "gap_std": 0.0, "n_positive": 0, "n_seeds": int(len(d))}
    _, p = wilcoxon(a, b)
    res = bootstrap((d,), np.mean, confidence_level=0.95, n_resamples=10000,
                    method="BCa", random_state=np.random.default_rng(boot_seed))
    return {"gap_mean": float(d.mean()),
            "gap_bca_ci_95": [float(res.confidence_interval.low),
                              float(res.confidence_interval.high)],
            "wilcoxon_p": float(p), "gap_std": float(d.std(ddof=1)),
            "n_positive": int((d > 0).sum()), "n_seeds": int(len(d))}


def main():
    t0 = time.time()
    X, y = s43.load_raw_real_features()
    ref = json.load(open(REF_43))["capacities"]

    out = {
        "config": {
            "n_samples": int(len(y)), "capacities": CAPACITIES, "n_seeds": N_SEEDS,
            "epochs": EPOCHS, "n_inner_folds": N_INNER_FOLDS, "test_size": TEST_SIZE,
            "calibration_alpha": s43.ALPHA_TRAIN_ONLY,
            "calibration_method": "label_free_axis_noising (code/43)",
            "es_fraction_shipped": F_SHIPPED,
            "es_fraction_fold_matched": F_FOLD,
            "oof_hold_fraction": OOF_HOLD_FRACTION,
            "what_was_wrong": (
                "code/43:79 sets ES_HOLD_FRACTION=0.15 and code/43:366 uses it in the "
                "same in-fold train_test_split(tr_idx, ...) construction as code/02d, "
                "so the real-feature estimate carries the identical selection-set-size "
                "and selection-run-budget asymmetries. The paper's claim that this line "
                "is untouched by that confound was false."),
        },
        "part_A_in_fold_es_sweep": {"by_capacity": {}},
        "part_B_factorial": {"by_capacity": {}},
    }

    # ---------------- PART A ----------------
    print("=== PART A: full pool, in-fold carve-out at f in {0.15, 0.25} ===", flush=True)
    for hidden in CAPACITIES:
        cols, metas = {}, []
        for seed in range(N_SEEDS):
            aucs, meta = partA_seed(X, y, seed, hidden)
            for k, v in aucs.items():
                cols.setdefault(k, []).append(v)
            metas.append(meta)
            if (seed + 1) % 25 == 0:
                print(f"  [A cap={hidden}] {seed+1}/{N_SEEDS} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
        arrs = {k: np.array(v) for k, v in cols.items()}

        # Assert reproduction of code/43's shipped arrays BEFORE reporting.
        for local, ref_key in (("leaky", "leaky"), (f"cm_{F_SHIPPED:.2f}", "clean_matched")):
            shipped = np.array(ref[str(hidden)]["aucs"][ref_key])
            drift = float(np.max(np.abs(arrs[local] - shipped)))
            assert drift < 1e-12, (
                f"recomputed {local} drifted from code/43's shipped per-seed {ref_key} "
                f"by {drift:.2e} at capacity {hidden}")

        cell = {"reproduces_code43_per_seed": True,
                "n_val": metas[0]["n_val"], "n_tr_fold": metas[0]["n_tr_fold"],
                "leaky_mean": float(arrs["leaky"].mean()),
                "leaky_best_epoch_mean": float(np.mean([m["leaky_ep"] for m in metas])),
                "by_es_fraction": {}}
        for i, f in enumerate(ES_FRACTIONS):
            k = f"cm_{f:.2f}"
            g = gap_stats(arrs["leaky"], arrs[k], boot_seed=hidden * 7919 + 41 + i)
            cell["by_es_fraction"][f"{f:.2f}"] = {
                **g,
                "clean_matched_mean": float(arrs[k].mean()),
                "n_selection_points": metas[0][k]["n_es"],
                "n_selection_run_training": metas[0][k]["n_tr2"],
                "selection_set_size_ratio_leaky_over_clean":
                    float(metas[0]["n_val"] / metas[0][k]["n_es"]),
                "clean_best_epoch_mean": float(np.mean([m[k]["ep"] for m in metas])),
                "is_shipped": abs(f - F_SHIPPED) < 1e-9,
                "is_fold_matched": abs(f - F_FOLD) < 1e-9,
            }
            print(f"  cap={hidden} ES={f:.2f} (|sel|={metas[0][k]['n_es']:.0f}, "
                  f"|tr2|={metas[0][k]['n_tr2']:.0f}): gap={g['gap_mean']:+.4f} "
                  f"CI=[{g['gap_bca_ci_95'][0]:+.4f},{g['gap_bca_ci_95'][1]:+.4f}] "
                  f"p={g['wilcoxon_p']:.4g}", flush=True)
        out["part_A_in_fold_es_sweep"]["by_capacity"][str(hidden)] = cell

    # ---------------- PART B ----------------
    print("\n=== PART B: shared out-of-fold geometry, 2x2x2 factorial ===", flush=True)
    for hidden in CAPACITIES:
        cols, metas = {}, []
        for seed in range(N_SEEDS):
            aucs, meta = partB_pass1(X, y, seed, hidden)
            for k, v in aucs.items():
                cols.setdefault(k, []).append(v)
            metas.append(meta)
            if (seed + 1) % 25 == 0:
                print(f"  [B1 cap={hidden}] {seed+1}/{N_SEEDS} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
        arrs = {k: np.array(v) for k, v in cols.items()}
        oof_ident = float(np.mean([m["oof_retrain_identical"] for m in metas]))
        assert oof_ident == 1.0, (
            f"out-of-fold arms' retrains not bitwise identical to their selection "
            f"runs at capacity {hidden} (frac={oof_ident})")

        mean_ep = {k: float(np.mean([e for m in metas for e in m["epochs"][k]]))
                   for k in ["leaky"] + SEL_KEYS}
        shifts = {k: int(round(mean_ep["leaky"] - mean_ep[k])) for k in SEL_KEYS}
        const_depth = int(round(mean_ep["leaky"]))

        ni = nc = 0
        for seed in range(N_SEEDS):
            aucs, (a, b) = partB_pass2(
                X, y, seed, hidden,
                {k: metas[seed]["epochs"][k] for k in SEL_KEYS}, shifts, const_depth)
            ni += a; nc += b
            for k, v in aucs.items():
                cols.setdefault(k, []).append(v)
            if (seed + 1) % 25 == 0:
                print(f"  [B2 cap={hidden}] {seed+1}/{N_SEEDS} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
        for k in list(cols):
            arrs[k] = np.array(cols[k])
        assert ni == nc, ("a depth-matched retrain at const_depth was not bitwise "
                          "identical to CONST_DEPTH")

        cells = {}
        for i, arm in enumerate(ARMS):
            size, loc, depth = arm.split("__")
            sk = f"{size}__{loc}"
            g = gap_stats(arrs["leaky"], arrs[arm], boot_seed=hidden * 104729 + 13 * i)
            cells[arm] = {
                "selection_set_size": size, "selection_run_budget": loc,
                "training_depth": depth,
                "arm_mean_auroc": float(arrs[arm].mean()),
                "n_selection_points": metas[0]["sel_sizes"][sk],
                "n_selection_run_training": metas[0]["sel_train_sizes"][sk],
                "selection_run_budget_pct": float(
                    100.0 * metas[0]["sel_train_sizes"][sk] / metas[0]["n_tr_fold"]),
                "mean_selected_epoch": mean_ep[sk],
                "depth_shift_applied": shifts[sk] if depth == "matched" else 0,
                "gap_leaky_minus_arm": g,
                "is_shipped_uncorrected": arm == "small__infold__free",
                "is_fully_corrected": arm == "fold_matched__oof__matched",
            }

        def mean_gap(pred):
            return float(np.mean([cells[a]["gap_leaky_minus_arm"]["gap_mean"]
                                  for a in ARMS if pred(a)]))

        shipped_g = cells["small__infold__free"]["gap_leaky_minus_arm"]["gap_mean"]
        corr_g = cells["fold_matched__oof__matched"]["gap_leaky_minus_arm"]["gap_mean"]
        main_eff = {
            "selection_set_size_small_minus_fold_matched":
                mean_gap(lambda a: a.startswith("small__"))
                - mean_gap(lambda a: a.startswith("fold_matched__")),
            "selection_run_budget_infold_minus_oof":
                mean_gap(lambda a: "__infold__" in a) - mean_gap(lambda a: "__oof__" in a),
            "training_depth_free_minus_matched":
                mean_gap(lambda a: a.endswith("__free"))
                - mean_gap(lambda a: a.endswith("__matched")),
        }
        fc = cells["fold_matched__oof__matched"]["gap_leaky_minus_arm"]
        out["part_B_factorial"]["by_capacity"][str(hidden)] = {
            "n_ges_out_of_fold_selection_set": metas[0]["n_ges"],
            "n_cv_pool": metas[0]["n_cv"],
            "n_val_per_fold": metas[0]["n_val"],
            "n_tr_idx_per_fold": metas[0]["n_tr_fold"],
            "oof_retrain_bitwise_identical_frac": oof_ident,
            "leaky_mean_auroc": float(arrs["leaky"].mean()),
            "placebo_mean_auroc": float(arrs["placebo"].mean()),
            "const_depth_epochs": const_depth,
            "const_depth_mean_auroc": float(arrs["const_depth"].mean()),
            "gap_leaky_minus_const_depth": gap_stats(
                arrs["leaky"], arrs["const_depth"], boot_seed=hidden * 104729 + 991),
            "gap_leaky_minus_placebo": gap_stats(
                arrs["leaky"], arrs["placebo"], boot_seed=hidden * 104729 + 993),
            "mean_selected_epoch_by_arm": mean_ep,
            "depth_shifts_applied": shifts,
            "factorial_cells": cells,
            "decomposition": {
                "shipped_uncorrected_gap": shipped_g,
                "fully_corrected_gap": corr_g,
                "total_movement": shipped_g - corr_g,
                "main_effects_average_over_other_factors": main_eff,
                "additivity_residual": float(
                    (shipped_g - corr_g) - sum(main_eff.values())),
            },
            "verdict": ("ESTABLISHED" if (fc["gap_mean"] > 0 and fc["gap_bca_ci_95"][0] > 0)
                        else "NOT_ESTABLISHED"),
        }
        print(f"  cap={hidden}: shipped-cell {shipped_g:+.4f} -> fully-corrected "
              f"{corr_g:+.4f} CI=[{fc['gap_bca_ci_95'][0]:+.4f},"
              f"{fc['gap_bca_ci_95'][1]:+.4f}] p={fc['wilcoxon_p']:.4g}  "
              f"verdict={out['part_B_factorial']['by_capacity'][str(hidden)]['verdict']}",
              flush=True)

    # ---------------- headline comparison ----------------
    comp = {}
    for hidden in CAPACITIES:
        A = out["part_A_in_fold_es_sweep"]["by_capacity"][str(hidden)]["by_es_fraction"]
        B = out["part_B_factorial"]["by_capacity"][str(hidden)]
        comp[str(hidden)] = {
            "shipped_uncorrected_gap": A[f"{F_SHIPPED:.2f}"]["gap_mean"],
            "shipped_uncorrected_ci": A[f"{F_SHIPPED:.2f}"]["gap_bca_ci_95"],
            "fold_matched_in_fold_gap": A[f"{F_FOLD:.2f}"]["gap_mean"],
            "fold_matched_in_fold_ci": A[f"{F_FOLD:.2f}"]["gap_bca_ci_95"],
            "fully_corrected_gap": B["decomposition"]["fully_corrected_gap"],
            "fully_corrected_ci": B["factorial_cells"]["fold_matched__oof__matched"]
                                   ["gap_leaky_minus_arm"]["gap_bca_ci_95"],
            "retention_fully_corrected_over_shipped": (
                B["decomposition"]["fully_corrected_gap"]
                / A[f"{F_SHIPPED:.2f}"]["gap_mean"]
                if abs(A[f"{F_SHIPPED:.2f}"]["gap_mean"]) > 1e-12 else None),
            "verdict": B["verdict"],
        }
    out["headline_comparison"] = comp

    out["limitations"] = [
        "Part A and Part B measure gaps at different pool sizes. Part A keeps "
        "code/43's full 320-sample training pool; Part B removes 1/(K+1) of it "
        "for the out-of-fold selection set, for every arm equally. Part B's gaps "
        "are internally matched but its absolute AUROCs are not comparable to "
        "Part A's or to code/43's shipped table.",
        "One dataset (Mistral-7B hidden states on HaluEval, 400 samples) and one "
        "calibration strength (alpha read from code/43). The calibration is what "
        "makes the task hard enough to have headroom; the corrected gaps inherit "
        "whatever the calibration does to the operating point.",
        "The depth shift is a single integer per arm estimated from the same 500 "
        "(seed, fold) pairs it is applied to, and equalizes MEAN depth only.",
        "The out-of-fold selection pool is reused across all five folds; it is "
        "disjoint from every val_idx so it cannot leak, but the folds' checkpoint "
        "choices are correlated through it. The small/oof arm redraws a "
        "size-matched subsample per fold, which partially decorrelates it.",
        "This harness reproduces MultiHaluDet's checkpoint-selection leak only. "
        "The scheduler/early-stopping pathway is code/49's, corrected separately.",
    ]
    out["runtime_seconds"] = time.time() - t0
    out["outputs"] = {"json": str(OUT_PATH.relative_to(ROOT)),
                      "script": str(Path(__file__).resolve().relative_to(ROOT))}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 80)
    for hidden in CAPACITIES:
        c = comp[str(hidden)]
        print(f"capacity {hidden}:  shipped {c['shipped_uncorrected_gap']:+.4f}  ->  "
              f"fold-matched in-fold {c['fold_matched_in_fold_gap']:+.4f}  ->  "
              f"fully corrected {c['fully_corrected_gap']:+.4f}  "
              f"[{c['fully_corrected_ci'][0]:+.4f},{c['fully_corrected_ci'][1]:+.4f}]  "
              f"{c['verdict']}")
    print("=" * 80)
    print(f"\nSaved: {OUT_PATH}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
