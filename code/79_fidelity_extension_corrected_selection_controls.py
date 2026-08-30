"""
the corrected selection controls applied to the FIDELITY EXTENSION.

WHY THIS SCRIPT EXISTS. Two reasons, both raised by the third independent
review.

(1) `code/49_mechanism3_fidelity_extension.py:212` sets ES_HOLD_FRACTION = 0.15
    and `code/49:541` uses it in the same in-fold
    `train_test_split(tr_idx, test_size=ES_HOLD_FRACTION, ...)` construction as
    `code/02d` and `code/43`. So the fidelity extension's +0.0250 carries the
    identical selection-set-size and selection-run-budget asymmetries that the
    corrected controls dissolved on the primary synthetic harness, and the
    paper's claim that this line does not share that confound was false.

(2) The fidelity extension's +0.0250 is the number that tops the paper's
    "magnitude triangle" (the 38.3x upper bound of the mechanism-and-harness
    axis), and it existed at ONE capacity only. A single-capacity estimate
    cannot support an axis. This script runs it at capacity 384 as well.

WHAT THIS SCRIPT DOES. It re-runs code/49's comparison under the corrections at
BOTH capacities and reports corrected alongside original. code/49 is not
modified: its trainers (`train_with_scheduler_and_earlystop`,
`train_replay_lr_schedule`), its calibration, its scalers, its ported optimizer
settings and its seeds are all imported, so the harness under test is byte-for-
byte the audited-repo-faithful one and only the CONTROL construction changes.

  PART A -- the directly comparable correction. Same 400-sample pool, same
  splits, same folds, same seeds; only the in-fold carve-out fraction moves,
  over {0.15 (shipped), 0.25 (fold-matched)}. With n_train = 320 and K_CV = 5,
  |val_idx| = 64 and |tr_idx| = 256, so f = 1/(K_CV-1) = 0.25 is the fraction
  that makes |es_idx| = |val_idx|. The f=0.15 arm is asserted per seed against
  code/49's shipped `clean_matched_budget_matched` array, and LEAKY against its
  shipped `leaky_plus_lrsched` array, before anything is reported. Capacity 128
  is the shipped cell; capacity 384 is new and closes the single-capacity gap.

  PART B -- the fully corrected control, on the shared out-of-fold geometry.
  code/77's 2 x 2 x 2 factorial (selection-set size x selection-run budget x
  training depth) ported to this harness. Every arm shares one pool, one fold
  assignment and one LEAKY reference; the out-of-fold carve-out is taken for
  every arm, so the in-fold/out-of-fold factor is not confounded with pool size.

HOW EACH FACTOR IS REALIZED IN THIS HARNESS, which is not the same harness the
other two scripts use. The control here is not a constant-LR fixed-epoch
retrain: it is `train_replay_lr_schedule`, which retrains on the full tr_idx
while REPLAYING the selection run's per-epoch learning rates, so that the
contrast with LEAKY stays "which fold drove the adaptation" rather than
"was there any adaptation." Therefore:

  selection-set size    -- |es_idx| = 0.15*|tr_idx| vs |val_idx|, as elsewhere.
  selection-run budget  -- in-fold: the selection run trains on tr2_idx;
                           out-of-fold: it trains on 100% of tr_idx and its
                           scheduler and early stopping react to a carve-out
                           taken from OUTSIDE the CV pool. In the out-of-fold
                           case the replay retrain reproduces the selection
                           run's own trajectory step for step, so the two come
                           out BITWISE IDENTICAL -- asserted at runtime.
  training depth        -- the replayed trajectory is truncated or extended to
                           e* + delta_e, where delta_e is the single integer
                           equalizing that arm's mean kept epoch with LEAKY's.
                           Extension pads with the last learning rate actually
                           reached, which is what ReduceLROnPlateau would hold
                           at in the absence of further improvement.

  DEPTH MATTERS MORE HERE THAN ELSEWHERE, which is why it is worth the compute:
  code/49's shipped `best_epoch_stats` records LEAKY at a mean kept epoch of
  29.1 against the control's 19.7, a 47.6% difference -- far larger than the
  ~0.7-epoch difference the isotropic synthetic harness shows. If any harness
  in this paper has a training-depth confound, it is this one.

  As in code/77, a CONSTANT depth cannot be a factor level: a replay of a fixed
  trajectory on tr_idx from a fixed seed depends on nothing but the trajectory,
  so all four constant-depth arms would be the same weights. CONST_DEPTH is run
  once, as the "selection of any kind vs none" reference, with LEAKY's own mean
  kept epoch and LEAKY's mean learning-rate trajectory.

Output: results/fidelity_extension_corrected_selection_controls.json
"""
import importlib.util
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import bootstrap, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
OUT_PATH = ROOT / "results" / "fidelity_extension_corrected_selection_controls.json"
REF_49 = ROOT / "results" / "mechanism3_fidelity_extension.json"

_spec = importlib.util.spec_from_file_location(
    "s49", CODE / "49_mechanism3_fidelity_extension.py")
s49 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s49)

CAPACITIES = [128, 384]
SHIPPED_CAPACITY = s49.HIDDEN             # 128
N_SEEDS = s49.N_SEEDS                     # 100
MAX_EPOCHS = s49.MAX_EPOCHS               # 45
N_INNER_FOLDS = s49.N_INNER_FOLDS         # 5
TEST_SIZE = s49.TEST_SIZE                 # 0.20

F_SHIPPED = s49.ES_HOLD_FRACTION          # 0.15
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
    """code/49's exact split / calibrate / RobustScaler path, by import."""
    tr_i, te_i = train_test_split(
        np.arange(len(y)), test_size=TEST_SIZE, stratify=y, random_state=seed)
    Xc = s49.CALIBRATION_FN(X, y, s49.ALPHA_TRAIN_ONLY, tr_i, seed=seed).astype(np.float32)
    scaler = RobustScaler()
    X_train = scaler.fit_transform(Xc[tr_i])
    X_test = scaler.transform(Xc[te_i])
    return X_train, X_test, y[tr_i], y[te_i]


def _fit_and_score(oof, test_feat, rows, y_train, y_test):
    """code/49's readout: the audited pipeline standardizes the deep OOF
    features before the meta-learner (run_pipeline.py:130)."""
    out = {}
    for k in oof:
        tf = test_feat[k] / N_INNER_FOLDS
        ds = StandardScaler()
        oof_k = ds.fit_transform(oof[k][rows])
        test_k = ds.transform(tf)
        clf = LogisticRegression(max_iter=2000).fit(oof_k, y_train[rows])
        out[k] = float(roc_auc_score(y_test, clf.predict_proba(test_k)[:, 1]))
    return out


def _resize_traj(traj, n):
    """Truncate or extend a replayed LR trajectory to exactly n epochs.

    Extension holds the last learning rate actually reached, which is what
    ReduceLROnPlateau does in the absence of further improvement -- it only
    ever reduces, and only on a no-improvement streak the arm has by
    construction already ended its run inside.
    """
    n = max(1, int(n))
    if len(traj) >= n:
        return list(traj[:n])
    return list(traj) + [traj[-1]] * (n - len(traj))


def _identical(m1, m2):
    s1, s2 = m1.state_dict(), m2.state_dict()
    return all(torch.equal(s1[k], s2[k]) for k in s1)


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
        m_l, ep_l, _ = s49.train_with_scheduler_and_earlystop(
            X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_train[val_idx],
            hidden, MAX_EPOCHS, fold_seed)
        oof["leaky"][val_idx] = s49.extract_features(m_l, X_train[val_idx])
        test_feat["leaky"] += s49.extract_features(m_l, X_test)
        n_val.append(len(val_idx)); n_trf.append(len(tr_idx)); leaky_ep.append(ep_l)

        for f in ES_FRACTIONS:
            k = f"cm_{f:.2f}"
            tr2_idx, es_idx = train_test_split(
                tr_idx, test_size=f, stratify=y_train[tr_idx], random_state=fold_seed)
            _, ep, traj = s49.train_with_scheduler_and_earlystop(
                X_train[tr2_idx], y_train[tr2_idx], X_train[es_idx], y_train[es_idx],
                hidden, MAX_EPOCHS, fold_seed)
            m = s49.train_replay_lr_schedule(
                X_train[tr_idx], y_train[tr_idx], hidden, traj[:ep], fold_seed)
            oof[k][val_idx] = s49.extract_features(m, X_train[val_idx])
            test_feat[k] += s49.extract_features(m, X_test)
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
    trajs = {k: [] for k in SEL_KEYS}
    leaky_trajs = []
    sizes = {k: [] for k in SEL_KEYS}
    sel_tr_n = {k: [] for k in SEL_KEYS}
    n_val, n_trf, ident = [], [], []

    for fold, (tr_loc, val_loc) in enumerate(skf.split(X_train[cv_idx], y_train[cv_idx])):
        tr_idx, val_idx = cv_idx[tr_loc], cv_idx[val_loc]
        fold_seed = seed * 100 + fold
        n_val.append(len(val_idx)); n_trf.append(len(tr_idx))

        m_l, ep_l, traj_l = s49.train_with_scheduler_and_earlystop(
            X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_train[val_idx],
            hidden, MAX_EPOCHS, fold_seed)
        oof["leaky"][val_idx] = s49.extract_features(m_l, X_train[val_idx])
        test_feat["leaky"] += s49.extract_features(m_l, X_test)
        epochs["leaky"].append(ep_l)
        leaky_trajs.append(list(traj_l[:ep_l]))

        for key, (sel_tr, sel_es) in _selection_sets(tr_idx, ges_idx, y_train, fold_seed).items():
            m_sel, ep, traj = s49.train_with_scheduler_and_earlystop(
                X_train[sel_tr], y_train[sel_tr], X_train[sel_es], y_train[sel_es],
                hidden, MAX_EPOCHS, fold_seed)
            epochs[key].append(ep); trajs[key].append(list(traj))
            sizes[key].append(len(sel_es)); sel_tr_n[key].append(len(sel_tr))
            m_free = s49.train_replay_lr_schedule(
                X_train[tr_idx], y_train[tr_idx], hidden, traj[:ep], fold_seed)
            if key.endswith("__oof"):
                ident.append(float(_identical(m_sel, m_free)))
            a = f"{key}__free"
            oof[a][val_idx] = s49.extract_features(m_free, X_train[val_idx])
            test_feat[a] += s49.extract_features(m_free, X_test)

        y_perm = rng.permutation(y_train[val_idx])
        m_p, _, _ = s49.train_with_scheduler_and_earlystop(
            X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_perm,
            hidden, MAX_EPOCHS, fold_seed)
        oof["placebo"][val_idx] = s49.extract_features(m_p, X_train[val_idx])
        test_feat["placebo"] += s49.extract_features(m_p, X_test)

    aucs = _fit_and_score(oof, test_feat, cv_idx, y_train, y_test)
    return aucs, {
        "epochs": {k: [int(e) for e in v] for k, v in epochs.items()},
        "trajs": trajs, "leaky_trajs": leaky_trajs,
        "n_val": float(np.mean(n_val)), "n_tr_fold": float(np.mean(n_trf)),
        "n_ges": int(len(ges_idx)), "n_cv": int(len(cv_idx)),
        "sel_sizes": {k: float(np.mean(v)) for k, v in sizes.items()},
        "sel_train_sizes": {k: float(np.mean(v)) for k, v in sel_tr_n.items()},
        "oof_retrain_identical": float(np.mean(ident))}


def partB_pass2(X, y, seed, hidden, stored, shifts, const_traj):
    X_train, X_test, y_train, y_test = _prep(X, y, seed)
    cv_idx, ges_idx, skf = _pool(y_train, seed)
    fdim, n_tr = hidden // 2, len(y_train)
    keys = [f"{k}__matched" for k in SEL_KEYS] + ["const_depth"]
    oof = {k: np.zeros((n_tr, fdim)) for k in keys}
    test_feat = {k: np.zeros((len(y_test), fdim)) for k in keys}
    depths = {k: [] for k in SEL_KEYS}

    for fold, (tr_loc, val_loc) in enumerate(skf.split(X_train[cv_idx], y_train[cv_idx])):
        tr_idx, val_idx = cv_idx[tr_loc], cv_idx[val_loc]
        fold_seed = seed * 100 + fold
        m_c = s49.train_replay_lr_schedule(
            X_train[tr_idx], y_train[tr_idx], hidden, const_traj, fold_seed)
        oof["const_depth"][val_idx] = s49.extract_features(m_c, X_train[val_idx])
        test_feat["const_depth"] += s49.extract_features(m_c, X_test)
        for key in SEL_KEYS:
            ep = max(1, stored["epochs"][key][fold] + shifts[key])
            depths[key].append(ep)
            traj = _resize_traj(stored["trajs"][key][fold], ep)
            m = s49.train_replay_lr_schedule(
                X_train[tr_idx], y_train[tr_idx], hidden, traj, fold_seed)
            a = f"{key}__matched"
            oof[a][val_idx] = s49.extract_features(m, X_train[val_idx])
            test_feat[a] += s49.extract_features(m, X_test)
    return (_fit_and_score(oof, test_feat, cv_idx, y_train, y_test),
            {k: float(np.mean(v)) for k, v in depths.items()})


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
    X, y = s49.load_raw_real_features()
    ref = json.load(open(REF_49))

    out = {
        "config": {
            "n_samples": int(len(y)), "capacities": CAPACITIES,
            "shipped_capacity": SHIPPED_CAPACITY, "n_seeds": N_SEEDS,
            "max_epochs": MAX_EPOCHS, "n_inner_folds": N_INNER_FOLDS,
            "test_size": TEST_SIZE, "lr_patience": s49.LR_PATIENCE,
            "es_patience": s49.ES_PATIENCE,
            "calibration_alpha": s49.ALPHA_TRAIN_ONLY,
            "es_fraction_shipped": F_SHIPPED,
            "es_fraction_fold_matched": F_FOLD,
            "oof_hold_fraction": OOF_HOLD_FRACTION,
            "what_was_wrong": (
                "code/49:212 sets ES_HOLD_FRACTION=0.15 and code/49:541 uses it in the "
                "same in-fold carve-out construction as code/02d and code/43, so the "
                "fidelity extension's +0.0250 carries the identical selection-set-size "
                "and selection-run-budget asymmetries. It also existed at one capacity "
                "only, while supplying the upper bound of the paper's "
                "mechanism-and-harness axis. Both are addressed here."),
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

        reproduces = False
        if hidden == SHIPPED_CAPACITY:
            for local, ref_key in (("leaky", "leaky_plus_lrsched"),
                                   (f"cm_{F_SHIPPED:.2f}", "clean_matched_budget_matched")):
                shipped = np.array(ref["raw_per_seed"][ref_key])
                drift = float(np.max(np.abs(arrs[local] - shipped)))
                assert drift < 1e-12, (
                    f"recomputed {local} drifted from code/49's shipped per-seed "
                    f"{ref_key} by {drift:.2e}")
            reproduces = True

        cell = {"reproduces_code49_per_seed": reproduces,
                "is_shipped_capacity": hidden == SHIPPED_CAPACITY,
                "n_val": metas[0]["n_val"], "n_tr_fold": metas[0]["n_tr_fold"],
                "leaky_mean": float(arrs["leaky"].mean()),
                "leaky_best_epoch_mean": float(np.mean([m["leaky_ep"] for m in metas])),
                "by_es_fraction": {}}
        for i, f in enumerate(ES_FRACTIONS):
            k = f"cm_{f:.2f}"
            g = gap_stats(arrs["leaky"], arrs[k], boot_seed=hidden * 7919 + 71 + i)
            cell["by_es_fraction"][f"{f:.2f}"] = {
                **g,
                "control_mean": float(arrs[k].mean()),
                "n_selection_points": metas[0][k]["n_es"],
                "n_selection_run_training": metas[0][k]["n_tr2"],
                "control_best_epoch_mean": float(np.mean([m[k]["ep"] for m in metas])),
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
            f"out-of-fold replay retrains not bitwise identical to their selection "
            f"runs at capacity {hidden} (frac={oof_ident})")

        mean_ep = {k: float(np.mean([e for m in metas for e in m["epochs"][k]]))
                   for k in ["leaky"] + SEL_KEYS}
        shifts = {k: int(round(mean_ep["leaky"] - mean_ep[k])) for k in SEL_KEYS}
        const_n = int(round(mean_ep["leaky"]))
        # CONST_DEPTH replays LEAKY's own mean trajectory shape: the per-epoch
        # median LR across every (seed, fold) LEAKY run, resized to const_n.
        max_len = max(len(t) for m in metas for t in m["leaky_trajs"])
        med = []
        for e in range(max_len):
            vals = [t[e] for m in metas for t in m["leaky_trajs"] if len(t) > e]
            med.append(float(np.median(vals)))
        const_traj = _resize_traj(med, const_n)

        ni = 0
        for seed in range(N_SEEDS):
            aucs, dep = partB_pass2(X, y, seed, hidden, metas[seed], shifts, const_traj)
            for k, v in aucs.items():
                cols.setdefault(k, []).append(v)
            ni += 1
            if (seed + 1) % 25 == 0:
                print(f"  [B2 cap={hidden}] {seed+1}/{N_SEEDS} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
        for k in list(cols):
            arrs[k] = np.array(cols[k])

        cells = {}
        for i, arm in enumerate(ARMS):
            size, loc, depth = arm.split("__")
            sk = f"{size}__{loc}"
            g = gap_stats(arrs["leaky"], arrs[arm], boot_seed=hidden * 104729 + 17 * i)
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
            "oof_replay_bitwise_identical_frac": oof_ident,
            "leaky_mean_auroc": float(arrs["leaky"].mean()),
            "placebo_mean_auroc": float(arrs["placebo"].mean()),
            "const_depth_epochs": const_n,
            "const_depth_mean_auroc": float(arrs["const_depth"].mean()),
            "gap_leaky_minus_const_depth": gap_stats(
                arrs["leaky"], arrs["const_depth"], boot_seed=hidden * 104729 + 881),
            "gap_leaky_minus_placebo": gap_stats(
                arrs["leaky"], arrs["placebo"], boot_seed=hidden * 104729 + 883),
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
    out["shipped_reference"] = {
        "code49_leaky_plus_lrsched_minus_clean_matched_budget_matched":
            ref["leaky_plus_lrsched_minus_clean_matched_budget_matched"]["gap_mean"],
        "code49_capacity": SHIPPED_CAPACITY,
        "note": ("The shipped +0.0250 is Part A's ES=0.15 cell at capacity 128 and is "
                 "asserted to reproduce per seed."),
    }

    out["limitations"] = [
        "Part A keeps code/49's full 320-sample training pool; Part B removes "
        "1/(K+1) of it for the out-of-fold selection set, for every arm equally. "
        "Part B gaps are internally matched but its absolute AUROCs are not "
        "comparable to Part A's or to code/49's shipped table.",
        "The depth-matched arms resize a replayed LR trajectory. Truncation is "
        "exact; extension holds the last learning rate reached, which is what "
        "ReduceLROnPlateau does absent further improvement but is still an "
        "assumption about a counterfactual the selection run never ran.",
        "CONST_DEPTH replays the per-epoch median of LEAKY's own trajectories "
        "rather than any single run's, so it is a synthetic schedule; it is a "
        "reference arm for 'selection of any kind vs none', not a control.",
        "code/49's deliberately-not-ported list (EMA, the composite loss, "
        "pos_weight, label smoothing, mixup/cutmix, the transformer model class) "
        "is unchanged here. This is the same partial-fidelity harness, with only "
        "the control construction corrected.",
        "The depth shift is a single integer per arm estimated from the same 500 "
        "(seed, fold) pairs it is applied to, and equalizes MEAN depth only.",
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
