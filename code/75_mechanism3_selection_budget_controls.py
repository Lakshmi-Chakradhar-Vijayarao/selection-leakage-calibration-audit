"""
Mechanism 3 has TWO budget asymmetries inside its honest-selection
control, and only the first has ever been sized. This script sizes the second
and asks whether the primary synthetic effect survives its removal.

WHY THIS SCRIPT EXISTS. Appendix A.3 and SS4.3 name two ways CLEAN_MATCHED is
disadvantaged relative to LEAKY, and code/67 measured one of them:

  (1) THE RETRAIN-BUDGET DEFICIT. CLEAN_MATCHED's blind fixed-epoch retrain
      runs on the full tr_idx, but the epoch count it replays was chosen on a
      run that saw only 85% of it (tr2_idx). code/67 isolated this with
      CLEAN_MATCHED_85 and put it at +0.0015 (capacity 128, CI
      [-0.0003,+0.0035], p=0.29) and +0.0031 (capacity 384, CI
      [+0.0004,+0.0057], p=0.005). Those numbers are LOADED here from
      results/adaptivity_control_budget_deficit.json and reported for context.
      Nothing below re-derives them.

  (2) THE SELECTION-RUN BUDGET ASYMMETRY -- never sized at all, by anyone, in
      any round. In code/47 (and code/02d, the bit-identical primary harness)
      LEAKY's SELECTION run trains on 100% of tr_idx and argmaxes validation
      AUROC on the 112-sample val_idx it later reports features on.
      CLEAN_MATCHED's SELECTION run trains on only tr2_idx (85% of tr_idx) and
      argmaxes on the disjoint 68-sample es_idx carved out of tr_idx. So the
      two arms' selection runs do not see the same amount of data and do not
      argmax over the same number of selection points. A selection run trained
      on less data has a different -- generally later, and noisier -- argmax,
      and that biases LEAKY minus CLEAN_MATCHED independently of fold reuse,
      which is the thing the comparison is supposed to isolate. Appendix A.3
      states the confound and explicitly flags the fix as not run:
      "a cleaner variant would size the carve-out to match the fold (25% of
      448=112) at the cost of a larger budget deficit; we did not run it, and
      flag it as an open robustness gap."

THIS SCRIPT CLOSES THAT GAP TWO WAYS.

PART A -- ES_HOLD_FRACTION robustness (in-fold carve-out, resized).
Re-runs the primary cell at ES_HOLD_FRACTION in {0.15 (shipped), 0.20, 0.25},
all four capacities, same seeds/folds/initialization. LEAKY does not depend on
ES_HOLD_FRACTION at all -- the constant enters only through the tr2_idx/es_idx
split -- so LEAKY is computed once and reused, and the recomputed
ES=0.15 arm is asserted per seed against code/02d's shipped values before
anything is reported, exactly as code/67 asserts against code/22.

  THE EXACT FOLD-MATCHED FRACTION IS 0.25, NOT 0.20, and the reason matters.
  ES_HOLD_FRACTION is a fraction of tr_idx, not of the training pool. With
  n_train=560 and N_INNER_FOLDS=5: |val_idx| = 560/5 = 112 and |tr_idx| = 448,
  so |es_idx| = |val_idx| requires f = 112/448 = 1/(N_INNER_FOLDS - 1) = 0.25.
  1/N_INNER_FOLDS = 0.20 is the fraction of the POOL that one fold occupies,
  but applied to tr_idx it gives |es_idx| = 90, not 112. Measured directly
  below rather than asserted. Both 0.20 and 0.25 are run: 0.25 because it is
  the exact match A.3 named, 0.20 because it is the value a reader would
  reach for by analogy with 1/K and because an intermediate point shows
  whether the trend in ES_HOLD_FRACTION is smooth or a single-cell artifact.

  What Part A does NOT do: it cannot remove asymmetry (2), only trade it. A
  larger carve-out matches the selection-SET size but makes the selection run's
  training deficit worse (tr2_idx falls 380 -> 358 -> 336). The two halves of
  the asymmetry are coupled inside one fold and cannot both be fixed there.
  That is why Part B goes outside the fold.

PART B -- CLEAN_MATCHED_OOF, whose selection run also sees 100% of the data.
The carve-out is drawn from OUTSIDE tr_idx, so nothing has to be taken out of
the selection run's training set. Construction, precisely:

    X_train (560)  --stratified-->  ges_idx (94)   +   cv_idx (466)
    StratifiedKFold(5) over cv_idx only  ->  per fold  |val_idx| ~ 93,
                                                        |tr_idx| ~ 373

    LEAKY_R        = train_to_best(tr_idx, select on val_idx)      [reused fold]
    CLEAN_MATCHED_OOF
                   = train_to_best(tr_idx, select on ges_idx) -> e*
                     then train_fixed(tr_idx, e*)                  [blind retrain]
    CLEAN_MATCHED_INFOLD_R
                   = the SHIPPED scheme (ES_HOLD_FRACTION=0.15 inside tr_idx),
                     replicated on this same reduced pool
    PLACEBO_R      = select on permuted val_idx labels

  The hold-out fraction is 1/(N_INNER_FOLDS + 1) = 1/6, which is not a round
  number chosen for looks: holding aside g and cross-validating the remaining
  n-g into K folds gives |val| = (n-g)/K, and setting g = (n-g)/K gives
  g = n/(K+1) = 93.3. So the out-of-fold selection set and each validation fold
  come out the same size (94 vs 93), which matches the selection-SET size too,
  not just the selection-run training budget.

  CLEAN_MATCHED_OOF therefore has NEITHER asymmetry: its selection run trains
  on 100% of tr_idx, identical to LEAKY_R's, and argmaxes over a selection set
  the same size as LEAKY_R's. Because the selection run and the retrain now
  train on identical data from an identical seed, the blind retrain reproduces
  the selection run's trajectory step for step and the two models come out
  BITWISE IDENTICAL -- asserted at runtime via state_dicts_identical, which is
  a stronger statement than "budget matched": the retrain-budget asymmetry is
  removed by construction, not merely made small. The retrain is kept anyway
  so the arm is structurally the same object CLEAN_MATCHED is.

  WHAT IT COSTS, stated plainly. The 94 selection samples have to come from
  somewhere, and there is nowhere inside the fold left to take them from:
  tr_idx and val_idx partition the whole CV pool, so "the other folds' training
  data" IS tr_idx. They are therefore removed from the cross-validation pool
  entirely -- they train no fold model, and they contribute no row to the OOF
  matrix the meta-learner is fit on -- for BOTH arms equally. The contrast is
  internally matched and the reported test set (140 samples) is untouched, but
  the whole Part B comparison runs at a 16.7% smaller effective training pool
  than the primary estimate. Absolute AUROCs are therefore not comparable to
  Part A's; only gaps within Part B are, and the primary-vs-Part-B comparison
  is a comparison of gaps at slightly different pool sizes. Sweep B and Sweep D
  in code/47 disagree about the sign of the pool-size effect on the gap
  (+0.0026 at N=350 vs +0.0036 at 700 vs +0.0005 at 2800; smaller n_val gives
  a smaller gap), so we do not claim a direction for this residual.

  THE WITHIN-RUN ISOLATION. CLEAN_MATCHED_OOF minus CLEAN_MATCHED_INFOLD_R,
  computed on the same pool, same folds, same seeds, against the same LEAKY_R,
  is the size of asymmetry (2) itself. Honest caveat: that difference moves two
  coupled things at once -- the selection run's training budget (100% vs 85% of
  tr_idx) and the selection set's provenance and size (94 out-of-fold vs 56
  in-fold). Both are constituents of the selection-run asymmetry and neither
  can be moved alone inside this design; Part A's ES sweep is what separates
  the selection-set-size axis.

PRE-REGISTERED VERDICT RULE (fixed before the run, evaluated after):
  primary_mean = mean over the four capacities of the shipped
                 LEAKY - CLEAN_MATCHED gap (+0.0019/+0.0011/+0.0036/+0.0024).
  oof_mean     = mean over the same four capacities of LEAKY_R - CLEAN_MATCHED_OOF.
  ratio        = oof_mean / primary_mean.
  SURVIVES         if ratio >= 0.50 and all four gaps > 0 and >= 2 of 4 BCa
                      CIs exclude zero from below.
  ATTENUATED       if oof_mean > 0 and ratio >= 0.25 and not SURVIVES.
  DOES_NOT_SURVIVE otherwise.
The rule's output is reported alongside the raw numbers, never instead of them.
If the effect does not survive, that is the finding and the paper reports it.

Everything -- generator, SweepMLP, train_to_best_checkpoint, train_fixed_epochs,
the decoupled data/split/fold/init seed scheme, the capacities, EPOCHS=45,
N_SAMPLES=700, TARGET_AUROC=0.80 -- is imported from code/47 rather than
copied, so the operating point is identical to the primary estimate's by
construction.

Output: results/mechanism3_selection_budget_controls.json
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
OUT_PATH = ROOT / "results" / "mechanism3_selection_budget_controls.json"
REF_PRIMARY = ROOT / "results" / "corrected_capacity_placebo_sweep.json"
REF_BUDGET = ROOT / "results" / "adaptivity_control_budget_deficit.json"

_spec = importlib.util.spec_from_file_location(
    "s47", CODE / "47_selection_multiplicity_sweep.py")
s47 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s47)

CAPACITIES = [16, 48, 128, 384]
N_SEEDS = 100
EPOCHS = s47.DEFAULT_EPOCHS
N_SAMPLES = s47.DEFAULT_N_SAMPLES
TARGET_AUROC = s47.DEFAULT_TARGET_AUROC
N_INNER_FOLDS = s47.N_INNER_FOLDS
TEST_SIZE = s47.TEST_SIZE

ES_FRACTIONS = [0.15, 0.20, 0.25]
ES_SHIPPED = s47.ES_HOLD_FRACTION
# Fold-matched carve-out: |es_idx| = f * |tr_idx| = f * n*(K-1)/K must equal
# |val_idx| = n/K, hence f = 1/(K-1). Verified against measured sizes below.
ES_FOLD_MATCHED = 1.0 / (N_INNER_FOLDS - 1)
# Part B: hold aside g = n/(K+1) so the out-of-fold selection set and each
# validation fold are the same size.
OOF_HOLD_FRACTION = 1.0 / (N_INNER_FOLDS + 1)


def _fit_and_score(oof, test_feat, rows, y_train, y_test, n_folds):
    """Downstream meta-learner, identical to code/47's, restricted to `rows`
    (Part B fills only the CV pool's rows; Part A fills all of them)."""
    out = {}
    for k in oof:
        tf = test_feat[k] / n_folds
        clf = LogisticRegression(max_iter=2000).fit(oof[k][rows], y_train[rows])
        out[k] = float(roc_auc_score(y_test, clf.predict_proba(tf)[:, 1]))
    return out


def _prep(data_seed, split_seed):
    X, y = s47.make_synthetic_data(data_seed, N_SAMPLES, TARGET_AUROC)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=split_seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# PART A: in-fold carve-out at several ES_HOLD_FRACTION values
# ---------------------------------------------------------------------------
def run_one_seed_partA(data_seed, split_seed, fold_seed_base, init_seed_base, hidden):
    """LEAKY (computed once) plus one CLEAN_MATCHED arm per ES fraction.

    LEAKY is invariant to ES_HOLD_FRACTION: the constant enters only through
    the tr2_idx/es_idx split, which LEAKY never touches. CLEAN and PLACEBO are
    not computed -- they are separate downstream fits and cannot affect these
    arms' values (torch's RNG is reseeded at every train call, and the placebo
    label permutation draws from its own numpy stream). The ES=0.15 arm is
    asserted per seed against code/02d's shipped values by the caller, which is
    what proves the omission is harmless rather than assumed to be.
    """
    X_train, X_test, y_train, y_test = _prep(data_seed, split_seed)
    skf = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=fold_seed_base)
    fdim = hidden // 2
    n_tr = len(y_train)
    arms = ["leaky"] + [f"cm_{f:.2f}" for f in ES_FRACTIONS]
    oof = {k: np.zeros((n_tr, fdim)) for k in arms}
    test_feat = {k: np.zeros((len(y_test), fdim)) for k in arms}
    geom = {f"cm_{f:.2f}": {"n_es": [], "n_tr2": [], "best_epoch": []} for f in ES_FRACTIONS}
    n_val, n_tr_fold, leaky_epochs = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        init_seed = init_seed_base * 100 + fold
        m_leaky, leaky_ep, _ = s47.train_to_best_checkpoint(
            X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_train[val_idx],
            hidden, EPOCHS, init_seed)
        oof["leaky"][val_idx] = s47.extract_features(m_leaky, X_train[val_idx])
        test_feat["leaky"] += s47.extract_features(m_leaky, X_test)
        n_val.append(len(val_idx)); n_tr_fold.append(len(tr_idx))
        leaky_epochs.append(leaky_ep)

        for f in ES_FRACTIONS:
            key = f"cm_{f:.2f}"
            tr2_idx, es_idx = train_test_split(
                tr_idx, test_size=f, stratify=y_train[tr_idx], random_state=init_seed)
            _, best_epoch, _ = s47.train_to_best_checkpoint(
                X_train[tr2_idx], y_train[tr2_idx], X_train[es_idx], y_train[es_idx],
                hidden, EPOCHS, init_seed)
            m_cm = s47.train_fixed_epochs(X_train[tr_idx], y_train[tr_idx],
                                          hidden, best_epoch, init_seed)
            oof[key][val_idx] = s47.extract_features(m_cm, X_train[val_idx])
            test_feat[key] += s47.extract_features(m_cm, X_test)
            geom[key]["n_es"].append(len(es_idx))
            geom[key]["n_tr2"].append(len(tr2_idx))
            geom[key]["best_epoch"].append(best_epoch)

    aucs = _fit_and_score(oof, test_feat, np.arange(n_tr), y_train, y_test, N_INNER_FOLDS)
    meta = {"n_val": float(np.mean(n_val)), "n_tr_fold": float(np.mean(n_tr_fold)),
            "leaky_best_epoch": float(np.mean(leaky_epochs))}
    for key, g in geom.items():
        meta[key] = {kk: float(np.mean(vv)) for kk, vv in g.items()}
    return aucs, meta


# ---------------------------------------------------------------------------
# PART B: out-of-fold carve-out -- the selection run also sees 100% of tr_idx
# ---------------------------------------------------------------------------
def run_one_seed_partB(data_seed, split_seed, fold_seed_base, init_seed_base, hidden):
    X_train, X_test, y_train, y_test = _prep(data_seed, split_seed)
    n_tr = len(y_train)
    # The out-of-fold selection pool, carved out of X_train BEFORE folding, so
    # it is disjoint from every fold's tr_idx AND from every fold's val_idx.
    # Its own stream (split_seed + 500000) collides with nothing else here.
    cv_idx, ges_idx = train_test_split(
        np.arange(n_tr), test_size=OOF_HOLD_FRACTION, stratify=y_train,
        random_state=split_seed + 500000)
    cv_idx = np.sort(cv_idx); ges_idx = np.sort(ges_idx)

    rng = np.random.default_rng(fold_seed_base + 10000)
    skf = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=fold_seed_base)
    fdim = hidden // 2
    arms = ["leaky_r", "cm_oof", "cm_infold_r", "placebo_r"]
    oof = {k: np.zeros((n_tr, fdim)) for k in arms}
    test_feat = {k: np.zeros((len(y_test), fdim)) for k in arms}
    sizes = {"n_ges": len(ges_idx), "n_cv": len(cv_idx), "n_val": [], "n_tr_fold": [],
             "n_es_infold": [], "n_tr2_infold": []}
    epochs = {"leaky_r": [], "cm_oof": [], "cm_infold_r": []}
    retrain_identical = []

    for fold, (tr_loc, val_loc) in enumerate(skf.split(X_train[cv_idx], y_train[cv_idx])):
        tr_idx, val_idx = cv_idx[tr_loc], cv_idx[val_loc]
        init_seed = init_seed_base * 100 + fold
        sizes["n_val"].append(len(val_idx)); sizes["n_tr_fold"].append(len(tr_idx))

        # LEAKY_R -- selects on the fold it reports on. 100% of tr_idx.
        m_leaky, ep_l, _ = s47.train_to_best_checkpoint(
            X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_train[val_idx],
            hidden, EPOCHS, init_seed)
        oof["leaky_r"][val_idx] = s47.extract_features(m_leaky, X_train[val_idx])
        test_feat["leaky_r"] += s47.extract_features(m_leaky, X_test)
        epochs["leaky_r"].append(ep_l)

        # CLEAN_MATCHED_OOF -- selects on ges_idx, which is outside tr_idx and
        # outside val_idx, so the selection run keeps 100% of tr_idx.
        m_sel, ep_o, _ = s47.train_to_best_checkpoint(
            X_train[tr_idx], y_train[tr_idx], X_train[ges_idx], y_train[ges_idx],
            hidden, EPOCHS, init_seed)
        m_oof = s47.train_fixed_epochs(X_train[tr_idx], y_train[tr_idx],
                                       hidden, ep_o, init_seed)
        retrain_identical.append(float(s47.state_dicts_identical(m_sel, m_oof)))
        oof["cm_oof"][val_idx] = s47.extract_features(m_oof, X_train[val_idx])
        test_feat["cm_oof"] += s47.extract_features(m_oof, X_test)
        epochs["cm_oof"].append(ep_o)

        # CLEAN_MATCHED_INFOLD_R -- the shipped scheme, same reduced pool.
        tr2_idx, es_idx = train_test_split(
            tr_idx, test_size=ES_SHIPPED, stratify=y_train[tr_idx], random_state=init_seed)
        _, ep_i, _ = s47.train_to_best_checkpoint(
            X_train[tr2_idx], y_train[tr2_idx], X_train[es_idx], y_train[es_idx],
            hidden, EPOCHS, init_seed)
        m_inf = s47.train_fixed_epochs(X_train[tr_idx], y_train[tr_idx],
                                       hidden, ep_i, init_seed)
        oof["cm_infold_r"][val_idx] = s47.extract_features(m_inf, X_train[val_idx])
        test_feat["cm_infold_r"] += s47.extract_features(m_inf, X_test)
        epochs["cm_infold_r"].append(ep_i)
        sizes["n_es_infold"].append(len(es_idx)); sizes["n_tr2_infold"].append(len(tr2_idx))

        # PLACEBO_R -- zero selection signal, real training signal.
        y_perm = rng.permutation(y_train[val_idx])
        m_pl, _, _ = s47.train_to_best_checkpoint(
            X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_perm,
            hidden, EPOCHS, init_seed)
        oof["placebo_r"][val_idx] = s47.extract_features(m_pl, X_train[val_idx])
        test_feat["placebo_r"] += s47.extract_features(m_pl, X_test)

    # The meta-learner is fit only on rows the CV actually produced features
    # for; ges_idx rows are structurally absent from the OOF matrix.
    aucs = _fit_and_score(oof, test_feat, cv_idx, y_train, y_test, N_INNER_FOLDS)
    meta = {
        "n_ges": sizes["n_ges"], "n_cv": sizes["n_cv"],
        "n_val": float(np.mean(sizes["n_val"])),
        "n_tr_fold": float(np.mean(sizes["n_tr_fold"])),
        "n_es_infold": float(np.mean(sizes["n_es_infold"])),
        "n_tr2_infold": float(np.mean(sizes["n_tr2_infold"])),
        "best_epoch_leaky_r": float(np.mean(epochs["leaky_r"])),
        "best_epoch_cm_oof": float(np.mean(epochs["cm_oof"])),
        "best_epoch_cm_infold_r": float(np.mean(epochs["cm_infold_r"])),
        "retrain_bitwise_identical_frac": float(np.mean(retrain_identical)),
    }
    return aucs, meta


def gap_stats(a, b, boot_seed):
    """Paired gap a - b with a BCa CI and a Wilcoxon p, seeded per cell so the
    interval does not depend on the order cells were computed in."""
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
    primary = json.load(open(REF_PRIMARY))["by_capacity"]
    budget = json.load(open(REF_BUDGET))["by_capacity"]

    out = {
        "config": {
            "n_seeds": N_SEEDS, "capacities": CAPACITIES, "epochs": EPOCHS,
            "n_samples": N_SAMPLES, "target_auroc": TARGET_AUROC,
            "n_inner_folds": N_INNER_FOLDS, "test_size": TEST_SIZE,
            "es_fractions_run": ES_FRACTIONS,
            "es_shipped": ES_SHIPPED,
            "es_fold_matched_exact": ES_FOLD_MATCHED,
            "oof_hold_fraction": OOF_HOLD_FRACTION,
            "n_seeds_note": (
                "N_SEEDS=100, matching code/47 and code/02d exactly; no reduction "
                "was needed."),
            "operating_point_note": (
                "EPOCHS/N_SAMPLES/TARGET_AUROC/N_INNER_FOLDS/TEST_SIZE and the "
                "decoupled data/split/fold/init seed scheme are imported from "
                "code/47, whose default cell is bit-identical to code/02d's "
                "capacity-128 primary cell. Capacities are code/02d's full "
                "{16,48,128,384} grid, so Part A is apples-to-apples with the "
                "primary table and not merely with its flagship cell."),
        },
        "asymmetry_1_retrain_budget_deficit_LOADED_NOT_REDERIVED": {
            "source": "results/adaptivity_control_budget_deficit.json (code/67)",
            "what_it_measures": (
                "CLEAN_MATCHED's blind fixed-epoch retrain uses the full tr_idx but "
                "replays an epoch count chosen on a run that saw only 85% of it. "
                "CLEAN_MATCHED minus CLEAN_MATCHED_85 isolates that training-set-size "
                "difference. Reported here for context only; not recomputed."),
            "by_capacity": {
                h: {"budget_deficit_mean": budget[h]["budget_deficit_mean"],
                    "budget_deficit_bca_ci_95": budget[h]["budget_deficit_bca_ci_95"],
                    "budget_deficit_wilcoxon_p": budget[h]["budget_deficit_wilcoxon_p"]}
                for h in budget},
        },
        "part_A_es_hold_fraction": {"by_capacity": {}},
        "part_B_out_of_fold_carveout": {"by_capacity": {}},
    }

    # ---------------- PART A ----------------
    print("=== PART A: ES_HOLD_FRACTION in {0.15, 0.20, 0.25}, in-fold carve-out ===",
          flush=True)
    partA_raw = {}
    for hidden in CAPACITIES:
        cols = {k: [] for k in ["leaky"] + [f"cm_{f:.2f}" for f in ES_FRACTIONS]}
        metas = []
        for seed in range(N_SEEDS):
            aucs, meta = run_one_seed_partA(seed, seed + 100000, seed + 200000,
                                            seed + 300000, hidden)
            for k in cols:
                cols[k].append(aucs[k])
            metas.append(meta)
            if (seed + 1) % 25 == 0:
                print(f"  [A cap={hidden}] {seed+1}/{N_SEEDS} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
        arrs = {k: np.array(v) for k, v in cols.items()}

        # Assert against the shipped primary harness BEFORE reporting anything.
        for k_local, k_ref in [("leaky", "leaky"), (f"cm_{ES_SHIPPED:.2f}", "clean_matched")]:
            shipped = np.array(primary[str(hidden)]["aucs"][k_ref])[:len(arrs[k_local])]
            drift = float(np.max(np.abs(arrs[k_local] - shipped)))
            assert drift < 1e-12, (
                f"recomputed {k_local} drifted from code/02d's shipped per-seed "
                f"{k_ref} by {drift:.2e} at capacity {hidden}")

        cell = {"reproduces_code02d_per_seed": True,
                "n_val": metas[0]["n_val"], "n_tr_fold": metas[0]["n_tr_fold"],
                "leaky_mean": float(arrs["leaky"].mean()),
                "leaky_best_epoch_mean": float(np.mean([m["leaky_best_epoch"] for m in metas])),
                "by_es_fraction": {}}
        for f in ES_FRACTIONS:
            key = f"cm_{f:.2f}"
            # Deterministic per-cell bootstrap seed (NOT hash(), which is
            # PYTHONHASHSEED-randomized for strings and would make the CIs
            # irreproducible across processes).
            g = gap_stats(arrs["leaky"], arrs[key],
                          boot_seed=hidden * 7919 + int(round(f * 100)))
            cell["by_es_fraction"][f"{f:.2f}"] = {
                **g,
                "clean_matched_mean": float(arrs[key].mean()),
                "n_es": metas[0][key]["n_es"],
                "n_tr2_selection_run": metas[0][key]["n_tr2"],
                "selection_set_size_ratio_leaky_over_clean":
                    float(metas[0]["n_val"] / metas[0][key]["n_es"]),
                "clean_best_epoch_mean": float(np.mean([m[key]["best_epoch"] for m in metas])),
                "is_shipped": abs(f - ES_SHIPPED) < 1e-9,
                "is_fold_matched": abs(f - ES_FOLD_MATCHED) < 1e-9,
            }
            print(f"  cap={hidden} ES={f:.2f} (|es|={metas[0][key]['n_es']:.0f}, "
                  f"|tr2|={metas[0][key]['n_tr2']:.0f}): gap={g['gap_mean']:+.4f} "
                  f"CI=[{g['gap_bca_ci_95'][0]:+.4f},{g['gap_bca_ci_95'][1]:+.4f}] "
                  f"p={g['wilcoxon_p']:.4g}", flush=True)
        out["part_A_es_hold_fraction"]["by_capacity"][str(hidden)] = cell
        partA_raw[hidden] = arrs

    out["part_A_es_hold_fraction"]["geometry"] = {
        "n_train": int(N_SAMPLES * (1 - TEST_SIZE)),
        "n_val_per_fold": out["part_A_es_hold_fraction"]["by_capacity"]["16"]["n_val"],
        "n_tr_idx_per_fold": out["part_A_es_hold_fraction"]["by_capacity"]["16"]["n_tr_fold"],
        "exact_fold_matched_es_fraction": ES_FOLD_MATCHED,
        "derivation": (
            "ES_HOLD_FRACTION is a fraction of tr_idx, not of the training pool. "
            "|tr_idx| = n_train*(K-1)/K and |val_idx| = n_train/K, so |es_idx| = "
            "|val_idx| requires f = 1/(K-1). With K=N_INNER_FOLDS=5 that is exactly "
            "0.25, and the measured sizes confirm it: f=0.25 gives |es_idx|=112 = "
            "|val_idx|=112. 1/K = 0.20 is NOT the fold-matched value -- applied to "
            "tr_idx it gives |es_idx|=90 against a 112-sample fold. Both were run: "
            "0.25 because it is the exact match Appendix A.3 named as unrun, 0.20 "
            "because it is the value the 1/K analogy suggests and because an "
            "intermediate point distinguishes a smooth trend from a single-cell "
            "artifact."),
        "what_this_cannot_do": (
            "Raising ES_HOLD_FRACTION matches the selection-SET size but worsens the "
            "selection RUN's training deficit (|tr2_idx| falls 380 -> 358 -> 336 as f "
            "goes 0.15 -> 0.20 -> 0.25). Inside one fold the two halves of the "
            "selection-run asymmetry are coupled and cannot both be fixed, which is "
            "exactly the trade Appendix A.3 described. Part B leaves the fold."),
    }

    # ---------------- PART B ----------------
    print("\n=== PART B: out-of-fold carve-out (selection run keeps 100% of tr_idx) ===",
          flush=True)
    partB_raw = {}
    for hidden in CAPACITIES:
        arms = ["leaky_r", "cm_oof", "cm_infold_r", "placebo_r"]
        cols = {k: [] for k in arms}
        metas = []
        for seed in range(N_SEEDS):
            aucs, meta = run_one_seed_partB(seed, seed + 100000, seed + 200000,
                                            seed + 300000, hidden)
            for k in cols:
                cols[k].append(aucs[k])
            metas.append(meta)
            if (seed + 1) % 25 == 0:
                print(f"  [B cap={hidden}] {seed+1}/{N_SEEDS} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
        arrs = {k: np.array(v) for k, v in cols.items()}
        ident = float(np.mean([m["retrain_bitwise_identical_frac"] for m in metas]))
        assert ident == 1.0, (
            f"CLEAN_MATCHED_OOF's blind retrain was NOT bitwise identical to its "
            f"selection run at capacity {hidden} (frac={ident}); the arm's "
            f"zero-retrain-budget-asymmetry claim would not hold")

        g_oof = gap_stats(arrs["leaky_r"], arrs["cm_oof"], boot_seed=hidden * 7919 + 1)
        g_inf = gap_stats(arrs["leaky_r"], arrs["cm_infold_r"], boot_seed=hidden * 7919 + 2)
        g_asym = gap_stats(arrs["cm_oof"], arrs["cm_infold_r"], boot_seed=hidden * 7919 + 3)
        g_pl_oof = gap_stats(arrs["cm_oof"], arrs["placebo_r"], boot_seed=hidden * 7919 + 4)

        out["part_B_out_of_fold_carveout"]["by_capacity"][str(hidden)] = {
            "n_ges_out_of_fold_selection_set": metas[0]["n_ges"],
            "n_cv_pool": metas[0]["n_cv"],
            "n_val_per_fold": metas[0]["n_val"],
            "n_tr_idx_per_fold": metas[0]["n_tr_fold"],
            "n_es_infold_reference_arm": metas[0]["n_es_infold"],
            "n_tr2_infold_reference_arm": metas[0]["n_tr2_infold"],
            "retrain_bitwise_identical_frac": ident,
            "leaky_r_mean": float(arrs["leaky_r"].mean()),
            "cm_oof_mean": float(arrs["cm_oof"].mean()),
            "cm_infold_r_mean": float(arrs["cm_infold_r"].mean()),
            "placebo_r_mean": float(arrs["placebo_r"].mean()),
            "best_epoch_leaky_r": float(np.mean([m["best_epoch_leaky_r"] for m in metas])),
            "best_epoch_cm_oof": float(np.mean([m["best_epoch_cm_oof"] for m in metas])),
            "best_epoch_cm_infold_r": float(np.mean([m["best_epoch_cm_infold_r"] for m in metas])),
            "gap_leaky_minus_cm_oof": g_oof,
            "gap_leaky_minus_cm_infold_r": g_inf,
            "selection_run_asymmetry_cm_oof_minus_cm_infold_r": g_asym,
            "gap_cm_oof_minus_placebo_r": g_pl_oof,
        }
        print(f"  cap={hidden}: LEAKY_R-CM_OOF={g_oof['gap_mean']:+.4f} "
              f"CI=[{g_oof['gap_bca_ci_95'][0]:+.4f},{g_oof['gap_bca_ci_95'][1]:+.4f}] "
              f"p={g_oof['wilcoxon_p']:.4g} | LEAKY_R-CM_INFOLD={g_inf['gap_mean']:+.4f} "
              f"| asym(2)={g_asym['gap_mean']:+.4f} "
              f"| CM_OOF-PLACEBO={g_pl_oof['gap_mean']:+.4f}", flush=True)
        partB_raw[hidden] = arrs

    out["part_B_out_of_fold_carveout"]["construction"] = (
        "X_train (560) is split, stratified and before any folding, into an out-of-fold "
        "selection pool ges_idx and a cross-validation pool cv_idx, with "
        "|ges_idx| = n_train/(K+1) so that ges_idx and each subsequent validation fold "
        "are the same size (94 vs 93). StratifiedKFold(K=5) then runs over cv_idx only, "
        "so ges_idx is disjoint from every fold's tr_idx AND from every fold's val_idx. "
        "CLEAN_MATCHED_OOF's selection run trains on 100% of tr_idx and argmaxes on "
        "ges_idx; its blind fixed-epoch retrain then runs on the same 100% of tr_idx "
        "from the same seed, and is asserted at runtime to be bitwise identical to the "
        "selection run's kept checkpoint -- so the retrain-budget asymmetry is removed "
        "by construction rather than made small. LEAKY_R is the identical procedure "
        "selecting on the reported fold. CLEAN_MATCHED_INFOLD_R replicates the shipped "
        "ES_HOLD_FRACTION=0.15 in-fold scheme on the same reduced pool as the "
        "within-run reference.")
    out["part_B_out_of_fold_carveout"]["what_it_costs"] = (
        "The selection points have to come from somewhere and there is nowhere inside "
        "the fold left to take them from: tr_idx and val_idx partition the entire CV "
        "pool, so 'the other folds' training data' IS tr_idx. The 94 selection samples "
        "are therefore removed from the cross-validation pool outright -- they train no "
        "fold model and contribute no row to the OOF matrix the meta-learner is fit on "
        "(560 rows become 466) -- for every arm equally. The held-out test set (140) is "
        "untouched. So gaps WITHIN Part B are matched, but Part B as a whole runs at a "
        "16.7% smaller effective training pool than the primary estimate, and its "
        "absolute AUROCs are not comparable to Part A's. code/47's sweeps do not agree "
        "on the sign of the pool-size effect on the gap (Sweep B: +0.0026/+0.0036/"
        "+0.0005 at N=350/700/2800; Sweep D: smaller n_val gives a SMALLER gap), so no "
        "direction is claimed for this residual.")
    out["part_B_out_of_fold_carveout"]["coupling_caveat"] = (
        "CLEAN_MATCHED_OOF minus CLEAN_MATCHED_INFOLD_R sizes asymmetry (2), but it "
        "moves two coupled things at once: the selection run's training budget (100% vs "
        "85% of tr_idx) and the selection set's provenance and size (94 out-of-fold vs "
        "~56 in-fold). Both are constituents of the selection-run asymmetry and neither "
        "can be moved alone inside this design. Part A's ES sweep is what varies the "
        "selection-set-size axis on its own terms.")

    # ---------------- VERDICT ----------------
    primary_gaps = [primary[str(h)]["gaps"]["leaky_minus_clean_matched"]["mean"]
                    for h in CAPACITIES]
    oof_gaps = [out["part_B_out_of_fold_carveout"]["by_capacity"][str(h)]
                ["gap_leaky_minus_cm_oof"]["gap_mean"] for h in CAPACITIES]
    oof_ci_low = [out["part_B_out_of_fold_carveout"]["by_capacity"][str(h)]
                  ["gap_leaky_minus_cm_oof"]["gap_bca_ci_95"][0] for h in CAPACITIES]
    es25_gaps = [out["part_A_es_hold_fraction"]["by_capacity"][str(h)]
                 ["by_es_fraction"]["0.25"]["gap_mean"] for h in CAPACITIES]
    asym2 = [out["part_B_out_of_fold_carveout"]["by_capacity"][str(h)]
             ["selection_run_asymmetry_cm_oof_minus_cm_infold_r"]["gap_mean"]
             for h in CAPACITIES]

    primary_mean = float(np.mean(primary_gaps))
    oof_mean = float(np.mean(oof_gaps))
    ratio = float(oof_mean / primary_mean)
    n_ci_excl = int(sum(1 for lo in oof_ci_low if lo > 0))
    all_pos = bool(all(g > 0 for g in oof_gaps))

    if ratio >= 0.50 and all_pos and n_ci_excl >= 2:
        verdict = "SURVIVES"
    elif oof_mean > 0 and ratio >= 0.25:
        verdict = "ATTENUATED"
    else:
        verdict = "DOES_NOT_SURVIVE"

    out["summary"] = {
        "primary_gaps_by_capacity": {str(h): g for h, g in zip(CAPACITIES, primary_gaps)},
        "es_0.25_fold_matched_gaps_by_capacity": {str(h): g for h, g in zip(CAPACITIES, es25_gaps)},
        "out_of_fold_gaps_by_capacity": {str(h): g for h, g in zip(CAPACITIES, oof_gaps)},
        "asymmetry_2_size_by_capacity": {str(h): g for h, g in zip(CAPACITIES, asym2)},
        "primary_mean_gap": primary_mean,
        "out_of_fold_mean_gap": oof_mean,
        "ratio_oof_over_primary": ratio,
        "n_capacities_with_positive_gap": int(sum(1 for g in oof_gaps if g > 0)),
        "n_capacities_with_bca_ci_excluding_zero_from_below": n_ci_excl,
        "prereg_rule": (
            "SURVIVES if ratio >= 0.50 and all four out-of-fold gaps > 0 and >= 2 of 4 "
            "BCa CIs exclude zero from below; ATTENUATED if the mean gap is positive "
            "and ratio >= 0.25 but the SURVIVES conditions fail; DOES_NOT_SURVIVE "
            "otherwise. Fixed before the run."),
    }
    out["verdict"] = verdict
    out["runtime_seconds"] = time.time() - t0
    # REPO-RELATIVE, never absolute: an absolute path here leaks the author's
    # username into a double-blind submission, which code/91's identity audit
    # correctly refused to package.
    out["outputs"] = {"json": str(OUT_PATH.relative_to(ROOT)),
                      "script": str(Path(__file__).resolve().relative_to(ROOT))}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)  # written before `statement` so a crash below loses nothing

    # Plain-language statement, assembled from the numbers just computed.
    band = (f"{min(oof_gaps):+.4f} to {max(oof_gaps):+.4f}")
    out["statement"] = (
        "Mechanism 3's primary synthetic estimate contains two budget asymmetries "
        "inside its honest-selection control, not one. The first -- CLEAN_MATCHED "
        "retrains on the full fold but replays an epoch count chosen on 85% of it -- "
        "was already sized by code/67 at +0.0015 (capacity 128) and +0.0031 (384). "
        "The second, sized here for the first time, is that the two arms' SELECTION "
        "runs are trained on different amounts of data at all: LEAKY's sees 100% of "
        "tr_idx and argmaxes on the 112-sample fold it later reports on, while "
        "CLEAN_MATCHED's sees 85% and argmaxes on a 68-sample carve-out taken out of "
        "its own training set. Two controls were run at the primary estimate's exact "
        "operating point and all four of its capacities. (A) Resizing the in-fold "
        "carve-out to the exactly fold-matched 25% -- the variant Appendix A.3 named "
        "and did not run; 1/(K-1)=0.25, not 1/K=0.20, is the value that makes "
        "|es_idx| equal |val_idx| -- gives gaps of "
        + ", ".join(f"{g:+.4f}" for g in es25_gaps) +
        " at capacities 16/48/128/384 against the shipped "
        + ", ".join(f"{g:+.4f}" for g in primary_gaps) +
        ". (B) Moving the carve-out OUTSIDE the fold, so the honest arm's selection "
        "run trains on 100% of tr_idx and argmaxes on an out-of-fold set the same "
        "size as LEAKY's validation fold, removes the second asymmetry entirely (and, "
        "as a by-product, the first: the blind retrain comes out bitwise identical to "
        "the selection run). Against that arm the fold-reuse gap is " + band +
        f" across capacities, mean {oof_mean:+.4f} against the primary mean "
        f"{primary_mean:+.4f} -- a ratio of {ratio:.2f}, with "
        f"{n_ci_excl} of 4 BCa intervals excluding zero. Under the pre-registered "
        f"rule the verdict is {verdict}. The cost of arm (B) is that its "
        "out-of-fold selection samples are removed from the cross-validation pool "
        "outright, so the whole Part B comparison runs on a 16.7% smaller training "
        "pool; that is matched across arms but makes its absolute AUROCs "
        "incomparable to the primary table's.")

    out["limitations"] = [
        "Part B's out-of-fold selection set is removed from the cross-validation pool "
        "entirely -- there is nowhere else for it to come from, because tr_idx and "
        "val_idx already partition that pool -- so every Part B arm trains on a 16.7% "
        "smaller pool than the primary estimate. The contrast is internally matched, "
        "but Part B gaps and primary gaps are gaps measured at different pool sizes, "
        "and code/47's own sweeps disagree on the sign of the pool-size effect.",
        "Part B's single out-of-fold selection set is REUSED across all five folds. It "
        "is disjoint from every val_idx, so it cannot leak into the reported metric, "
        "but the five folds' checkpoint choices are correlated through it in a way the "
        "shipped in-fold scheme's five independent carve-outs are not.",
        "CLEAN_MATCHED_OOF minus CLEAN_MATCHED_INFOLD_R moves the selection run's "
        "training budget and the selection set's size/provenance together. It bounds "
        "the selection-run asymmetry as a whole; it does not decompose it.",
        "Part A cannot remove asymmetry (2), only trade its two halves against each "
        "other: a fold-matched 25% carve-out costs the selection run 25% of its "
        "training data instead of 15%. Its value is as a robustness check on the "
        "shipped constant, not as a fix.",
        "Everything here is the isotropic-Gaussian synthetic reconstruction, at "
        "AUROC_0=0.80, EPOCHS=45, N_SAMPLES=700, d=64. Operating point is the single "
        "strongest severity modifier code/47 measures (+0.0093 at 0.70 down to +0.0002 "
        "at 0.985), so these conclusions are conditional on 0.80 in exactly the way the "
        "primary estimate is. Nothing here is a measurement on MultiHaluDet's real "
        "7B-scale pipeline.",
        "The verdict rule's thresholds (0.50 and 0.25 of the primary mean) are "
        "pre-registered but not derived. The raw per-capacity numbers are reported so a "
        "reader can apply a different rule.",
        "The downstream feature averaging that code/55 showed suppresses this effect "
        "2.3-5.8x is left in place here, because the point is comparability with the "
        "primary estimate. Both Part A and Part B therefore inherit that suppression.",
    ]

    out["runtime_seconds"] = time.time() - t0
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 78)
    print(f"VERDICT: {verdict}")
    print(f"  primary  (in-fold ES=0.15): " + " ".join(f"{g:+.4f}" for g in primary_gaps)
          + f"   mean {primary_mean:+.4f}")
    print(f"  ES=0.25  (fold-matched)   : " + " ".join(f"{g:+.4f}" for g in es25_gaps))
    print(f"  out-of-fold carve-out     : " + " ".join(f"{g:+.4f}" for g in oof_gaps)
          + f"   mean {oof_mean:+.4f}  (ratio {ratio:.2f})")
    print(f"  asymmetry (2) size        : " + " ".join(f"{g:+.4f}" for g in asym2))
    print(f"  BCa CIs excluding zero: {n_ci_excl}/4;  all gaps positive: {all_pos}")
    print("=" * 78)
    print(f"\nSaved: {OUT_PATH}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
