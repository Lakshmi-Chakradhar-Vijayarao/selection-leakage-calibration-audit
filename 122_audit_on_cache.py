"""
The Mechanism-2 audit, run on any hidden-state cache.

    python3 122_audit_on_cache.py <cache.npz> <out_name> [n_reps] [n_procs]

The cache needs `hidden_states` (n, 32, d) and `labels` (n,), 1 = hallucinated.

Why it exists. The paper's Mechanism-2 numbers come from one cache (GUARDIAN's
Mistral-7B TruthfulQA, 700 items) through code/48 -> 95 -> 97 -> 112. Two other
caches now need the identical audit:

  - GUARDIAN's Mistral-7B HaluEval cache, behind a replication the paper
    reports ("+0.0112 and +0.0200") whose producing script is not in the
    artifact;
  - 25 independent 700-item TriviaQA samples (the population-level
    replication), produced by the Kaggle kernel in
    kaggle_runs/triviaqa-independent-samples/.

Nothing is re-implemented. Probe fitting, the fold loop and the split
generator are imported from code/48 unchanged (unscaled logistic, C = 1.0,
StratifiedKFold(5, shuffle, 42), default_rng(42) split stream, 400 selection
/ the rest held out); calibration metrics from 103; the selective-prediction
threshold transfer from 97. Run on the TruthfulQA cache this reproduces the
paper's Mechanism-2 numbers, which is how the script is checked.

Output: results/audit_<out_name>.npz with, per re-split and per layer, the
selection-fold and held-out value of AUROC, Brier, reliability and ECE, the
selected layer, and the risk violation at 30/50/70% coverage -- everything
needed for the gap, A, B and the risk violation, with no refitting.
"""
import os
# One BLAS thread per worker: the work is parallelised across re-splits, and
# letting each worker's linear algebra spawn its own threads oversubscribes the
# CPU many times over (load average ~190 on 8 cores, observed).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import importlib.util
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cs2 = _load(ROOT / "code" / "48_case_study_2_layer_decomposition.py", "cs2_decomp")
rr = _load(ROOT / "103_reviewer_response_mechanism2.py", "reviewer_response")
sp = _load(ROOT / "97_selective_prediction_consequence.py", "selective_prediction")
METRICS = ("auroc", "brier", "reliability", "ece")
COVERAGES = sp.COVERAGE_TARGETS            # [0.30, 0.50, 0.70]

_H = _y = None                             # set per worker


def _init(cache):
    global _H, _y
    d = np.load(cache)
    _H = d["hidden_states"].astype(np.float32)
    _y = d["labels"].astype(int)


def _calib(s, y):
    rel, _, _ = rr.murphy_decomposition(s, y, 10, "width")
    return float(np.mean((s - y) ** 2)), rel, rr.ece_binned(s, y, 10, "width")


def one_rep(args):
    r, sel_idx, ho_idx = args
    L = _H.shape[1]
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=cs2.RANDOM_STATE)
    y_sel, y_ho = _y[sel_idx], _y[ho_idx]
    cs2.ARTIFACT.clear()
    per_layer = cs2.per_layer_gaps(_H[sel_idx], y_sel, _H[ho_idx], y_ho, L, kfold,
                                   variant="v")
    cvs, hos = cs2.ARTIFACT["v__cv_scores"].astype(float), cs2.ARTIFACT["v__ho_scores"].astype(float)
    cv = np.empty((len(METRICS), L)); ho = np.empty((len(METRICS), L))
    for l in range(L):
        cv[0, l] = per_layer[l]["cv_auroc_selectpool"]
        ho[0, l] = per_layer[l]["heldout_auroc"]
        cv[1:, l] = _calib(cvs[l], y_sel)
        ho[1:, l] = _calib(hos[l], y_ho)
    l_star = int(cv[0].argmax())
    viol = []
    for c in COVERAGES:
        tau, _, believed = sp.risk_at_coverage(cvs[l_star], y_sel, c)
        _, actual = sp.apply_threshold(hos[l_star], y_ho, tau)
        viol.append(actual - believed)
    return r, cv, ho, l_star, np.array(viol)


def main():
    cache, name = Path(sys.argv[1]), sys.argv[2]
    n_reps = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    n_procs = int(sys.argv[4]) if len(sys.argv) > 4 else max(1, os.cpu_count() - 1)
    y = np.load(cache)["labels"].astype(int)
    # the split stream is drawn up front, in order, from code/48's generator,
    # so rep r is the same partition however the work is scheduled
    rng = np.random.default_rng(cs2.RANDOM_STATE)
    splits = [(r, *cs2.random_stratified_split(rng, y, cs2.N_TRAIN_SELECT)) for r in range(n_reps)]
    with Pool(n_procs, initializer=_init, initargs=(cache,)) as pool:
        res = sorted(pool.map(one_rep, splits), key=lambda t: t[0])
    CV = np.stack([t[1] for t in res]); HO = np.stack([t[2] for t in res])
    ls = np.array([t[3] for t in res]); viol = np.stack([t[4] for t in res])
    out = ROOT / "results" / f"audit_{name}.npz"
    np.savez_compressed(out, cv=CV, ho=HO, l_star=ls, risk_violation=viol,
                        metrics=np.array(METRICS), coverages=np.array(COVERAGES),
                        n_items=len(y), hall_rate=y.mean(), n_sel=cs2.N_TRAIN_SELECT)
    rows = np.arange(n_reps)
    print(f"{name}: n={len(y)} hall_rate={y.mean():.3f} reps={n_reps}")
    for k, m in enumerate(METRICS):
        gap = (HO[:, k][rows, ls] - CV[:, k][rows, ls]) if m != "auroc" else (CV[:, k][rows, ls] - HO[:, k][rows, ls])
        print(f"  {m:<12} reuse gap at l* (positive = optimistic) {gap.mean():+.5f}")
    for c, v in zip(COVERAGES, viol.T):
        print(f"  risk violation @{int(c*100)}%  {v.mean():+.5f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
