"""
OPTIONAL P0-3 stretch measurement. Mechanism 1 (full-dataset probe
fit, reused as a downstream feature -- HaRP's own bug, §4.1) has no
code-verified, matched-control measurement anywhere else in this paper: unlike
Mechanisms 2-5, no synthetic harness reconstructs it under the same BCa
methodology used everywhere else. This is a first, self-contained attempt,
built to the same recipe as the rest of §5's synthetic harnesses (isotropic
binormal features calibrated to a target Bayes-optimal AUROC via
AUROC_0 = Phi(Delta/sqrt(2))), CPU-only, n=100 seeds.

DESIGN, mirroring HaRP's own diagnostic (§4.1's "probe_conf alone: in-sample
AUROC 1.0000 vs out-of-fold AUROC 0.7573"):

  1. Generate n_samples isotropic-Gaussian features (dim d) with a class-mean
     separation calibrated so that a Bayes-optimal linear rule achieves
     AUROC_0 (0.70 / 0.80 / 0.95). d is deliberately large relative to
     n_samples (matching the paper's own framing: "few samples and many
     dimensions"), which is what makes a full-dataset-fit probe overfit
     severely in-sample.

  2. LEAKY: fit one LogisticRegression probe on ALL n_samples (X, y), then
     score those same n_samples -- the "probe_conf" feature -- and report its
     in-sample AUROC directly (a monotone transform of a single score doesn't
     change AUROC, so no separate downstream classifier is needed to
     reproduce HaRP's own single-feature diagnostic).

  3. CLEAN_MATCHED: an out-of-fold reconstruction -- 5-fold CV, refitting the
     identical probe on each fold's own training split only, scoring only that
     fold's held-out rows, assembling one out-of-fold score vector covering
     every sample exactly once, and reporting ITS AUROC. This is the same
     per-fold-refit convention the rest of this paper calls CLEAN_MATCHED.

  4. gap = LEAKY - CLEAN_MATCHED, per seed; BCa 95% CI via the same
     scipy.stats.bootstrap(method="BCa") convention as code/77 and code/78.

WHAT THIS DOES NOT CLAIM. This is one synthetic reconstruction, not a
recomputation of HaRP's own +0.19 figure (§3 already states that number is not
re-verifiable from this artifact), and it does not port HaRP's real
architecture, its A+B+C feature composition, or its 4096-dimensional hidden
states. It measures whether the general MECHANISM -- full-dataset probe fit
reused as a downstream feature -- produces severities comparable in scale to
the abstract's four Mechanism 2-4 point estimates, or larger. We report
whatever this harness measures, in either direction.

Output: results/mechanism1_severity_probe.json
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap, norm, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "mechanism1_severity_probe.json"

N_SAMPLES = 200
DIM = 60
N_FOLDS = 5
N_SEEDS = 100
AUROC0_GRID = [0.70, 0.80, 0.95]


def make_data(seed, n_samples, dim, auroc0):
    rng = np.random.default_rng(seed)
    delta = np.sqrt(2) * norm.ppf(auroc0)  # Bayes-optimal binormal identity
    y = rng.integers(0, 2, size=n_samples)
    mu = np.zeros(dim)
    mu[0] = delta  # signal lives on one axis; the rest are pure noise
    X = rng.standard_normal((n_samples, dim))
    X[y == 1] += mu
    return X, y


def leaky_and_clean(X, y, fold_seed):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # LEAKY: fit on everything, score everything.
    clf_full = LogisticRegression(max_iter=2000).fit(Xs, y)
    s_leaky = clf_full.decision_function(Xs)
    leaky_auroc = roc_auc_score(y, s_leaky)

    # CLEAN_MATCHED: out-of-fold, refitting per fold.
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=fold_seed)
    s_oof = np.zeros(len(y))
    for tr_idx, te_idx in skf.split(Xs, y):
        clf = LogisticRegression(max_iter=2000).fit(Xs[tr_idx], y[tr_idx])
        s_oof[te_idx] = clf.decision_function(Xs[te_idx])
    clean_auroc = roc_auc_score(y, s_oof)
    return leaky_auroc, clean_auroc


def gap_stats(leaky, clean, boot_seed):
    d = np.asarray(leaky) - np.asarray(clean)
    _, p = wilcoxon(leaky, clean)
    res = bootstrap((d,), np.mean, confidence_level=0.95, n_resamples=10000,
                    method="BCa", random_state=np.random.default_rng(boot_seed))
    return {
        "leaky_mean": float(np.mean(leaky)), "clean_matched_mean": float(np.mean(clean)),
        "gap_mean": float(d.mean()),
        "gap_bca_ci_95": [float(res.confidence_interval.low),
                          float(res.confidence_interval.high)],
        "wilcoxon_p": float(p), "gap_std": float(d.std(ddof=1)),
        "n_positive": int((d > 0).sum()), "n_seeds": int(len(d)),
    }


def main():
    t0 = time.time()
    results = {}
    for auroc0 in AUROC0_GRID:
        leaky_arr, clean_arr = [], []
        for seed in range(N_SEEDS):
            X, y = make_data(seed, N_SAMPLES, DIM, auroc0)
            l, c = leaky_and_clean(X, y, fold_seed=seed + 500000)
            leaky_arr.append(l)
            clean_arr.append(c)
        g = gap_stats(leaky_arr, clean_arr, boot_seed=104729 + int(auroc0 * 1000))
        g["auroc0_target"] = auroc0
        established = bool(g["gap_mean"] > 0 and g["gap_bca_ci_95"][0] > 0)
        g["verdict"] = "ESTABLISHED" if established else "NOT_ESTABLISHED"
        results[str(auroc0)] = g
        print(f"AUROC_0={auroc0}: LEAKY={g['leaky_mean']:.4f} "
              f"CLEAN_MATCHED={g['clean_matched_mean']:.4f} "
              f"gap={g['gap_mean']:+.4f} BCa=[{g['gap_bca_ci_95'][0]:+.4f},"
              f"{g['gap_bca_ci_95'][1]:+.4f}] p={g['wilcoxon_p']:.3g} "
              f"[{g['verdict']}]", flush=True)

    gaps = [results[str(a)]["gap_mean"] for a in AUROC0_GRID]
    out = {
        "config": {"n_samples": N_SAMPLES, "dim": DIM, "n_folds": N_FOLDS,
                  "n_seeds": N_SEEDS, "auroc0_grid": AUROC0_GRID},
        "by_auroc0": results,
        "range_over_grid": [float(min(gaps)), float(max(gaps))],
        "comparison_to_mechanisms_2_4": (
            "Mechanisms 2-4's four code-verified point estimates span "
            "+0.0011 to +0.0255 AUROC (Table tab:severity-summary). This "
            f"harness's Mechanism-1 gaps span {min(gaps):+.4f} to "
            f"{max(gaps):+.4f} over the same three-operating-point design "
            "used elsewhere in §5."),
        "runtime_seconds": time.time() - t0,
        "outputs": {"json": str(OUT_PATH.relative_to(ROOT)),
                    "script": "code/86_mechanism1_severity_probe.py"},
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{out['comparison_to_mechanisms_2_4']}")
    print(f"\nSaved: {OUT_PATH}  ({(time.time()-t0):.0f}s)")


if __name__ == "__main__":
    main()
