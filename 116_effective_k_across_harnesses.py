"""
Does the candidate redundancy of 115 generalise beyond one harness?

115 measures K_eff = 1.67 of a nominal 32 for CV-argmax layer selection on
Mistral-7B TruthfulQA hidden states. If that is a property of one harness it
is a curiosity; if it holds across model families, datasets and feature
representations it is a property of layer selection, and every correction
calibrated to nominal K is mis-calibrated wherever layer selection is used.

This script recomputes the same quantities on the real-feature harness for two
model families on a different dataset, so the comparison spans:

    model family        Mistral-7B and Qwen2.5-7B
    dataset             TruthfulQA and HaluEval
    representation      4096-d raw hidden states and 12-d summary statistics

Same estimator throughout: per-layer 5-fold CV AUROC over randomized
stratified selection pools, then the correlation matrix across layers and the
participation ratio of its eigenvalues.

Reuses shipped feature caches. No refitting of anything in the main results.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "results" / "effective_k_across_harnesses.json"
N_REPS, N_SEL, SEED = 40, 229, 42

HARNESSES = [
    ("mistral-7b / halueval / real features",
     "results/real_features_mistral7b_halueval.npz"),
    ("qwen2.5-7b / halueval / real features",
     "results/real_features_qwen2.5_7b_halueval.npz"),
]


def effective_k(C):
    w = np.linalg.eigvalsh(C)
    w = w[w > 1e-12]
    return float((w.sum() ** 2) / (w ** 2).sum())


def per_layer_cv_auroc(X, y, rng, cv):
    n_layers = X.shape[1]
    M = np.empty((N_REPS, n_layers))
    for r in range(N_REPS):
        sel = rng.permutation(len(y))[:N_SEL]
        Xs, ys = X[sel], y[sel]
        for l in range(n_layers):
            A = Xs[:, l, :]
            aucs = []
            for tr, te in cv.split(A, ys):
                sc = StandardScaler().fit(A[tr])
                clf = LogisticRegression(max_iter=1000).fit(sc.transform(A[tr]), ys[tr])
                aucs.append(roc_auc_score(
                    ys[te], clf.predict_proba(sc.transform(A[te]))[:, 1]))
            M[r, l] = float(np.mean(aucs))
    return M


def main():
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    out = {"n_reps": N_REPS, "n_sel": N_SEL, "seed": SEED, "harnesses": {}}

    # carried over from 115 so all three sit in one table
    out["harnesses"]["mistral-7b / truthfulqa / hidden states"] = {
        "nominal_K": 32, "mean_offdiagonal_correlation": 0.755,
        "mean_adjacent_correlation": 0.957, "effective_K": 1.67,
        "source": "115_candidate_dependence.py",
    }

    for name, rel in HARNESSES:
        d = np.load(ROOT / rel)
        X, y = d["X_seq"].astype(float), d["y"].astype(int)
        M = per_layer_cv_auroc(X, y, np.random.default_rng(SEED), cv)
        C = np.corrcoef(M.T)
        nL = X.shape[1]
        off = C[~np.eye(nL, dtype=bool)]
        adj = np.array([C[i, i + 1] for i in range(nL - 1)])
        out["harnesses"][name] = {
            "nominal_K": nL,
            "mean_offdiagonal_correlation": float(off.mean()),
            "mean_adjacent_correlation": float(adj.mean()),
            "effective_K": effective_k(C),
            "source": "116_effective_k_across_harnesses.py",
        }

    ks = [v["effective_K"] for v in out["harnesses"].values()]
    out["summary"] = {
        "effective_K_range": [min(ks), max(ks)],
        "nominal_K": 32,
        "overstatement_factor_range": [32 / max(ks), 32 / min(ks)],
        "all_far_below_nominal": bool(max(ks) < 32 / 4),
    }

    print("=" * 84)
    print("EFFECTIVE CANDIDATE COUNT ACROSS HARNESSES")
    print("=" * 84)
    print(f"\n{'harness':<42}{'K':>4}{'rho_bar':>9}{'adj':>8}{'K_eff':>8}")
    print("-" * 71)
    for k, v in out["harnesses"].items():
        print(f"{k:<42}{v['nominal_K']:>4}{v['mean_offdiagonal_correlation']:>9.3f}"
              f"{v['mean_adjacent_correlation']:>8.3f}{v['effective_K']:>8.2f}")
    print("-" * 71)
    s = out["summary"]
    print(f"\nK_eff spans {s['effective_K_range'][0]:.2f}-{s['effective_K_range'][1]:.2f} "
          f"against a nominal {s['nominal_K']}:")
    print(f"  the candidate count is overstated by "
          f"{s['overstatement_factor_range'][0]:.1f}x to "
          f"{s['overstatement_factor_range'][1]:.1f}x")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
