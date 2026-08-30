"""
UncertaiNLP addition: Holm-Bonferroni correction across the 10 new
tests from code/95, code/96 and code/97 (calibration bridge + selective-
prediction consequence), matching this paper's existing family-wise-
correction discipline (Table~\\ref{tab:holm} in the main submission).

For each test, a two-sided paired-permutation p-value is computed on the
per-rep gap (sign-flip permutation on paired differences, assumption-free,
identical method to code/44's paired_permutation_test), since the existing
calibration/selective-prediction scripts report BCa intervals but not
p-values directly.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def paired_permutation_p(diff, n_perm=100_000, seed=20260811):
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff)
    observed = diff.mean()
    signs = rng.choice([-1, 1], size=(n_perm, len(diff)))
    perm_means = (signs * diff[None, :]).mean(axis=1)
    return float((np.abs(perm_means) >= abs(observed)).mean())


def holm_bonferroni(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj_p = (m - rank) * pvals[idx]
        running_max = max(running_max, adj_p)
        adj[idx] = min(running_max, 1.0)
    return adj


def recompute_diffs_mechanism2():
    from sklearn.metrics import roc_auc_score, brier_score_loss
    a = np.load(ROOT / "results" / "case_study_2_probe_scores.npz")
    n_layers, n_reps = int(a["n_layers"]), int(a["n_reps"])
    ref = json.load(open(ROOT / "results" / "case_study_2_layer_decomposition.json"))["randomized_stratified_splits"]

    def ece_binned(scores, labels, n_bins=10):
        scores, labels = np.asarray(scores, float), np.asarray(labels, float)
        edges = np.linspace(0, 1, n_bins + 1)
        ece, n = 0.0, len(scores)
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            mask = (scores >= lo) & (scores <= hi) if i == n_bins - 1 else (scores >= lo) & (scores < hi)
            if mask.sum() == 0:
                continue
            ece += (mask.sum() / n) * abs(labels[mask].mean() - scores[mask].mean())
        return ece

    def risk_at_coverage(scores, labels, cov):
        conf = np.abs(scores - 0.5) * 2.0
        pred = (scores >= 0.5).astype(int)
        correct = (pred == labels).astype(int)
        n = len(scores)
        k = max(1, int(round(cov * n)))
        order_ = np.argsort(-conf)
        tau = conf[order_[k - 1]]
        kept = conf >= tau
        risk = 1.0 - correct[kept].mean() if kept.sum() else float("nan")
        return tau, risk

    def apply_threshold(scores, labels, tau):
        conf = np.abs(scores - 0.5) * 2.0
        pred = (scores >= 0.5).astype(int)
        correct = (pred == labels).astype(int)
        kept = conf >= tau
        return 1.0 - correct[kept].mean() if kept.sum() else float("nan")

    brier_gap, ece_gap = [], []
    sp_gap = {c: [] for c in (0.30, 0.50, 0.70)}
    for r in range(n_reps):
        variant = f"rand{r}"
        y_sel, y_ho = a[f"{variant}__y_sel"], a[f"{variant}__y_ho"]
        fold_id, cv_scores, ho_scores = a[f"{variant}__fold_id"], a[f"{variant}__cv_scores"], a[f"{variant}__ho_scores"]
        folds = sorted(set(fold_id.tolist()))
        cv_auroc = np.array([np.mean([roc_auc_score(y_sel[fold_id == f], cv_scores[l][fold_id == f]) for f in folds])
                              for l in range(n_layers)])
        l_star = int(cv_auroc.argmax())
        assert l_star == ref["selected_layer_per_rep"][r]
        s_leaky, s_clean = cv_scores[l_star], ho_scores[l_star]
        brier_gap.append(brier_score_loss(y_ho, s_clean) - brier_score_loss(y_sel, s_leaky))
        ece_gap.append(ece_binned(s_clean, y_ho) - ece_binned(s_leaky, y_sel))
        for cov in sp_gap:
            tau, believed = risk_at_coverage(s_leaky, y_sel, cov)
            actual = apply_threshold(s_clean, y_ho, tau)
            sp_gap[cov].append(actual - believed)
    return brier_gap, ece_gap, sp_gap


def recompute_diffs_mechanism1():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    d = np.load(ROOT / "results" / "real_features_mistral7b_halueval.npz")
    X_seq, X_glob, y = d["X_seq"], d["X_glob"], d["y"]
    X = np.hstack([X_seq.reshape(X_seq.shape[0], -1), X_glob]).astype(np.float64)

    def ece_binned(scores, labels, n_bins=10):
        scores, labels = np.asarray(scores, float), np.asarray(labels, float)
        edges = np.linspace(0, 1, n_bins + 1)
        ece, n = 0.0, len(scores)
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            mask = (scores >= lo) & (scores <= hi) if i == n_bins - 1 else (scores >= lo) & (scores < hi)
            if mask.sum() == 0:
                continue
            ece += (mask.sum() / n) * abs(labels[mask].mean() - scores[mask].mean())
        return ece

    def risk_at_coverage(scores, labels, cov):
        conf = np.abs(scores - 0.5) * 2.0
        pred = (scores >= 0.5).astype(int)
        correct = (pred == labels).astype(int)
        n = len(scores)
        k = max(1, int(round(cov * n)))
        order_ = np.argsort(-conf)
        tau = conf[order_[k - 1]]
        kept = conf >= tau
        risk = 1.0 - correct[kept].mean() if kept.sum() else float("nan")
        return tau, risk

    def apply_threshold(scores, labels, tau):
        conf = np.abs(scores - 0.5) * 2.0
        pred = (scores >= 0.5).astype(int)
        correct = (pred == labels).astype(int)
        kept = conf >= tau
        return 1.0 - correct[kept].mean() if kept.sum() else float("nan")

    brier_gap, ece_gap = [], []
    sp_gap = {c: [] for c in (0.30, 0.50, 0.70)}
    for seed in range(40):
        Xs = StandardScaler().fit_transform(X)
        clf_full = LogisticRegression(max_iter=500, C=1.0).fit(Xs, y)
        p_leaky = clf_full.predict_proba(Xs)[:, 1]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        p_oof = np.zeros(len(y))
        for tr_idx, te_idx in skf.split(X, y):
            scaler = StandardScaler().fit(X[tr_idx])
            Xtr, Xte = scaler.transform(X[tr_idx]), scaler.transform(X[te_idx])
            clf = LogisticRegression(max_iter=500, C=1.0).fit(Xtr, y[tr_idx])
            p_oof[te_idx] = clf.predict_proba(Xte)[:, 1]
        brier_gap.append(brier_score_loss(y, p_oof) - brier_score_loss(y, p_leaky))
        ece_gap.append(ece_binned(p_oof, y) - ece_binned(p_leaky, y))
        for cov in sp_gap:
            tau, believed = risk_at_coverage(p_leaky, y, cov)
            actual = apply_threshold(p_oof, y, tau)
            sp_gap[cov].append(actual - believed)
    return brier_gap, ece_gap, sp_gap


def main():
    b2, e2, sp2 = recompute_diffs_mechanism2()
    b1, e1, sp1 = recompute_diffs_mechanism1()

    tests = {
        "M2_brier": b2, "M2_ece": e2,
        "M2_selpred_cov30": sp2[0.30], "M2_selpred_cov50": sp2[0.50], "M2_selpred_cov70": sp2[0.70],
        "M1_brier": b1, "M1_ece": e1,
        "M1_selpred_cov30": sp1[0.30], "M1_selpred_cov50": sp1[0.50], "M1_selpred_cov70": sp1[0.70],
    }
    names = list(tests.keys())
    pvals = np.array([paired_permutation_p(tests[n]) for n in names])
    adj = holm_bonferroni(pvals)

    out = {}
    for n, p, a in zip(names, pvals, adj):
        out[n] = {"mean_gap": float(np.mean(tests[n])), "p_uncorrected": float(p),
                   "p_holm_adjusted": float(a), "survives_holm_at_0.05": bool(a < 0.05)}
        print(f"{n:22s} mean_gap={np.mean(tests[n]):+.5f}  p={p:.5f}  p_holm={a:.5f}  "
              f"survives={a < 0.05}")

    n_survive = sum(1 for v in out.values() if v["survives_holm_at_0.05"])
    out["_summary"] = {"n_tests": len(names), "n_survive_holm_at_0.05": n_survive}
    print(f"\n{n_survive}/{len(names)} of the new findings survive Holm-Bonferroni correction at alpha=0.05")

    with open(ROOT / "results" / "holm_correction_new_findings.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {ROOT}/results/holm_correction_new_findings.json")


if __name__ == "__main__":
    main()
