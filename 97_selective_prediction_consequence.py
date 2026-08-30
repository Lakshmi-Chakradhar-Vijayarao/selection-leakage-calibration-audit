"""
UncertaiNLP downstream-consequence bridge: does selection-induced
optimism break an actual selective-prediction risk guarantee, not just look
bad on a calibration metric in isolation?

WHY THIS SCRIPT EXISTS. code/95 and code/96 show a calibration GAP (Brier,
ECE). That establishes miscalibration exists but not that it matters
operationally. Selective prediction (Geifman & El-Yaniv 2017) is the
standard framework for "what happens if you deploy this": fix a confidence
threshold, abstain below it, and the risk (error rate) among non-abstained
predictions is what a practitioner actually relies on. This script asks: if
a practitioner picks that threshold using the SAME (reused-fold, LEAKY)
scores this paper already shows are optimistic, what risk do they actually
get when the model is deployed on genuinely new (CLEAN_MATCHED) data at that
identical threshold?

METHOD, per mechanism, per rep (identical LEAKY/CLEAN_MATCHED score pairs as
code/95 and code/96):
  1. On LEAKY scores, compute confidence c_i = |score_i - 0.5| * 2 in [0,1]
     and correctness_i = (round(score_i) == label_i).
  2. Find the confidence threshold tau achieving a TARGET COVERAGE (fraction
     of samples not abstained) on LEAKY -- e.g. tau = the coverage-target
     quantile of LEAKY's own confidence distribution.
  3. Read off LEAKY's OWN reported risk (error rate among samples with
     confidence >= tau) -- this is what a practitioner using only the
     reused fold would believe they are getting.
  4. Apply the IDENTICAL numeric threshold tau to CLEAN_MATCHED's scores and
     labels; report CLEAN's actual coverage and actual risk at that
     threshold -- this is what is really achieved when the model sees
     genuinely new data.
  5. The risk-guarantee violation is (actual risk on CLEAN) - (believed risk
     from LEAKY), reported with the same BCa 95% bootstrap this paper uses
     throughout, at three target coverage levels (30%, 50%, 70%) for
     sensitivity.

This directly answers the natural-next-step question code/95/96's Limitations
paragraph names: not just "is the confidence estimate wrong" but "what risk
does a practitioner actually sign up for by trusting it."
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "selective_prediction_consequence.json"
COVERAGE_TARGETS = [0.30, 0.50, 0.70]
RNG = np.random.default_rng(20260811)


def risk_at_coverage(scores, labels, coverage_target):
    conf = np.abs(scores - 0.5) * 2.0
    pred = (scores >= 0.5).astype(int)
    correct = (pred == labels).astype(int)
    n = len(scores)
    k = max(1, int(round(coverage_target * n)))
    order = np.argsort(-conf)  # most confident first
    tau = conf[order[k - 1]]  # confidence value at the target-coverage cutoff
    kept = conf >= tau
    achieved_coverage = float(kept.mean())
    risk = float(1.0 - correct[kept].mean()) if kept.sum() > 0 else float("nan")
    return tau, achieved_coverage, risk


def apply_threshold(scores, labels, tau):
    conf = np.abs(scores - 0.5) * 2.0
    pred = (scores >= 0.5).astype(int)
    correct = (pred == labels).astype(int)
    kept = conf >= tau
    achieved_coverage = float(kept.mean())
    risk = float(1.0 - correct[kept].mean()) if kept.sum() > 0 else float("nan")
    return achieved_coverage, risk


def bca_ci(x):
    res = bootstrap((np.asarray(x),), np.mean, confidence_level=0.95, n_resamples=10000,
                     method="BCa", random_state=RNG)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def mechanism2_reps():
    a = np.load(ROOT / "results" / "case_study_2_probe_scores.npz")
    n_layers = int(a["n_layers"])
    n_reps = int(a["n_reps"])
    ref = json.load(open(ROOT / "results" / "case_study_2_layer_decomposition.json"))["randomized_stratified_splits"]
    out = []
    for r in range(n_reps):
        variant = f"rand{r}"
        y_sel = a[f"{variant}__y_sel"]
        y_ho = a[f"{variant}__y_ho"]
        fold_id = a[f"{variant}__fold_id"]
        cv_scores = a[f"{variant}__cv_scores"]
        ho_scores = a[f"{variant}__ho_scores"]
        folds = sorted(set(fold_id.tolist()))
        cv_auroc = np.array([
            np.mean([roc_auc_score(y_sel[fold_id == f], cv_scores[l][fold_id == f]) for f in folds])
            for l in range(n_layers)])
        l_star = int(cv_auroc.argmax())
        assert l_star == ref["selected_layer_per_rep"][r]
        out.append((cv_scores[l_star], y_sel, ho_scores[l_star], y_ho))
    return out


def mechanism1_reps():
    d = np.load(ROOT / "results" / "real_features_mistral7b_halueval.npz")
    X_seq, X_glob, y = d["X_seq"], d["X_glob"], d["y"]
    X = np.hstack([X_seq.reshape(X_seq.shape[0], -1), X_glob]).astype(np.float64)
    out = []
    for seed in range(40):
        Xs_leaky = StandardScaler().fit_transform(X)
        clf_full = LogisticRegression(max_iter=500, C=1.0).fit(Xs_leaky, y)
        p_leaky = clf_full.predict_proba(Xs_leaky)[:, 1]

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        p_oof = np.zeros(len(y))
        for tr_idx, te_idx in skf.split(X, y):
            scaler = StandardScaler().fit(X[tr_idx])
            Xtr, Xte = scaler.transform(X[tr_idx]), scaler.transform(X[te_idx])
            clf = LogisticRegression(max_iter=500, C=1.0).fit(Xtr, y[tr_idx])
            p_oof[te_idx] = clf.predict_proba(Xte)[:, 1]
        out.append((p_leaky, y, p_oof, y))
    return out


def run_mechanism(name, reps):
    result = {"mechanism": name, "coverage_targets": {}}
    for cov_target in COVERAGE_TARGETS:
        believed_risks, actual_risks, actual_coverages = [], [], []
        for s_leaky, y_leaky, s_clean, y_clean in reps:
            tau, ach_cov_leaky, believed_risk = risk_at_coverage(s_leaky, y_leaky, cov_target)
            ach_cov_clean, actual_risk = apply_threshold(s_clean, y_clean, tau)
            believed_risks.append(believed_risk)
            actual_risks.append(actual_risk)
            actual_coverages.append(ach_cov_clean)
        believed_risks = np.array(believed_risks)
        actual_risks = np.array(actual_risks)
        actual_coverages = np.array(actual_coverages)
        violation = actual_risks - believed_risks
        v_lo, v_hi = bca_ci(violation)
        entry = {
            "believed_risk_mean_from_leaky": float(believed_risks.mean()),
            "actual_risk_mean_on_clean": float(actual_risks.mean()),
            "actual_coverage_mean_on_clean": float(actual_coverages.mean()),
            "risk_violation_mean": float(violation.mean()),
            "risk_violation_bca_95ci": [v_lo, v_hi],
            "relative_risk_inflation_pct": float(100 * violation.mean() / believed_risks.mean()) if believed_risks.mean() > 0 else None,
        }
        result["coverage_targets"][f"{cov_target:.2f}"] = entry
        print(f"  [{name}] coverage_target={cov_target:.2f}: believed_risk={believed_risks.mean():.4f} "
              f"actual_risk={actual_risks.mean():.4f} (actual coverage {actual_coverages.mean():.3f})  "
              f"violation={violation.mean():+.4f} BCa 95% [{v_lo:+.4f},{v_hi:+.4f}]  "
              f"({100*violation.mean()/believed_risks.mean():+.1f}% relative)")
    return result


def main():
    print("=== Mechanism 2 (GUARDIAN, CV-argmax layer selection) ===")
    m2 = run_mechanism("Mechanism 2 (CV-argmax layer selection)", mechanism2_reps())
    print("\n=== Mechanism 1 (full-dataset probe reuse, real features) ===")
    m1 = run_mechanism("Mechanism 1 (full-dataset probe reuse)", mechanism1_reps())

    out = {"mechanism_2": m2, "mechanism_1": m1}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
