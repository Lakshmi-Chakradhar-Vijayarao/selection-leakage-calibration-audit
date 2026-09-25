"""
Corrected deployment error count  [Reviewer DUxq, concern 2].

The submitted paper multiplies the risk difference by a fixed 70,000 answered
queries, which assumes the threshold that yields 70% coverage on the reused
(LEAKY) fold also yields 70% coverage on genuinely held-out (CLEAN) data. It
does not: coverage shifts when the score distribution shifts. This script
recomputes expected daily errors using EACH ARM'S OWN ACHIEVED COVERAGE,
per rep, so the comparison multiplies each risk by the number of predictions
actually accepted under that arm.

  believed errors = Q * coverage_achieved_on_LEAKY * risk_believed_on_LEAKY
  actual   errors = Q * coverage_achieved_on_CLEAN * risk_actual_on_CLEAN

Paired per rep across the 50 randomized splits, BCa 95% bootstrap as elsewhere.
No new inference; reuses results/case_study_2_probe_scores.npz.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "results" / "case_study_2_probe_scores.npz"
REF_JSON = ROOT / "results" / "case_study_2_layer_decomposition.json"
OUT_PATH = ROOT / "results" / "deployment_error_count_corrected.json"

QUERIES_PER_DAY = 100_000
COVERAGE_TARGETS = [0.30, 0.50, 0.70]
SEED = 20260925


def conf_pred_correct(scores, labels):
    conf = np.abs(scores - 0.5) * 2.0
    correct = ((scores >= 0.5).astype(int) == labels).astype(int)
    return conf, correct


def tau_at_coverage(scores, labels, coverage_target):
    conf, correct = conf_pred_correct(scores, labels)
    n = len(scores)
    k = max(1, int(round(coverage_target * n)))
    order = np.argsort(-conf)
    tau = conf[order[k - 1]]
    kept = conf >= tau
    return float(tau), float(kept.mean()), float(1.0 - correct[kept].mean())


def apply_tau(scores, labels, tau):
    conf, correct = conf_pred_correct(scores, labels)
    kept = conf >= tau
    if kept.sum() == 0:
        return 0.0, float("nan")
    return float(kept.mean()), float(1.0 - correct[kept].mean())


def bca(d, seed=SEED):
    d = np.asarray(d, dtype=float)
    if np.allclose(d, d[0]):
        return float(d[0]), float(d[0])
    r = bootstrap((d,), np.mean, confidence_level=0.95, n_resamples=10000,
                  method="BCa", random_state=np.random.default_rng(seed))
    return float(r.confidence_interval.low), float(r.confidence_interval.high)


def main():
    a = np.load(ARTIFACT)
    n_layers, n_reps = int(a["n_layers"]), int(a["n_reps"])
    ref = json.load(open(REF_JSON))["randomized_stratified_splits"]

    pairs = []
    for r in range(n_reps):
        v = f"rand{r}"
        y_sel, y_ho = a[f"{v}__y_sel"], a[f"{v}__y_ho"]
        fold_id = a[f"{v}__fold_id"]
        cv, ho = a[f"{v}__cv_scores"], a[f"{v}__ho_scores"]
        folds = sorted(set(fold_id.tolist()))
        cv_auroc = np.array([
            np.mean([roc_auc_score(y_sel[fold_id == f], cv[l][fold_id == f]) for f in folds])
            for l in range(n_layers)])
        l_star = int(cv_auroc.argmax())
        assert l_star == ref["selected_layer_per_rep"][r]
        pairs.append((cv[l_star], y_sel, ho[l_star], y_ho))

    out = {"queries_per_day": QUERIES_PER_DAY, "n_reps": n_reps, "seed": SEED,
           "coverage_targets": {}}

    print("=" * 96)
    print("CORRECTED DEPLOYMENT ERROR COUNT  (each arm uses its own achieved coverage)")
    print(f"Q = {QUERIES_PER_DAY:,} queries/day")
    print("=" * 96)

    for ct in COVERAGE_TARGETS:
        cov_L, risk_L, cov_C, risk_C = [], [], [], []
        for sL, yL, sC, yC in pairs:
            tau, cl, rl = tau_at_coverage(sL, yL, ct)
            cc, rc = apply_tau(sC, yC, tau)
            cov_L.append(cl); risk_L.append(rl); cov_C.append(cc); risk_C.append(rc)
        cov_L, risk_L = np.array(cov_L), np.array(risk_L)
        cov_C, risk_C = np.array(cov_C), np.array(risk_C)

        err_believed = QUERIES_PER_DAY * cov_L * risk_L
        err_actual = QUERIES_PER_DAY * cov_C * risk_C
        excess = err_actual - err_believed

        # the submitted paper's estimator, for comparison
        err_naive = QUERIES_PER_DAY * ct * (risk_C - risk_L)

        lo, hi = bca(excess)
        nlo, nhi = bca(err_naive)
        cov_lo, cov_hi = bca(cov_C - cov_L)

        e = {
            "coverage_target": ct,
            "achieved_coverage_leaky_mean": float(cov_L.mean()),
            "achieved_coverage_clean_mean": float(cov_C.mean()),
            "coverage_shift_clean_minus_leaky": {
                "mean": float((cov_C - cov_L).mean()), "bca_95ci": [cov_lo, cov_hi]},
            "believed_risk_mean": float(risk_L.mean()),
            "actual_risk_mean": float(risk_C.mean()),
            "answered_per_day_believed": float(QUERIES_PER_DAY * cov_L.mean()),
            "answered_per_day_actual": float(QUERIES_PER_DAY * cov_C.mean()),
            "errors_per_day_believed": float(err_believed.mean()),
            "errors_per_day_actual": float(err_actual.mean()),
            "excess_errors_per_day_corrected": {
                "mean": float(excess.mean()), "bca_95ci": [lo, hi]},
            "excess_errors_per_day_submitted_estimator": {
                "mean": float(err_naive.mean()), "bca_95ci": [nlo, nhi],
                "note": "Q * target_coverage * (risk_clean - risk_leaky); assumes coverage "
                        "is unchanged by the threshold transfer."},
        }
        out["coverage_targets"][f"{ct:.2f}"] = e

        print(f"\ncoverage target {ct:.0%}")
        print(f"  achieved coverage   LEAKY {cov_L.mean():.4f}   CLEAN {cov_C.mean():.4f}   "
              f"shift {(cov_C-cov_L).mean():+.4f}  BCa95 [{cov_lo:+.4f},{cov_hi:+.4f}]")
        print(f"  risk                LEAKY {risk_L.mean():.4f}   CLEAN {risk_C.mean():.4f}")
        print(f"  answered/day        believed {QUERIES_PER_DAY*cov_L.mean():,.0f}   "
              f"actual {QUERIES_PER_DAY*cov_C.mean():,.0f}")
        print(f"  errors/day          believed {err_believed.mean():,.0f}   "
              f"actual {err_actual.mean():,.0f}")
        print(f"  EXCESS errors/day   corrected  {excess.mean():,.0f}  "
              f"BCa95 [{lo:,.0f},{hi:,.0f}]")
        print(f"                      submitted  {err_naive.mean():,.0f}  "
              f"BCa95 [{nlo:,.0f},{nhi:,.0f}]")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
