"""
The corrected audit report, applied to this paper's own harness.

Section "The Estimand, Its Null, and a Corrected Test" recommends that an
audit report three things instead of testing Delta_sel against zero:

  1. gap at the selected candidate  -- the optimism actually paid. Directly
     measured, unbiased, needs no correction.
  2. A = cv_{l*} - mean(cv)         -- the winner's curse on the selection
     criterion, which is the reference Delta_sel must beat.
  3. B = ho_{l*} - mean(ho)         -- tested against zero with a one-sample
     t-test over replicates. Built only from held-out values, so its null
     value is genuinely zero.

This script produces exactly that report for the Mechanism-2 harness, on
AUROC and on each calibration metric, so the paper practises what it
prescribes. It supersedes the permutation null of 110: by the identity
Delta_sel = A - B, "Delta_sel below its permutation null" is the same event as
"B > 0", so the permutation recovers with Monte Carlo error a conclusion the
t-test gives exactly. 110 is retained in the artifact only so the equivalence
can be checked.

Signs. AUROC is a score (higher is better), so B > 0 means the selected layer
transfers real quality. Reliability, ECE and Brier are losses (lower is
better), so for those B < 0 means the same thing. We report the raw signed
value and say which direction favours transfer.

Reuses results/case_study_2_probe_scores.npz. No refitting.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "results" / "case_study_2_probe_scores.npz"
OUT_PATH = ROOT / "results" / "corrected_report.json"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rr = _load(ROOT / "103_reviewer_response_mechanism2.py", "reviewer_response")

# (name, is_a_loss) -- for losses, a NEGATIVE B favours transfer
METRICS = [("auroc", False), ("reliability", True), ("ece", True), ("brier", True)]


def per_layer(a, rep, n_layers):
    v = f"rand{rep}"
    y_sel, y_ho = a[f"{v}__y_sel"], a[f"{v}__y_ho"]
    fold_id = a[f"{v}__fold_id"]
    cv = a[f"{v}__cv_scores"].astype(np.float64)
    ho = a[f"{v}__ho_scores"].astype(np.float64)
    folds = sorted(set(fold_id.tolist()))

    out_cv = {m: np.empty(n_layers) for m, _ in METRICS}
    out_ho = {m: np.empty(n_layers) for m, _ in METRICS}
    for l in range(n_layers):
        out_cv["auroc"][l] = np.mean([
            roc_auc_score(y_sel[fold_id == f], cv[l][fold_id == f]) for f in folds])
        out_ho["auroc"][l] = roc_auc_score(y_ho, ho[l])
        rl, _, _ = rr.murphy_decomposition(cv[l], y_sel, 10, "width")
        rc, _, _ = rr.murphy_decomposition(ho[l], y_ho, 10, "width")
        out_cv["reliability"][l], out_ho["reliability"][l] = rl, rc
        out_cv["ece"][l] = rr.ece_binned(cv[l], y_sel, 10, "width")
        out_ho["ece"][l] = rr.ece_binned(ho[l], y_ho, 10, "width")
        out_cv["brier"][l] = np.mean((cv[l] - y_sel) ** 2)
        out_ho["brier"][l] = np.mean((ho[l] - y_ho) ** 2)
    # selection is always on the CV AUROC criterion, whatever metric is reported
    l_star = int(out_cv["auroc"].argmax())
    return out_cv, out_ho, l_star


def main():
    a = np.load(ARTIFACT)
    n_layers, n_reps = int(a["n_layers"]), int(a["n_reps"])
    CV = {m: np.empty((n_reps, n_layers)) for m, _ in METRICS}
    HO = {m: np.empty((n_reps, n_layers)) for m, _ in METRICS}
    lstar = np.empty(n_reps, dtype=int)
    for r in range(n_reps):
        c, h, ls = per_layer(a, r, n_layers)
        for m, _ in METRICS:
            CV[m][r], HO[m][r] = c[m], h[m]
        lstar[r] = ls

    rows = np.arange(n_reps)
    out = {"n_reps": n_reps, "n_layers": n_layers, "metrics": {}}

    print("=" * 104)
    print("THE CORRECTED AUDIT REPORT  (Mechanism 2, CV-argmax over 32 layers)")
    print("=" * 104)
    print(f"{'metric':<13} {'gap@l*':>10} {'A (curse)':>11} {'B':>10} "
          f"{'B 95% CI':>22} {'t':>7} {'p':>9}  transfer?")
    print("-" * 104)

    for m, is_loss in METRICS:
        cv, ho = CV[m], HO[m]
        gap = cv[rows, lstar] - ho[rows, lstar]
        A = cv[rows, lstar] - cv.mean(axis=1)
        B = ho[rows, lstar] - ho.mean(axis=1)
        t, pv = stats.ttest_1samp(B, 0.0)
        lo, hi = rr.bca_ci(B)
        sig = (lo > 0) or (hi < 0)
        favours = (B.mean() < 0) if is_loss else (B.mean() > 0)
        verdict = ("transfers" if (sig and favours) else
                   "not established" if not sig else "against transfer")
        out["metrics"][m] = {
            "is_loss": is_loss,
            "gap_at_selected_mean": float(gap.mean()),
            "A_curse_mean": float(A.mean()),
            "B_mean": float(B.mean()),
            "B_bca_95ci": [lo, hi],
            "B_t": float(t), "B_p": float(pv),
            "B_excludes_zero": bool(sig),
            "B_favours_transfer": bool(favours),
            "verdict": verdict,
            "delta_sel_mean": float((A - B).mean()),
            "identity_check": float(abs((A - B).mean() - (A.mean() - B.mean()))),
        }
        print(f"{m:<13} {gap.mean():>+10.5f} {A.mean():>+11.5f} {B.mean():>+10.5f} "
              f"[{lo:>+9.5f},{hi:>+9.5f}] {t:>7.2f} {pv:>9.2e}  {verdict}")

    print("-" * 104)
    print("gap@l* is what the practitioner pays and needs no correction.")
    print("A is the reference Delta_sel must beat; Delta_sel = A - B by identity.")
    print("B is tested against zero; for losses (reliability/ECE/Brier) a "
          "negative B favours transfer.")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
