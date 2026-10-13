"""
The corrected report on the second model family.

108 reported, for Mistral-7B and Qwen2.5-7B on HaluEval real features, that
Delta_sel on AUROC is "established in both" (+0.0070, +0.0072). That is a test
of Delta_sel against zero -- the test Section 4 of the paper shows cannot
distinguish selection recovering everything from selection recovering
nothing, because Delta_sel = A - B and E[A] > 0 under any noisy argmax.

This script replaces that reading with the one the paper prescribes, on the
same harness, so the paper practises what it prescribes in both places it
reports a selection result:

  gap at l*  = cv_{l*} - ho_{l*}     the optimism paid; needs no correction
  A          = cv_{l*} - mean(cv)    the winner's curse on the criterion
  B          = ho_{l*} - mean(ho)    tested against zero by one-sample t;
                                     its null value genuinely is zero

for AUROC, reliability, ECE and Brier, with Holm-Bonferroni across the four B
tests within each model family, exactly as 112 does for the main harness.

Protocol: identical to 108 -- same splits (code/48's generator, RNG seeded
42), same StratifiedKFold(5, shuffle, 42), same StandardScaler + logistic
probe fit in-fold, same 229/171 selection/held-out ratio, same 50 reps. The
probe-fitting code is imported from 108, not copied, so the two cannot drift.

Signs follow 112: gap is cv - ho, so for losses a negative value is the
optimism, and for losses B < 0 favours transfer.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "results" / "second_family_corrected_report.json"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sf = _load(ROOT / "108_calibration_second_model_family.py", "second_family")
rr = sf.rr          # 103: murphy_decomposition, ece_binned, bca_ci
cs2 = sf.cs2        # code/48: random_stratified_split

METRICS = [("auroc", False), ("reliability", True), ("ece", True), ("brier", True)]


def per_layer_metrics(cv_auroc, cv_scores, y_sel, ho_scores, y_ho):
    """Selection-fold (cv) and held-out (ho) value of every metric at every
    layer. cv AUROC is 108's mean of per-fold AUROCs -- the selection
    criterion itself -- so A is computed on exactly what the argmax saw."""
    L = cv_scores.shape[0]
    cv = {m: np.empty(L) for m, _ in METRICS}
    ho = {m: np.empty(L) for m, _ in METRICS}
    cv["auroc"][:] = cv_auroc
    for l in range(L):
        ho["auroc"][l] = roc_auc_score(y_ho, ho_scores[l])
        for arm, s, y in (("cv", cv_scores[l], y_sel), ("ho", ho_scores[l], y_ho)):
            rel, _, _ = rr.murphy_decomposition(s, y, 10, "width")
            tgt = cv if arm == "cv" else ho
            tgt["reliability"][l] = rel
            tgt["ece"][l] = rr.ece_binned(s, y, 10, "width")
            tgt["brier"][l] = float(np.mean((s - y) ** 2))
    return cv, ho


def holm(pvals):
    order = sorted(pvals.items(), key=lambda kv: kv[1])
    out, still = {}, True
    for i, (name, p) in enumerate(order):
        thr = 0.05 / (len(order) - i)
        ok = still and p <= thr
        if not ok:
            still = False
        out[name] = {"p": p, "threshold": thr, "survives_holm": ok}
    return out


def main():
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=sf.RANDOM_STATE)
    res = {"n_reps": sf.N_REPS, "n_sel": sf.N_SEL, "protocol": "identical to 108",
           "by_model": {}}

    print("=" * 100)
    print("CORRECTED REPORT, SECOND MODEL FAMILY (HaluEval real features, 12-dim probe)")
    print("=" * 100)

    for name, path in sf.MODELS.items():
        d = np.load(path)
        X, y = d["X_seq"].astype(np.float64), d["y"].astype(int)
        L = X.shape[1]
        rng = np.random.default_rng(sf.RANDOM_STATE)
        CV = {m: np.empty((sf.N_REPS, L)) for m, _ in METRICS}
        HO = {m: np.empty((sf.N_REPS, L)) for m, _ in METRICS}
        lstar = np.empty(sf.N_REPS, dtype=int)
        for r in range(sf.N_REPS):
            sel_idx, ho_idx = cs2.random_stratified_split(rng, y, sf.N_SEL)
            cv_auroc, cvs, hos = sf.layer_scores(X[sel_idx], y[sel_idx], X[ho_idx], kfold)
            c, h = per_layer_metrics(cv_auroc, cvs, y[sel_idx], hos, y[ho_idx])
            for m, _ in METRICS:
                CV[m][r], HO[m][r] = c[m], h[m]
            lstar[r] = int(cv_auroc.argmax())   # selection is on CV AUROC, as in 108

        rows = np.arange(sf.N_REPS)
        model = {"selected_layer_per_rep": lstar.tolist(), "metrics": {}}
        print(f"\n{name}   (selected layers: {sorted(set(lstar.tolist()))})")
        print(f"  {'metric':<12} {'gap@l*':>10} {'A':>10} {'B':>10} {'B BCa 95%':>22} "
              f"{'t':>7} {'p':>9}  verdict")
        pv = {}
        for m, is_loss in METRICS:
            cv, ho = CV[m], HO[m]
            gap = cv[rows, lstar] - ho[rows, lstar]
            A = cv[rows, lstar] - cv.mean(axis=1)
            B = ho[rows, lstar] - ho.mean(axis=1)
            t, p = stats.ttest_1samp(B, 0.0)
            lo, hi = rr.bca_ci(B)
            sig = (lo > 0) or (hi < 0)
            favours = (B.mean() < 0) if is_loss else (B.mean() > 0)
            verdict = ("transfers" if (sig and favours) else
                       "not established" if not sig else "against transfer")
            pv[m] = float(p)
            model["metrics"][m] = {
                "is_loss": is_loss,
                "gap_at_selected_mean": float(gap.mean()),
                "gap_at_selected_bca_95ci": list(rr.bca_ci(gap)),
                "A_curse_mean": float(A.mean()),
                "B_mean": float(B.mean()), "B_bca_95ci": [lo, hi],
                "B_t": float(t), "B_p": float(p),
                "verdict": verdict,
                "delta_sel_mean": float((A - B).mean()),
                # per-split values, so 121 can apply the re-split correction
                "B_per_rep": B.tolist(), "gap_per_rep": gap.tolist(),
            }
            print(f"  {m:<12} {gap.mean():>+10.5f} {A.mean():>+10.5f} {B.mean():>+10.5f} "
                  f"[{lo:>+9.5f},{hi:>+9.5f}] {t:>7.2f} {p:>9.2e}  {verdict}")
        h = holm(pv)
        model["holm_across_B_tests"] = h
        model["holm_changes_no_verdict"] = all(
            h[m]["survives_holm"] == (model["metrics"][m]["verdict"] != "not established")
            for m, _ in METRICS)
        print(f"  Holm changes no verdict: {model['holm_changes_no_verdict']}")
        res["by_model"][name] = model

    with open(OUT_PATH, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
