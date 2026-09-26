"""
Does the selection-specific calibration effect appear in a SECOND model family?
[Reviewer zDz3 weakness 3; reviewer DUxq concern 3.]

Both reviews make the same point: the calibration result is established on one
model family. zDz3: "Its replication changes the question pool but retains
Mistral-7B; the four additional architectures are evaluated only on the
AUROC-based selection component." DUxq: "established primarily for one
layer-selection construction on one model across two question pools."

The four extra architectures cannot answer this -- the artifact holds only
their summary statistics, not per-sample scores, so no calibration metric can
be computed from them. But the real-feature harness DOES ship per-sample
features for two model families on the same dataset:

    results/real_features_mistral7b_halueval.npz   (400, 32, 12)
    results/real_features_qwen2.5_7b_halueval.npz  (400, 32, 12)

Same extractor (code/03), same HaluEval pool, same 12 per-layer summary
statistics at 32 sampled layers. That permits the identical Mechanism-2
construction -- argmax over 32 layers on reused CV folds, then the calibration
bridge -- run on Mistral-7B and Qwen2.5-7B under a matched protocol.

Two things make this a useful test rather than a repeat:

  1. It is a genuinely different model family, which is what was asked for.
  2. The probe here is 12-dimensional at n=229, so it CANNOT interpolate.
     Reviewer t9UX's concern is that the calibration effect may be driven by
     probe training near the interpolation threshold (the main study is
     d=4096, n=400). If the effect appears here too, that explanation is
     insufficient on its own.

Magnitudes are NOT comparable with the main study: different features,
different dataset, different n. The question is direction and significance.

Protocol constants mirror the main study where a choice exists: the same
4:3 selection/held-out ratio, the same StratifiedKFold(5, shuffle,
random_state=42), the same split generator and RNG stream (code/48), C=1.0,
the same 50 reps, the same BCa. The one deliberate departure is feature
scaling: these are heterogeneous summary statistics, so we follow THIS
harness's own convention (StandardScaler, fit in-fold) rather than the main
study's unscaled hidden-state probe. See `_fit`.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "results" / "calibration_second_model_family.json"

MODELS = {
    "mistral-7b": ROOT / "results" / "real_features_mistral7b_halueval.npz",
    "qwen2.5-7b": ROOT / "results" / "real_features_qwen2.5_7b_halueval.npz",
}
N_REPS = 50
RANDOM_STATE = 42
COVERAGE_TARGET = 0.50
# 400 total. The main study selects on 400 of 700 (4:3); 229/171 is the same
# ratio, so the selection pool is not incidentally larger or smaller here.
N_SEL = 229


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cs2 = _load(ROOT / "code" / "48_case_study_2_layer_decomposition.py", "cs2_decomp")
rr = _load(ROOT / "103_reviewer_response_mechanism2.py", "reviewer_response")


def calib_row(scores, labels):
    rel, res, _ = rr.murphy_decomposition(scores, labels, 10, "width")
    return {"brier": float(np.mean((scores - labels) ** 2)),
            "reliability": rel, "resolution": res,
            "ece": rr.ece_binned(scores, labels, 10, "width")}


def _fit(X_tr, y_tr):
    """StandardScaler + LogisticRegression, fit on the training split only.

    This harness's features are heterogeneous summary statistics (vector norm,
    mean, std, entropy, kurtosis, ...) whose scales differ by orders of
    magnitude, so its own scripts standardize before fitting
    (code/03 line 340, code/25 line 126). We match that convention rather than
    the main study's unscaled-hidden-state probe, which is only appropriate
    because raw hidden-state dimensions share a scale. The scaler is fit
    inside the fold on training data alone, so it leaks nothing.
    """
    sc = StandardScaler().fit(X_tr)
    return sc, LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(X_tr), y_tr)


def layer_scores(X_sel, y_sel, X_ho, cv):
    """CV-fold and held-out probe scores at every layer, matching code/48's
    per_layer_gaps arithmetic (explicit fold loop, mean of per-fold AUROCs)."""
    n_layers = X_sel.shape[1]
    splits = list(cv.split(np.zeros(len(y_sel)), y_sel))
    cv_scores = np.zeros((n_layers, len(y_sel)), dtype=np.float64)
    ho_scores = np.zeros((n_layers, X_ho.shape[0]), dtype=np.float64)
    cv_auroc = np.zeros(n_layers)
    for l in range(n_layers):
        A, B = X_sel[:, l, :], X_ho[:, l, :]
        fold_aurocs = []
        for tr, te in splits:
            sc, clf = _fit(A[tr], y_sel[tr])
            s = clf.predict_proba(sc.transform(A[te]))[:, 1]
            cv_scores[l, te] = s
            fold_aurocs.append(roc_auc_score(y_sel[te], s))
        cv_auroc[l] = float(np.mean(fold_aurocs))
        sc, clf = _fit(A, y_sel)
        ho_scores[l] = clf.predict_proba(sc.transform(B))[:, 1]
    return cv_auroc, cv_scores, ho_scores


def decompose(cv_scores, y_sel, ho_scores, y_ho, l_star):
    per_layer = []
    for l in range(cv_scores.shape[0]):
        leaky, clean = calib_row(cv_scores[l], y_sel), calib_row(ho_scores[l], y_ho)
        per_layer.append({
            "brier": clean["brier"] - leaky["brier"],
            "reliability": clean["reliability"] - leaky["reliability"],
            "ece": clean["ece"] - leaky["ece"],
            "resolution": clean["resolution"] - leaky["resolution"],
        })
    out = {}
    for m in ("brier", "reliability", "ece", "resolution"):
        v = np.array([p[m] for p in per_layer])
        out[m] = {"gap_at_selected": float(v[l_star]),
                  "placebo_all_layer_mean": float(v.mean()),
                  "selection_specific": float(v[l_star] - v.mean())}
    return out


def risk_violation(s_l, y_l, s_c, y_c, target):
    conf_l = np.abs(s_l - 0.5) * 2.0
    corr_l = ((s_l >= 0.5).astype(int) == y_l).astype(int)
    k = max(1, int(round(target * len(s_l))))
    tau = conf_l[np.argsort(-conf_l)[k - 1]]
    kept_l = conf_l >= tau
    conf_c = np.abs(s_c - 0.5) * 2.0
    corr_c = ((s_c >= 0.5).astype(int) == y_c).astype(int)
    kept_c = conf_c >= tau
    if kept_c.sum() == 0:
        return float("nan")
    return float((1.0 - corr_c[kept_c].mean()) - (1.0 - corr_l[kept_l].mean()))


def main():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {"n_reps": N_REPS, "n_sel": N_SEL, "seed": rr.SEED,
               "coverage_target": COVERAGE_TARGET, "by_model": {}}

    for name, path in MODELS.items():
        d = np.load(path)
        X, y = d["X_seq"].astype(np.float64), d["y"].astype(int)
        n_layers, n_feat = X.shape[1], X.shape[2]
        print(f"\n=== {name}: X={X.shape} (layers={n_layers}, feats/layer={n_feat}), "
              f"n={len(y)}, hall_rate={y.mean():.3f} ===", flush=True)

        rng = np.random.default_rng(RANDOM_STATE)
        rows, risks, aurocs, layers = [], [], [], []
        for rep in range(N_REPS):
            sel_idx, ho_idx = cs2.random_stratified_split(rng, y, N_SEL)
            cv_auroc, cvs, hos = layer_scores(X[sel_idx], y[sel_idx], X[ho_idx], cv)
            l_star = int(cv_auroc.argmax())
            y_sel, y_ho = y[sel_idx], y[ho_idx]
            ho_auroc = np.array([roc_auc_score(y_ho, hos[l]) for l in range(n_layers)])
            gaps = cv_auroc - ho_auroc
            aurocs.append(float(gaps[l_star] - gaps.mean()))
            layers.append(l_star)
            rows.append(decompose(cvs, y_sel, hos, y_ho, l_star))
            risks.append(risk_violation(cvs[l_star], y_sel, hos[l_star], y_ho,
                                        COVERAGE_TARGET))
            if rep % 10 == 0:
                print(f"  rep {rep}: L{l_star}  "
                      f"rel-specific={rows[-1]['reliability']['selection_specific']:+.5f}",
                      flush=True)

        e = {"n_layers": n_layers, "n_features_per_layer": n_feat,
             "probe": "StandardScaler + LogisticRegression(max_iter=1000, C=1.0), scaler fit in-fold",
             "interpolation_possible": bool(n_feat >= N_SEL),
             "selected_layer_per_rep": layers,
             "delta_sel_auroc": rr.summarise(np.array(aurocs)),
             "risk_violation_at_coverage": rr.summarise(np.array(risks, dtype=float)),
             "metrics": {}}
        for m in ("brier", "reliability", "ece", "resolution"):
            sub = {}
            for part in ("gap_at_selected", "placebo_all_layer_mean", "selection_specific"):
                sub[part] = rr.summarise(np.array([r[m][part] for r in rows]))
            g = sub["gap_at_selected"]["mean"]
            sub["share_selection_specific"] = float(
                sub["selection_specific"]["mean"] / g) if g != 0 else None
            e["metrics"][m] = sub
        results["by_model"][name] = e

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 96)
    print("SELECTION-SPECIFIC CALIBRATION, SECOND MODEL FAMILY")
    print("(12-dim probe at n=229: cannot interpolate)")
    print("=" * 96)
    for name, e in results["by_model"].items():
        print(f"\n{name}")
        d = e["delta_sel_auroc"]
        print(f"  Delta_sel (AUROC)   {d['mean']:+.5f} "
              f"[{d['bca_95ci'][0]:+.5f},{d['bca_95ci'][1]:+.5f}]"
              f"{'*' if d['excludes_zero'] else ''}")
        for m in ("reliability", "ece", "brier", "resolution"):
            s = e["metrics"][m]
            ss = s["selection_specific"]
            print(f"  {m:<12s} gap {s['gap_at_selected']['mean']:+.5f}   "
                  f"placebo {s['placebo_all_layer_mean']['mean']:+.5f}   "
                  f"sel-specific {ss['mean']:+.5f} "
                  f"[{ss['bca_95ci'][0]:+.5f},{ss['bca_95ci'][1]:+.5f}]"
                  f"{'*' if ss['excludes_zero'] else ''}")
        r = e["risk_violation_at_coverage"]
        print(f"  risk violation @50% {r['mean']:+.5f} "
              f"[{r['bca_95ci'][0]:+.5f},{r['bca_95ci'][1]:+.5f}]"
              f"{'*' if r['excludes_zero'] else ''}")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
