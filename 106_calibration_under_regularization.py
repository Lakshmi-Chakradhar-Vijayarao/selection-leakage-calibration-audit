"""
Does the SELECTION-SPECIFIC calibration effect survive regularization?
[Reviewer t9UX, sole substantive weakness; reinforces zDz3 weaknesses 1-2.]

t9UX observes that at C=0.01 the raw ECE gap vanishes for Mechanism 1 and the
raw Brier gap shrinks ~4x, and concludes that "part of the observed probability
distortion is driven by probe training near the interpolation threshold rather
than by the selection mechanism alone."

The paper already answers half of this for DISCRIMINATION: code/48's block (4)
sweeps StandardScaler x C over five decades and finds the selection-specific
AUROC component stable (+0.0324 at C=0.01 vs +0.0369 at C=1.0). It has never
been answered for CALIBRATION, which is this paper's actual claim. The raw gap
shrinking under regularization is consistent with two very different stories:

  (a) the calibration effect is an interpolation artifact -- then the
      SELECTION-SPECIFIC increment should shrink too; or
  (b) the raw gap mixes an interpolation-driven baseline with a genuine
      selection effect -- then the placebo shrinks but the increment survives.

103 established that at the shipped probe roughly 57% of the reliability gap is
selection-specific. This script re-runs that decomposition under regularization
so (a) and (b) can be told apart.

Protocol is imported wholesale from code/48 rather than reimplemented: same
hidden states, same StratifiedKFold(5, shuffle, random_state=42), same
_fit_probe, same random_stratified_split, same RNG stream (so rep r is the same
partition as rep r everywhere in this project). The metric functions are
imported from 103 for the same reason. The only thing that varies is (scale, C).

The shipped arm (scale=False, C=1.0) is NOT recomputed: it is read from the
cached artifact results/case_study_2_probe_scores.npz, which is what produced
the submitted numbers. Its appearance here is therefore also a consistency
check against 103.

Needs the 171 MB raw cache (GUARDIAN_ROOT), unlike 103/104. ~20 min on one core.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "results" / "calibration_under_regularization.json"
ARTIFACT = ROOT / "results" / "case_study_2_probe_scores.npz"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Both modules guard their entry points with __name__ == "__main__", so these
# imports execute definitions only.
cs2 = _load(ROOT / "code" / "48_case_study_2_layer_decomposition.py", "cs2_decomp")
rr = _load(ROOT / "103_reviewer_response_mechanism2.py", "reviewer_response")

N_REPS = 50
COVERAGE_TARGET = 0.50
CONFIGS = [
    ("scaled_C1.0", True, 1.0),
    ("scaled_C0.01", True, 0.01),
]


def calib_row(scores, labels):
    """The four per-arm quantities the paper's calibration claim rests on."""
    rel, res, _ = rr.murphy_decomposition(scores, labels, 10, "width")
    return {
        "brier": float(np.mean((scores - labels) ** 2)),
        "reliability": rel,
        "resolution": res,
        "ece": rr.ece_binned(scores, labels, 10, "width"),
    }


def risk_at_coverage(s_leaky, y_leaky, s_clean, y_clean, target):
    """Selective-prediction risk violation, as in code/97 and 104."""
    conf_l = np.abs(s_leaky - 0.5) * 2.0
    correct_l = ((s_leaky >= 0.5).astype(int) == y_leaky).astype(int)
    k = max(1, int(round(target * len(s_leaky))))
    tau = conf_l[np.argsort(-conf_l)[k - 1]]
    kept_l = conf_l >= tau
    risk_l = 1.0 - correct_l[kept_l].mean()

    conf_c = np.abs(s_clean - 0.5) * 2.0
    correct_c = ((s_clean >= 0.5).astype(int) == y_clean).astype(int)
    kept_c = conf_c >= tau
    if kept_c.sum() == 0:
        return float("nan")
    return float((1.0 - correct_c[kept_c].mean()) - risk_l)


def decompose(cv_scores, y_sel, ho_scores, y_ho, l_star, n_layers):
    """gap@selected, all-layer placebo, and the selection-specific increment,
    for each calibration metric. Sign convention matches 103: positive means
    'worse on genuinely held-out data'."""
    per_layer = []
    for l in range(n_layers):
        leaky = calib_row(cv_scores[l], y_sel)
        clean = calib_row(ho_scores[l], y_ho)
        per_layer.append({
            "brier": clean["brier"] - leaky["brier"],
            "reliability": clean["reliability"] - leaky["reliability"],
            "ece": clean["ece"] - leaky["ece"],
            # Resolution is signed clean-minus-leaky, matching 103's
            # placebo_control block (NOT its decomposition table, which reports
            # the same quantity leaky-minus-clean). Under this convention a
            # NEGATIVE selection-specific increment is the expected signature:
            # argmax-AUROC buys in-sample discrimination at the selected layer.
            "resolution": clean["resolution"] - leaky["resolution"],
        })
    out = {}
    for m in ("brier", "reliability", "ece", "resolution"):
        vals = np.array([p[m] for p in per_layer])
        out[m] = {
            "gap_at_selected": float(vals[l_star]),
            "placebo_all_layer_mean": float(vals.mean()),
            "selection_specific": float(vals[l_star] - vals.mean()),
        }
    return out


def main():
    d = np.load(cs2.GUARDIAN_NPZ)
    H, y = d["hidden_states"], d["labels"]
    valid = y >= 0
    H, y = H[valid], y[valid]
    n_layers = H.shape[1]
    print(f"Loaded {cs2.GUARDIAN_NPZ.name}: {H.shape}, hall_rate={y.mean():.3f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cs2.RANDOM_STATE)
    results = {"n_reps": N_REPS, "seed": rr.SEED,
               "random_state": cs2.RANDOM_STATE,
               "coverage_target_for_risk": COVERAGE_TARGET,
               "by_config": {}}

    # ── the shipped arm, replayed from the cached artifact ──────────────────
    print("\n=== shipped (scale=False, C=1.0), replayed from cached artifact ===",
          flush=True)
    a = np.load(ARTIFACT)
    n_reps_art = int(a["n_reps"])
    rows, risks = [], []
    for rep in range(n_reps_art):
        cv_auroc, y_sel, y_ho, cvs, hos = rr.per_rep_arrays(a, rep, n_layers)
        l_star = int(cv_auroc.argmax())
        rows.append(decompose(cvs, y_sel, hos, y_ho, l_star, n_layers))
        risks.append(risk_at_coverage(cvs[l_star], y_sel, hos[l_star], y_ho,
                                      COVERAGE_TARGET))
    results["by_config"]["shipped_unscaled_C1.0"] = summarise_config(rows, risks)

    # ── the regularized arms, recomputed ────────────────────────────────────
    for name, scale, C in CONFIGS:
        print(f"\n=== {name} (scale={scale}, C={C}), {N_REPS} reps ===", flush=True)
        rng = np.random.default_rng(cs2.RANDOM_STATE)
        conv = {"n_fits": 0, "max_n_iter": 0, "n_convergence_warnings": 0}
        rows, risks = [], []
        for rep in range(N_REPS):
            sel_idx, ho_idx = cs2.random_stratified_split(rng, y, cs2.N_TRAIN_SELECT)
            # per_layer_gaps writes the per-sample scores into cs2.ARTIFACT
            # when `variant` is set; we use that to get the scores back out.
            cs2.ARTIFACT.clear()
            pl = cs2.per_layer_gaps(H[sel_idx], y[sel_idx], H[ho_idx], y[ho_idx],
                                    n_layers, cv, variant="tmp", scale=scale,
                                    C=C, conv=conv)
            l_star = int(np.argmax([r["cv_auroc_selectpool"] for r in pl]))
            y_sel = cs2.ARTIFACT["tmp__y_sel"]
            y_ho = cs2.ARTIFACT["tmp__y_ho"]
            cvs = cs2.ARTIFACT["tmp__cv_scores"]
            hos = cs2.ARTIFACT["tmp__ho_scores"]
            rows.append(decompose(cvs, y_sel, hos, y_ho, l_star, n_layers))
            risks.append(risk_at_coverage(cvs[l_star], y_sel, hos[l_star], y_ho,
                                          COVERAGE_TARGET))
            if rep % 10 == 0:
                print(f"  rep {rep}: L{l_star}  "
                      f"rel-specific={rows[-1]['reliability']['selection_specific']:+.5f}",
                      flush=True)
        cfg = summarise_config(rows, risks)
        cfg["convergence"] = conv
        results["by_config"][name] = cfg

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    report(results)
    print(f"\nSaved: {OUT_PATH}")


def summarise_config(rows, risks):
    out = {"n_reps": len(rows), "metrics": {}}
    for m in ("brier", "reliability", "ece", "resolution"):
        e = {}
        for part in ("gap_at_selected", "placebo_all_layer_mean", "selection_specific"):
            vals = np.array([r[m][part] for r in rows])
            e[part] = rr.summarise(vals)
        gap = e["gap_at_selected"]["mean"]
        e["share_selection_specific"] = (
            float(e["selection_specific"]["mean"] / gap) if gap != 0 else None)
        out["metrics"][m] = e
    out["risk_violation_at_coverage"] = rr.summarise(np.array(risks, dtype=float))
    return out


def report(results):
    print("\n" + "=" * 96)
    print("SELECTION-SPECIFIC CALIBRATION UNDER REGULARIZATION")
    print("=" * 96)
    for name, cfg in results["by_config"].items():
        print(f"\n{name}  (n_reps={cfg['n_reps']})")
        for m in ("reliability", "ece", "brier", "resolution"):
            e = cfg["metrics"][m]
            ss = e["selection_specific"]
            star = "*" if ss["excludes_zero"] else " "
            share = e["share_selection_specific"]
            print(f"  {m:<12s} gap {e['gap_at_selected']['mean']:+.5f}   "
                  f"placebo {e['placebo_all_layer_mean']['mean']:+.5f}   "
                  f"selection-specific {ss['mean']:+.5f} "
                  f"[{ss['bca_95ci'][0]:+.5f},{ss['bca_95ci'][1]:+.5f}]{star}"
                  + (f"  ({share:.1%})" if share is not None else ""))
        r = cfg["risk_violation_at_coverage"]
        print(f"  risk violation @ {results['coverage_target_for_risk']:.0%} coverage: "
              f"{r['mean']:+.5f} [{r['bca_95ci'][0]:+.5f},{r['bca_95ci'][1]:+.5f}]"
              f"{'*' if r['excludes_zero'] else ''}")


if __name__ == "__main__":
    main()
