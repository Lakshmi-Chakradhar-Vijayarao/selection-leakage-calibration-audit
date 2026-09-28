"""
Permutation null for the selection-specific CALIBRATION increment.

This exists because of a criticism the paper hands a reviewer itself. Appendix
C's `selection_specific_component` docstring states, correctly, that the
quantity is positive in expectation even under a pure-noise null: with
l* = argmax_l cv_l and gap_l = cv_l - ho_l sharing the +cv_l term, selecting
the argmax of cv necessarily selects a layer whose gap is upward-biased
relative to the mean gap. The paper builds a permutation null for Mechanism
4's AUROC statistic for exactly this reason.

The headline CALIBRATION increment has had no such null. It is reported only
as a BCa interval excluding zero. If the estimator drifts positive under noise,
"the interval excludes zero" does not establish a selection effect beyond
argmax-over-a-noisy-criterion. That is the gap this script closes.

Null construction. We keep the cached probe scores exactly as they are and
permute the LABELS, independently within each arm so that each arm's base rate
is preserved (Murphy's UNC term is then unchanged and cannot drive the
contrast). Under the permuted labels the scores carry no information, so the
per-layer CV AUROCs differ only by chance, and argmax over 32 of them is
precisely the noisy-criterion selection whose mechanical effect we want to
measure. Everything downstream -- l*, the per-layer calibration gaps, the
all-layer placebo, the increment -- is then recomputed unchanged.

What this isolates. Refitting probes on permuted labels would additionally
destroy the probes themselves; that is a different (and weaker) null, because
it removes the score geometry along with the signal. Permuting labels against
fixed scores holds the selection procedure and the score distribution constant
and removes only the label relationship, which is the null the criticism is
about.

Reported: the observed increment, the null distribution of the same statistic,
a one-sided permutation p-value, and the null's mean -- the last being the
magnitude of the mechanical bias the criticism predicts.

Reuses results/case_study_2_probe_scores.npz and 103's metric functions. No
refitting, no new inference.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "results" / "case_study_2_probe_scores.npz"
OUT_PATH = ROOT / "results" / "selection_increment_null.json"

N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 500
SEED = 20260928
METRICS = ("reliability", "ece", "brier")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rr = _load(ROOT / "103_reviewer_response_mechanism2.py", "reviewer_response")


def gaps_per_layer(cv_scores, y_sel, ho_scores, y_ho, n_layers):
    """Per-layer (clean - leaky) gap for each calibration metric."""
    out = {m: np.empty(n_layers) for m in METRICS}
    for l in range(n_layers):
        rel_l, _, _ = rr.murphy_decomposition(cv_scores[l], y_sel, 10, "width")
        rel_c, _, _ = rr.murphy_decomposition(ho_scores[l], y_ho, 10, "width")
        out["reliability"][l] = rel_c - rel_l
        out["ece"][l] = (rr.ece_binned(ho_scores[l], y_ho, 10, "width")
                         - rr.ece_binned(cv_scores[l], y_sel, 10, "width"))
        out["brier"][l] = (np.mean((ho_scores[l] - y_ho) ** 2)
                           - np.mean((cv_scores[l] - y_sel) ** 2))
    return out


def cv_auroc_per_layer(cv_scores, y_sel, fold_id, n_layers):
    folds = sorted(set(fold_id.tolist()))
    return np.array([
        np.mean([roc_auc_score(y_sel[fold_id == f], cv_scores[l][fold_id == f])
                 for f in folds]) for l in range(n_layers)])


def increments(cv_scores, y_sel, ho_scores, y_ho, fold_id, n_layers):
    """gap@argmax - mean gap over all layers, per metric."""
    l_star = int(cv_auroc_per_layer(cv_scores, y_sel, fold_id, n_layers).argmax())
    g = gaps_per_layer(cv_scores, y_sel, ho_scores, y_ho, n_layers)
    return {m: float(g[m][l_star] - g[m].mean()) for m in METRICS}, l_star


def permute_within_arm(rng, y):
    """Shuffle labels within the arm, preserving its base rate exactly."""
    return y[rng.permutation(len(y))]


def main():
    a = np.load(ARTIFACT)
    n_layers, n_reps = int(a["n_layers"]), int(a["n_reps"])
    reps = []
    for r in range(n_reps):
        v = f"rand{r}"
        reps.append((a[f"{v}__cv_scores"].astype(np.float64), a[f"{v}__y_sel"],
                     a[f"{v}__ho_scores"].astype(np.float64), a[f"{v}__y_ho"],
                     a[f"{v}__fold_id"]))
    print(f"loaded {n_reps} reps x {n_layers} layers; running {N_PERM} permutations",
          flush=True)

    # ---- observed statistic: mean increment over reps ----------------------
    obs_per_rep = {m: [] for m in METRICS}
    for cv, ys, ho, yh, fid in reps:
        inc, _ = increments(cv, ys, ho, yh, fid, n_layers)
        for m in METRICS:
            obs_per_rep[m].append(inc[m])
    observed = {m: float(np.mean(obs_per_rep[m])) for m in METRICS}
    print("observed mean increment:",
          {m: round(v, 5) for m, v in observed.items()}, flush=True)

    # ---- null: same statistic under within-arm label permutation ----------
    rng = np.random.default_rng(SEED)
    null = {m: np.empty(N_PERM) for m in METRICS}
    for b in range(N_PERM):
        per_rep = {m: [] for m in METRICS}
        for cv, ys, ho, yh, fid in reps:
            ysp = permute_within_arm(rng, ys)
            yhp = permute_within_arm(rng, yh)
            inc, _ = increments(cv, ysp, ho, yhp, fid, n_layers)
            for m in METRICS:
                per_rep[m].append(inc[m])
        for m in METRICS:
            null[m][b] = np.mean(per_rep[m])
        if (b + 1) % 25 == 0:
            print(f"  perm {b+1}/{N_PERM}  "
                  f"null reliability mean so far {null['reliability'][:b+1].mean():+.5f}",
                  flush=True)

    out = {"n_perm": N_PERM, "n_reps": n_reps, "n_layers": n_layers, "seed": SEED,
           "null_construction": "labels permuted within each arm (base rate "
                                "preserved); cached probe scores unchanged",
           "metrics": {}}
    print("\n" + "=" * 94)
    print("PERMUTATION NULL FOR THE SELECTION-SPECIFIC INCREMENT")
    print("=" * 94)
    for m in METRICS:
        nd = null[m]
        # one-sided (upper) permutation p-value, +1 correction
        p = float((1 + np.sum(nd >= observed[m])) / (N_PERM + 1))
        e = {
            "observed": observed[m],
            "null_mean": float(nd.mean()),
            "null_sd": float(nd.std(ddof=1)),
            "null_q975": float(np.quantile(nd, 0.975)),
            "null_max": float(nd.max()),
            "permutation_p_one_sided": p,
            "observed_minus_null_mean": float(observed[m] - nd.mean()),
            "exceeds_null_97_5th_pct": bool(observed[m] > np.quantile(nd, 0.975)),
        }
        out["metrics"][m] = e
        star = "*" if p < 0.05 else " "
        print(f"  {m:<12s} observed {e['observed']:+.5f}   "
              f"null {e['null_mean']:+.5f} (SD {e['null_sd']:.5f}, "
              f"97.5th {e['null_q975']:+.5f})   "
              f"p={p:.4f}{star}   excess {e['observed_minus_null_mean']:+.5f}")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
