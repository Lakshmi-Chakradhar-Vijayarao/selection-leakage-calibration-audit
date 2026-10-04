"""
Does candidate dependence explain the size of the winner's curse here?
[Phase 2. Answers the referee's M2 by measuring what it asserts.]

THE QUESTION. Every correction for selection optimism -- the permutation null
we discarded, the winner's-curse estimators of the genetics literature, and
the simulation in this paper's own methods section -- is calibrated to K, the
nominal number of candidates. If candidates are nearly redundant, the curse is
much smaller than K implies, and corrections calibrated to K over-correct.

Layer selection is the extreme case: adjacent transformer layers produce
near-identical representations, so their cross-validated criteria move
together. This script measures that dependence and asks whether it accounts
for the curse actually observed.

DESIGN. From the cached scores we form, for each of the 50 splits, the
per-layer CV AUROC (the selection criterion) and the per-layer held-out AUROC.
Centring each column gives a 50 x 64 residual matrix whose rows carry the
joint dependence across all 32 layers and both arms.

Two resampling schemes differ in exactly one respect:

  DEPENDENT   resample whole ROWS of residuals. Preserves the empirical
              cross-candidate dependence structure exactly.
  INDEPENDENT permute each COLUMN separately. Preserves every marginal
              distribution exactly and destroys only the cross-candidate
              dependence.

Both preserve the per-layer means, so genuine quality differences between
layers survive in both. The contrast isolates dependence and nothing else, and
it is nonparametric: no Gaussian assumption, no covariance estimated from 50
rows.

WHAT WOULD FALSIFY THE CLAIM. If the DEPENDENT scheme fails to reproduce the
observed A, the dependence story does not explain the curse and this line of
argument should be dropped. That is the kill criterion, and it is checked
below rather than assumed.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "results" / "case_study_2_probe_scores.npz"
OUT_PATH = ROOT / "results" / "candidate_dependence.json"

N_TRIALS = 4000
SEED = 20261002


def per_layer_auroc(a, n_layers, n_reps):
    CV = np.empty((n_reps, n_layers))
    HO = np.empty((n_reps, n_layers))
    for r in range(n_reps):
        v = f"rand{r}"
        y_sel, y_ho = a[f"{v}__y_sel"], a[f"{v}__y_ho"]
        fid = a[f"{v}__fold_id"]
        cv, ho = a[f"{v}__cv_scores"], a[f"{v}__ho_scores"]
        folds = sorted(set(fid.tolist()))
        for l in range(n_layers):
            CV[r, l] = np.mean([roc_auc_score(y_sel[fid == f], cv[l][fid == f])
                                for f in folds])
            HO[r, l] = roc_auc_score(y_ho, ho[l])
    return CV, HO


def stats_from(cv, ho):
    """A, B and Delta_sel for one set of replicates (rows)."""
    lstar = cv.argmax(axis=1)
    rows = np.arange(cv.shape[0])
    A = cv[rows, lstar] - cv.mean(axis=1)
    B = ho[rows, lstar] - ho.mean(axis=1)
    return A.mean(), B.mean(), (A - B).mean()


def effective_k(C):
    """Participation ratio of the eigenvalues: the number of independent
    candidates the correlation structure behaves like."""
    w = np.linalg.eigvalsh(C)
    w = w[w > 1e-12]
    return float((w.sum() ** 2) / (w ** 2).sum())


def main():
    a = np.load(ARTIFACT)
    n_layers, n_reps = int(a["n_layers"]), int(a["n_reps"])
    CV, HO = per_layer_auroc(a, n_layers, n_reps)

    C = np.corrcoef(CV.T)
    off = C[~np.eye(n_layers, dtype=bool)]
    adj = np.array([C[i, i + 1] for i in range(n_layers - 1)])
    K_eff = effective_k(C)

    obs_A, obs_B, obs_D = stats_from(CV, HO)

    mu = np.concatenate([CV.mean(axis=0), HO.mean(axis=0)])
    R = np.concatenate([CV - CV.mean(axis=0), HO - HO.mean(axis=0)], axis=1)

    rng = np.random.default_rng(SEED)
    out_dep, out_ind = np.empty((N_TRIALS, 3)), np.empty((N_TRIALS, 3))
    for t in range(N_TRIALS):
        # DEPENDENT: resample rows -> joint dependence preserved
        idx = rng.integers(0, n_reps, size=n_reps)
        X = mu + R[idx]
        out_dep[t] = stats_from(X[:, :n_layers], X[:, n_layers:])
        # INDEPENDENT: permute each column -> marginals kept, dependence gone
        Rp = np.column_stack([rng.permutation(R[:, j]) for j in range(R.shape[1])])
        X = mu + Rp
        out_ind[t] = stats_from(X[:, :n_layers], X[:, n_layers:])

    def summarise(arr, i):
        v = arr[:, i]
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                "ci_95": [float(np.quantile(v, .025)), float(np.quantile(v, .975))]}

    dep_A, ind_A = summarise(out_dep, 0), summarise(out_ind, 0)
    dep_B, ind_B = summarise(out_dep, 1), summarise(out_ind, 1)

    covers_dep = dep_A["ci_95"][0] <= obs_A <= dep_A["ci_95"][1]
    covers_ind = ind_A["ci_95"][0] <= obs_A <= ind_A["ci_95"][1]
    inflation = ind_A["mean"] / dep_A["mean"] if dep_A["mean"] else float("nan")

    out = {
        "n_layers": n_layers, "n_reps": n_reps, "n_trials": N_TRIALS, "seed": SEED,
        "dependence": {
            "mean_offdiagonal_correlation": float(off.mean()),
            "mean_adjacent_correlation": float(adj.mean()),
            "max_adjacent_correlation": float(adj.max()),
            "fraction_pairs_above_0.5": float((np.abs(off) > 0.5).mean()),
            "effective_K": K_eff, "nominal_K": n_layers,
            "sqrt_1_minus_rhobar": float(np.sqrt(max(0.0, 1 - off.mean()))),
        },
        "observed": {"A": float(obs_A), "B": float(obs_B), "delta_sel": float(obs_D)},
        "resampling_dependent": {"A": dep_A, "B": dep_B},
        "resampling_independent": {"A": ind_A, "B": ind_B},
        "verdict": {
            "dependent_ci_covers_observed_A": bool(covers_dep),
            "independent_ci_covers_observed_A": bool(covers_ind),
            "independence_inflation_factor": float(inflation),
            "kill_criterion_fired": bool(not covers_dep),
        },
    }

    print("=" * 92)
    print("CANDIDATE DEPENDENCE AND THE SIZE OF THE WINNER'S CURSE")
    print("=" * 92)
    print(f"\nDependence of the selection criterion across {n_layers} layers:")
    print(f"  mean off-diagonal correlation   {off.mean():+.3f}")
    print(f"  mean adjacent-layer correlation {adj.mean():+.3f}  (max {adj.max():+.3f})")
    print(f"  effective K (participation)     {K_eff:.2f}  of a nominal {n_layers}")
    print(f"  sqrt(1 - rho_bar)               {np.sqrt(max(0,1-off.mean())):.3f}"
          f"   <- predicted shrinkage of the curse")

    print(f"\nWinner's curse A, observed: {obs_A:+.5f}")
    print(f"  resampling WITH dependence    {dep_A['mean']:+.5f}  "
          f"95% [{dep_A['ci_95'][0]:+.5f},{dep_A['ci_95'][1]:+.5f}]  "
          f"covers observed: {covers_dep}")
    print(f"  resampling WITHOUT dependence {ind_A['mean']:+.5f}  "
          f"95% [{ind_A['ci_95'][0]:+.5f},{ind_A['ci_95'][1]:+.5f}]  "
          f"covers observed: {covers_ind}")
    print(f"\n  independence inflates the curse by {inflation:.2f}x")

    print(f"\nTransferred quality B, observed: {obs_B:+.5f}")
    print(f"  with dependence    {dep_B['mean']:+.5f}")
    print(f"  without dependence {ind_B['mean']:+.5f}")

    print("\n" + ("KILL CRITERION FIRED: dependence does NOT explain the curse."
                  if out["verdict"]["kill_criterion_fired"] else
                  "Kill criterion not fired: the dependent model reproduces the "
                  "observed curse."))
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
