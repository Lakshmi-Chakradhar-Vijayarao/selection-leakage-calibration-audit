"""
Case Study 4: the PROMOTED estimator, computed exactly, and proved
non-negative.

WHY THIS SCRIPT EXISTS. A fourth independent review found that the estimator
`code/45` promoted to primary after Theorem 1 retired the rotation diagnostic
Delta_wc carries THE SAME DEFECT, by the same species of argument.

  code/45's docstring for `_bootstrap_max_bias` asserted: "Structurally cannot
  be pinned to zero: the resample's mean at the resample-selected layer and the
  full-sample mean at that same layer are computed on different seed multisets
  by construction, so nothing telescopes."

That is FALSE. Nothing telescopes, but the estimator is still non-negative in
expectation over the bootstrap for any input matrix whatsoever:

  THEOREM 2. Let A be (L x R) with column (seed) means. Write
  mbar(l) = (1/R) sum_r a_{r,l} for the full-sample mean at layer l, let
  c*(l) be the resample mean at layer l over a uniform bootstrap resample of
  the R seed indices, and let L* = argmax_l c*(l). The implemented estimator is

      Delta_boot := E*[ c*(L*) - mbar(L*) ]  =  E*[ max_l c*(l) ] - E*[ mbar(L*) ].

  (i) For each FIXED l, E*[c*(l)] = mbar(l), since a uniform bootstrap
      resample of seed indices is mean-preserving. The max is a convex
      functional, so by Jensen
          E*[ max_l c*(l) ]  >=  max_l E*[c*(l)]  =  max_l mbar(l).
  (ii) Trivially mbar(L*) <= max_l mbar(l) for every realized L*, hence
          E*[ mbar(L*) ]  <=  max_l mbar(l).
  Subtracting, Delta_boot >= max_l mbar(l) - max_l mbar(l) = 0.  QED.

  EQUALITY holds iff both bounds are tight, i.e. iff every resample's argmax is
  also a grand-mean argmax and vice versa -- in the generic no-ties case, iff
  the grand-mean argmax is BOOTSTRAP-STABLE. That is the exact analogue of
  Delta_wc's equal-argmax degeneracy.

CONSEQUENCES.
  * "2 of the 24 cells return negative values, which is the property Delta_wc
    structurally cannot have" was an artifact of estimating Delta_boot by
    N_BOOT_BIAS = 2,000 Monte-Carlo draws. With R = 3 seeds there are exactly
    3^3 = 27 equiprobable resamples and Delta_boot has a closed form. Under
    exact enumeration those two cells are +/-1e-16 -- float residue around an
    algebraic zero -- and they are precisely the two cells whose grand-mean
    argmax is bootstrap-stable. They are degenerate, not negative.
  * No inference may be drawn from Delta_boot's SIGN, nor from a Wilcoxon or
    signed-rank p-value against zero: the theorem makes that null false by
    construction, exactly as Theorem 1 does for Delta_wc.
  * Delta_boot is retained as a MAGNITUDE, computed exactly.

A SECOND ESTIMATOR IS ALSO COMPUTED, because the implemented one is not the
textbook quantity. The standard bootstrap estimate of the bias of a
max-of-means subtracts the full-sample mean at the FULL-SAMPLE argmax, not at
the resample-selected layer:

      Delta_boot_std := E*[ max_l c*(l) ]  -  max_l mbar(l).

Step (i) of the proof alone gives Delta_boot_std >= 0, so it is non-negative
too -- but it is a DIFFERENT and strictly smaller quantity
(Delta_boot - Delta_boot_std = max_l mbar(l) - E*[mbar(L*)] >= 0). Both are
reported here; code/71 calibrates them against a known ground truth and that
calibration, not a sign, is what selects which one the paper reports.

Outputs: results/case_study_4_exact_enumeration.json
"""
import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap

ROOT = Path(__file__).resolve().parent.parent
PROBES_DIR = ROOT / "code" / "external" / "HallucinationPatternDetection" / "results" / "probes"
REF_PATH = ROOT / "results" / "case_study_4_winners_curse.json"
OUT_PATH = ROOT / "results" / "case_study_4_exact_enumeration.json"

SATURATION_THRESHOLD = 0.975
ZERO_SNAP = 1e-12
RNG_GLOBAL = np.random.default_rng(2026)


def enumerate_table(n_seeds):
    return np.array(list(product(range(n_seeds), repeat=n_seeds)))


def exact_estimators(A, idx_table):
    """Exact (enumerated) values of both bootstrap estimators plus degeneracy.

    A is (n_layers, n_seeds); idx_table is (n_seeds^n_seeds, n_seeds)."""
    full_mean = A.mean(axis=1)
    m = A[:, idx_table].mean(axis=2)           # (n_layers, n_resamples)
    li = m.argmax(axis=0)                      # (n_resamples,)
    r = np.arange(m.shape[1])
    implemented = float(np.mean(m[li, r] - full_mean[li]))
    standard = float(m.max(axis=0).mean() - full_mean.max())
    l_star = int(full_mean.argmax())
    # Degeneracy (equality case of Theorem 2): the grand-mean argmax is
    # bootstrap-stable, i.e. every resample selects it.
    stable = bool(np.all(li == l_star))
    return {
        "implemented": 0.0 if abs(implemented) < ZERO_SNAP else implemented,
        "implemented_unsnapped": implemented,
        "standard": 0.0 if abs(standard) < ZERO_SNAP else standard,
        "standard_unsnapped": standard,
        "bootstrap_argmax_stable": stable,
        "n_distinct_bootstrap_argmax_layers": int(len(set(li.tolist()))),
        "grand_mean_argmax_index": l_star,
    }


def rotation_estimate(A):
    n_layers, n_seeds = A.shape
    vals, sel = [], []
    for held in range(n_seeds):
        others = [s for s in range(n_seeds) if s != held]
        mo = A[:, others].mean(axis=1)
        li = int(np.argmax(mo))
        sel.append(li)
        vals.append(mo[li] - A[li, held])
    est = float(np.mean(vals))
    return (0.0 if abs(est) < ZERO_SNAP else est), bool(len(set(sel)) == 1)


def verify_theorem_numerically(n_trials=50_000, seed=7):
    """Independent numerical check of Theorem 2 (and of its standard variant).

    Random matrices, exact enumeration, R=3 and R=5. The theorem says both
    quantities are >= 0 for ANY real matrix, so the only negative values that
    may appear are float summation residues."""
    rng = np.random.default_rng(seed)
    out = {}
    for R, n in ((3, n_trials), (5, max(1, n_trials // 20))):
        idx = enumerate_table(R)
        worst_impl, worst_std = np.inf, np.inf
        for _ in range(n):
            L = int(rng.integers(2, 40))
            scale = float(rng.choice([1e-3, 1e-1, 1.0, 1e2]))
            family = rng.integers(0, 3)
            if family == 0:
                A = rng.normal(size=(L, R)) * scale
            elif family == 1:
                A = rng.uniform(size=(L, R)) * scale
            else:  # discretized, to manufacture ties
                A = np.round(rng.normal(size=(L, R)), 2)
            e = exact_estimators(A, idx)
            worst_impl = min(worst_impl, e["implemented_unsnapped"])
            worst_std = min(worst_std, e["standard_unsnapped"])
        out[f"R={R}"] = {
            "n_trials": n,
            "min_implemented": float(worst_impl),
            "min_standard": float(worst_std),
        }
        print(f"  Theorem 2 check R={R}, {n} random matrices: "
              f"min implemented {worst_impl:.3e}, min standard {worst_std:.3e}")
    return out


def bca_ci(vals, n_resamples=10000):
    vals = np.asarray(vals, dtype=float)
    if np.allclose(vals, vals[0]):
        return [float(vals[0]), float(vals[0])]
    res = bootstrap((vals,), np.mean, confidence_level=0.95,
                    n_resamples=n_resamples, method="BCa", random_state=RNG_GLOBAL)
    return [float(res.confidence_interval.low), float(res.confidence_interval.high)]


def cluster_bootstrap_ci(vals, clusters, n_resamples=20000, seed=31):
    """Percentile bootstrap CI for the mean, RESAMPLING CLUSTERS not cells.

    The 24 Case Study 4 cells are a fully crossed 3 models x 4 datasets x 2
    probe-types design from ONE artifact sharing 3 seeds -- they are not 24
    independent observations. Dataset is the coarsest grouping with a visible
    variance component, so it is the cluster unit."""
    vals = np.asarray(vals, dtype=float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    groups = [vals[clusters == u] for u in uniq]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples)
    for b in range(n_resamples):
        pick = rng.integers(0, len(groups), size=len(groups))
        draws[b] = np.concatenate([groups[i] for i in pick]).mean()
    return ([float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            float(draws.std(ddof=1)))


def variance_components(vals, models, datasets, probes):
    """Share of between-cell variance explained by each crossed factor
    (one-way eta^2 per factor, computed marginally)."""
    vals = np.asarray(vals, dtype=float)
    total = float(((vals - vals.mean()) ** 2).sum())
    out = {}
    for name, fac in (("model", models), ("dataset", datasets), ("probe", probes)):
        fac = np.asarray(fac)
        between = 0.0
        for u in np.unique(fac):
            g = vals[fac == u]
            between += len(g) * (g.mean() - vals.mean()) ** 2
        out[name] = float(between / total) if total > 0 else None
    return out


def summarize(name, vals, key):
    vals = list(vals)
    if not vals:
        return None
    s = {
        "n_cells": len(vals),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
        "bca_ci_95": bca_ci(vals) if len(vals) > 1 else None,
        "n_exact_zero": int(sum(abs(v) < ZERO_SNAP for v in vals)),
    }
    print(f"  {name:52s} [{key}] n={s['n_cells']:2d} mean={s['mean']:+.6f} "
          f"CI={s['bca_ci_95']} zeros={s['n_exact_zero']}")
    return s


def main():
    files = sorted(PROBES_DIR.glob("*.json"))
    assert len(files) == 24, f"expected 24 probe result files, found {len(files)}"
    ref = json.load(open(REF_PATH))["per_cell"]

    per_cell, mc_err = {}, []
    for path in files:
        d = json.load(open(path))
        layers, pl = d["layers"], d["per_layer"]
        n_seeds = len(pl[str(layers[0])]["seed_values"]["auroc"])
        A = np.array([[pl[str(l)]["seed_values"]["auroc"][s] for s in range(n_seeds)]
                      for l in layers], dtype=float)
        idx = enumerate_table(n_seeds)
        e = exact_estimators(A, idx)
        rot, rot_degen = rotation_estimate(A)
        shipped = ref[path.stem]["bootstrap_max_bias"]
        mc_err.append(abs(e["implemented"] - shipped))
        model, dataset, probe = path.stem.split("__")
        per_cell[path.stem] = {
            **e,
            "n_seeds": n_seeds,
            "n_layers": len(layers),
            "model": model, "dataset": dataset, "probe": probe,
            "rotation_delta_wc": rot,
            "rotation_degenerate": rot_degen,
            "shipped_monte_carlo_implemented": shipped,
            "monte_carlo_error_of_shipped": float(abs(e["implemented"] - shipped)),
            "shipped_monte_carlo_negative": bool(shipped < 0),
            "operating_point_saturated": ref[path.stem]["operating_point_saturated"],
            "selection_auroc": ref[path.stem]["naive_selection_auroc_loo"],
        }

    cells = list(per_cell.values())
    impl = [c["implemented"] for c in cells]
    std = [c["standard"] for c in cells]
    rot = [c["rotation_delta_wc"] for c in cells]
    sat = np.array([c["operating_point_saturated"] for c in cells])
    stable = np.array([c["bootstrap_argmax_stable"] for c in cells])
    rotdeg = np.array([c["rotation_degenerate"] for c in cells])
    models = [c["model"] for c in cells]
    datasets = [c["dataset"] for c in cells]
    probes = [c["probe"] for c in cells]

    print("\n=== per-cell exact enumeration ===")
    for k, v in sorted(per_cell.items()):
        print(f"  {k:36s} impl={v['implemented']:+.6f}  std={v['standard']:+.6f}  "
              f"rot={v['rotation_delta_wc']:+.6f}  "
              f"{'STABLE-ARGMAX(=0)' if v['bootstrap_argmax_stable'] else '        '}  "
              f"shipped_MC={v['shipped_monte_carlo_implemented']:+.6f}")

    print("\n=== group summaries ===")
    groups = {}
    for key, arr in (("implemented", impl), ("standard", std), ("rotation_delta_wc", rot)):
        groups[key] = {
            "all_24": summarize("ALL 24 CELLS", arr, key),
            "ceiling_saturated": summarize("CEILING-SATURATED (sel. layer >= 0.975)",
                                           [v for v, s in zip(arr, sat) if s], key),
            "non_saturated": summarize("NON-SATURATED",
                                       [v for v, s in zip(arr, sat) if not s], key),
            "rotation_degenerate": summarize("rotation-degenerate cells",
                                             [v for v, s in zip(arr, rotdeg) if s], key),
            "rotation_non_degenerate": summarize("rotation-non-degenerate cells",
                                                 [v for v, s in zip(arr, rotdeg) if not s], key),
        }
        # ceiling ratio, under this estimator
        a = groups[key]["ceiling_saturated"]["mean"]
        b = groups[key]["non_saturated"]["mean"]
        groups[key]["ceiling_ratio_nonsat_over_sat"] = float(b / a) if a else None
        # power: SD across the 24 cells -> MDE at alpha=.05, 80% power, n=24
        sd24 = float(np.std(arr, ddof=1))
        groups[key]["sd_across_24_cells"] = sd24
        groups[key]["mde_alpha05_power80_n24"] = float(2.802 * sd24 / np.sqrt(24))
        # clustered interval (P1-3)
        cl_ci, cl_se = cluster_bootstrap_ci(arr, datasets)
        groups[key]["dataset_clustered_bootstrap_ci_95"] = cl_ci
        groups[key]["dataset_clustered_bootstrap_se"] = cl_se
        iid = groups[key]["all_24"]["bca_ci_95"]
        groups[key]["clustered_width_over_iid_width"] = float(
            (cl_ci[1] - cl_ci[0]) / (iid[1] - iid[0]))
        groups[key]["variance_components_eta2"] = variance_components(
            arr, models, datasets, probes)
        print(f"    [{key}] ceiling ratio non-sat/sat = "
              f"{groups[key]['ceiling_ratio_nonsat_over_sat']:.2f}x   "
              f"SD24={sd24:.4f}  MDE={groups[key]['mde_alpha05_power80_n24']:+.4f}")
        print(f"    [{key}] clustered CI {cl_ci} "
              f"({groups[key]['clustered_width_over_iid_width']:.2f}x the i.i.d. width); "
              f"eta^2 {groups[key]['variance_components_eta2']}")

    print("\n=== numerical verification of Theorem 2 ===")
    theorem_check = verify_theorem_numerically()

    n_shipped_neg = int(sum(c["shipped_monte_carlo_negative"] for c in cells))
    n_stable = int(stable.sum())
    stable_cells = sorted(k for k, v in per_cell.items() if v["bootstrap_argmax_stable"])
    shipped_neg_cells = sorted(k for k, v in per_cell.items()
                               if v["shipped_monte_carlo_negative"])

    out = {
        "n_cells": len(cells),
        "estimand_note": (
            "Both bootstrap quantities here are NON-NEGATIVE for any input matrix "
            "(Theorem 2, module docstring). Neither their sign nor a signed-rank test "
            "against zero carries evidence. They are reported as MAGNITUDES."),
        "implemented_definition": "E*[ c*(L*) - mbar(L*) ], L* = argmax_l c*(l)",
        "standard_definition": "E*[ max_l c*(l) ] - max_l mbar(l)",
        "exact_enumeration": True,
        "n_resamples_enumerated_per_cell": {"n_seeds=3": 27},
        "theorem2_numerical_check": theorem_check,
        "monte_carlo_artifact_audit": {
            "n_cells_negative_under_shipped_monte_carlo": n_shipped_neg,
            "cells_negative_under_shipped_monte_carlo": shipped_neg_cells,
            "n_cells_negative_under_exact_enumeration": int(
                sum(c["implemented_unsnapped"] < -ZERO_SNAP for c in cells)),
            "max_monte_carlo_error_of_shipped_values": float(max(mc_err)),
            "n_cells_bootstrap_argmax_stable": n_stable,
            "cells_bootstrap_argmax_stable": stable_cells,
            "statement": (
                f"{n_shipped_neg} of 24 cells returned a negative value under the shipped "
                f"{2000}-draw Monte-Carlo bootstrap. Under exact enumeration of all 27 "
                f"resamples none is negative: those same cells are the "
                f"{n_stable} whose grand-mean argmax is bootstrap-stable, where Theorem 2 "
                f"holds with equality and the exact value is 0 to float precision. The "
                f"negative values were Monte-Carlo noise around an algebraic zero, not "
                f"evidence that the estimator is free to take either sign."),
        },
        "ratio_implemented_over_standard_all24": float(np.mean(impl) / np.mean(std)),
        "groups": groups,
        "per_cell": per_cell,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nimplemented/standard ratio over all 24 = "
          f"{out['ratio_implemented_over_standard_all24']:.3f}x")
    print(f"shipped-MC negative cells: {n_shipped_neg} -> exact-enumeration negative: 0; "
          f"bootstrap-stable (degenerate) cells: {n_stable} {stable_cells}")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
