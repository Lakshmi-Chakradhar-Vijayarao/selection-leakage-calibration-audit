"""
Mechanism 2 reviewer-response analyses (TAE 2026 camera-ready).

Addresses three reviewer criticisms, all from the SAME cached artifact
(results/case_study_2_probe_scores.npz) that produced the submitted numbers.
No new inference, no new model calls.

(1) PLACEBO CONTROL  [Reviewer zDz3, major weakness 1]
    The submitted LEAKY-vs-CLEAN_MATCHED gap uses the argmax-selected layer on
    two different samples, so it may combine a selection effect with an
    ordinary reused-vs-held-out difference that would appear for ANY layer.
    We therefore compute the same gap at every one of the 32 layers, and
    report:
      - gap at l_star (the submitted quantity)
      - gap averaged over all 32 layers (the "prespecified layer" placebo)
      - gap at a layer drawn at random per rep (fixed seed)
      - the paired selection-specific increment  gap(l_star) - gap(placebo)
    A near-zero placebo supports the selection attribution; a large placebo
    means only part of the headline gap is selection-specific.

(2) BRIER DOES NOT ISOLATE CALIBRATION  [zDz3 weakness 2; DUxq concern 1]
    Murphy decomposition  BS = REL - RES + UNC.  A Brier gap can arise through
    resolution (discrimination), which is exactly what argmax-AUROC selection
    optimises. We report REL, RES and UNC separately for LEAKY and CLEAN so the
    reliability component can be read on its own, plus calibration
    slope/intercept and a binning-free smooth calibration error.

(3) ECE IS BIN-SENSITIVE  [zDz3 weakness 2; DUxq concern 1]
    ECE swept over bin counts {5,10,15,20} and both equal-width and equal-mass
    binning schemes.

All gaps are paired across the 50 randomized splits and carry the same BCa 95%
bootstrap the rest of the paper uses.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "results" / "case_study_2_probe_scores.npz"
REF_JSON = ROOT / "results" / "case_study_2_layer_decomposition.json"
OUT_PATH = ROOT / "results" / "reviewer_response_mechanism2.json"

SEED = 20260925
RNG = np.random.default_rng(SEED)
EPS = 1e-6


# --------------------------------------------------------------------------- #
# calibration estimators
# --------------------------------------------------------------------------- #
def _bin_edges(scores, n_bins, scheme):
    if scheme == "width":
        return np.linspace(0.0, 1.0, n_bins + 1)
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(np.asarray(scores, dtype=float), q)
    edges[0], edges[-1] = 0.0, 1.0
    return np.unique(edges)


def ece_binned(scores, labels, n_bins=10, scheme="width"):
    """Expected calibration error. scheme in {'width','mass'}."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    edges = _bin_edges(s, n_bins, scheme)
    n = len(s)
    ece = 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (s >= lo) & (s <= hi) if i == len(edges) - 2 else (s >= lo) & (s < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y[mask].mean() - s[mask].mean())
    return float(ece)


def murphy_decomposition(scores, labels, n_bins=10, scheme="width"):
    """BS = REL - RES + UNC.  Lower REL is better; higher RES is better."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    n = len(s)
    obar = y.mean()
    edges = _bin_edges(s, n_bins, scheme)
    rel = res = 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (s >= lo) & (s <= hi) if i == len(edges) - 2 else (s >= lo) & (s < hi)
        nk = int(mask.sum())
        if nk == 0:
            continue
        pbar_k = s[mask].mean()
        obar_k = y[mask].mean()
        rel += nk * (pbar_k - obar_k) ** 2
        res += nk * (obar_k - obar) ** 2
    rel /= n
    res /= n
    unc = obar * (1.0 - obar)
    return float(rel), float(res), float(unc)


def calibration_slope_intercept(scores, labels):
    """Logistic recalibration: logit(y) ~ intercept + slope * logit(p).
    Perfect calibration is slope=1, intercept=0. Slope>1 => underconfident,
    slope<1 => overconfident."""
    s = np.clip(np.asarray(scores, dtype=float), EPS, 1 - EPS)
    y = np.asarray(labels, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    x = np.log(s / (1 - s)).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    lr.fit(x, y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def smooth_calibration_error(scores, labels):
    """Binning-free calibration error: mean squared deviation between the
    predicted probability and an isotonic recalibration of it. Reported as a
    bin-insensitive companion to ECE; note the isotonic fit is in-sample, so
    this is a lower bound on calibration error."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    p_cal = iso.fit_transform(s, y)
    return float(np.mean((s - p_cal) ** 2))


# --------------------------------------------------------------------------- #
# bootstrap
# --------------------------------------------------------------------------- #
def bca_ci(diff, seed=SEED):
    d = np.asarray(diff, dtype=float)
    if np.allclose(d, d[0]):
        return float(d[0]), float(d[0])
    res = bootstrap((d,), np.mean, confidence_level=0.95, n_resamples=10000,
                    method="BCa", random_state=np.random.default_rng(seed))
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def summarise(diff, seed=SEED):
    d = np.asarray(diff, dtype=float)
    lo, hi = bca_ci(d, seed)
    return {"mean": float(d.mean()), "bca_95ci": [lo, hi],
            "excludes_zero": bool(lo > 0 or hi < 0)}


# --------------------------------------------------------------------------- #
def per_rep_arrays(a, rep, n_layers):
    v = f"rand{rep}"
    y_sel = a[f"{v}__y_sel"]
    y_ho = a[f"{v}__y_ho"]
    fold_id = a[f"{v}__fold_id"]
    cv_scores = a[f"{v}__cv_scores"]
    ho_scores = a[f"{v}__ho_scores"]
    folds = sorted(set(fold_id.tolist()))
    cv_auroc = np.array([
        np.mean([roc_auc_score(y_sel[fold_id == f], cv_scores[l][fold_id == f])
                 for f in folds]) for l in range(n_layers)])
    return cv_auroc, y_sel, y_ho, cv_scores, ho_scores


def main():
    a = np.load(ARTIFACT)
    n_layers = int(a["n_layers"])
    n_reps = int(a["n_reps"])
    ref = json.load(open(REF_JSON))["randomized_stratified_splits"]

    bin_settings = [(nb, sch) for sch in ("width", "mass") for nb in (5, 10, 15, 20)]

    # per-rep, per-layer stores
    brier_L = np.zeros((n_reps, n_layers))
    brier_C = np.zeros((n_reps, n_layers))
    rel_L = np.zeros((n_reps, n_layers))
    rel_C = np.zeros((n_reps, n_layers))
    res_L = np.zeros((n_reps, n_layers))
    res_C = np.zeros((n_reps, n_layers))
    unc_L = np.zeros((n_reps, n_layers))
    unc_C = np.zeros((n_reps, n_layers))
    sce_L = np.zeros((n_reps, n_layers))
    sce_C = np.zeros((n_reps, n_layers))
    slope_L = np.zeros((n_reps, n_layers))
    slope_C = np.zeros((n_reps, n_layers))
    icpt_L = np.zeros((n_reps, n_layers))
    icpt_C = np.zeros((n_reps, n_layers))
    ece_L = {k: np.zeros((n_reps, n_layers)) for k in bin_settings}
    ece_C = {k: np.zeros((n_reps, n_layers)) for k in bin_settings}

    l_stars, base_sel, base_ho = [], [], []

    for r in range(n_reps):
        cv_auroc, y_sel, y_ho, cv_scores, ho_scores = per_rep_arrays(a, r, n_layers)
        l_stars.append(int(cv_auroc.argmax()))
        base_sel.append(float(np.mean(y_sel)))
        base_ho.append(float(np.mean(y_ho)))

        for l in range(n_layers):
            sL, sC = cv_scores[l], ho_scores[l]
            brier_L[r, l] = brier_score_loss(y_sel, sL)
            brier_C[r, l] = brier_score_loss(y_ho, sC)
            rel_L[r, l], res_L[r, l], unc_L[r, l] = murphy_decomposition(sL, y_sel)
            rel_C[r, l], res_C[r, l], unc_C[r, l] = murphy_decomposition(sC, y_ho)
            sce_L[r, l] = smooth_calibration_error(sL, y_sel)
            sce_C[r, l] = smooth_calibration_error(sC, y_ho)
            slope_L[r, l], icpt_L[r, l] = calibration_slope_intercept(sL, y_sel)
            slope_C[r, l], icpt_C[r, l] = calibration_slope_intercept(sC, y_ho)
            for (nb, sch) in bin_settings:
                ece_L[(nb, sch)][r, l] = ece_binned(sL, y_sel, nb, sch)
                ece_C[(nb, sch)][r, l] = ece_binned(sC, y_ho, nb, sch)

    assert l_stars == list(ref["selected_layer_per_rep"]), \
        "selected layers differ from code/64 reference -- replication failed"
    print(f"Layer-selection replication OK across {n_reps} reps.\n")

    reps = np.arange(n_reps)
    ls = np.array(l_stars)
    rnd = RNG.integers(0, n_layers, size=n_reps)

    def at(mat, idx):
        return mat[reps, idx]

    def gap(matC, matL, idx):
        """clean - leaky; positive = LEAKY looks better than it honestly is."""
        return at(matC, idx) - at(matL, idx)

    def all_layer_mean_gap(matC, matL):
        return (matC - matL).mean(axis=1)

    out = {
        "seed": SEED,
        "n_reps": n_reps,
        "n_layers": n_layers,
        "selected_layer_per_rep": l_stars,
        "random_placebo_layer_per_rep": rnd.tolist(),
        "base_rate_sel_mean": float(np.mean(base_sel)),
        "base_rate_ho_mean": float(np.mean(base_ho)),
        "note_base_rate": "Murphy UNC = obar(1-obar); LEAKY and CLEAN base rates are "
                          "matched to within the value reported here, so the REL/RES "
                          "comparison is not confounded by an uncertainty-term shift.",
    }

    # ---------------- (1) placebo control -------------------------------- #
    placebo = {}
    for name, matC, matL in (("brier", brier_C, brier_L),
                             ("reliability", rel_C, rel_L),
                             ("resolution", res_C, res_L),
                             ("ece_10_width", ece_C[(10, "width")], ece_L[(10, "width")]),
                             ("smooth_calibration_error", sce_C, sce_L)):
        g_sel = gap(matC, matL, ls)
        g_allmean = all_layer_mean_gap(matC, matL)
        g_rand = gap(matC, matL, rnd)
        placebo[name] = {
            "gap_at_selected_layer": summarise(g_sel),
            "placebo_gap_all_layer_mean": summarise(g_allmean),
            "placebo_gap_random_layer": summarise(g_rand),
            "selection_specific_increment_vs_all_layer_mean": summarise(g_sel - g_allmean),
            "selection_specific_increment_vs_random_layer": summarise(g_sel - g_rand),
            "share_of_gap_that_is_selection_specific":
                float((g_sel.mean() - g_allmean.mean()) / g_sel.mean())
                if abs(g_sel.mean()) > 1e-12 else None,
        }
    out["placebo_control"] = placebo

    # ---------------- (2) Brier decomposition ---------------------------- #
    out["brier_decomposition_at_selected_layer"] = {
        "brier_leaky_mean": float(at(brier_L, ls).mean()),
        "brier_clean_mean": float(at(brier_C, ls).mean()),
        "reliability_leaky_mean": float(at(rel_L, ls).mean()),
        "reliability_clean_mean": float(at(rel_C, ls).mean()),
        "resolution_leaky_mean": float(at(res_L, ls).mean()),
        "resolution_clean_mean": float(at(res_C, ls).mean()),
        "uncertainty_leaky_mean": float(at(unc_L, ls).mean()),
        "uncertainty_clean_mean": float(at(unc_C, ls).mean()),
        "reliability_gap_clean_minus_leaky": summarise(gap(rel_C, rel_L, ls)),
        "resolution_gap_leaky_minus_clean": summarise(at(res_L, ls) - at(res_C, ls)),
        "uncertainty_gap_clean_minus_leaky": summarise(gap(unc_C, unc_L, ls)),
        "interpretation": "A positive reliability gap means the held-out predictions are "
                          "less reliable than the reused-fold predictions, i.e. a genuine "
                          "calibration effect. A positive resolution gap means part of the "
                          "Brier difference is discrimination, which argmax-AUROC selection "
                          "optimises directly and which is therefore not a calibration claim.",
    }

    # ---------------- (3) ECE sweep -------------------------------------- #
    sweep = {}
    for (nb, sch) in bin_settings:
        g = gap(ece_C[(nb, sch)], ece_L[(nb, sch)], ls)
        sweep[f"{sch}_{nb}bins"] = {
            "ece_leaky_mean": float(at(ece_L[(nb, sch)], ls).mean()),
            "ece_clean_mean": float(at(ece_C[(nb, sch)], ls).mean()),
            **summarise(g),
        }
    out["ece_bin_sensitivity_at_selected_layer"] = sweep
    out["ece_sweep_all_settings_exclude_zero"] = bool(
        all(v["excludes_zero"] for v in sweep.values()))

    # ---------------- calibration slope / intercept ---------------------- #
    out["calibration_slope_intercept_at_selected_layer"] = {
        "slope_leaky_mean": float(np.nanmean(at(slope_L, ls))),
        "slope_clean_mean": float(np.nanmean(at(slope_C, ls))),
        "intercept_leaky_mean": float(np.nanmean(at(icpt_L, ls))),
        "intercept_clean_mean": float(np.nanmean(at(icpt_C, ls))),
        "slope_gap_leaky_minus_clean": summarise(at(slope_L, ls) - at(slope_C, ls)),
        "interpretation": "Slope 1 / intercept 0 is perfect calibration; slope below 1 "
                          "indicates overconfidence. Reported because it is a "
                          "calibration-specific statistic that does not depend on binning.",
    }

    # ---------------- smooth calibration error --------------------------- #
    out["smooth_calibration_error_at_selected_layer"] = {
        "sce_leaky_mean": float(at(sce_L, ls).mean()),
        "sce_clean_mean": float(at(sce_C, ls).mean()),
        "sce_gap_clean_minus_leaky": summarise(gap(sce_C, sce_L, ls)),
        "caveat": "Isotonic fit is in-sample, so this underestimates absolute calibration "
                  "error; it is used only for the LEAKY-vs-CLEAN comparison, where the "
                  "bias applies to both arms.",
    }

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    # ---------------- console report ------------------------------------- #
    def fmt(d):
        return f"{d['mean']:+.5f}  BCa95 [{d['bca_95ci'][0]:+.5f},{d['bca_95ci'][1]:+.5f}]" \
               f"{'  *' if d['excludes_zero'] else '   (crosses 0)'}"

    print("=" * 78)
    print("(1) PLACEBO CONTROL  -- is the gap selection-specific?")
    print("=" * 78)
    for name, d in placebo.items():
        print(f"\n{name}:")
        print(f"  gap @ selected layer      {fmt(d['gap_at_selected_layer'])}")
        print(f"  placebo: all-layer mean   {fmt(d['placebo_gap_all_layer_mean'])}")
        print(f"  placebo: random layer     {fmt(d['placebo_gap_random_layer'])}")
        print(f"  selection-specific (vs all-layer) {fmt(d['selection_specific_increment_vs_all_layer_mean'])}")
        s = d["share_of_gap_that_is_selection_specific"]
        if s is not None:
            print(f"  share of gap that is selection-specific: {s*100:.1f}%")

    print("\n" + "=" * 78)
    print("(2) BRIER DECOMPOSITION  BS = REL - RES + UNC   (at selected layer)")
    print("=" * 78)
    b = out["brier_decomposition_at_selected_layer"]
    print(f"  LEAKY : BS={b['brier_leaky_mean']:.5f}  REL={b['reliability_leaky_mean']:.5f}  "
          f"RES={b['resolution_leaky_mean']:.5f}  UNC={b['uncertainty_leaky_mean']:.5f}")
    print(f"  CLEAN : BS={b['brier_clean_mean']:.5f}  REL={b['reliability_clean_mean']:.5f}  "
          f"RES={b['resolution_clean_mean']:.5f}  UNC={b['uncertainty_clean_mean']:.5f}")
    print(f"  RELIABILITY gap (clean-leaky)  {fmt(b['reliability_gap_clean_minus_leaky'])}")
    print(f"  RESOLUTION  gap (leaky-clean)  {fmt(b['resolution_gap_leaky_minus_clean'])}")
    print(f"  UNCERTAINTY gap (clean-leaky)  {fmt(b['uncertainty_gap_clean_minus_leaky'])}")

    print("\n" + "=" * 78)
    print("(3) ECE BIN SENSITIVITY  (at selected layer)")
    print("=" * 78)
    for k, v in sweep.items():
        print(f"  {k:16s} leaky={v['ece_leaky_mean']:.5f} clean={v['ece_clean_mean']:.5f}  "
              f"gap {v['mean']:+.5f} [{v['bca_95ci'][0]:+.5f},{v['bca_95ci'][1]:+.5f}]"
              f"{'  *' if v['excludes_zero'] else '   (crosses 0)'}")
    print(f"\n  All {len(sweep)} bin settings exclude zero: "
          f"{out['ece_sweep_all_settings_exclude_zero']}")

    c = out["calibration_slope_intercept_at_selected_layer"]
    print("\n" + "=" * 78)
    print("CALIBRATION SLOPE / INTERCEPT  (binning-free)")
    print("=" * 78)
    print(f"  LEAKY slope={c['slope_leaky_mean']:.4f}  intercept={c['intercept_leaky_mean']:+.4f}")
    print(f"  CLEAN slope={c['slope_clean_mean']:.4f}  intercept={c['intercept_clean_mean']:+.4f}")
    print(f"  slope gap (leaky-clean)  {fmt(c['slope_gap_leaky_minus_clean'])}")

    s = out["smooth_calibration_error_at_selected_layer"]
    print(f"\n  Smooth calibration error: leaky={s['sce_leaky_mean']:.5f} "
          f"clean={s['sce_clean_mean']:.5f}  gap {fmt(s['sce_gap_clean_minus_leaky'])}")

    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
