"""
withdrawal of the "48.6x" operating-point multiplier.

An independent review observed that the headline ratio

    gap(AUROC_0 = 0.70) / gap(AUROC_0 = 0.985)  =  0.009316 / 0.000192  =  48.6

has NO FINITE CONFIDENCE INTERVAL, because its denominator is not
distinguishable from zero (Wilcoxon p = 0.2186, n = 100 seeds). This script
regenerates the two endpoint cells of `code/47`'s Sweep C at the per-seed level
-- which the shipped `selection_multiplicity_sweep.json` summarizes but does
not store -- and quantifies the problem three ways:

  (1) FIELLER'S THEOREM. The 95% confidence set for a ratio of means is
      unbounded exactly when
          g = (t_.975 * SE_denominator / denominator)^2  >=  1.
      The t quantile is used consistently on both sides of this expression;
      an earlier revision mixed the normal and t quantiles (see below).
      Computed both from the shipped BCa halfwidth and from the regenerated
      per-seed SD.

  (2) A PAIRED BOOTSTRAP OF THE RATIO. Resample seeds (paired across the two
      cells, since both use the same seed sequence), recompute the ratio, and
      report the fraction of resamples whose denominator is <= 0 together with
      the percentile interval.

  (3) THE DENOMINATOR CELL ON ITS OWN TERMS. Sign test, Wilcoxon, and the
      ratio of the per-seed SD to the mean.

Reproducibility: `code/47`'s per-seed sub-seeds are pure functions of the loop
index (data_seed = i, split_seed = i + 100000, fold_seed_base = i + 200000,
init_seed_base = i + 300000), so this script reproduces the exact same 100
seeds and its cell means must match the shipped JSON to float tolerance. That
equality is asserted, not assumed.

CONCLUSION CARRIED INTO THE PAPER: the monotone decline of the gap across the
operating point survives untouched -- it is measured, ordered and (for three of
five cells) individually significant. What is withdrawn is the MULTIPLIER. The
paper now says the gap "falls to a level indistinguishable from zero
(p = 0.219, n = 100)" and reports no ratio.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "operating_point_ratio_fieller_check.json"
SWEEP_JSON = ROOT / "results" / "selection_multiplicity_sweep.json"

N_SEEDS = 100
NUM_AUROC = 0.70
DEN_AUROC = 0.985
N_BOOT = 200_000


def _load_47():
    spec = importlib.util.spec_from_file_location(
        "s47", ROOT / "code" / "47_selection_multiplicity_sweep.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def per_seed_gaps(m, target_auroc, n_seeds=N_SEEDS):
    """Reproduce one Sweep C cell at per-seed resolution (code/47's seed map)."""
    gaps = np.empty(n_seeds)
    for i in range(n_seeds):
        a = m.run_one_seed(i, i + 100000, i + 200000, i + 300000,
                           m.CAPACITY, m.DEFAULT_EPOCHS, m.DEFAULT_N_SAMPLES,
                           target_auroc, n_inner_folds=m.N_INNER_FOLDS)
        gaps[i] = a["leaky"] - a["clean_matched"]
        if (i + 1) % 25 == 0:
            print(f"    AUROC_0={target_auroc}: {i+1}/{n_seeds} seeds, "
                  f"running mean {gaps[:i+1].mean():+.6f}", flush=True)
    return gaps


def fieller_g(mean_den, se_den, n):
    t = float(stats.t.ppf(0.975, df=n - 1))
    return float((t * se_den / mean_den) ** 2), t


def main():
    shipped = json.load(open(SWEEP_JSON))["sweep_C_operating_point"]
    num_shipped = shipped[str(NUM_AUROC)]
    den_shipped = shipped[str(DEN_AUROC)]

    m = _load_47()
    print("Regenerating Sweep C endpoint cells at per-seed resolution ...", flush=True)
    num = per_seed_gaps(m, NUM_AUROC)
    den = per_seed_gaps(m, DEN_AUROC)

    # the regeneration must reproduce the shipped cell means exactly
    for name, arr, ship in (("0.70", num, num_shipped), ("0.985", den, den_shipped)):
        assert abs(arr.mean() - ship["gap_mean"]) < 1e-12, (
            f"cell {name}: regenerated mean {arr.mean():.10g} != shipped "
            f"{ship['gap_mean']:.10g}")
    print("  regenerated cell means match the shipped JSON exactly.", flush=True)

    ratio_point = float(num.mean() / den.mean())

    # ── (1) Fieller ──────────────────────────────────────────────────────────
    se_regen = float(den.std(ddof=1) / np.sqrt(len(den)))
    ci = den_shipped["gap_bca_ci_95"]
    _t = float(stats.t.ppf(0.975, df=len(den) - 1))
    # QUANTILE CONSISTENCY (an independent review found this). An earlier
    # revision converted the BCa halfwidth to an SE by dividing by the NORMAL
    # quantile 1.95996 and then re-inflated it by the T quantile t_.975,99 =
    # 1.98422 inside fieller_g -- mixing two reference distributions in one
    # quantity. The t quantile is used on both sides here. The legacy value is
    # retained alongside because it is what earlier revisions quoted, and
    # because the conclusion is identical either way: both exceed 1, so the
    # ratio has no finite confidence interval.
    se_from_bca = float((ci[1] - ci[0]) / 2 / _t)
    se_from_bca_legacy_normal = float((ci[1] - ci[0]) / 2 / 1.959963985)
    g_regen, tcrit = fieller_g(den.mean(), se_regen, len(den))
    g_bca, _ = fieller_g(den.mean(), se_from_bca, len(den))
    g_bca_legacy, _ = fieller_g(den.mean(), se_from_bca_legacy_normal, len(den))

    # ── (2) paired bootstrap of the ratio ────────────────────────────────────
    rng = np.random.default_rng(20260803)
    idx = rng.integers(0, N_SEEDS, size=(N_BOOT, N_SEEDS))
    bn = num[idx].mean(axis=1)
    bd = den[idx].mean(axis=1)
    frac_den_nonpos = float(np.mean(bd <= 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = bn / bd
    finite = ratios[np.isfinite(ratios)]
    pct = [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]

    # ── (3) the denominator cell on its own terms ────────────────────────────
    n_pos = int(np.sum(den > 0))
    n_neg = int(np.sum(den < 0))
    n_zero = int(np.sum(den == 0))
    sign_p = float(stats.binomtest(n_pos, n_pos + n_neg, 0.5).pvalue) if (n_pos + n_neg) else 1.0
    try:
        wil_p = float(stats.wilcoxon(den).pvalue)
    except ValueError:
        wil_p = None

    out = {
        "verdict": "RATIO_HAS_NO_FINITE_CONFIDENCE_INTERVAL__MULTIPLIER_WITHDRAWN",
        "numerator_cell": {
            "target_auroc": NUM_AUROC, "n_seeds": N_SEEDS,
            "mean": float(num.mean()), "sd": float(num.std(ddof=1)),
            "shipped_mean": num_shipped["gap_mean"],
            "wilcoxon_p": num_shipped["wilcoxon_p"],
        },
        "denominator_cell": {
            "target_auroc": DEN_AUROC, "n_seeds": N_SEEDS,
            "mean": float(den.mean()), "sd": float(den.std(ddof=1)),
            "sd_over_mean": float(den.std(ddof=1) / den.mean()),
            "shipped_mean": den_shipped["gap_mean"],
            "shipped_wilcoxon_p": den_shipped["wilcoxon_p"],
            "regenerated_wilcoxon_p": wil_p,
            "n_positive": n_pos, "n_negative": n_neg, "n_zero": n_zero,
            "sign_test_p": sign_p,
        },
        "point_ratio": ratio_point,
        "fieller": {
            "t_crit_975_df99": tcrit,
            "se_denominator_regenerated": se_regen,
            "se_denominator_from_shipped_bca_halfwidth": se_from_bca,
            "se_denominator_from_shipped_bca_halfwidth_legacy_normal_quantile":
                se_from_bca_legacy_normal,
            "g_from_bca_legacy_mixed_quantiles": g_bca_legacy,
            "quantile_consistency_note": (
                "The halfwidth-to-SE conversion now uses the same t quantile "
                "(t_.975,99 = %.5f) that fieller_g re-inflates by. An earlier revision "
                "divided by the normal 1.95996 and multiplied by the t quantile, giving "
                "g = %.3f instead of %.3f. Both exceed 1, so the conclusion -- the ratio "
                "has no finite confidence interval -- is unchanged." % (
                    _t, g_bca_legacy, g_bca)),
            "g_regenerated": g_regen,
            "g_from_shipped_bca": g_bca,
            "unbounded_iff_g_ge_1": True,
            "is_unbounded": bool(g_regen >= 1.0 and g_bca >= 1.0),
            "note": ("Fieller's theorem: the 95% confidence set for a ratio of means is "
                     "unbounded exactly when g >= 1. Both estimates of g exceed 1, so the "
                     "48.6x point estimate admits no finite interval and must not be "
                     "reported as a calibrated multiplier."),
        },
        "paired_bootstrap": {
            "n_resamples": N_BOOT,
            "fraction_of_resamples_with_denominator_le_zero": frac_den_nonpos,
            "percentile_interval_95_of_finite_ratios": pct,
            "note": ("The interval spans zero and both signs; a non-trivial share of "
                     "resamples places the denominator at or below zero, at which point "
                     "the ratio is undefined or negative."),
        },
        "what_survives": (
            "The monotone decline of the gap across the operating point is unaffected: "
            "+0.0093 / +0.0036 / +0.0011 / +0.0007 / +0.0002 at AUROC_0 = "
            "0.70 / 0.80 / 0.90 / 0.95 / 0.985, individually significant at 0.70, 0.80 and "
            "0.95. The paper now reports the endpoint as 'indistinguishable from zero "
            "(p = 0.219, n = 100)' and reports no multiplier."),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_PATH, "w"), indent=2)

    print(f"\npoint ratio                     = {ratio_point:.4g}")
    print(f"Fieller g (regenerated SE)      = {g_regen:.4g}")
    print(f"Fieller g (shipped BCa halfwid) = {g_bca:.4g}   -> unbounded iff >= 1")
    print(f"bootstrap: {100*frac_den_nonpos:.1f}% of resamples have denominator <= 0")
    print(f"bootstrap 95% percentile interval of the ratio = "
          f"[{pct[0]:.1f}, {pct[1]:.1f}]")
    print(f"denominator cell: {n_pos} positive / {n_neg} negative / {n_zero} zero, "
          f"sign test p = {sign_p:.4g}, SD/mean = {den.std(ddof=1)/den.mean():.2f}")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
