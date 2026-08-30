"""
P0-5 correction. The 2x2x2 factorial (code/77) reports its three
main-effect shares of the shipped-to-corrected movement (~75.5% selection-set
size, ~28.1% selection-run budget, ~-6.2% training depth) as point estimates
only, with no uncertainty interval, even though every other headline number in
this paper carries a BCa 95% interval.

This script reruns code/77's exact seed-generation pipeline (pass1_seed /
pass2_seed, imported unchanged) to reproduce its per-seed, per-arm AUROC
arrays bit-for-bit (asserted below against the shipped JSON's cell means), then
performs a seed-level BCa bootstrap over the paired arrays: each of 10,000
resamples draws a bootstrap sample of the 100 seed indices (with replacement,
same resampled index set applied to every arm so the pairing across arms and
LEAKY is preserved), recomputes all eight cells' gap means, the three main
effects and their shares of the resampled total movement, and the shared
denominator (total movement) itself.

BCa acceleration is estimated via jackknife (leave-one-seed-out, n=100).

Output: results/mechanism3_factorial_share_bootstrap.json
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
R = ROOT / "results"
SHIPPED = R / "mechanism3_factorial_selection_controls.json"
OUT_PATH = R / "mechanism3_factorial_share_bootstrap.json"

_spec = importlib.util.spec_from_file_location(
    "f77", CODE / "77_mechanism3_factorial_selection_controls.py")
f77 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f77)

N_SEEDS = f77.N_SEEDS
ARMS = f77.ARMS
SEL_KEYS = f77.SEL_KEYS
N_BOOT = 10000
RNG_SEED = 20260805


def regenerate_arrays():
    """Reproduces code/77's `arrs` dict exactly (same seed streams)."""
    t0 = time.time()
    cols, metas = {}, []
    for seed in range(N_SEEDS):
        aucs, meta = f77.pass1_seed(seed, seed + 100000, seed + 200000, seed + 300000)
        for k, v in aucs.items():
            cols.setdefault(k, []).append(v)
        metas.append(meta)
        if (seed + 1) % 25 == 0:
            print(f"  [pass1] {seed+1}/{N_SEEDS} elapsed={time.time()-t0:.0f}s", flush=True)
    arrs = {k: np.array(v) for k, v in cols.items()}

    mean_ep = {k: float(np.mean([e for m in metas for e in m["epochs"][k]]))
               for k in ["leaky"] + SEL_KEYS}
    shifts = {k: int(round(mean_ep["leaky"] - mean_ep[k])) for k in SEL_KEYS}
    const_depth = int(round(mean_ep["leaky"]))

    cols2 = {}
    for seed in range(N_SEEDS):
        aucs, _ = f77.pass2_seed(seed, seed + 100000, seed + 200000, seed + 300000,
                                 {k: metas[seed]["epochs"][k] for k in SEL_KEYS},
                                 shifts, const_depth)
        for k, v in aucs.items():
            cols2.setdefault(k, []).append(v)
        if (seed + 1) % 25 == 0:
            print(f"  [pass2] {seed+1}/{N_SEEDS} elapsed={time.time()-t0:.0f}s", flush=True)
    for k, v in cols2.items():
        arrs[k] = np.array(v)
    print(f"  regeneration done in {time.time()-t0:.0f}s", flush=True)
    return arrs


def cell_gap_means(arrs, idx):
    """Per-arm mean(leaky - arm) over the (possibly resampled) index set idx."""
    leaky = arrs["leaky"][idx]
    return {arm: float(np.mean(leaky - arrs[arm][idx])) for arm in ARMS}


def mean_gap(gaps, pred):
    vals = [gaps[a] for a in ARMS if pred(a)]
    return float(np.mean(vals))


def compute_stats(arrs, idx):
    gaps = cell_gap_means(arrs, idx)
    shipped = gaps["small__infold__free"]
    corrected = gaps["fold_matched__oof__matched"]
    total_move = shipped - corrected
    eff_size = (mean_gap(gaps, lambda a: a.startswith("small__"))
                - mean_gap(gaps, lambda a: a.startswith("fold_matched__")))
    eff_budget = (mean_gap(gaps, lambda a: "__infold__" in a)
                  - mean_gap(gaps, lambda a: "__oof__" in a))
    eff_depth = (mean_gap(gaps, lambda a: a.endswith("__free"))
                 - mean_gap(gaps, lambda a: a.endswith("__matched")))
    out = {
        "shipped_gap": shipped, "corrected_gap": corrected, "total_movement": total_move,
        "eff_size": eff_size, "eff_budget": eff_budget, "eff_depth": eff_depth,
    }
    if abs(total_move) > 1e-12:
        out["share_size"] = eff_size / total_move
        out["share_budget"] = eff_budget / total_move
        out["share_depth"] = eff_depth / total_move
    else:
        out["share_size"] = out["share_budget"] = out["share_depth"] = float("nan")
    return out


def bca_interval(theta_hat, boot_vals, jack_vals, alpha=0.05):
    boot_vals = np.asarray(boot_vals)
    boot_vals = boot_vals[np.isfinite(boot_vals)]
    jack_vals = np.asarray(jack_vals)
    n = len(jack_vals)
    z0 = norm.ppf(np.mean(boot_vals < theta_hat))
    jack_mean = jack_vals.mean()
    num = np.sum((jack_mean - jack_vals) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack_vals) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0
    z_lo, z_hi = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)

    def adj(z):
        return norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))

    lo_pct, hi_pct = adj(z_lo) * 100, adj(z_hi) * 100
    lo_pct, hi_pct = np.clip([lo_pct, hi_pct], 0.001, 99.999)
    lo, hi = np.percentile(boot_vals, [lo_pct, hi_pct])
    return {"ci_95": [float(lo), float(hi)], "z0": float(z0), "a": float(a)}


def main():
    arrs = regenerate_arrays()

    shipped_ref = json.load(open(SHIPPED))
    for arm in ARMS:
        ref_mean = shipped_ref["factorial_cells"][arm]["arm_mean_auroc"]
        drift = abs(float(arrs[arm].mean()) - ref_mean)
        assert drift < 1e-9, f"{arm} regenerated mean drifted from shipped by {drift:.2e}"
    print("Regenerated arrays reproduce the shipped JSON's cell means to <1e-9.")

    idx_full = np.arange(N_SEEDS)
    point = compute_stats(arrs, idx_full)

    rng = np.random.default_rng(RNG_SEED)
    boot = {k: np.empty(N_BOOT) for k in point}
    for b in range(N_BOOT):
        idx = rng.integers(0, N_SEEDS, size=N_SEEDS)
        s = compute_stats(arrs, idx)
        for k in point:
            boot[k][b] = s[k]

    jack = {k: np.empty(N_SEEDS) for k in point}
    for i in range(N_SEEDS):
        idx = np.delete(idx_full, i)
        s = compute_stats(arrs, idx)
        for k in point:
            jack[k][i] = s[k]

    intervals = {}
    for k in ["share_size", "share_budget", "share_depth", "eff_size", "eff_budget",
              "eff_depth", "total_movement"]:
        intervals[k] = bca_interval(point[k], boot[k], jack[k])

    def dist_zero(key):
        vals = boot[key]
        frac_pos = float(np.mean(vals > 0))
        return {"frac_positive_of_10000_boot": frac_pos,
                "distinguishable_from_zero_bca": not (intervals[key]["ci_95"][0] < 0 < intervals[key]["ci_95"][1])}

    out = {
        "why": ("code/77 reports the factorial's three main-effect shares "
                "(75.5% / 28.1% / -6.2%) as point estimates with no interval. "
                "This reruns the identical seed-generation pipeline and attaches "
                "a seed-level BCa bootstrap (n=100 seeds, 10000 resamples) to "
                "each share and to the underlying main effects and total "
                "movement."),
        "point_estimates": point,
        "bca_intervals": intervals,
        "budget_share_distinguishable_from_zero": dist_zero("share_budget"),
        "size_share_distinguishable_from_zero": dist_zero("share_size"),
        "depth_share_distinguishable_from_zero": dist_zero("share_depth"),
        "n_boot": N_BOOT,
        "n_seeds": N_SEEDS,
        "rng_seed": RNG_SEED,
        "outputs": {"json": str(OUT_PATH.relative_to(ROOT)),
                    "script": str(Path(__file__).resolve().relative_to(ROOT))},
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print("\nPoint estimates and BCa 95% intervals:")
    for k in ["share_size", "share_budget", "share_depth"]:
        lo, hi = intervals[k]["ci_95"]
        print(f"  {k:<14} {point[k]*100:+7.1f}%   BCa [{lo*100:+7.1f}%, {hi*100:+7.1f}%]")
    for k in ["eff_size", "eff_budget", "eff_depth"]:
        lo, hi = intervals[k]["ci_95"]
        print(f"  {k:<14} {point[k]:+.4f}   BCa [{lo:+.4f}, {hi:+.4f}]")
    print(f"  total_movement {point['total_movement']:+.4f}   BCa "
          f"[{intervals['total_movement']['ci_95'][0]:+.4f}, "
          f"{intervals['total_movement']['ci_95'][1]:+.4f}]")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
