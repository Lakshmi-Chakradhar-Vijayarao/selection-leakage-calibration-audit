"""
Case Study 4: the estimator calibration was only ever measured on
TWO MARGINAL SLICES of a two-dimensional condition space. This script runs the
JOINT sweep and asks what the calibration factor actually is at the corner
where the real data lives.

WHY THIS SCRIPT EXISTS.

code/71 calibrates the three Case Study 4 selection-bias estimators against
synthetic data with a KNOWN true bias. It runs two sweeps, and they are
MARGINAL, not joint:

  * Section A, `controlled_sweep`: sweeps SEP_GRID -- the between-layer
    separation, i.e. how strongly one layer truly dominates -- with strictly
    i.i.d. layer x seed noise. That is rho = 0 at every point of the sweep.

  * Section D, `real_data_matching.rho_sweep`: sweeps RHO_GRID -- the AR(1)
    layer-correlation of the interaction noise, via `ar1_chol()` -- with mu
    held FIXED at the 24 real cells' plug-in per-layer means. Separation never
    varies along this sweep; only rho does.

Each sweep therefore holds the other axis at a single value. Neither ever
visits the interior of the (rho, sep) plane, and in particular neither visits
the HIGH-rho x HIGH-sep corner -- which is exactly where the audited files
plausibly sit, since those files show both strong adjacent-layer correlation
(measured lag-1 residual autocorrelation, median +0.71) and a genuinely
non-flat per-layer AUROC profile.

This matters because the paper generalises across both axes from evidence on
one. The reported ranges

    Delta_boot 0.50-0.55x, Delta_boot^std 0.28-0.30x, Delta_wc 1.27-1.34x

are the RHO-GRID ranges (main.tex: "every estimator's calibration ratio is
nearly constant over the entire rho grid ... the factor does not depend on
identifying rho correctly"). Over the SEP grid, restricted as code/71 itself
restricts it to conditions with true bias > 0.002, the same three quantities
span

    0.47-1.25x,                0.28-0.52x,                1.23-2.48x

-- a factor of 2.6, 1.9 and 2.0 respectively, not "nearly constant". code/71
stores both ranges side by side in `calibration_ratio_ranges`, so the fact is
in the artifact; the prose reads the rho range and describes it as though it
governed the whole space. The corrected headline +0.0030 to +0.0040 inherits
that reading.

WHAT THIS SCRIPT DOES.

1. VERIFICATION (Section 0). Imports code/71 by file path (same importlib
   pattern code/75 uses for code/47) and re-runs enough of it to prove the
   imported machinery is the shipped machinery:
     (a) an EXACT bit-for-bit replay of Section A. Section A is the first
         consumer of code/71's shared `np.random.default_rng(2026)`, and it
         consumes it in a fixed order, so every number in `controlled_sweep`
         is reproducible exactly. Asserted against the shipped JSON.
     (b) the Section D rho-sweep true_bias values, which are driven by
         independent seeds (5000 + i) and are therefore also exactly
         reproducible. Asserted exactly.
     (c) the Section D per-estimator simulated pooled values and calibration
         ratios, which ride the shared rng and so are reproducible only up to
         Monte-Carlo error. Asserted to a relative tolerance.

2. JOINT SWEEP (Section A). The Cartesian product of code/71's RHO_GRID and
   SEP_GRID, in code/71's controlled mu family (flat profile, one layer raised
   by sep * sigma) with AR(1)(rho) interaction noise. code/71's SEP_GRID is
   very coarse where it matters -- it jumps 2.0 -> 4.0, and the simulated
   Delta_boot value falls from 0.0065 to 0.0003 across that single step,
   straddling the observed +0.0021 -- so the grid is REFINED with extra sep
   points in [1.25, 3.5]. The original 8 x 8 product is a subset of what is
   run and is reported separately.

3. THE MANIFOLD (Section B). code/71 picks ONE point (the rho whose simulated
   Delta_boot best matches the observed +0.002107) and reads the calibration
   factor there. In two dimensions the set of conditions reproducing the
   observed value is not a point but a CURVE. This script finds that curve --
   every (rho, sep) cell whose simulated Delta_boot is within 10% of the
   observed value -- and reports the range of calibration ratios, corrected
   values and true biases along it. If that range is wide, the calibration
   correction is not identified by the observed value alone, and the
   degeneracy is the finding.

4. REAL-PROFILE JOINT SWEEP (Section C). The controlled family of Section A is
   stylized (flat + one spike). Section C repeats the joint design on the
   REALISTIC family: the 24 real cells' plug-in per-layer means, with their
   between-layer spread scaled by gamma (gamma = 1 is exactly code/71's
   Section D; gamma = 0 flattens every profile; gamma > 1 sharpens it), crossed
   with rho. This is the same two-dimensional question asked in the family the
   observed +0.002107 actually came from, and it is the sweep that decides
   whether the manifold degeneracy survives in the realistic model.

Everything is seeded per cell so results do not depend on iteration order.

Output: results/cs4_joint_calibration_sweep.json
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
OUT_PATH = ROOT / "results" / "cs4_joint_calibration_sweep.json"
REF_PATH = ROOT / "results" / "cs4_estimator_calibration.json"

# ── import code/71 rather than copying any of it ────────────────────────────
_spec = importlib.util.spec_from_file_location(
    "s71", CODE / "71_cs4_estimator_calibration.py")
s71 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s71)

N_LAYERS = s71.N_LAYERS
N_SEEDS = s71.N_SEEDS
RHO_GRID = list(s71.RHO_GRID)
SEP_GRID_71 = list(s71.SEP_GRID)

# Refinement of the sep axis in the region the observed value falls in.
SEP_EXTRA = [1.25, 1.5, 1.75, 2.25, 2.5, 2.75, 3.0, 3.5]
SEP_GRID = sorted(set(SEP_GRID_71 + SEP_EXTRA))

# Monte-Carlo budget for the joint sweep (code/71 uses 400k / 20k on its
# 8-point marginal; the joint grid is 15x larger, so truth draws are halved and
# trials doubled -- the trial count is what sets the precision of the manifold
# test, so it is the one that gets more).
N_TRUTH_JOINT = 200_000
N_TRIALS_JOINT = 40_000
TRIAL_CHUNK = 5_000          # cap peak memory in estimators_batch

# Real-profile joint sweep (Section C)
RHO_GRID_REAL = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98]
GAMMA_GRID_REAL = [0.0, 0.5, 1.0, 2.0, 3.0]
N_TRUTH_REAL = 30_000
N_TRIALS_REAL = 6_000

MANIFOLD_RTOL = 0.10
EST_KEYS = ("implemented", "standard", "rotation_delta_wc")


def estimators_chunked(mu, sigma, n_trials, rng, chol, chunk=TRIAL_CHUNK):
    """s71.estimators_batch over n_trials draws, in memory-safe chunks."""
    acc = {k: [] for k in EST_KEYS}
    done = 0
    while done < n_trials:
        k = min(chunk, n_trials - done)
        A = s71.draw(mu, sigma, k, rng, chol)
        e = s71.estimators_batch(A)
        for name in EST_KEYS:
            acc[name].append(e[name])
        done += k
    return {k: np.concatenate(v) for k, v in acc.items()}


def cell_record(mu, sigma, rho, n_truth, n_trials, seed):
    """One (rho, mu) condition: true bias + each estimator's pooled value."""
    chol = s71.ar1_chol(len(mu), rho)
    tb = s71.true_bias_mc(mu, sigma, n=n_truth, seed=seed, chol=chol)
    rng = np.random.default_rng(seed + 7_000_000)
    e = estimators_chunked(mu, sigma, n_trials, rng, chol)
    rec = {"true_bias": tb}
    for k in EST_KEYS:
        v = e[k]
        rec[k] = {
            "simulated_pooled": float(v.mean()),
            "se": float(v.std(ddof=1) / np.sqrt(len(v))),
            "ratio_to_truth": float(v.mean() / tb) if tb > 1e-9 else None,
        }
    return rec


# ── Section 0: verification against the shipped artifact ────────────────────
def verify(ref, SIGMA, real):
    print("=== 0. VERIFICATION: does the imported machinery reproduce code/71? ===")
    report = {}

    # (a) exact replay of Section A. Section A is the first consumer of
    #     code/71's shared rng(2026) and consumes it in a fixed order.
    rng = np.random.default_rng(2026)
    max_abs = 0.0
    n_checked = 0
    for sep, ref_row in zip(SEP_GRID_71, ref["controlled_sweep"]):
        mu = np.zeros(N_LAYERS)
        mu[N_LAYERS // 2] = sep * SIGMA
        tb = s71.true_bias_mc(mu, SIGMA, seed=int(1000 + sep * 10))
        A = mu[None, :, None] + rng.normal(scale=SIGMA,
                                           size=(s71.N_TRIALS, N_LAYERS, N_SEEDS))
        e = s71.estimators_batch(A)
        assert sep == ref_row["sep_in_sigma"]
        max_abs = max(max_abs, abs(tb - ref_row["true_bias"]))
        n_checked += 1
        for k in EST_KEYS:
            max_abs = max(max_abs, abs(float(e[k].mean()) - ref_row[k]["mean"]))
            n_checked += 1
    ok_a = max_abs < 1e-12
    print(f"  (a) Section A exact replay ({n_checked} values): "
          f"max |delta| = {max_abs:.3e}  -> {'EXACT MATCH' if ok_a else 'MISMATCH'}")
    report["section_A_exact_replay"] = {
        "n_values_checked": n_checked, "max_abs_diff": max_abs, "passed": bool(ok_a)}
    assert ok_a, "Section A replay failed -- imported machinery is not the shipped machinery"

    # (b) exact replay of Section D's true_bias (independent seeds 5000+i)
    profiles = [(n, np.asarray(v["mu_plugin"], float), v["sigma"])
                for n, v in real.items()]
    max_tb = 0.0
    sim_rel = {k: 0.0 for k in EST_KEYS}
    ratio_rel = {k: 0.0 for k in EST_KEYS}
    rngD = np.random.default_rng(20260)    # NOT code/71's stream state; see docstring
    for rho, ref_row in zip(RHO_GRID, ref["real_data_matching"]["rho_sweep"]):
        assert rho == ref_row["rho"]
        truths, ests = [], {k: [] for k in EST_KEYS}
        for i, (name, mu, sg) in enumerate(profiles):
            ch = s71.ar1_chol(len(mu), rho)
            truths.append(s71.true_bias_mc(mu, sg, n=s71.N_TRUTH_MIX,
                                           seed=5000 + i, chol=ch))
            A = s71.draw(mu, sg, s71.N_TRIALS_MIX, rngD, ch)
            e = s71.estimators_batch(A)
            for k in ests:
                ests[k].append(float(e[k].mean()))
        tb = float(np.mean(truths))
        max_tb = max(max_tb, abs(tb - ref_row["true_bias"]))
        for k in EST_KEYS:
            sm = float(np.mean(ests[k]))
            sim_rel[k] = max(sim_rel[k],
                             abs(sm - ref_row[k]["simulated_pooled"])
                             / abs(ref_row[k]["simulated_pooled"]))
            ratio_rel[k] = max(ratio_rel[k],
                               abs(sm / tb - ref_row[k]["ratio_to_truth"])
                               / abs(ref_row[k]["ratio_to_truth"]))
    ok_b = max_tb < 1e-12
    ok_c = max(sim_rel.values()) < 0.03 and max(ratio_rel.values()) < 0.03
    print(f"  (b) Section D true_bias exact replay (8 rho): "
          f"max |delta| = {max_tb:.3e}  -> {'EXACT MATCH' if ok_b else 'MISMATCH'}")
    print(f"  (c) Section D simulated pooled / ratios (MC-only agreement, fresh rng): "
          f"max rel. diff sim = {max(sim_rel.values()):.4f}, "
          f"ratio = {max(ratio_rel.values()):.4f}  -> "
          f"{'WITHIN 3% MC TOLERANCE' if ok_c else 'OUT OF TOLERANCE'}")
    report["section_D_true_bias_exact_replay"] = {
        "max_abs_diff": max_tb, "passed": bool(ok_b)}
    report["section_D_estimator_mc_agreement"] = {
        "max_rel_diff_simulated_pooled": {k: sim_rel[k] for k in EST_KEYS},
        "max_rel_diff_ratio_to_truth": {k: ratio_rel[k] for k in EST_KEYS},
        "tolerance": 0.03, "passed": bool(ok_c),
        "note": ("These ride code/71's shared rng, whose state at Section D depends "
                 "on Sections A and C; a fresh stream is used here, so agreement is "
                 "expected only to Monte-Carlo error.")}
    assert ok_b and ok_c
    print("  VERIFICATION PASSED.\n")
    return report


# ── Section B: manifold extraction ──────────────────────────────────────────
def manifold(cells, observed, rtol=MANIFOLD_RTOL, key="implemented"):
    """Cells whose simulated `key` value reproduces the observed value."""
    on = [c for c in cells
          if abs(c[key]["simulated_pooled"] - observed) <= rtol * abs(observed)]
    best = min(cells, key=lambda c: abs(c[key]["simulated_pooled"] - observed))
    return on, best


def summarise_manifold(on, best, observed_all, label):
    def corrected(c, k):
        r = c[k]["ratio_to_truth"]
        return float(observed_all[k] / r) if r and r > 1e-9 else None

    out = {"n_cells": len(on), "criterion_rtol": MANIFOLD_RTOL,
           "cells": [], "best_cell": None}
    for c in on:
        row = {"rho": c["rho"], "sep_or_gamma": c.get("sep", c.get("gamma")),
               "true_bias": c["true_bias"]}
        for k in EST_KEYS:
            row[k] = {"simulated_pooled": c[k]["simulated_pooled"],
                      "ratio_to_truth": c[k]["ratio_to_truth"],
                      "corrected_from_observed": corrected(c, k)}
        out["cells"].append(row)
    if on:
        out["true_bias_range"] = [min(c["true_bias"] for c in on),
                                  max(c["true_bias"] for c in on)]
        for k in EST_KEYS:
            rs = [c[k]["ratio_to_truth"] for c in on if c[k]["ratio_to_truth"]]
            cs = [corrected(c, k) for c in on if corrected(c, k) is not None]
            out[k] = {"ratio_to_truth_range": [min(rs), max(rs)] if rs else None,
                      "corrected_range": [min(cs), max(cs)] if cs else None,
                      "ratio_spread_factor": (max(rs) / min(rs)) if rs else None}
    bb = {"rho": best["rho"], "sep_or_gamma": best.get("sep", best.get("gamma")),
          "true_bias": best["true_bias"]}
    for k in EST_KEYS:
        bb[k] = {"simulated_pooled": best[k]["simulated_pooled"],
                 "ratio_to_truth": best[k]["ratio_to_truth"],
                 "corrected_from_observed": corrected(best, k)}
    out["best_cell"] = bb
    return out


def print_manifold(m, label, xname):
    print(f"\n--- MANIFOLD [{label}]: cells reproducing observed Delta_boot "
          f"within {int(MANIFOLD_RTOL*100)}% ---")
    if m["n_cells"] == 0:
        print("  (empty)")
    else:
        print(f"{'rho':>6} {xname:>7} {'true_bias':>10} {'sim_impl':>10} "
              f"{'r_impl':>7} {'corr_impl':>10} {'r_std':>7} {'corr_std':>9} "
              f"{'r_wc':>7} {'corr_wc':>9}")
        for c in m["cells"]:
            i, s, w = c["implemented"], c["standard"], c["rotation_delta_wc"]
            print(f"{c['rho']:6.2f} {c['sep_or_gamma']:7.2f} {c['true_bias']:10.6f} "
                  f"{i['simulated_pooled']:10.6f} {i['ratio_to_truth']:7.3f} "
                  f"{i['corrected_from_observed']:10.6f} {s['ratio_to_truth']:7.3f} "
                  f"{s['corrected_from_observed']:9.6f} {w['ratio_to_truth']:7.3f} "
                  f"{w['corrected_from_observed']:9.6f}")
        print(f"  true_bias range   : {m['true_bias_range'][0]:.6f} to "
              f"{m['true_bias_range'][1]:.6f}")
        for k in EST_KEYS:
            print(f"  {k:18s} ratio {m[k]['ratio_to_truth_range'][0]:.3f}-"
                  f"{m[k]['ratio_to_truth_range'][1]:.3f}x "
                  f"(spread {m[k]['ratio_spread_factor']:.2f}x)   corrected "
                  f"{m[k]['corrected_range'][0]:.6f} to {m[k]['corrected_range'][1]:.6f}")
    b = m["best_cell"]
    print(f"  BEST cell: rho={b['rho']}, {xname}={b['sep_or_gamma']}, "
          f"sim_impl={b['implemented']['simulated_pooled']:.6f}, "
          f"true_bias={b['true_bias']:.6f}, "
          f"ratio={b['implemented']['ratio_to_truth']:.3f}x, "
          f"corrected={b['implemented']['corrected_from_observed']:.6f}")


def main():
    t0 = time.time()
    ref = json.load(open(REF_PATH))
    real = s71.load_real_cells()
    sigmas = np.array([v["sigma"] for v in real.values()])
    SIGMA = float(np.median(sigmas))
    assert abs(SIGMA - ref["design"]["sigma_from_real_data"]) < 1e-12
    observed = ref["real_data_matching"]["observed_pooled_on_real_data"]
    obs_impl = observed["implemented"]
    print(f"SIGMA (layer x seed interaction SD, median of 24 real cells) = {SIGMA:.6f}")
    print("observed pooled on real data: " +
          ", ".join(f"{k}={observed[k]:+.6f}" for k in EST_KEYS) + "\n")

    out = {
        "purpose": ("Joint (rho, sep) calibration sweep for the Case Study 4 selection-"
                    "bias estimators. code/71 sweeps rho and sep only MARGINALLY; the "
                    "paper's 0.50-0.55x / 0.28-0.30x / 1.27-1.34x are the rho-grid "
                    "ranges alone."),
        "design": {"n_layers": N_LAYERS, "n_seeds": N_SEEDS, "sigma": SIGMA,
                   "rho_grid": RHO_GRID, "sep_grid": SEP_GRID,
                   "sep_grid_from_code71": SEP_GRID_71, "sep_grid_added": SEP_EXTRA,
                   "n_truth_draws_joint": N_TRUTH_JOINT,
                   "n_trials_joint": N_TRIALS_JOINT,
                   "rho_grid_real_profile": RHO_GRID_REAL,
                   "gamma_grid_real_profile": GAMMA_GRID_REAL,
                   "n_truth_draws_real": N_TRUTH_REAL, "n_trials_real": N_TRIALS_REAL},
        "observed_pooled_on_real_data": observed,
        "code71_marginal_ratio_ranges": ref["real_data_matching"]["calibration_ratio_ranges"],
        "code71_calibrated": ref["real_data_matching"]["calibrated"],
    }

    out["verification"] = verify(ref, SIGMA, real)

    # ── A. joint sweep, controlled mu family ────────────────────────────────
    print(f"=== A. JOINT SWEEP: {len(RHO_GRID)} rho x {len(SEP_GRID)} sep = "
          f"{len(RHO_GRID)*len(SEP_GRID)} cells (controlled mu family) ===")
    print(f"{'rho':>6} {'sep':>6} {'true_bias':>11} | {'implemented':>12} {'ratio':>8} | "
          f"{'standard':>11} {'ratio':>8} | {'rotation':>11} {'ratio':>8}")
    joint = []
    for ir, rho in enumerate(RHO_GRID):
        for isep, sep in enumerate(SEP_GRID):
            mu = np.zeros(N_LAYERS)
            mu[N_LAYERS // 2] = sep * SIGMA
            rec = cell_record(mu, SIGMA, rho, N_TRUTH_JOINT, N_TRIALS_JOINT,
                              seed=810_000 + 1000 * ir + isep)
            rec["rho"], rec["sep"] = rho, sep
            joint.append(rec)
            f = lambda k: ("     n/a" if rec[k]["ratio_to_truth"] is None
                           else f"{rec[k]['ratio_to_truth']:7.3f}x")
            print(f"{rho:6.2f} {sep:6.2f} {rec['true_bias']:11.6f} | "
                  f"{rec['implemented']['simulated_pooled']:12.6f} {f('implemented')} | "
                  f"{rec['standard']['simulated_pooled']:11.6f} {f('standard')} | "
                  f"{rec['rotation_delta_wc']['simulated_pooled']:11.6f} "
                  f"{f('rotation_delta_wc')}")
    out["joint_sweep_controlled"] = joint
    print(f"  [{time.time()-t0:.0f}s]")

    # Ratio ranges over the FULL joint grid, restricted as code/71 restricts.
    joint_ranges = {}
    for k in EST_KEYS:
        rs = [c[k]["ratio_to_truth"] for c in joint
              if c[k]["ratio_to_truth"] and c["true_bias"] > 0.002]
        joint_ranges[k] = {"over_joint_grid_true_bias_above_0.002":
                           [float(min(rs)), float(max(rs))]}
    out["joint_ratio_ranges"] = joint_ranges
    print("\n  calibration-ratio range over the JOINT grid (true_bias > 0.002), "
          "vs code/71's two marginals:")
    for k in EST_KEYS:
        j = joint_ranges[k]["over_joint_grid_true_bias_above_0.002"]
        r = ref["real_data_matching"]["calibration_ratio_ranges"][k]
        print(f"    {k:20s} joint {j[0]:.3f}-{j[1]:.3f}x | rho-marginal "
              f"{r['over_rho_grid'][0]:.3f}-{r['over_rho_grid'][1]:.3f}x | sep-marginal "
              f"{r['over_sep_grid_true_bias_above_0.002'][0]:.3f}-"
              f"{r['over_sep_grid_true_bias_above_0.002'][1]:.3f}x")

    # ── B. manifold ─────────────────────────────────────────────────────────
    on, best = manifold(joint, obs_impl)
    m = summarise_manifold(on, best, observed, "controlled")
    out["manifold_controlled"] = m
    print_manifold(m, "controlled mu family", "sep")

    # ── C. real-profile joint sweep ─────────────────────────────────────────
    print(f"\n=== C. REAL-PROFILE JOINT SWEEP: {len(RHO_GRID_REAL)} rho x "
          f"{len(GAMMA_GRID_REAL)} gamma (24 real cells per condition) ===")
    print("  gamma scales each real cell's between-layer spread; gamma=1 is exactly "
          "code/71 Section D")
    profiles = [(n, np.asarray(v["mu_plugin"], float), v["sigma"])
                for n, v in real.items()]
    print(f"{'rho':>6} {'gamma':>6} {'true_bias':>11} | {'implemented':>12} {'ratio':>8} | "
          f"{'standard':>11} {'ratio':>8} | {'rotation':>11} {'ratio':>8}")
    joint_real = []
    for ir, rho in enumerate(RHO_GRID_REAL):
        for ig, gamma in enumerate(GAMMA_GRID_REAL):
            truths, ests = [], {k: [] for k in EST_KEYS}
            for i, (name, mu0, sg) in enumerate(profiles):
                mu = mu0.mean() + gamma * (mu0 - mu0.mean())
                ch = s71.ar1_chol(len(mu), rho)
                truths.append(s71.true_bias_mc(mu, sg, n=N_TRUTH_REAL,
                                               seed=820_000 + 100 * ir + 10 * ig + i,
                                               chol=ch))
                rng = np.random.default_rng(830_000 + 1000 * ir + 100 * ig + i)
                e = estimators_chunked(mu, sg, N_TRIALS_REAL, rng, ch)
                for k in ests:
                    ests[k].append(float(e[k].mean()))
            tb = float(np.mean(truths))
            rec = {"rho": rho, "gamma": gamma, "true_bias": tb}
            for k in EST_KEYS:
                sm = float(np.mean(ests[k]))
                rec[k] = {"simulated_pooled": sm,
                          "ratio_to_truth": float(sm / tb) if tb > 1e-9 else None}
            joint_real.append(rec)
            f = lambda k: ("     n/a" if rec[k]["ratio_to_truth"] is None
                           else f"{rec[k]['ratio_to_truth']:7.3f}x")
            print(f"{rho:6.2f} {gamma:6.2f} {tb:11.6f} | "
                  f"{rec['implemented']['simulated_pooled']:12.6f} {f('implemented')} | "
                  f"{rec['standard']['simulated_pooled']:11.6f} {f('standard')} | "
                  f"{rec['rotation_delta_wc']['simulated_pooled']:11.6f} "
                  f"{f('rotation_delta_wc')}")
    out["joint_sweep_real_profile"] = joint_real
    print(f"  [{time.time()-t0:.0f}s]")

    on_r, best_r = manifold(joint_real, obs_impl)
    m_r = summarise_manifold(on_r, best_r, observed, "real_profile")
    out["manifold_real_profile"] = m_r
    print_manifold(m_r, "real mu profiles", "gamma")

    # ── verdict ─────────────────────────────────────────────────────────────
    paper_range = [0.0030, 0.0040]

    def judge(mm, n_min=2):
        if not mm["n_cells"]:
            return {"n_manifold_cells": 0, "status": "empty"}
        cr = mm["implemented"]["corrected_range"]
        rr = mm["implemented"]["ratio_to_truth_range"]
        sp = mm["implemented"]["ratio_spread_factor"]
        overlaps = not (cr[1] < paper_range[0] or cr[0] > paper_range[1])
        if cr[1] < paper_range[0]:
            pos = "manifold entirely BELOW paper range (paper too high)"
        elif cr[0] > paper_range[1]:
            pos = "manifold entirely ABOVE paper range (paper too low)"
        elif cr[0] <= paper_range[0] and paper_range[1] <= cr[1]:
            pos = "manifold CONTAINS paper range (paper range is a sub-interval)"
        else:
            pos = "manifold OVERLAPS paper range only partially"
        ident = ("single-cell -- grid resolution, not identification"
                 if mm["n_cells"] < n_min
                 else ("WELL IDENTIFIED" if sp < 1.2 else "DEGENERATE"))
        return {"n_manifold_cells": mm["n_cells"],
                "implemented_ratio_range": rr,
                "implemented_corrected_range": cr,
                "true_bias_range": mm["true_bias_range"],
                "ratio_spread_factor": sp,
                "identification": ident,
                "overlaps_paper_range": bool(overlaps),
                "position_vs_paper_range": pos}

    verdict = {"paper_calibration_corrected_range": paper_range,
               "controlled": judge(m), "real_profile": judge(m_r)}

    # Combined manifold: every cell in EITHER family that reproduces the
    # observed value. This is the honest identified set given only the
    # observed +0.0021 and no further commitment about which (rho, sep/gamma)
    # or which mu family generated the real files.
    all_on = list(on) + list(on_r)
    if all_on:
        m_all = summarise_manifold(all_on, best if abs(
            best["implemented"]["simulated_pooled"] - obs_impl) <=
            abs(best_r["implemented"]["simulated_pooled"] - obs_impl) else best_r,
            observed, "combined")
        out["manifold_combined"] = m_all
        verdict["combined"] = judge(m_all)

    out["verdict"] = verdict

    print("\n=== VERDICT ===")
    for label in ("controlled", "real_profile", "combined"):
        v = verdict.get(label)
        if v is None or not v.get("n_manifold_cells"):
            print(f"  [{label}] manifold empty")
            continue
        print(f"  [{label}] {v['n_manifold_cells']} cells on the manifold; "
              f"Delta_boot ratio {v['implemented_ratio_range'][0]:.3f}-"
              f"{v['implemented_ratio_range'][1]:.3f}x "
              f"(spread {v['ratio_spread_factor']:.2f}x -> {v['identification']})")
        print(f"           honest corrected Delta_boot: "
              f"{v['implemented_corrected_range'][0]:.6f} to "
              f"{v['implemented_corrected_range'][1]:.6f}   "
              f"(paper says {paper_range[0]:.4f}-{paper_range[1]:.4f}; "
              f"{v['position_vs_paper_range']})")
        print(f"           true_bias on manifold: {v['true_bias_range'][0]:.6f} to "
              f"{v['true_bias_range'][1]:.6f}")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}   [total {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
