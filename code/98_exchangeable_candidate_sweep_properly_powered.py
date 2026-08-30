"""
code/73 at the paper's own power convention (N_SEEDS=200),
instead of the compute-limited N_SEEDS=30 it disclosed and ran with.

WHY THIS SCRIPT EXISTS. code/73 reruns the paper's K-candidate-severity sweep
with K indexing independently-seeded, exchangeable candidates instead of
epochs of one training run (see code/73's docstring for the full motivation --
Appendix A.11's observation that epochs of one run are not an exchangeable
candidate set, and the paper's use of the K-law to reason about selection
among layers/thresholds, which ARE closer to exchangeable). code/73's own
`power` field in results/exchangeable_candidate_sweep.json already computes
that its N_SEEDS=30 design is underpowered: minimum detectable effect (MDE)
+0.0099 exceeds the target effect it is trying to detect, +0.0042 (code/47's
K=45 epoch gap, the scale code/73 itself compares against). main.tex discloses
this honestly as an open weakness ("underpowered (MDE ...)" near "code/73").
This script asks the direct follow-up: does N_SEEDS=200 -- code/73's own
stated point of comparison, and the paper's usual convention elsewhere --
actually close that power gap, and if so, does the K-law reproduce on
exchangeable candidates once the test has the power to see it?

WHAT THIS SCRIPT DOES AND DOES NOT CHANGE. This is code/73, run at higher
power, and NOTHING ELSE. It loads code/73 as a module (the same
importlib.util.spec_from_file_location idiom code/73 itself uses to load
code/47, so there is no copy-paste drift), overrides exactly one module
constant -- N_SEEDS: 30 -> 200 -- and points the output at a new path so
code/73's own results file and its provenance are untouched. Every other
moving part is inherited unchanged from the loaded code/73 module object:
the isotropic generator and SweepMLP (themselves imported by code/73 from
code/47), the LEAKY/CLEAN/CLEAN_MATCHED/PLACEBO protocol in
run_one_seed/run_cell, the inline seed derivation in main()
(data_seed=seed, split_seed=seed+100000, fold_seed_base=seed+200000,
init_seed_base=seed+300000 -- code/73 does not define a separate named
`seeds_for()` function; this IS its seed-derivation logic, reused verbatim),
the K grid [2, 5, 15, 45], FIXED_EPOCHS=45, the ln K slope fit, the EVT fit,
the degeneracy gate, and the verdict rule. Compute cost: code/73 trains 2K+1
models per (seed, fold); at N_SEEDS=200 (6.67x code/73's 30) this is
CPU-only and expected to take on the order of code/73's 958.9s x 6.67 =
roughly 1.5-2 hours, not requiring a GPU.

Output: results/exchangeable_candidate_sweep_n200.json (code/73's own
results/exchangeable_candidate_sweep.json is not touched or overwritten).
"""
import importlib.util
import json
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
OUT_PATH = ROOT / "results" / "exchangeable_candidate_sweep_n200.json"

N_SEEDS_NEW = 200

_spec = importlib.util.spec_from_file_location(
    "s73", CODE / "73_exchangeable_candidate_sweep.py")
s73 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s73)

_orig_n_seeds = s73.N_SEEDS
_orig_out_path = s73.OUT_PATH

# The ONLY substitution relative to code/73: N_SEEDS 30 -> 200. Everything
# else (K_VALUES, FIXED_EPOCHS, N_BOOT_SLOPE, BOOT_SEED_BASE, SLOPE_SEED,
# DEGENERACY_MAX_IDENTICAL_FRAC, run_one_seed, run_cell, fit_block, main)
# is code/73's own module code, executed as-is.
s73.N_SEEDS = N_SEEDS_NEW
s73.OUT_PATH = OUT_PATH


def main():
    print(f"=== code/98: code/73's exchangeable-candidate sweep at "
          f"N_SEEDS={s73.N_SEEDS} (code/73 itself ran N_SEEDS={_orig_n_seeds}) ===",
          flush=True)
    print(f"    Writing to {OUT_PATH} -- code/73's own "
          f"{_orig_out_path.name} is untouched.\n", flush=True)

    s73.main()

    # ---- post-hoc power/verdict summary, read back from the JSON we just
    # wrote, so this script's own console report is self-contained and does
    # not require re-deriving anything code/73's main() already computed.
    with open(OUT_PATH) as f:
        out = json.load(f)

    target_effect = out["power"]["code47_gap_at_K45_for_scale"]
    mde_new = out["power"]["min_detectable_gap_80pct_power_two_sided_05"]
    gap_closed = bool(mde_new <= abs(target_effect)) if target_effect is not None else None

    b = out["comparison_to_code47_sweep_A"]["exchangeable_lnK_slope_b"]
    b_ci = out["comparison_to_code47_sweep_A"]["exchangeable_lnK_slope_b_ci_95"]
    b_excludes_zero = bool(b_ci[0] > 0 or b_ci[1] < 0)
    ref_b = out["comparison_to_code47_sweep_A"]["code47_lnK_slope_b"]
    ref_b_ci = out["comparison_to_code47_sweep_A"]["code47_lnK_slope_b_ci_95"]

    print("\n=== code/98 power/verdict summary (N_SEEDS=200) ===")
    print(f"  MDE at N_SEEDS={s73.N_SEEDS}: {mde_new:+.5f}")
    print(f"  target effect (code/47 K=45 epoch gap): {target_effect:+.5f}")
    print(f"  MDE <= target effect (power gap closed): {gap_closed}")
    print(f"\n  exchangeable ln K slope b = {b:+.5f}  95% CI [{b_ci[0]:+.5f}, {b_ci[1]:+.5f}]")
    print(f"  slope CI excludes zero: {b_excludes_zero}")
    print(f"  code/47 (epoch-based) headline: b = {ref_b:+.5f}  "
          f"95% CI [{ref_b_ci[0]:+.5f}, {ref_b_ci[1]:+.5f}]")
    print(f"\n  VERDICT (N_SEEDS=200): {out['verdict']}")
    print(f"  {out['verdict_statement']}")
    print(f"\n  runtime_seconds: {out['runtime_seconds']:.1f} "
          f"({out['runtime_seconds']/60:.1f} min)")

    out["_n200_power_summary"] = {
        "n_seeds": s73.N_SEEDS,
        "code73_n_seeds_for_contrast": _orig_n_seeds,
        "mde_80pct_power": mde_new,
        "target_effect_code47_K45_gap": target_effect,
        "power_gap_closed": gap_closed,
        "exchangeable_lnK_slope_b": b,
        "exchangeable_lnK_slope_b_ci_95": b_ci,
        "exchangeable_slope_ci_excludes_zero": b_excludes_zero,
        "code47_epoch_lnK_slope_b": ref_b,
        "code47_epoch_lnK_slope_b_ci_95": ref_b_ci,
        "note": (
            "Computed by code/98 from the fields code/73's own main() already "
            "produces (power.*, comparison_to_code47_sweep_A.*, verdict*), at "
            "N_SEEDS=200 instead of code/73's own N_SEEDS=30. No new "
            "statistical machinery; this is a read-back summary for convenience."),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved (with _n200_power_summary appended): {OUT_PATH}")


if __name__ == "__main__":
    main()
