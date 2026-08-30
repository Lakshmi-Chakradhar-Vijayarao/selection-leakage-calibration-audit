"""
ships the PER-SEED severity values for Sweep A (candidate count K)
and Sweep C (operating point), which no previous revision shipped.

WHY. An independent review noted that `results/selection_multiplicity_sweep.json`
carries only per-cell means, BCa intervals and Wilcoxon p-values. Nobody --
including us -- could re-derive from the shipped artifact either (a) the
Monte-Carlo error on the points the `gap = a + b ln K` law is fitted to, which
is what determines whether that functional form is identified at all, or (b) a
variance-stabilized re-expression of the operating-point contrast, which is what
distinguishes "the operating point governs severity" from "AUROC differences
compress near the ceiling." Both require the per-seed arms, not the cell means.

WHAT THIS DOES. It re-runs `code/47`'s own `run_one_seed` at exactly the seeds
`run_sweep_cell` uses (data_seed=s, split_seed=s+100000, fold_seed_base=s+200000,
init_seed_base=s+300000), for:

  * Sweep A, the six NON-DEGENERATE K cells the K-law is fitted on
    (K in {15, 25, 45, 75, 135, 225}), 200 seeds each, plus the four excluded
    small-K cells (K in {1, 3, 5, 10}) at the same 200 seeds so the degeneracy
    argument is also re-derivable from per-seed data;
  * Sweep C, all five operating points (0.70, 0.80, 0.90, 0.95, 0.985),
    100 seeds each.

and records, per seed, the four arm AUROCs (leaky / clean / clean_matched /
placebo) plus the degeneracy record. Nothing about the computation changes: the
run path is code/47's, imported rather than copied, so the recomputed cell means
must reproduce the shipped JSON to full float64. That equality is asserted, and
failing it aborts the run rather than writing a divergent artifact.

Output: results/sweep_per_seed.npz  (+ a small JSON manifest with the
reproduction check).
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
OUT_NPZ = ROOT / "results" / "sweep_per_seed.npz"
OUT_JSON = ROOT / "results" / "sweep_per_seed_manifest.json"
REF_JSON = ROOT / "results" / "selection_multiplicity_sweep.json"

_spec = importlib.util.spec_from_file_location(
    "s47", CODE / "47_selection_multiplicity_sweep.py")
s47 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s47)

SWEEP_A_K = [1, 3, 5, 10, 15, 25, 45, 75, 135, 225]
SWEEP_A_N_SEEDS = 200
SWEEP_C_AUROC = [0.70, 0.80, 0.90, 0.95, 0.985]
SWEEP_C_N_SEEDS = 100

ARMS = ["leaky", "clean", "clean_matched", "placebo"]


def run_cell(n_seeds, epochs, target_auroc):
    rows = {a: [] for a in ARMS}
    extra = {k: [] for k in ["_val_auc_std", "_degen_identical_frac",
                             "_degen_max_param_diff", "_leaky_argmax_is_last_frac",
                             "_clean_argmax_is_last_frac"]}
    for seed in range(n_seeds):
        aucs = s47.run_one_seed(seed, seed + 100000, seed + 200000, seed + 300000,
                                s47.CAPACITY, epochs, s47.DEFAULT_N_SAMPLES,
                                target_auroc, degeneracy_check=True)
        for a in ARMS:
            rows[a].append(aucs[a])
        for k in extra:
            extra[k].append(aucs[k])
    return ({a: np.array(v, dtype=np.float64) for a, v in rows.items()},
            {k: np.array(v, dtype=np.float64) for k, v in extra.items()})


def main():
    t0 = time.time()
    ref = json.load(open(REF_JSON))
    arrays, manifest = {}, {"sweep_A": {}, "sweep_C": {}}

    print("=== Sweep A per-seed (K), 200 seeds/cell ===", flush=True)
    for K in SWEEP_A_K:
        arms, extra = run_cell(SWEEP_A_N_SEEDS, K, s47.DEFAULT_TARGET_AUROC)
        gap = arms["leaky"] - arms["clean_matched"]
        for a in ARMS:
            arrays[f"A_K{K}__{a}"] = arms[a]
        arrays[f"A_K{K}__gap"] = gap
        arrays[f"A_K{K}__degen_identical_frac"] = extra["_degen_identical_frac"]
        shipped = ref["sweep_A_K"][str(K)]["gap_mean"]
        d = abs(float(gap.mean()) - shipped)
        manifest["sweep_A"][str(K)] = {
            "gap_mean_recomputed": float(gap.mean()),
            "gap_mean_shipped": shipped,
            "abs_diff": d,
            "gap_sem": float(gap.std(ddof=1) / np.sqrt(len(gap))),
            "gap_sd": float(gap.std(ddof=1)),
            "n_seeds": SWEEP_A_N_SEEDS,
            "n_exactly_zero_gap": int((gap == 0).sum()),
            "identical_state_dict_frac": float(extra["_degen_identical_frac"].mean()),
        }
        print(f"  K={K:4d}: gap={gap.mean():+.8f} (shipped {shipped:+.8f}, "
              f"|d|={d:.2e})  SEM={manifest['sweep_A'][str(K)]['gap_sem']:.2e}  "
              f"elapsed={time.time()-t0:.0f}s", flush=True)

    print("\n=== Sweep C per-seed (operating point), 100 seeds/cell ===", flush=True)
    for tgt in SWEEP_C_AUROC:
        arms, extra = run_cell(SWEEP_C_N_SEEDS, s47.DEFAULT_EPOCHS, tgt)
        gap = arms["leaky"] - arms["clean_matched"]
        for a in ARMS:
            arrays[f"C_A{tgt}__{a}"] = arms[a]
        arrays[f"C_A{tgt}__gap"] = gap
        arrays[f"C_A{tgt}__degen_identical_frac"] = extra["_degen_identical_frac"]
        shipped = ref["sweep_C_operating_point"][str(tgt)]["gap_mean"]
        d = abs(float(gap.mean()) - shipped)
        manifest["sweep_C"][str(tgt)] = {
            "gap_mean_recomputed": float(gap.mean()),
            "gap_mean_shipped": shipped,
            "abs_diff": d,
            "gap_sem": float(gap.std(ddof=1) / np.sqrt(len(gap))),
            "gap_sd": float(gap.std(ddof=1)),
            "n_seeds": SWEEP_C_N_SEEDS,
            "leaky_mean": float(arms["leaky"].mean()),
            "clean_matched_mean": float(arms["clean_matched"].mean()),
        }
        print(f"  AUROC0={tgt}: gap={gap.mean():+.8f} (shipped {shipped:+.8f}, "
              f"|d|={d:.2e})  elapsed={time.time()-t0:.0f}s", flush=True)

    worst = max([v["abs_diff"] for v in manifest["sweep_A"].values()] +
                [v["abs_diff"] for v in manifest["sweep_C"].values()])
    manifest["max_abs_diff_vs_shipped"] = worst
    manifest["reproduces_shipped_cell_means"] = bool(worst < 1e-12)
    manifest["runtime_seconds"] = time.time() - t0
    manifest["note"] = (
        "Per-seed arm AUROCs for Sweep A (K) and Sweep C (operating point), produced by "
        "importing code/47's own run_one_seed at the same seeds run_sweep_cell uses. The "
        "recomputed cell means reproduce results/selection_multiplicity_sweep.json to "
        "full float64, so this artifact adds resolution and changes no published number.")

    assert worst < 1e-12, f"per-seed rerun does NOT reproduce shipped cell means (max |d|={worst})"
    np.savez_compressed(OUT_NPZ, **arrays)
    with open(OUT_JSON, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nAll cell means reproduce the shipped JSON to <1e-12.")
    print(f"Saved: {OUT_NPZ}\nSaved: {OUT_JSON}\nRuntime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
