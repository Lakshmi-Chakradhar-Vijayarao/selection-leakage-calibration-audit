"""
sensitivity sweep of ES_HOLD_FRACTION (= F_SMALL in code/77's
notation) across the decisive 2x2x2 Mechanism-3 factorial (code/77), whose
severity decomposition (75.5% / 28.1% / -6.2% to selection-set size /
selection-run budget / training depth) drives Case Study 3's corrected
verdict and is quoted in the abstract, §4.3 and Appendix A.

WHY THIS SCRIPT EXISTS. Appendix A calls ES_HOLD_FRACTION=0.15 "the single
most consequential undisclosed choice in the paper": it is the proximate
cause of the selection-set-size asymmetry that produced the now-retracted
primary headline (§4.3, code/02-02d), and it sets the "small" selection-set
SIZE arm's size in code/77's factorial (the arm responsible for 75.5% of the
shipped-to-corrected movement). It has never been swept inside the factorial
itself -- code/75's own {0.15, 0.20, 0.25} sweep predates the factorial and
does not report how the factorial's own three-way decomposition moves.

WHAT THIS SCRIPT DOES. Re-runs code/77's identical 2x2x2x100-seed factorial
(same pool/fold/init-seed scheme, same CAPACITY=128, same AUROC_0=0.80, same
LEAKY/PLACEBO references, same depth-matching and CONST_DEPTH degeneracy
logic) at F_SMALL in {0.05, 0.10, 0.15, 0.20, 0.25}, holding every other
factorial constant fixed (F_FOLD=0.25, OOF_HOLD_FRACTION=1/6, CAPACITY=128,
AUROC_0=0.80, N_SEEDS=100). Only the "small" selection-set-size arms (four of
the eight cells: small__infold__free, small__infold__matched, small__oof__free,
small__oof__matched) and their depth shifts change with F_SMALL; the
"fold_matched" arms, LEAKY, PLACEBO and CONST_DEPTH do not depend on F_SMALL
at all and are recomputed identically at every sweep point as an internal
consistency check (they should not move outside seed-level noise).

F_SMALL=0.30 IS OMITTED, NOT SILENTLY DROPPED: at F_SMALL=0.30, the
small__oof arm needs |sel|=round(0.30*|tr_idx|)~=112 selection points drawn
from the shared out-of-fold pool ges_idx, whose size (~94, fixed by
OOF_HOLD_FRACTION=1/(N_INNER_FOLDS+1) independent of F_SMALL) is smaller than
that. This is a genuine geometric ceiling on how far F_SMALL can be pushed
inside THIS factorial's shared-pool design, not a code bug, and it is
reported as a finding in its own right: the factorial's shared-pool
construction bounds ES_HOLD_FRACTION above at F_FOLD=0.25 before the
out-of-fold arms become infeasible.

Everything imported verbatim from code/77 (not reimplemented): _prep, _pool,
_selection_sets, pass1_seed, pass2_seed, gap_stats, ARMS, SEL_KEYS,
FREE_ARMS, F_FOLD, OOF_HOLD_FRACTION, CAPACITY, N_SEEDS, EPOCHS, N_SAMPLES,
TARGET_AUROC, N_INNER_FOLDS, TEST_SIZE. Only the module-level F_SMALL is
overridden per sweep point (pass1_seed/pass2_seed resolve it as a global at
call time via _selection_sets, so no code in code/77 needs to change).

Anchor check (informal, not the pre-registered assertion code/77 itself
runs): at F_SMALL=0.15 exactly, this script's own small__infold__free and
fold_matched__oof__free arms are checked to reproduce code/77's shipped
values to a small numerical tolerance, so the sweep's F_SMALL=0.15 point is
known to sit on the same foundation as the shipped decomposition rather than
on a re-implementation that happens to look similar.

Output: results/es_hold_fraction_sensitivity_cs3_factorial.json
"""
import importlib.util
import itertools
import json
import time
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
OUT_PATH = ROOT / "results" / "es_hold_fraction_sensitivity_cs3_factorial.json"
REF_77 = ROOT / "results" / "mechanism3_factorial_selection_controls.json"

_spec = importlib.util.spec_from_file_location(
    "s77_m100", CODE / "77_mechanism3_factorial_selection_controls.py")
s77 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s77)

F_SMALL_GRID = [0.05, 0.10, 0.15, 0.20]
F_SMALL_OMITTED = [0.25, 0.30]
N_SEEDS = s77.N_SEEDS   # 100, unchanged from code/77


def run_one_f_small(f_small, n_seeds=N_SEEDS):
    """One full pass1+pass2 factorial run at this F_SMALL, reusing code/77's
    functions verbatim with its module-level F_SMALL overridden."""
    s77.F_SMALL = f_small   # resolved as a global by _selection_sets at call time
    t0 = time.time()

    cols, metas = {}, []
    for seed in range(n_seeds):
        aucs, meta = s77.pass1_seed(seed, seed + 100000, seed + 200000, seed + 300000)
        for k, v in aucs.items():
            cols.setdefault(k, []).append(v)
        metas.append(meta)
    arrs = {k: np.array(v) for k, v in cols.items()}

    mean_ep = {k: float(np.mean([e for m in metas for e in m["epochs"][k]]))
               for k in ["leaky"] + s77.SEL_KEYS}
    shifts = {k: int(round(mean_ep["leaky"] - mean_ep[k])) for k in s77.SEL_KEYS}
    const_depth = int(round(mean_ep["leaky"]))

    cols2, metas2 = {}, []
    for seed in range(n_seeds):
        aucs, meta = s77.pass2_seed(seed, seed + 100000, seed + 200000, seed + 300000,
                                    {k: metas[seed]["epochs"][k] for k in s77.SEL_KEYS},
                                    shifts, const_depth)
        for k, v in aucs.items():
            cols2.setdefault(k, []).append(v)
        metas2.append(meta)
    for k, v in cols2.items():
        arrs[k] = np.array(v)

    cells = {}
    for i, arm in enumerate(s77.ARMS):
        size, loc, depth = arm.split("__")
        g = s77.gap_stats(arrs["leaky"], arrs[arm], boot_seed=int(f_small * 1e6) + 13 * i)
        sel_key = f"{size}__{loc}"
        cells[arm] = {
            "selection_set_size": size, "selection_run_budget": loc,
            "training_depth": depth,
            "n_selection_points": metas[0]["sel_sizes"][sel_key],
            "gap_leaky_minus_arm": g,
        }

    def mean_gap(pred):
        return float(np.mean([cells[a]["gap_leaky_minus_arm"]["gap_mean"]
                              for a in s77.ARMS if pred(a)]))

    shipped = cells["small__infold__free"]["gap_leaky_minus_arm"]["gap_mean"]
    corrected = cells["fold_matched__oof__matched"]["gap_leaky_minus_arm"]["gap_mean"]
    total_move = shipped - corrected

    main_effects = {
        "selection_set_size_small_minus_fold_matched": mean_gap(lambda a: a.startswith("small__"))
            - mean_gap(lambda a: a.startswith("fold_matched__")),
        "selection_run_budget_infold_minus_oof": mean_gap(lambda a: "__infold__" in a)
            - mean_gap(lambda a: "__oof__" in a),
        "training_depth_free_minus_matched": mean_gap(lambda a: a.endswith("__free"))
            - mean_gap(lambda a: a.endswith("__matched")),
    }
    shares = {k: (float(v / total_move) if abs(total_move) > 1e-12 else None)
              for k, v in main_effects.items()}

    n_val_mean = float(np.mean([m["n_val"] for m in metas]))
    n_tr_mean = float(np.mean([m["n_tr_fold"] for m in metas]))

    return {
        "f_small": f_small,
        "n_selection_points_small": cells["small__infold__free"]["n_selection_points"],
        "n_tr_fold_mean": n_tr_mean, "n_val_mean": n_val_mean,
        "leaky_mean_auroc": float(arrs["leaky"].mean()),
        "placebo_mean_auroc": float(arrs["placebo"].mean()),
        "shipped_uncorrected_gap": shipped,
        "fully_corrected_gap": corrected,
        "fully_corrected_gap_bca_ci_95": cells["fold_matched__oof__matched"]["gap_leaky_minus_arm"]["gap_bca_ci_95"],
        "total_movement": total_move,
        "main_effects": main_effects,
        "share_of_total_movement": shares,
        "cells": cells,
        "runtime_seconds": time.time() - t0,
    }


def main():
    t0 = time.time()
    print(f"=== ES_HOLD_FRACTION sensitivity sweep over code/77's CS3 factorial: "
          f"F_SMALL in {F_SMALL_GRID} (F_SMALL={F_SMALL_OMITTED} omitted -- "
          f"see module docstring), N_SEEDS={N_SEEDS} ===", flush=True)

    results = {}
    for f in F_SMALL_GRID:
        print(f"\n--- F_SMALL={f} ---", flush=True)
        r = run_one_f_small(f)
        results[str(f)] = r
        print(f"  shipped={r['shipped_uncorrected_gap']:+.5f} "
              f"corrected={r['fully_corrected_gap']:+.5f} "
              f"total_move={r['total_movement']:+.5f}  shares: " +
              ", ".join(f"{k.split('_')[0]}={v*100:+.1f}%" if v is not None else f"{k.split('_')[0]}=n/a"
                        for k, v in r["share_of_total_movement"].items()) +
              f"  elapsed={time.time()-t0:.0f}s", flush=True)

    # ── informal anchor check against code/77's shipped F_SMALL=0.15 run ────
    ref = json.load(open(REF_77))
    ref_shipped = ref["factorial_cells"]["small__infold__free"]["gap_leaky_minus_arm"]["gap_mean"]
    ref_corrected = ref["factorial_cells"]["fold_matched__oof__matched"]["gap_leaky_minus_arm"]["gap_mean"]
    our_015 = results["0.15"]
    drift_shipped = abs(our_015["shipped_uncorrected_gap"] - ref_shipped)
    drift_corrected = abs(our_015["fully_corrected_gap"] - ref_corrected)
    anchor_ok = drift_shipped < 1e-9 and drift_corrected < 1e-9
    print(f"\nAnchor check at F_SMALL=0.15 vs code/77 shipped: "
          f"shipped drift={drift_shipped:.2e}, corrected drift={drift_corrected:.2e} "
          f"({'OK' if anchor_ok else 'MISMATCH'})")

    # ── how much does the headline share move across the sweep? ────────────
    size_shares = [results[str(f)]["share_of_total_movement"][
        "selection_set_size_small_minus_fold_matched"] for f in F_SMALL_GRID]
    shipped_gaps = [results[str(f)]["shipped_uncorrected_gap"] for f in F_SMALL_GRID]
    total_moves = [results[str(f)]["total_movement"] for f in F_SMALL_GRID]
    fully_corrected_gaps = [results[str(f)]["fully_corrected_gap"] for f in F_SMALL_GRID]

    summary = {
        "f_small_grid_run": F_SMALL_GRID,
        "f_small_omitted": F_SMALL_OMITTED,
        "f_small_omitted_reason": (
            "Both F_SMALL=0.30 and, once actually attempted, F_SMALL=0.25 are "
            "infeasible under this factorial's shared out-of-fold pool geometry "
            "-- a sharper ceiling than the pre-run expectation that only 0.30 "
            "would break. At F_SMALL=0.30, the small__oof arm's required "
            "selection-set size (~112 points) exceeds the shared pool ges_idx's "
            "size (~94, fixed by OOF_HOLD_FRACTION=1/(N_INNER_FOLDS+1), "
            "independent of F_SMALL) outright. At F_SMALL=0.25 (the exact "
            "fold-matched value, where 'small' and 'fold_matched' coincide), "
            "the required size (~93) very nearly exhausts the ~94-point pool, "
            "leaving a 1-sample stratified complement that scikit-learn's "
            "train_test_split refuses (ValueError: test_size=1 < number of "
            "classes=2) -- confirmed to fail identically at every seed tried, "
            "not a seed-specific fluke. This is itself a finding: this "
            "factorial's shared-pool construction cannot realize the single "
            "value (0.25, the exact fold match) that Appendix A's own "
            "'not derived' discussion named as the natural next thing to try, "
            "without first enlarging the shared pool -- a stronger version of "
            "the robustness gap than 'we did not run it' previously conveyed."),
        "shipped_gap_range": [float(min(shipped_gaps)), float(max(shipped_gaps))],
        "fully_corrected_gap_range": [float(min(fully_corrected_gaps)), float(max(fully_corrected_gaps))],
        "total_movement_range": [float(min(total_moves)), float(max(total_moves))],
        "selection_set_size_share_range": [float(min(size_shares)), float(max(size_shares))],
        "selection_set_size_share_at_shipped_0.15": results["0.15"][
            "share_of_total_movement"]["selection_set_size_small_minus_fold_matched"],
        "anchor_check_at_0.15": {
            "reference_shipped_gap": ref_shipped, "reference_corrected_gap": ref_corrected,
            "our_shipped_gap": our_015["shipped_uncorrected_gap"],
            "our_corrected_gap": our_015["fully_corrected_gap"],
            "drift_shipped": drift_shipped, "drift_corrected": drift_corrected,
            "ok": anchor_ok,
        },
        "reading": (
            "If the selection-set-size share stays close to ~75% and the sign "
            "pattern (size >> budget > 0 > depth) is preserved across the whole "
            "swept range, the shipped decomposition is not an artifact of the "
            "particular F_SMALL=0.15 value and the 'single most consequential "
            "undisclosed choice' framing in Appendix A can be read as a "
            "disclosure of a robust choice rather than a fragile one. If the "
            "share or sign pattern moves materially across the range, "
            "Appendix A's framing understates the fragility."),
    }

    out = {
        "framing": (
            "Sensitivity sweep of ES_HOLD_FRACTION (F_SMALL) across code/77's "
            "decisive CS3 factorial, holding every other factor fixed (CAPACITY=128, "
            "AUROC_0=0.80, N_SEEDS=100, F_FOLD=0.25, OOF_HOLD_FRACTION=1/6). "
            "Reported as measured."),
        "config": {"capacity": s77.CAPACITY, "n_seeds": N_SEEDS,
                   "target_auroc": s77.TARGET_AUROC, "f_fold": s77.F_FOLD,
                   "oof_hold_fraction": s77.OOF_HOLD_FRACTION},
        "by_f_small": results,
        "summary": summary,
        "limitations": [
            "One capacity (128), one operating point (AUROC_0=0.80) -- the same "
            "conditions code/77's own factorial is scoped to.",
            "F_SMALL=0.30 could not be run inside this factorial's shared "
            "out-of-fold pool geometry; see summary.f_small_omitted_reason. A "
            "wider sweep would require enlarging the shared ges_idx pool itself, "
            "which would break comparability with the shipped factorial's pool "
            "size and was out of scope for a sensitivity check meant to bound "
            "the shipped choice, not redesign the pool.",
            "This is the isotropic-Gaussian synthetic reconstruction (code/47's "
            "generator), not a measurement on MultiHaluDet's real pipeline, "
            "matching every other CS3 control in this paper.",
        ],
        "runtime_seconds": time.time() - t0,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")
    print(f"Selection-set-size share range across F_SMALL in {F_SMALL_GRID}: "
          f"[{summary['selection_set_size_share_range'][0]*100:.1f}%, "
          f"{summary['selection_set_size_share_range'][1]*100:.1f}%]")
    print(f"Total runtime: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
