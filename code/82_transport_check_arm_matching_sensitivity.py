"""
on WHICH ARM is the transport check's operating point matched, and
does the answer change the mechanism-and-harness axis?

WHY THIS SCRIPT EXISTS. `code/58` reports that three harnesses measuring the
same LEAKY-minus-CLEAN_MATCHED contrast sit at operating points "matched to
within 0.0030 AUROC" and nevertheless differ in severity by 14.3x and 38.3x.
That 14.3-38.3x range is the mechanism-and-harness axis of the paper's
magnitude triangle, and the retitled paper's central comparative claim rests on
it being the largest of the three axes.

A third independent review observed that `code/58:119` reads
`achieved_operating_point` from a LEAKY quantity in all three rows
(`sweep["0.95"]["leaky_mean"]`, `mean(rf["aucs"]["leaky"])`,
`fx["means"]["leaky_plus_lrsched"]`). No control or placebo arm is read
anywhere in that script. So the 0.0030 agreement is an agreement between the
three harnesses' LEAKY arms and nothing else.

That matters because the reported quantity is a DIFFERENCE, LEAKY minus
CONTROL. If the LEAKY arms are pinned to within 0.0030 while the CONTROL arms
are free to differ by an order of magnitude more, then the gap spread the check
reports as its finding is substantially the control spread, and the choice to
match on LEAKY is the choice that maximizes it.

WHAT THIS SCRIPT DOES. It recomputes the entire transport check three times,
matching on each arm in turn, changing nothing else:

  (a) LEAKY arm      -- code/58's shipped convention, recomputed here and
                        asserted against code/58's shipped JSON so the other
                        two conventions sit on a verified reimplementation.
  (b) CONTROL arm    -- each harness's honest-selection control, i.e. the
                        denominator's own partner in the difference being
                        reported.
  (c) PLACEBO arm    -- each harness's zero-signal reference.

Under each convention, both of code/58's estimators are recomputed: the
nearest-Sweep-C-cell ratio, and the ratio against a probit-space log-linear
interpolation of Sweep C REFIT ON THAT ARM (so the prediction is made in the
same coordinate the matching is done in, rather than borrowing the LEAKY fit).

WHAT IT CANNOT SETTLE. None of the three conventions is uniquely correct. A
severity comparison at a "matched operating point" is only well posed if the
whole harness is matched, and these three harnesses differ in generative
process, feature dimension, sample size, optimizer, scaler and leakage mechanic
all at once -- which is exactly why the axis is labelled "mechanism AND
harness." The point of running all three is that the paper should not quote the
convention that gives the largest number without saying that the number depends
on the convention, and by how much.

Output: results/transport_check_arm_matching_sensitivity.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT_PATH = R / "transport_check_arm_matching_sensitivity.json"
REF_58 = R / "operating_point_transport_check.json"

# Which JSON field carries each harness's each-arm mean. The control column
# uses code/49's BUDGET-MATCHED control, matching the arm whose gap code/58
# actually reports (its 6th correction); the non-budget-matched alternative is
# reported separately below because the paper retracts it.
ARMS = ["leaky", "control", "placebo"]


def main():
    sweep = json.load(open(R / "selection_multiplicity_sweep.json"))["sweep_C_operating_point"]
    rf = json.load(open(R / "real_feature_test_train_only_calibrated.json"))["capacities"]["128"]
    fx = json.load(open(R / "mechanism3_fidelity_extension.json"))
    ref58 = json.load(open(REF_58))

    targets = sorted(float(k) for k in sweep)
    sc = [{
        "target_auroc": t,
        "leaky": sweep[str(t)]["leaky_mean"],
        "control": sweep[str(t)]["clean_matched_mean"],
        "placebo": sweep[str(t)]["placebo_mean"],
        "gap": sweep[str(t)]["gap_mean"],
    } for t in targets]

    harnesses = [
        {"name": "real-feature checkpoint-only harness (code/43), capacity 128",
         "short": "code/43",
         "leaky": float(np.mean(rf["aucs"]["leaky"])),
         "control": float(np.mean(rf["aucs"]["clean_matched"])),
         "placebo": float(np.mean(rf["aucs"]["placebo"])),
         "gap": rf["leaky_minus_clean_matched"]["mean"]},
        {"name": "fidelity extension (code/49), capacity 128",
         "short": "code/49",
         "leaky": fx["means"]["leaky_plus_lrsched"],
         "control": fx["means"]["clean_matched_budget_matched"],
         "placebo": fx["means"]["placebo_plus_lrsched"],
         "gap": fx["leaky_plus_lrsched_minus_clean_matched_budget_matched"]["gap_mean"]},
    ]

    # ── How well is each arm actually matched across the three harnesses? ─────
    # Sweep C's reference row is its AUROC_0=0.95 cell, the one code/58 selects.
    ref_cell = min(sc, key=lambda c: abs(c["leaky"] - 0.9413))
    spreads = {}
    for arm in ARMS:
        vals = [ref_cell[arm]] + [h[arm] for h in harnesses]
        pair = [abs(a - b) for i, a in enumerate(vals) for b in vals[i + 1:]]
        spreads[arm] = {
            "values": {"sweep_C_0.95": ref_cell[arm],
                       **{h["short"]: h[arm] for h in harnesses}},
            "max_pairwise_spread": float(max(pair)),
            "pairwise": [float(p) for p in pair],
        }
    leaky_spread = spreads["leaky"]["max_pairwise_spread"]
    for arm in ARMS:
        spreads[arm]["ratio_to_leaky_spread"] = float(
            spreads[arm]["max_pairwise_spread"] / leaky_spread)

    # ── The check, recomputed once per matching arm ───────────────────────────
    by_arm = {}
    for arm in ARMS:
        z = norm.ppf(np.array([c[arm] for c in sc]))
        lg = np.log(np.array([c["gap"] for c in sc]))
        coef = np.polyfit(z, lg, 1)
        r2 = float(1 - np.sum((lg - np.polyval(coef, z)) ** 2) / np.sum((lg - lg.mean()) ** 2))

        obs = []
        for h in harnesses:
            near = min(sc, key=lambda c: abs(c[arm] - h[arm]))
            pred = float(np.exp(np.polyval(coef, norm.ppf(h[arm]))))
            obs.append({
                "harness": h["name"], "short": h["short"],
                "matched_on_value": h[arm],
                "nearest_sweep_C_cell_target": near["target_auroc"],
                "nearest_sweep_C_cell_value": near[arm],
                "nearest_sweep_C_cell_gap": near["gap"],
                "gap": h["gap"],
                "ratio_vs_nearest_cell": float(h["gap"] / near["gap"]),
                "interpolated_prediction": pred,
                "ratio_vs_interpolated": float(h["gap"] / pred),
            })
        nearest = [o["ratio_vs_nearest_cell"] for o in obs]
        interp = [o["ratio_vs_interpolated"] for o in obs]
        by_arm[arm] = {
            "matched_arm": arm,
            "arm_spread_across_harnesses": spreads[arm]["max_pairwise_spread"],
            "probit_loglinear_fit": {"slope_m": float(coef[0]),
                                     "intercept_b": float(coef[1]),
                                     "r_squared_in_log_space": r2},
            "observations": obs,
            "mechanism_axis_range_nearest_cell": [float(min(nearest)), float(max(nearest))],
            "mechanism_axis_range_interpolated": [float(min(interp)), float(max(interp))],
        }

    # ── Assert the LEAKY convention reproduces code/58 before reporting ──────
    shipped = {o["name"]: o for o in
               ref58["matched_operating_point_comparison"]["observations"]}
    checks = []
    for o in by_arm["leaky"]["observations"]:
        s = shipped[o["harness"]]
        d1 = abs(o["ratio_vs_nearest_cell"] - s["ratio_vs_sweep_C_matched_cell"])
        d2 = abs(o["ratio_vs_interpolated"] - s["ratio_vs_sweep_C_interpolated"])
        assert d1 < 1e-9 and d2 < 1e-9, (
            f"LEAKY-matched reimplementation drifted from code/58 for "
            f"{o['harness']}: {d1:.2e}, {d2:.2e}")
        checks.append({"harness": o["harness"], "nearest_drift": d1, "interp_drift": d2})
    assert abs(ref58["matched_operating_point_comparison"]["operating_point_spread"]
               - leaky_spread) < 1e-12, "LEAKY spread drifted from code/58"

    # ── The non-budget-matched fidelity control, for completeness ─────────────
    alt_gap = fx["leaky_plus_lrsched_minus_clean_matched_plus_lrsched"]["gap_mean"]
    alt = {
        "what": ("code/49's non-budget-matched control "
                 "(clean_matched_plus_lrsched), which this paper retracts: its "
                 "sanity ratio is 1.60, outside the 0.06-0.85 comparator band, "
                 "the signature of a degraded control."),
        "gap": alt_gap,
        "ratio_vs_leaky_matched_nearest_cell": float(alt_gap / ref_cell["gap"]),
        "why_reported": ("code/58's docstring quoted this arm's +0.0338 until the "
                         "third review; 0.0338/0.00065 = 52x is the long-retracted "
                         "multiplier its arithmetic reproduced. Reported here so the "
                         "retraction is a number in the record, not only prose."),
        "control_arm_mean": fx["means"]["clean_matched_plus_lrsched"],
    }

    # ── The fidelity extension across capacity, which the axis's upper end
    # rests on. code/58 used capacity 128 only; code/79 adds 384 and both the
    # shipped and fold-matched controls, so the axis's top can be reported as a
    # grid rather than as its maximum cell.
    fxc_p = R / "fidelity_extension_corrected_selection_controls.json"
    fid_grid = {}
    if fxc_p.exists():
        fxc = json.load(open(fxc_p))
        for cap, cell in fxc["part_A_in_fold_es_sweep"]["by_capacity"].items():
            for f, key in (("0.15", "shipped"), ("0.25", "fold_matched")):
                g = cell["by_es_fraction"][f]["gap_mean"]
                fid_grid[f"capacity_{cap}__{key}"] = {
                    "gap": g,
                    "ratio_vs_sweep_C_0.95_cell": float(g / ref_cell["gap"]),
                    "n_selection_points": cell["by_es_fraction"][f]["n_selection_points"],
                }
        r128 = fid_grid["capacity_128__fold_matched"]["ratio_vs_sweep_C_0.95_cell"]
        r384 = fid_grid["capacity_384__shipped"]["ratio_vs_sweep_C_0.95_cell"]
        fid_grid["reading"] = (
            f"The fidelity extension's transport ratio falls from "
            f"{fid_grid['capacity_128__shipped']['ratio_vs_sweep_C_0.95_cell']:.1f}x "
            f"(capacity 128, shipped control) to {r128:.1f}x (capacity 128, "
            f"fold-matched) to {r384:.1f}x (capacity 384, shipped) and "
            f"{fid_grid['capacity_384__fold_matched']['ratio_vs_sweep_C_0.95_cell']:.1f}x "
            f"(capacity 384, fold-matched). The 38.3x that topped the "
            f"mechanism-and-harness axis is therefore specific to one capacity AND "
            f"one control construction; at capacity 384 the ratio falls below the "
            f"operating-point axis's own sound maximum of 14.3x.")

    out = {
        "fidelity_extension_across_capacity_and_control": fid_grid,
        "why": ("code/58 matches its three harnesses on the LEAKY arm only. This "
                "recomputes the same check matching on the control and placebo "
                "arms instead, changing nothing else."),
        "reference_cell_sweep_C": ref_cell,
        "arm_match_quality": spreads,
        "by_matching_arm": by_arm,
        "reproduces_code58_under_leaky_matching": checks,
        "retracted_non_budget_matched_control": alt,
        "headline": {
            "leaky_matched_shipped": {
                "nearest_cell": by_arm["leaky"]["mechanism_axis_range_nearest_cell"],
                "interpolated": by_arm["leaky"]["mechanism_axis_range_interpolated"]},
            "control_matched": {
                "nearest_cell": by_arm["control"]["mechanism_axis_range_nearest_cell"],
                "interpolated": by_arm["control"]["mechanism_axis_range_interpolated"]},
            "placebo_matched": {
                "nearest_cell": by_arm["placebo"]["mechanism_axis_range_nearest_cell"],
                "interpolated": by_arm["placebo"]["mechanism_axis_range_interpolated"]},
        },
        "reading": (
            "The three harnesses' LEAKY arms agree to "
            f"{spreads['leaky']['max_pairwise_spread']:.4f} AUROC, their control arms "
            f"to {spreads['control']['max_pairwise_spread']:.4f} "
            f"({spreads['control']['ratio_to_leaky_spread']:.1f}x worse) and their "
            f"placebo arms to {spreads['placebo']['max_pairwise_spread']:.4f} "
            f"({spreads['placebo']['ratio_to_leaky_spread']:.1f}x worse). Since the "
            "quantity being compared is LEAKY minus CONTROL, matching on LEAKY alone "
            "leaves the control spread inside the ratio. The mechanism-and-harness "
            "axis is therefore not a single range but a convention-dependent one, and "
            "the paper reports all three rather than the largest."),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print("Arm-match quality across the three harnesses (max pairwise spread):")
    for arm in ARMS:
        s = spreads[arm]
        print(f"  {arm:<8} {s['max_pairwise_spread']:.4f} AUROC "
              f"({s['ratio_to_leaky_spread']:.1f}x the LEAKY spread)   "
              + "  ".join(f"{k}={v:.4f}" for k, v in s["values"].items()))
    print("\nTransport ratios by matching arm:")
    print(f"{'matched on':<10} {'harness':<10} {'nearest cell':>14} {'interpolated':>14}")
    for arm in ARMS:
        for o in by_arm[arm]["observations"]:
            print(f"{arm:<10} {o['short']:<10} {o['ratio_vs_nearest_cell']:>13.2f}x "
                  f"{o['ratio_vs_interpolated']:>13.2f}x")
        r1 = by_arm[arm]["mechanism_axis_range_nearest_cell"]
        r2 = by_arm[arm]["mechanism_axis_range_interpolated"]
        print(f"{'':<10} {'-> axis':<10} {r1[0]:>7.1f}-{r1[1]:.1f}x "
              f"{r2[0]:>10.1f}-{r2[1]:.1f}x")
    print(f"\nRetracted non-budget-matched control: gap {alt_gap:+.4f} -> "
          f"{alt['ratio_vs_leaky_matched_nearest_cell']:.1f}x "
          "(the long-retracted multiplier code/58's docstring reproduced)")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
