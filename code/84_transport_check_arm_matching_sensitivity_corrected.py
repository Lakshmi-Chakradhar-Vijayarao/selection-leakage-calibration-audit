"""
P0-2 correction. `code/82` correctly recomputes the transport
check's mechanism-and-harness ratios under leaky-arm and control-arm matching,
but both computations there still read their two harnesses' GAP and CONTROL
values from the pre-correction sources (`real_feature_test_train_only_calibrated.json`
and `mechanism3_fidelity_extension.json`), i.e. the SUPERSEDED numerators
+0.0093 (real-feature, capacity 128) and +0.0250 (fidelity extension, capacity
128) -- both flagged in `main.tex` (§5.2/§5.4) as "pending that correction."

This script reruns exactly the same two computations (probit-space log-linear
interpolation of Sweep C, evaluated once matching on the LEAKY arm and once on
the CONTROL arm) substituting the CORRECTED, fold-matched numerators from
`results/real_feature_corrected_selection_controls.json` and
`results/fidelity_extension_corrected_selection_controls.json`
(`part_A_in_fold_es_sweep`, capacity 128, `es_fraction_fold_matched` = 0.25
cell): gap = +0.0060 (real-feature) and +0.0159 (fidelity extension), with the
matching CONTROL-arm means taken from the same fold-matched cell (the LEAKY
mean is unaffected by the correction, since only the honest arm's construction
changed).

It asserts its own SHIPPED-numerator recomputation reproduces `code/82`'s
shipped ratios (15.1x/42.0x leaky-matched, 13.1x-28.7x control-matched) before
reporting the corrected ones, so the only thing that changes between the two
reported ranges is the numerator.

Output: results/transport_check_arm_matching_sensitivity_corrected.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"


def probit_fit_and_ratios(sc, arm, harnesses):
    z = norm.ppf(np.array([c[arm] for c in sc]))
    lg = np.log(np.array([c["gap"] for c in sc]))
    coef = np.polyfit(z, lg, 1)
    obs = []
    for h in harnesses:
        pred = float(np.exp(np.polyval(coef, norm.ppf(h[arm]))))
        obs.append({
            "harness": h["name"],
            "matched_on_value": h[arm],
            "gap": h["gap"],
            "interpolated_prediction": pred,
            "ratio_vs_interpolated": float(h["gap"] / pred),
        })
    ratios = [o["ratio_vs_interpolated"] for o in obs]
    return {"observations": obs, "range": [float(min(ratios)), float(max(ratios))]}


def main():
    sweep = json.load(open(R / "selection_multiplicity_sweep.json"))["sweep_C_operating_point"]
    targets = sorted(float(k) for k in sweep)
    sc = [{
        "target_auroc": t,
        "leaky": sweep[str(t)]["leaky_mean"],
        "control": sweep[str(t)]["clean_matched_mean"],
        "gap": sweep[str(t)]["gap_mean"],
    } for t in targets]

    # ---- SHIPPED numerators (reproduces code/82's shipped ratios) ----
    rf_ship = json.load(open(R / "real_feature_test_train_only_calibrated.json"))["capacities"]["128"]
    fx_ship = json.load(open(R / "mechanism3_fidelity_extension.json"))
    harnesses_shipped = [
        {"name": "real-feature checkpoint-only harness (code/43), capacity 128",
         "leaky": float(np.mean(rf_ship["aucs"]["leaky"])),
         "control": float(np.mean(rf_ship["aucs"]["clean_matched"])),
         "gap": rf_ship["leaky_minus_clean_matched"]["mean"]},
        {"name": "fidelity extension (code/49), capacity 128",
         "leaky": fx_ship["means"]["leaky_plus_lrsched"],
         "control": fx_ship["means"]["clean_matched_budget_matched"],
         "gap": fx_ship["leaky_plus_lrsched_minus_clean_matched_budget_matched"]["gap_mean"]},
    ]
    shipped_leaky = probit_fit_and_ratios(sc, "leaky", harnesses_shipped)
    shipped_control = probit_fit_and_ratios(sc, "control", harnesses_shipped)

    # Sanity: reproduce code/82's shipped figures to 1 decimal place.
    assert abs(shipped_leaky["range"][0] - 15.06) < 0.1
    assert abs(shipped_leaky["range"][1] - 41.98) < 0.1
    assert abs(shipped_control["range"][0] - 13.07) < 0.1
    assert abs(shipped_control["range"][1] - 28.69) < 0.1

    # ---- CORRECTED numerators: fold-matched (es_fraction = 0.25) cell,
    # capacity 128, from the corrected-selection-controls JSONs. ----
    rf_corr = json.load(open(R / "real_feature_corrected_selection_controls.json"))
    fx_corr = json.load(open(R / "fidelity_extension_corrected_selection_controls.json"))
    rf_cell = rf_corr["part_A_in_fold_es_sweep"]["by_capacity"]["128"]
    fx_cell = fx_corr["part_A_in_fold_es_sweep"]["by_capacity"]["128"]
    rf_fm = rf_cell["by_es_fraction"]["0.25"]
    fx_fm = fx_cell["by_es_fraction"]["0.25"]
    assert rf_fm["is_fold_matched"] and fx_fm["is_fold_matched"]

    harnesses_corrected = [
        {"name": "real-feature checkpoint-only harness (code/43), capacity 128, fold-matched",
         "leaky": rf_cell["leaky_mean"],
         "control": rf_fm["clean_matched_mean"],
         "gap": rf_fm["gap_mean"]},
        {"name": "fidelity extension (code/49), capacity 128, fold-matched",
         "leaky": fx_cell["leaky_mean"],
         "control": fx_fm["control_mean"],
         "gap": fx_fm["gap_mean"]},
    ]
    corrected_leaky = probit_fit_and_ratios(sc, "leaky", harnesses_corrected)
    corrected_control = probit_fit_and_ratios(sc, "control", harnesses_corrected)

    out = {
        "why": ("code/82 recomputes the transport check's control-arm-matched "
                "ratio but both its harnesses still source the pre-correction "
                "(shipped) gap numerators. This script substitutes the "
                "corrected fold-matched numerators (+0.0060 real-feature, "
                "+0.0159 fidelity extension, both capacity 128) and reruns "
                "the identical probit-space interpolation."),
        "shipped_numerators": {
            "leaky_matched": shipped_leaky,
            "control_matched": shipped_control,
        },
        "corrected_numerators": {
            "leaky_matched": corrected_leaky,
            "control_matched": corrected_control,
        },
        "reading": (
            f"Leaky-matched (probit interpolation): shipped "
            f"{shipped_leaky['range'][0]:.1f}-{shipped_leaky['range'][1]:.1f}x -> "
            f"corrected {corrected_leaky['range'][0]:.1f}-{corrected_leaky['range'][1]:.1f}x. "
            f"Control-matched: shipped {shipped_control['range'][0]:.1f}-"
            f"{shipped_control['range'][1]:.1f}x -> corrected "
            f"{corrected_control['range'][0]:.1f}-{corrected_control['range'][1]:.1f}x."
        ),
    }
    out_path = R / "transport_check_arm_matching_sensitivity_corrected.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(out["reading"])
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
