"""
the three severity axes on ONE comparable scale.

WHY THIS SCRIPT EXISTS. The paper measured three candidate severity axes in
three different places and never put them side by side. Its title and abstract
asserted that severity is governed by candidate count and operating point
"rather than mechanism" -- a comparative claim that Section 5.4 had already
withdrawn in the body (a transport check finds mechanism-and-harness
differences of roughly 9-24x at matched operating points, using this paper's
own currently-reported fold-matched-carveout-corrected severities as
numerators) and that the paper's own numbers, once placed on one scale, do not
support as strongly as an earlier revision claimed -- this axis is comparable
to, not established as larger than, the operating-point axis.

Each axis is expressed as the ratio between its largest and smallest cell with
the OTHER axes held fixed, which is the only way the three are comparable.

PRECISION IS REPORTED, NOT ASSUMED. Section 5.3 of the paper withdraws a
"48.6x" operating-point multiplier precisely because its denominator cell is
indistinguishable from zero, so the ratio of means has no finite upper
confidence limit. Quoting any of the three axis spans bare would repeat that
error. Every ratio here therefore carries a conservative interval
[num_lo / den_hi, num_hi / den_lo] from the two cells' own independent BCa
intervals, and every ratio whose denominator interval includes zero is FLAGGED
as having no finite upper limit. Each axis additionally gets a
DENOMINATOR-SOUND variant, restricted to cells whose own interval excludes
zero, which is the version safe to compare across axes:

  OPERATING POINT      -- within each K column of the K x AUROC_0 grid,
                          max/min across AUROC_0 in {0.70 ... 0.985}.
  MECHANISM + HARNESS  -- the transport check: same LEAKY-minus-CLEAN_MATCHED
                          contrast, operating points matched to within 0.0030
                          AUROC, leakage mechanic AND harness varied together.
                          This CONFOUNDS mechanic with harness, which the design
                          does not separate, so it bounds the mechanism axis
                          from above rather than measuring it -- enough for the
                          negative conclusion, not enough for a positive one.
  CANDIDATE COUNT      -- within each AUROC_0 row of the same grid, max/min
                          across K in {15, 45, 135, 405}; also reported with the
                          low-contrast K=15 column excluded.

Output: results/magnitude_triangle.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "magnitude_triangle.json"


def ratio_with_interval(num, num_ci, den, den_ci):
    """Conservative interval for num/den from two INDEPENDENT BCa intervals.

    If the denominator interval reaches zero the ratio has no finite upper
    limit -- the same fact that made Section 5.3 withdraw the 48.6x."""
    finite = den_ci[0] > 0
    return {
        "ratio": float(num / den),
        "ci_95": [float(num_ci[0] / den_ci[1]),
                  float(num_ci[1] / den_ci[0]) if finite else None],
        "denominator_ci_excludes_zero": bool(finite),
        "note": (None if finite else
                 "denominator interval includes zero: no finite upper confidence limit, "
                 "the same defect that retired the 48.6x multiplier (Section 5.3)"),
    }


def axis_span(cells, groups):
    """cells: {key: (gap, ci)}. groups: list of lists of keys held-fixed-along.

    Returns the raw max/min span per group and the denominator-sound variant."""
    raw, sound = {}, {}
    for name, keys in groups.items():
        vals = [(k, cells[k][0], cells[k][1]) for k in keys]
        hi = max(vals, key=lambda t: t[1])
        lo = min(vals, key=lambda t: t[1])
        raw[name] = {**ratio_with_interval(hi[1], hi[2], lo[1], lo[2]),
                     "numerator": hi[0], "denominator": lo[0]}
        ok = [v for v in vals if v[2][0] > 0]
        if len(ok) >= 2:
            h2 = max(ok, key=lambda t: t[1])
            l2 = min(ok, key=lambda t: t[1])
            sound[name] = {**ratio_with_interval(h2[1], h2[2], l2[1], l2[2]),
                           "numerator": h2[0], "denominator": l2[0]}
    return raw, sound


def main():
    js = json.load(open(ROOT / "results" / "joint_severity_surface.json"))["cells"]
    C = {(v["K"], v["target_auroc"]): (v["gap_mean"], v["gap_bca_ci_95"])
         for v in js.values()}
    G = {k: v[0] for k, v in C.items()}
    Ks = sorted({k for k, _ in G})
    As = sorted({a for _, a in G})

    op_raw, op_sound = axis_span(C, {str(k): [(k, a) for a in As] for k in Ks})
    k_raw, k_sound = axis_span(C, {str(a): [(k, a) for k in Ks] for a in As})
    Ks_nolow = [k for k in Ks if k != 15]
    knl_raw, knl_sound = axis_span(C, {str(a): [(k, a) for k in Ks_nolow] for a in As})

    op_spans = {k: v["ratio"] for k, v in op_raw.items()}
    k_spans = {k: v["ratio"] for k, v in k_raw.items()}
    k_spans_nolow = {k: v["ratio"] for k, v in knl_raw.items()}

    # ── mechanism-and-harness axis, with intervals (Section 5.4) ────────────
    tr = json.load(open(ROOT / "results" / "operating_point_transport_check.json"))
    mech_hi = float(tr["matched_operating_point_comparison"]["gap_spread_ratio"])
    ref = json.load(open(ROOT / "results" / "selection_multiplicity_sweep.json"))
    ref_c = ref["sweep_C_operating_point"]["0.95"]
    den, den_ci = ref_c["gap_mean"], ref_c["gap_bca_ci_95"]
    # Numerators: the fold-matched-carveout-corrected severities §4.3 actually
    # reports for these two harnesses at capacity 128 (+0.0060 and +0.0159),
    # NOT the superseded shipped/uncorrected gaps (+0.0093 and +0.0250) that
    # statistical_rigor_retrofit.json and mechanism3_fidelity_extension.json
    # ship. See results/real_feature_corrected_selection_controls.json and
    # results/fidelity_extension_corrected_selection_controls.json.
    rf_h = json.load(open(ROOT / "results" / "real_feature_corrected_selection_controls.json"))
    rf_h = rf_h["headline_comparison"]["128"]
    rf = {"gap_mean": rf_h["fold_matched_in_fold_gap"], "bca_ci_95": rf_h["fold_matched_in_fold_ci"]}
    fx_h = json.load(open(ROOT / "results" / "fidelity_extension_corrected_selection_controls.json"))
    fx_h = fx_h["headline_comparison"]["128"]
    fx = {"gap_mean": fx_h["fold_matched_in_fold_gap"], "bca_ci_95": fx_h["fold_matched_in_fold_ci"]}
    transport = {
        "real_feature_checkpoint_only_cap128":
            ratio_with_interval(rf["gap_mean"], rf["bca_ci_95"], den, den_ci),
        "fidelity_extension_cap128":
            ratio_with_interval(fx["gap_mean"], fx["bca_ci_95"], den, den_ci),
        "reference_cell": {"gap": den, "bca_ci_95": den_ci,
                           "wilcoxon_p": ref_c["wilcoxon_p"]},
    }

    def rng(d):
        v = list(d.values())
        return {"min": float(min(v)), "max": float(max(v)), "per_cell": d}

    out = {
        "purpose": ("The three severity axes on one comparable scale. Each is a max/min "
                    "ratio with the other axes held fixed."),
        "operating_point_axis": {
            **rng(op_spans),
            "with_intervals": op_raw,
            "denominator_sound": op_sound,
            "denominator_sound_range": [
                float(min(v["ratio"] for v in op_sound.values())),
                float(max(v["ratio"] for v in op_sound.values()))],
            "varied": "AUROC_0 from 0.70 to 0.985, mechanism and harness fixed",
            "caveat": ("all four AUROC_0=0.985 denominators have intervals that include "
                       "zero, so the full-range spans have no finite upper confidence "
                       "limit -- exactly the defect that retired the 48.6x multiplier. "
                       "Use denominator_sound for cross-axis comparison."),
            "source": "results/joint_severity_surface.json, within each K column",
        },
        "mechanism_and_harness_axis": {
            "min": None, "max": mech_hi,
            "reported_range_in_paper": [
                round(transport["real_feature_checkpoint_only_cap128"]["ratio"], 1),
                round(transport["fidelity_extension_cap128"]["ratio"], 1)],
            "probit_interpolated_range_in_paper": [15.1, 42.0],
            "probit_interpolated_range_stale": (
                "computed by code/82 from the same superseded shipped numerators "
                "(+0.0093, +0.0250) that this script used before this correction; "
                "code/82 has not itself been re-run against the corrected "
                "fold-matched-carveout severities (+0.0060, +0.0159), so this figure "
                "and the control/placebo-arm-matched figures derived from it "
                "(results/transport_check_arm_matching_sensitivity.json) remain stale "
                "pending that correction and should not be quoted as current"),
            "varied": ("leakage mechanic AND harness, operating point matched to within "
                       "0.0030 AUROC"),
            "confound": ("mechanic and harness are varied together; this design does not "
                         "separate them, so this row bounds the mechanism axis from above"),
            "with_intervals": transport,
            "denominator_ci_excludes_zero": True,
            "source": "results/operating_point_transport_check.json",
        },
        "candidate_count_axis": {
            **rng(k_spans),
            "with_intervals": k_raw,
            "denominator_sound": k_sound,
            "denominator_sound_range": [
                float(min(v["ratio"] for v in k_sound.values())),
                float(max(v["ratio"] for v in k_sound.values()))] if k_sound else None,
            "excluding_K15_column": {**rng(k_spans_nolow), "with_intervals": knl_raw},
            "varied": "K from 15 to 405, operating point fixed",
            "source": "results/joint_severity_surface.json, within each AUROC_0 row",
        },
    }
    op, kk = out["operating_point_axis"], out["candidate_count_axis"]
    ops = op["denominator_sound_range"]
    mh_lo = transport["real_feature_checkpoint_only_cap128"]["ratio"]
    mh_hi = transport["fidelity_extension_cap128"]["ratio"]
    out["denominator_sound_comparison"] = {
        "operating_point": ops,
        "mechanism_and_harness": [mh_lo, mh_hi],
        "candidate_count": kk["denominator_sound_range"],
        "statement": (
            "Restricted to ratios whose denominator interval excludes zero -- the only "
            "ones this paper's own Section 5.3 standard permits quoting, and using the "
            "fold-matched-carveout-corrected severities this paper currently reports as "
            "numerators (+0.0060, +0.0159; not the superseded shipped +0.0093, +0.0250) "
            f"-- the mechanism-and-harness axis ({mh_lo:.1f}-{mh_hi:.1f}x) is AT LEAST "
            f"COMPARABLE TO, AND PLAUSIBLY LARGER THAN, the operating-point axis "
            f"({ops[0]:.1f}-{ops[1]:.1f}x). It is NOT asserted to be strictly the largest "
            f"of the three: its own lower end ({mh_lo:.1f}x) sits below the "
            f"operating-point axis's upper end ({ops[1]:.1f}x), so the two ranges overlap "
            "substantially. Candidate count remains the smallest on every convention."),
    }
    dsc = out["denominator_sound_comparison"]
    out["verdict"] = {
        "statement": (
            f"AS RAW SPANS: operating point {op['min']:.1f}-{op['max']:.1f}x within this "
            f"harness; mechanism-and-harness up to {mech_hi:.1f}x across harnesses at a "
            f"matched operating point; candidate count {kk['min']:.1f}-{kk['max']:.1f}x "
            f"within an operating point ({kk['excluding_K15_column']['min']:.1f}-"
            f"{kk['excluding_K15_column']['max']:.1f}x excluding the low-contrast K=15 "
            f"column). The first two are the same order of magnitude and the third is "
            f"roughly an order weaker -- but the first and third use denominators whose "
            f"intervals include zero. RESTRICTED TO SOUND DENOMINATORS, which is the only "
            f"comparison this paper's own Section 5.3 standard permits: operating point "
            f"{dsc['operating_point'][0]:.1f}-{dsc['operating_point'][1]:.1f}x, "
            f"mechanism-and-harness {dsc['mechanism_and_harness'][0]:.1f}-"
            f"{dsc['mechanism_and_harness'][1]:.1f}x, candidate count "
            f"{dsc['candidate_count'][0]:.1f}-{dsc['candidate_count'][1]:.1f}x -- and the "
            f"mechanism-and-harness axis is AT LEAST COMPARABLE TO, AND PLAUSIBLY LARGER "
            f"THAN, the operating-point axis (not asserted as strictly the largest of the "
            f"three: their sound ranges overlap). On either reading the paper's former "
            f"claim that severity is governed by candidate count and operating point "
            f"RATHER THAN mechanism is not merely unproven; these numbers point the other "
            f"way. It is withdrawn from the title and abstract. NOTE: the RAW-SPAN "
            f"mechanism-and-harness figure above ({mech_hi:.1f}x) is sourced from "
            f"operating_point_transport_check.json's own gap_spread_ratio and, like the "
            f"probit-interpolated and arm-matched sensitivity figures, has not been "
            f"re-derived from the corrected fold-matched-carveout severities in this "
            f"round; only the denominator-sound with_intervals figures above use the "
            f"corrected numerators."),
        "operating_point_over_candidate_count": float(op["max"] / kk["max"]),
        "mechanism_over_candidate_count": float(mech_hi / kk["max"]),
        "operating_point_over_mechanism": float(op["max"] / mech_hi),
        "precision_caveat": (
            "The operating-point and candidate-count full-range spans quoted above use "
            "denominators whose intervals include zero and therefore have no finite upper "
            "confidence limit. See denominator_sound_comparison for the version that "
            "meets this paper's own Section 5.3 standard."),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(out["verdict"]["statement"])
    print(f"\n  operating point / candidate count = "
          f"{out['verdict']['operating_point_over_candidate_count']:.1f}x")
    print(f"  mechanism+harness / candidate count = "
          f"{out['verdict']['mechanism_over_candidate_count']:.1f}x")
    print(f"  operating point / mechanism+harness = "
          f"{out['verdict']['operating_point_over_mechanism']:.2f}x")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
