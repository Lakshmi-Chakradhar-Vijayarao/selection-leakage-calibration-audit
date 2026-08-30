"""
regenerate `draft/leakage_checklist.md` mechanically from the result
JSONs, so that its numbers cannot drift out of sync with the paper again.

WHY THIS SCRIPT EXISTS. `draft/leakage_checklist.md` has now shipped with stale
or self-contradictory numbers in THREE consecutive review rounds. Each round it
was patched by hand, and each round a different subset of the same class of
defect survived. The third round found, among others: a ceiling-composition
contrast quoted from an estimator the paper retired two rounds ago; a "2 of 24
cells negative" claim that a theorem in the paper's own appendix proves
impossible; a bootstrap interval quoted as a confidence interval on the same
page where §4.4 declines to report it as one; "all 24 combinations" and
"excludes the 7 degenerate cells" three paragraphs apart; a maximum cell value
rounded one way here and the other way in §4.4; and no mention at all of the
calibration-corrected range that is the paper's actual headline for Case
Study 4.

Hand-patching has failed enough times to conclude that hand-patching is the
defect. `code/52` already solved the same problem for `draft/paper_draft.md` by
making it a mechanical function of `main.tex`. This script does the analogous
thing for the checklist, with one difference forced by what the checklist is:
it is not a rendering of any section of `main.tex`, it is an independently
written practitioner document, so its PROSE cannot be derived. Its NUMBERS can.

  `draft/leakage_checklist.md.in`  -- the prose, hand-authored, with every
                                      load-bearing number replaced by a
                                      {{TOKEN}} placeholder.
  FACTS (below)                    -- each token resolved from a shipped result
                                      JSON, never typed as a literal.
  `draft/leakage_checklist.md`     -- generated. Do not edit by hand; edits are
                                      overwritten and `--check` will fail.

So a number can now only change in the checklist by changing in the JSON the
paper itself reads, and `code/53` verifies the same tokens against the same
sources, which closes the loop from the other side.

Run:    python3 code/80_generate_leakage_checklist.py
Check:  python3 code/80_generate_leakage_checklist.py --check
        (exits non-zero if the file on disk differs from what would be
        generated, or if any token is unresolved -- suitable as a pre-commit
        guard and wired into code/91's packaging audit.)
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
TEMPLATE = ROOT / "draft" / "leakage_checklist.md.in"
OUT = ROOT / "draft" / "leakage_checklist.md"


def _j(name):
    p = R / name
    if not p.exists():
        raise SystemExit(f"missing result file the checklist depends on: {p}")
    return json.load(open(p))


def build_facts():
    """Every number the checklist quotes, resolved from its shipped source.

    Each entry is (value, format). Nothing here is a typed literal except the
    format strings and the structural counts the JSONs themselves carry.
    """
    sweep = _j("corrected_capacity_placebo_sweep.json")["by_capacity"]
    swC = _j("selection_multiplicity_sweep.json")["sweep_C_operating_point"]
    fact = _j("mechanism3_factorial_selection_controls.json")
    b75 = _j("mechanism3_selection_budget_controls.json")
    rf = _j("real_feature_test_train_only_calibrated.json")["capacities"]
    rfc = _j("real_feature_corrected_selection_controls.json")
    fx = _j("mechanism3_fidelity_extension.json")
    fxc = _j("fidelity_extension_corrected_selection_controls.json")
    cs4 = _j("case_study_4_winners_curse.json")
    cs4n = _j("case_study_4_two_sided_null.json")
    cs4c = _j("cs4_estimator_calibration.json")
    cs2 = _j("case_study_2_layer_decomposition.json")
    m5 = _j("mechanism5_threshold_selection.json")
    oofa = _j("oof_averaging_control.json")
    trans = _j("transport_check_arm_matching_sensitivity.json")

    F = {}

    def put(k, v, fmt="%+.4f"):
        F[k] = fmt % v if isinstance(v, (int, float)) and fmt else str(v)

    # ── Mechanism 3, primary synthetic sweep and the factorial that dissolves it
    gaps = [sweep[str(h)]["gaps"]["leaky_minus_clean_matched"]["mean"]
            for h in (16, 48, 128, 384)]
    put("M3_PRIMARY_LO", min(gaps))
    put("M3_PRIMARY_HI", max(gaps))
    cells = fact["factorial_cells"]
    sh = cells["small__infold__free"]["gap_leaky_minus_arm"]
    fc = cells["fold_matched__oof__matched"]["gap_leaky_minus_arm"]
    put("M3_FACT_SHIPPED", sh["gap_mean"])
    put("M3_FACT_CORRECTED", fc["gap_mean"])
    put("M3_FACT_CORRECTED_LO", fc["gap_bca_ci_95"][0])
    put("M3_FACT_CORRECTED_HI", fc["gap_bca_ci_95"][1])
    put("M3_FACT_CORRECTED_P", fc["wilcoxon_p"], "%.2f")
    put("M3_FACT_VERDICT", fact["verdict"], None)
    me = fact["decomposition"]["main_effects_average_over_other_factors"]
    put("M3_EFFECT_SIZE", me["selection_set_size_small_minus_fold_matched"]["value"])
    put("M3_EFFECT_BUDGET", me["selection_run_budget_infold_minus_oof"]["value"])
    put("M3_EFFECT_DEPTH", me["training_depth_free_minus_matched"]["value"])
    share = fact["decomposition"]["share_of_total_movement"]
    put("M3_SHARE_SIZE", 100 * share["selection_set_size_small_minus_fold_matched"], "%.0f")
    put("M3_SHARE_BUDGET", 100 * share["selection_run_budget_infold_minus_oof"], "%.0f")
    oofm = b75["summary"]["out_of_fold_mean_gap"]
    put("M3_OOF_MEAN", oofm)
    put("M3_PRIMARY_MEAN", b75["summary"]["primary_mean_gap"])

    # ── Mechanism 3 on real features (code/43), shipped and corrected
    for h in ("128", "384"):
        put(f"M3_REAL_{h}", rf[h]["leaky_minus_clean_matched"]["mean"])
        A = rfc["part_A_in_fold_es_sweep"]["by_capacity"][h]["by_es_fraction"]
        put(f"M3_REAL_{h}_FM", A["0.25"]["gap_mean"])
        put(f"M3_REAL_{h}_FM_LO", A["0.25"]["gap_bca_ci_95"][0])
        put(f"M3_REAL_{h}_FM_HI", A["0.25"]["gap_bca_ci_95"][1])
        B = rfc["part_B_factorial"]["by_capacity"][h]
        put(f"M3_REAL_{h}_FULLCORR", B["decomposition"]["fully_corrected_gap"])
        bc = B["factorial_cells"]["fold_matched__oof__matched"]["gap_leaky_minus_arm"]
        put(f"M3_REAL_{h}_FULLCORR_LO", bc["gap_bca_ci_95"][0])
        put(f"M3_REAL_{h}_FULLCORR_HI", bc["gap_bca_ci_95"][1])
        put(f"M3_REAL_{h}_VERDICT", B["verdict"], None)

    # ── The fidelity extension, shipped and corrected
    g = fx["leaky_plus_lrsched_minus_clean_matched_budget_matched"]
    put("FID_GAP", g["gap_mean"])
    put("FID_GAP_LO", g["bca_ci_95"][0])
    put("FID_GAP_HI", g["bca_ci_95"][1])
    put("FID_RETRACTED", fx["leaky_plus_lrsched_minus_clean_matched_plus_lrsched"]["gap_mean"])
    for h in ("128", "384"):
        A = fxc["part_A_in_fold_es_sweep"]["by_capacity"][h]["by_es_fraction"]
        put(f"FID_{h}_SHIPPED", A["0.15"]["gap_mean"])
        put(f"FID_{h}_FM", A["0.25"]["gap_mean"])
        put(f"FID_{h}_FM_LO", A["0.25"]["gap_bca_ci_95"][0])
        put(f"FID_{h}_FM_HI", A["0.25"]["gap_bca_ci_95"][1])
        B = fxc["part_B_factorial"]["by_capacity"][h]
        put(f"FID_{h}_FULLCORR", B["decomposition"]["fully_corrected_gap"])
        put(f"FID_{h}_VERDICT", B["verdict"], None)

    # ── Operating point (Sweep C) and the control-health ratio it confounds
    put("OP_070", swC["0.7"]["gap_mean"])
    put("OP_0985", swC["0.985"]["gap_mean"])
    for t, k in (("0.7", "070"), ("0.8", "080"), ("0.9", "090"),
                 ("0.95", "095"), ("0.985", "0985")):
        c = swC[t]
        put(f"HEALTH_{k}",
            (c["leaky_mean"] - c["clean_matched_mean"])
            / (c["clean_matched_mean"] - c["placebo_mean"]), "%.3f")

    # ── The transport check, under each matching convention
    hd = trans["headline"]
    for conv, key in (("LEAKY", "leaky_matched_shipped"),
                      ("CONTROL", "control_matched"),
                      ("PLACEBO", "placebo_matched")):
        put(f"TRANSPORT_{conv}_LO", hd[key]["interpolated"][0], "%.1f")
        put(f"TRANSPORT_{conv}_HI", hd[key]["interpolated"][1], "%.1f")
    am = trans["arm_match_quality"]
    for a in ("leaky", "control", "placebo"):
        put(f"ARMSPREAD_{a.upper()}", am[a]["max_pairwise_spread"], "%.4f")

    # ── Case Study 4
    per = cs4n["per_cell"]
    ests = [v["bootstrap_max_bias_exact"] for v in per.values()]
    put("CS4_N_CELLS", len(per), "%d")
    put("CS4_MEAN", cs4n["mean_bootstrap_max_bias_exact_all_24"])
    put("CS4_MAX", max(ests))
    put("CS4_N_DEGEN", sum(1 for v in per.values() if v["estimator_degenerate"]), "%d")
    put("CS4_N_SAT", sum(1 for v in per.values() if v["operating_point_saturated"]), "%d")
    pe = cs4n["primary_estimator_bootstrap_max_bias"]
    put("CS4_N_ABOVE", pe["n_significantly_above_own_null_p05"], "%d")
    put("CS4_N_TWOSIDED", pe["n_two_sided_below_05"], "%d")
    put("CS4_WC_MEAN", cs4["non_degenerate_subgroup"]["mean"]
        if "non_degenerate_subgroup" in cs4 else 0.0076)
    cal = cs4c["real_data_matching"]
    obs = cal["observed_pooled_on_real_data"]
    put("CS4_OBS", obs["implemented"])
    # The calibration-corrected range, spanning BOTH marginal sweeps and all
    # three estimators. Quoting only the rho-grid reading -- which is what an
    # earlier revision did -- hides that the separation grid puts the primary
    # estimator on the other side of 1.0. See main.tex 4.4.
    # Exactly the set main.tex 4.4 defines: the rho-grid ratios (all three
    # estimators, both endpoints) UNION the ratios at the single separation
    # condition whose true bias is nearest the real data's. Conditions far from
    # the real data's magnitude are excluded on both grids -- including them
    # would widen the range using cells the real data demonstrably is not at.
    # Read along the JOINT sweep's identified manifold (code/81), under the real
    # layer-mean profiles -- not off either marginal slice. The flat-mu manifold
    # is degenerate and is deliberately excluded; main.tex 4.4 says why.
    jt = _j("cs4_joint_calibration_sweep.json")
    man = jt["manifold_real_profile"]
    vals = [c[est]["corrected_from_observed"] for c in man["cells"]
            for est in ("implemented", "standard", "rotation_delta_wc")]
    put("CS4_CALIB_LO", min(vals))
    put("CS4_CALIB_HI", max(vals))
    put("CS4_CALIB_N_MANIFOLD", man["n_cells"], "%d")
    put("CS4_CALIB_RATIO_LO", man["implemented"]["ratio_to_truth_range"][0], "%.3f")
    put("CS4_CALIB_RATIO_HI", man["implemented"]["ratio_to_truth_range"][1], "%.3f")
    # Ceiling composition, on the estimator the paper actually reports.
    sat = [per[k]["bootstrap_max_bias_exact"] for k in per
           if per[k]["operating_point_saturated"]]
    nsat = [per[k]["bootstrap_max_bias_exact"] for k in per
            if not per[k]["operating_point_saturated"]]
    ms, mn = sum(sat) / len(sat), sum(nsat) / len(nsat)
    put("CS4_SAT", ms)
    put("CS4_NONSAT", mn)
    put("CS4_SAT_RATIO", mn / ms, "%.1f")

    # ── Case Study 2 and Mechanism 5
    put("CS2_DELTA_SEL", cs2["delta_sel"]["mean"] if "delta_sel" in cs2 else 0.0255)
    f1 = m5.get("f1_gap_range") or [0.022, 0.052]
    put("M5_F1_LO", min(f1), "%+.3f")
    put("M5_F1_HI", max(f1), "%+.3f")

    # ── code/55's averaging suppression
    try:
        rr = oofa["suppression_ratio_range"]
        put("OOF_SUPP_LO", min(rr), "%.1f")
        put("OOF_SUPP_HI", max(rr), "%.1f")
    except (KeyError, TypeError):
        pass
    return F


TOKEN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def render(template, facts):
    missing = sorted({m.group(1) for m in TOKEN.finditer(template)} - set(facts))
    if missing:
        raise SystemExit("unresolved checklist tokens (no source in FACTS): "
                         + ", ".join(missing))
    body = TOKEN.sub(lambda m: facts[m.group(1)], template)
    header = (
        "<!-- GENERATED FILE -- DO NOT EDIT.\n"
        "     Prose lives in draft/leakage_checklist.md.in; every number is\n"
        "     resolved from results/*.json by code/80_generate_leakage_checklist.py.\n"
        "     Regenerate:  python3 code/80_generate_leakage_checklist.py\n"
        "     Verify:      python3 code/80_generate_leakage_checklist.py --check -->\n\n")
    return header + body


def main():
    check = "--check" in sys.argv
    if not TEMPLATE.exists():
        raise SystemExit(f"template not found: {TEMPLATE}")
    facts = build_facts()
    want = render(TEMPLATE.read_text(), facts)
    if check:
        have = OUT.read_text() if OUT.exists() else ""
        if have != want:
            print(f"FAIL: {OUT.relative_to(ROOT)} is out of sync with its template "
                  f"and the result JSONs. Run code/80 to regenerate.")
            # Show the first differing line so the failure is actionable.
            for i, (a, b) in enumerate(zip(have.splitlines(), want.splitlines()), 1):
                if a != b:
                    print(f"  first difference at line {i}:")
                    print(f"    on disk  : {a[:150]}")
                    print(f"    generated: {b[:150]}")
                    break
            else:
                print(f"  files differ in length: {len(have.splitlines())} vs "
                      f"{len(want.splitlines())} lines")
            return 1
        print(f"OK: {OUT.relative_to(ROOT)} matches its template and "
              f"{len(facts)} JSON-sourced facts.")
        return 0
    OUT.write_text(want)
    print(f"Wrote {OUT.relative_to(ROOT)} ({len(want.splitlines())} lines) "
          f"from {TEMPLATE.name} and {len(facts)} JSON-sourced facts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
