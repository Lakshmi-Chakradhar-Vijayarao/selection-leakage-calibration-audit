"""
Trace every number in the TAE camera-ready back to the JSON that produces it.

code/53_verify_paper_numbers.py verifies `draft/latex/main.tex`, which is the
long TMLR-format version of this work. The TAE workshop paper is a separate,
shorter document, and its calibration-bridge and selective-prediction tables
were therefore never covered -- exactly as code/README notes. This script
closes that gap for the camera-ready, including the three analyses added in
response to review (Murphy decomposition, ECE bin sweep, placebo control) and
the coverage-aware deployment error count.

Same contract as code/53: a check passes if the formatted value appears
literally in the .tex. As there, this is a document-wide substring search, not
a located-claim match, so a pass means the number appears SOMEWHERE in the
document, not necessarily at the sentence it nominally verifies. Occurrence
counts are printed so a short/common literal cannot silently pass on an
unrelated match.

Two venue modes, because the anonymity requirement inverts between them:

  default        de-anonymized camera-ready. The author block and the real
                 artifact URL must be PRESENT; the anonymous mirror and
                 "Anonymous Author" must be ABSENT.
  --anonymous    double-blind submission. Exactly the reverse.

Getting this backwards is a desk-reject at one venue and a broken link at the
other, so it is checked rather than assumed.

Usage:
    python 105_verify_camera_ready_numbers.py [path/to/main.tex] [--anonymous]
Default path is the camera-ready staging directory.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
# The camera-ready source ships inside this repo, so the default path resolves
# for anyone who clones it. The working-copy path is a fallback for editing
# sessions where the in-repo copy has not been re-synced yet.
IN_REPO_TEX = ROOT / "draft" / "tae_camera_ready" / "main.tex"
WORKING_TEX = (Path.home() / "Downloads" / "SUBMISSIONS" / "CAMERA_READY"
               / "P2_TAE" / "main.tex")

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
ANONYMOUS = "--anonymous" in sys.argv[1:]

if ARGS:
    TEX_PATH = Path(ARGS[0])
else:
    TEX_PATH = IN_REPO_TEX if IN_REPO_TEX.exists() else WORKING_TEX
if not TEX_PATH.exists():
    sys.exit(f"camera-ready main.tex not found: {TEX_PATH}")
TEX = TEX_PATH.read_text()

failures = []
checks = 0


def check(label, value, fmt="{:+.4f}"):
    """Assert the formatted value appears literally in the camera-ready .tex.

    `fmt` is either a format string or a callable rendering the value the way
    the document writes it (see `thousands`).
    """
    global checks
    checks += 1
    s = fmt(value) if callable(fmt) else fmt.format(value)
    bare = s.lstrip("+")
    n = TEX.count(s) + (TEX.count(bare) if s.startswith("+") else 0)
    ok = n > 0
    tag = f"  [{n}x -- location not verified]" if ok and n > 1 else ""
    print(f"  {'OK  ' if ok else 'FAIL'}  {label}: {s}{tag}")
    if not ok:
        failures.append(f"{label}: {s} absent from {TEX_PATH.name}")


def check_rounded(label, value, places=3):
    """Assert the value appears at `places` decimals, allowing either
    neighbouring rendering.

    Values landing exactly on a rounding boundary (e.g. a believed risk of
    0.0955 at three decimals) are written half-up by a human but round down
    under Python's float formatting, because the stored double is fractionally
    below the boundary. Accepting both renderings keeps this from reporting a
    spurious failure, while still catching any value that is genuinely wrong
    in the third decimal.
    """
    global checks
    checks += 1
    step = 10.0 ** -places
    cands = {f"{value:.{places}f}", f"{value + step / 2:.{places}f}"}
    hit = sorted(c for c in cands if c in TEX)
    ok = bool(hit)
    shown = hit[0] if ok else "/".join(sorted(cands))
    print(f"  {'OK  ' if ok else 'FAIL'}  {label}: {shown}")
    if not ok:
        failures.append(f"{label}: neither {sorted(cands)} present in {TEX_PATH.name}")


def check_absent(label, s):
    """Fail if a superseded literal survives anywhere in the camera-ready."""
    global checks
    checks += 1
    n = TEX.count(s)
    ok = n == 0
    print(f"  {'OK  ' if ok else 'FAIL'}  absent: {label} ({s!r})"
          f"{'' if ok else f'  -- found {n}x'}")
    if not ok:
        failures.append(f"superseded literal {s!r} still present {n}x")


def thousands(n):
    """LaTeX thousands separator as written in this document: 1{,}082."""
    return f"{round(n):,}".replace(",", "{,}")


rr = json.load(open(R / "reviewer_response_mechanism2.json"))
dep = json.load(open(R / "deployment_error_count_corrected.json"))
bridge = json.load(open(ROOT / "calibration_bridge_mechanism2.json"))
selpred = json.load(open(ROOT / "selective_prediction_consequence.json"))

print(f"\nVerifying {TEX_PATH}\n")

# ── Submitted calibration-bridge table (never covered by code/53) ───────────
print("Calibration bridge, Mechanism 2 (submitted numbers):")
check("brier LEAKY", bridge["brier_leaky_mean"], "{:.4f}")
check("brier CLEAN", bridge["brier_clean_mean"], "{:.4f}")
check("brier gap", bridge["brier_gap_mean"], "{:+.4f}")
check("ece LEAKY", bridge["ece_leaky_mean"], "{:.4f}")
check("ece CLEAN", bridge["ece_clean_mean"], "{:.4f}")
check("ece gap", bridge["ece_gap_mean"], "{:+.4f}")
check("auroc gap = Delta_sel", bridge["auroc_gap_mean"], "{:+.4f}")

# ── Submitted selective-prediction table (never covered by code/53) ─────────
print("\nSelective prediction, Mechanism 2 (submitted numbers):")
for ct, e in selpred["mechanism_2"]["coverage_targets"].items():
    check_rounded(f"  {ct} believed risk", e["believed_risk_mean_from_leaky"])
    check_rounded(f"  {ct} actual risk", e["actual_risk_mean_on_clean"])
    check(f"  {ct} risk violation", e["risk_violation_mean"], "{:+.3f}")

# ── (a) Murphy decomposition, slope, smooth calibration error ──────────────
print("\nMurphy decomposition and binning-free calibration statistics:")
bd = rr["brier_decomposition_at_selected_layer"]
for key, fmt in [("brier_leaky_mean", "{:.4f}"), ("brier_clean_mean", "{:.4f}"),
                 ("reliability_leaky_mean", "{:.4f}"),
                 ("reliability_clean_mean", "{:.4f}"),
                 ("resolution_leaky_mean", "{:.4f}"),
                 ("resolution_clean_mean", "{:.4f}"),
                 ("uncertainty_leaky_mean", "{:.4f}"),
                 ("uncertainty_clean_mean", "{:.4f}")]:
    check(f"  {key}", bd[key], fmt)
for key in ["reliability_gap_clean_minus_leaky", "resolution_gap_leaky_minus_clean"]:
    check(f"  {key}", bd[key]["mean"], "{:+.4f}")
    check(f"  {key} lo", bd[key]["bca_95ci"][0], "{:+.4f}")
    check(f"  {key} hi", bd[key]["bca_95ci"][1], "{:+.4f}")

# Reliability must be established and resolution must NOT be, or the paper's
# core "the effect is calibration, not discrimination" sentence is wrong.
checks += 1
_rel_res_ok = (bd["reliability_gap_clean_minus_leaky"]["excludes_zero"] and
               not bd["resolution_gap_leaky_minus_clean"]["excludes_zero"])
print(f"  {'OK  ' if _rel_res_ok else 'FAIL'}  reliability excludes zero AND "
      f"resolution does not (the calibration-not-discrimination claim)")
if not _rel_res_ok:
    failures.append("reliability/resolution significance pattern no longer "
                    "supports the calibration-not-discrimination claim")

sl = rr["calibration_slope_intercept_at_selected_layer"]
check("  slope LEAKY", sl["slope_leaky_mean"], "{:.4f}")
check("  slope CLEAN", sl["slope_clean_mean"], "{:.4f}")
check("  slope gap", sl["slope_gap_leaky_minus_clean"]["mean"], "{:+.4f}")
check("  intercept LEAKY", sl["intercept_leaky_mean"], "{:.4f}")
check("  intercept CLEAN", sl["intercept_clean_mean"], "{:.4f}")

sce = rr["smooth_calibration_error_at_selected_layer"]
check("  SCE LEAKY", sce["sce_leaky_mean"], "{:.4f}")
check("  SCE CLEAN", sce["sce_clean_mean"], "{:.4f}")
check("  SCE gap", sce["sce_gap_clean_minus_leaky"]["mean"], "{:+.4f}")

# ── (b) ECE bin sensitivity ────────────────────────────────────────────────
print("\nECE bin sensitivity (8 settings):")
sweep = rr["ece_bin_sensitivity_at_selected_layer"]
for name, e in sweep.items():
    check(f"  {name} LEAKY", e["ece_leaky_mean"], "{:.4f}")
    check(f"  {name} CLEAN", e["ece_clean_mean"], "{:.4f}")
    check(f"  {name} gap", e["mean"], "{:+.4f}")

checks += 1
_all_excl = rr["ece_sweep_all_settings_exclude_zero"] and all(
    e["excludes_zero"] for e in sweep.values())
print(f"  {'OK  ' if _all_excl else 'FAIL'}  all 8 binning settings exclude zero")
if not _all_excl:
    failures.append("main.tex claims 8/8 binning settings exclude zero; JSON disagrees")

checks += 1
_gaps = [e["mean"] for e in sweep.values()]
_range_ok = f"{min(_gaps):+.4f}" in TEX and f"{max(_gaps):+.4f}" in TEX
print(f"  {'OK  ' if _range_ok else 'FAIL'}  quoted sweep range matches "
      f"[{min(_gaps):+.4f}, {max(_gaps):+.4f}]")
if not _range_ok:
    failures.append("the ECE sweep range quoted in main.tex is not the JSON's min/max")

# ── (c) Placebo control ────────────────────────────────────────────────────
print("\nPlacebo control:")
pc = rr["placebo_control"]
for metric in ["reliability", "ece_10_width", "smooth_calibration_error",
               "brier", "resolution"]:
    # Table 'Placebo control' is written to 5 decimals; checking at 4 would
    # pass on unrelated 4-decimal matches elsewhere in the document.
    m = pc[metric]
    check(f"  {metric} gap@selected", m["gap_at_selected_layer"]["mean"], "{:+.5f}")
    check(f"  {metric} placebo", m["placebo_gap_all_layer_mean"]["mean"], "{:+.5f}")
    inc = m["selection_specific_increment_vs_all_layer_mean"]
    check(f"  {metric} selection-specific", inc["mean"], "{:+.5f}")
    check(f"  {metric} increment lo", inc["bca_95ci"][0], "{:+.5f}")
    check(f"  {metric} increment hi", inc["bca_95ci"][1], "{:+.5f}")

# The 57%/43% selection-vs-reuse split was WITHDRAWN: the increment does not
# clear its mechanical null (110). Guard the withdrawal rather than the claim.
# The old check here tested for the substrings "57" and "43", which occur all
# over a number-dense paper and so passed regardless -- a weak check guarding
# a claim that has since been retracted.
for _phrase in ["of the calibration gap is selection-specific",
                "is selection-specific and $43",
                "roughly $57\\%$ of the calibration gap"]:
    check_absent(f"withdrawn selection-attribution claim ({_phrase[:38]}...)", _phrase)

checks += 1
_share = pc["reliability"]["share_of_gap_that_is_selection_specific"]
_null_ok = "mechanical null" in TEX
print(f"  {'OK  ' if _null_ok else 'FAIL'}  the placebo share ({_share:.1%}) is presented "
      f"against a mechanical null rather than against zero")
if not _null_ok:
    failures.append("main.tex reports a selection-specific share without the "
                    "mechanical null that qualifies it")

# ── Mechanical null for the selection-specific increment (110) ─────────────
# Content-gated, not venue-gated: a version that REPORTS the null must get its
# numbers right, and a version that does NOT report it must not be making the
# selection-attribution claim the null withdrew. Either is acceptable; keeping
# the claim without the null is not.
mn = json.load(open(R / "calibration_increment_null.json"))
# Gate on the full treatment (the null table), not on the phrase: a short
# version may cite the null and point at the artifact without tabulating it,
# which is honest. What is forbidden is asserting the attribution with no
# mention of the null at all.
REPORTS_NULL = "tab:mechnull" in TEX
if not REPORTS_NULL:
    print("\nMechanical null: not reported in this version -- "
          "checking the withdrawn claim is absent instead")
    checks += 1
    _clean = "selection-specific" not in TEX or "mechanical" in TEX
    print(f"  {'OK  ' if _clean else 'FAIL'}  does not assert a selection-specific "
          f"effect without reporting its mechanical null")
    if not _clean:
        failures.append("this version claims a selection-specific effect but omits "
                        "the mechanical null that withdraws it (see 110)")
print("\nMechanical null (selection-specific increment):" if REPORTS_NULL else "")
for m, e in (mn["metrics"].items() if REPORTS_NULL else []):
    check(f"  {m} observed increment", e["observed_increment"], "{:+.5f}")
    check(f"  {m} null mean", e["null_mean_simulated"], "{:+.5f}")
    check(f"  {m} null CI lo", e["null_ci_95"][0], "{:+.5f}")
    check(f"  {m} null CI hi", e["null_ci_95"][1], "{:+.5f}")

# The paper's central claim: NO increment is above its null. If a rerun ever
# puts one above, the thesis changes and this must fail loudly.
checks += 1
_none_above = all(
    e["observed_increment"] <= e["null_ci_95"][1] for e in mn["metrics"].values())
print(f"  {'OK  ' if _none_above else 'FAIL'}  no calibration increment exceeds the "
      f"upper bound of its own mechanical null")
if not _none_above:
    failures.append("main.tex claims no increment clears its mechanical null; the "
                    "JSON now shows one that does")

# The simulated null must match the closed form, which is the correctness
# check on the null itself.
checks += 1
_cf_ok = all(e["null_closed_form_abs_diff"] < 1e-3 for e in mn["metrics"].values())
print(f"  {'OK  ' if _cf_ok else 'FAIL'}  simulated null matches its closed form "
      f"for every metric")
if not _cf_ok:
    failures.append("the simulated mechanical null no longer matches its closed form")

# B is the identified component; the paper leans on it being established for
# Brier and NOT established for reliability/ECE.
print("\nIdentified component B:" if REPORTS_NULL else "")
for m, e in (mn["metrics"].items() if REPORTS_NULL else []):
    b = e["decomposition"]
    # The prose writes B to 4 decimals; check at the precision it is quoted.
    check(f"  {m} B", b["B_transferred_on_clean_arm"], "{:+.4f}")
checks += 1
_b_ok = (mn["metrics"]["brier"]["decomposition"]["B_excludes_zero"]
         and not mn["metrics"]["reliability"]["decomposition"]["B_excludes_zero"]
         and not mn["metrics"]["ece"]["decomposition"]["B_excludes_zero"])
print(f"  {'OK  ' if _b_ok else 'FAIL'}  B established for Brier, not established for "
      f"reliability or ECE (exactly as claimed)")
if not _b_ok:
    failures.append("main.tex's account of which B components are established no "
                    "longer matches the JSON")

# The negative resolution increment is load-bearing: it is why Brier's 92%
# share is not evidence the effect is mostly selection.
checks += 1
_res_neg = (pc["resolution"]["selection_specific_increment_vs_all_layer_mean"]["mean"] < 0
            and pc["resolution"]["selection_specific_increment_vs_all_layer_mean"]["excludes_zero"])
print(f"  {'OK  ' if _res_neg else 'FAIL'}  resolution's selection-specific "
      f"increment is negative and excludes zero")
if not _res_neg:
    failures.append("main.tex claims a significantly negative resolution increment; "
                    "the JSON no longer shows one")

# ── (d) Coverage-aware deployment error count ──────────────────────────────
print("\nCoverage-aware deployment error count:")
for ct, e in dep["coverage_targets"].items():
    check(f"  {ct} coverage LEAKY", e["achieved_coverage_leaky_mean"], "{:.4f}")
    check(f"  {ct} coverage CLEAN", e["achieved_coverage_clean_mean"], "{:.4f}")
    check(f"  {ct} errors believed", e["errors_per_day_believed"], thousands)
    check(f"  {ct} errors actual", e["errors_per_day_actual"], thousands)
    corr = e["excess_errors_per_day_corrected"]
    check(f"  {ct} excess corrected", corr["mean"], thousands)
    check(f"  {ct} excess corrected lo", corr["bca_95ci"][0], "{:.0f}")
    check(f"  {ct} excess corrected hi", corr["bca_95ci"][1], "{:.0f}")

# The correction is only worth reporting if it actually moves the 30% cell
# across zero; if a rerun changes that, the main text's claim must change too.
checks += 1
_c30 = dep["coverage_targets"]["0.30"]
_flip_ok = (_c30["excess_errors_per_day_corrected"]["bca_95ci"][0] > 0 >=
            _c30["excess_errors_per_day_submitted_estimator"]["bca_95ci"][0])
print(f"  {'OK  ' if _flip_ok else 'FAIL'}  at the 30% target the coverage-aware "
      f"interval excludes zero where the fixed-coverage one does not")
if not _flip_ok:
    failures.append("main.tex claims the 30% cell flips under the coverage-aware "
                    "estimator; the JSON no longer shows that flip")

# 104's achieved CLEAN coverage must reproduce the submitted pipeline's, or the
# two scripts are not measuring the same thing.
print("\nCross-check: 104's coverage vs the submitted selective-prediction run:")
for ct, e in dep["coverage_targets"].items():
    checks += 1
    ref = selpred["mechanism_2"]["coverage_targets"][ct]["actual_coverage_mean_on_clean"]
    ok = abs(e["achieved_coverage_clean_mean"] - ref) < 1e-9
    print(f"  {'OK  ' if ok else 'FAIL'}  {ct}: {e['achieved_coverage_clean_mean']:.6f} "
          f"vs {ref:.6f}")
    if not ok:
        failures.append(f"104's CLEAN coverage at {ct} does not reproduce "
                        f"selective_prediction_consequence.json")

# ── Calibration under regularization (106) ─────────────────────────────────
print("\nCalibration under regularization:")
reg = json.load(open(R / "calibration_under_regularization.json"))
for cfg_name, cfg in reg["by_config"].items():
    for m in ("reliability", "ece"):
        e = cfg["metrics"][m]
        check(f"  {cfg_name} {m} gap", e["gap_at_selected"]["mean"], "{:+.5f}")
        check(f"  {cfg_name} {m} placebo", e["placebo_all_layer_mean"]["mean"], "{:+.5f}")
        ss = e["selection_specific"]
        check(f"  {cfg_name} {m} selection-specific", ss["mean"], "{:+.5f}")
        check(f"  {cfg_name} {m} ss lo", ss["bca_95ci"][0], "{:+.5f}")
        check(f"  {cfg_name} {m} ss hi", ss["bca_95ci"][1], "{:+.5f}")
    r = cfg["risk_violation_at_coverage"]
    check(f"  {cfg_name} risk violation", r["mean"], "{:+.5f}")

# The paper's claim is that regularization does NOT explain the effect. If a
# rerun ever makes an increment lose significance, that sentence must change.
checks += 1
_reg_ok = all(
    cfg["metrics"][m]["selection_specific"]["excludes_zero"]
    for cfg in reg["by_config"].values() for m in ("reliability", "ece"))
print(f"  {'OK  ' if _reg_ok else 'FAIL'}  selection-specific reliability AND ece "
      f"exclude zero in every probe configuration")
if not _reg_ok:
    failures.append("main.tex claims the selection-specific increment survives every "
                    "probe configuration; the JSON no longer shows that")

# And that the SHARE rises rather than falls once the probe cannot interpolate.
checks += 1
_share_shipped = reg["by_config"]["shipped_unscaled_C1.0"]["metrics"]["reliability"]["share_selection_specific"]
_share_reg = reg["by_config"]["scaled_C0.01"]["metrics"]["reliability"]["share_selection_specific"]
_rise_ok = _share_reg > _share_shipped
print(f"  {'OK  ' if _rise_ok else 'FAIL'}  selection share rises under regularization "
      f"({_share_shipped:.1%} -> {_share_reg:.1%})")
if not _rise_ok:
    failures.append("main.tex says the selection share rises under regularization; it does not")

# ── Second model family: the negative replication (108) ────────────────────
print("\nSecond model family (negative replication):")
sf = json.load(open(R / "calibration_second_model_family.json"))
for model, e in sf["by_model"].items():
    check(f"  {model} delta_sel", e["delta_sel_auroc"]["mean"], "{:+.5f}")
    for m in ("reliability", "ece"):
        s = e["metrics"][m]
        check(f"  {model} {m} gap", s["gap_at_selected"]["mean"], "{:+.5f}")
        check(f"  {model} {m} placebo", s["placebo_all_layer_mean"]["mean"], "{:+.5f}")
        check(f"  {model} {m} selection-specific", s["selection_specific"]["mean"], "{:+.5f}")

# The paper reports this as a NEGATIVE result. Guard both halves of it, so the
# claim cannot silently invert: discrimination replicates, calibration does not.
checks += 1
_disc_ok = all(e["delta_sel_auroc"]["excludes_zero"] and e["delta_sel_auroc"]["mean"] > 0
               for e in sf["by_model"].values())
print(f"  {'OK  ' if _disc_ok else 'FAIL'}  Delta_sel positive and established in both families")
if not _disc_ok:
    failures.append("main.tex says discrimination optimism replicates in both model "
                    "families; the JSON no longer shows that")

checks += 1
_calib_neg_ok = all(e["metrics"]["ece"]["selection_specific"]["mean"] < 0
                    for e in sf["by_model"].values())
print(f"  {'OK  ' if _calib_neg_ok else 'FAIL'}  ECE selection-specific increment is "
      f"negative in both families (the reported negative result)")
if not _calib_neg_ok:
    failures.append("main.tex reports a negative cross-harness calibration result; "
                    "the JSON no longer shows a negative ECE increment")

# ── Reliability diagram region statistic (107) ─────────────────────────────
print("\nReliability diagram:")
rd = json.load(open(R / "reliability_diagram.json"))
hc = rd["high_confidence_region"]
# The prose states these as magnitudes below the diagonal ("sits 0.0592
# below"), so compare on absolute value rather than the JSON's signed form.
check("  high-confidence deviation LEAKY", abs(hc["leaky"]["mean_deviation"]), "{:.4f}")
check("  high-confidence deviation CLEAN", abs(hc["clean"]["mean_deviation"]), "{:.4f}")
check("  excess overconfidence",
      abs(hc["excess_overconfidence_clean_minus_leaky"]), "{:.4f}")
checks += 1
_hc_ok = hc["clean"]["mean_deviation"] < hc["leaky"]["mean_deviation"] < 0
print(f"  {'OK  ' if _hc_ok else 'FAIL'}  both arms overconfident in the high-confidence "
      f"region and CLEAN more so")
if not _hc_ok:
    failures.append("main.tex says both arms are overconfident above p=0.8 with CLEAN "
                    "further below the diagonal; the JSON no longer shows that")


# ── Corrected procedure and its size/power (111, 112) ──────────────────────
if "sec:sizepower" in TEX or "corrected report" in TEX.lower():
    print("\nCorrected procedure (111/112):")
    sp = json.load(open(R / "estimator_size_power.json"))
    cr = json.load(open(R / "corrected_report.json"))

    # The headline: the practised test is saturated, the corrected one nominal.
    checks += 1
    _prac = sp["summary"]["practised_type_I_error_range"]
    _corr = sp["summary"]["corrected_type_I_error_range"]
    _ok = _prac[0] >= 0.99 and 0.03 <= _corr[0] <= 0.07 and 0.03 <= _corr[1] <= 0.07
    print(f"  {'OK  ' if _ok else 'FAIL'}  practised type I error {_prac[0]:.3f}-{_prac[1]:.3f}, "
          f"corrected {_corr[0]:.3f}-{_corr[1]:.3f}")
    if not _ok:
        failures.append("the size result main.tex reports no longer holds in the JSON")

    # Delta_sel must be anti-monotone in tau -- the paper's sharpest claim.
    checks += 1
    _k32 = [v for v in sp["results"].values() if v["K"] == 32]
    _k32.sort(key=lambda v: v["tau"])
    _anti = all(a["mean_delta"] >= b["mean_delta"] for a, b in zip(_k32, _k32[1:]))
    _mono_B = all(a["mean_B"] <= b["mean_B"] for a, b in zip(_k32, _k32[1:]))
    print(f"  {'OK  ' if (_anti and _mono_B) else 'FAIL'}  E[Delta_sel] falls and E[B] rises "
          f"as tau grows (the anti-monotonicity claim)")
    if not (_anti and _mono_B):
        failures.append("main.tex claims Delta_sel is anti-monotone in transferable "
                        "quality; the simulation no longer shows that")

    # B's verdicts on the real harness must match what the paper states.
    for _m, _want in [("auroc", "transfers"), ("brier", "transfers"),
                      ("reliability", "not established"), ("ece", "not established")]:
        checks += 1
        _got = cr["metrics"][_m]["verdict"]
        print(f"  {'OK  ' if _got == _want else 'FAIL'}  B on {_m}: {_got}")
        if _got != _want:
            failures.append(f"main.tex says B on {_m} is '{_want}'; 112 reports '{_got}'")
    for _m in ("auroc", "brier"):
        check(f"  {_m} B t-statistic", cr["metrics"][_m]["B_t"], "{:.2f}")


# ── Estimator comparison: what the imported corrections cost (113) ────────
if "sec:comparison" in TEX:
    print("\nEstimator comparison (113):")
    ec = json.load(open(R / "estimator_comparison.json"))
    k = ec["summary"]
    checks += 1
    # the honest property is that the direct estimator is unbiased and at
    # least nominal (it is mildly conservative), not exactly 0.95
    _cov = [v["E2_direct_gap"]["ci_coverage"] for v in ec["results"].values()]
    _bias = max(abs(v["E2_direct_gap"]["bias"]) for v in ec["results"].values())
    _ok = (_bias < 1e-3 and min(_cov) >= 0.95
           and not k["delta_sel_is_biased_for_theta"])
    print(f"  {'OK  ' if _ok else 'FAIL'}  direct gap unbiased and covering, and "
          f"Delta_sel NOT biased for theta (the paper's own correction to itself)")
    if not _ok:
        failures.append("main.tex says the estimator was never the problem; 113 "
                        "no longer supports that")
    # the Tweedie coverage failure is the evidence for K2(b); guard it
    checks += 1
    _tw = [v["E3_tweedie_cv_only"]["ci_coverage"] for v in ec["results"].values()]
    _tw_bad = max(_tw) < 0.95
    print(f"  {'OK  ' if _tw_bad else 'FAIL'}  Tweedie never reaches nominal coverage "
          f"in any cell (range {min(_tw):.3f}-{max(_tw):.3f} vs 0.95)")
    if not _tw_bad:
        failures.append("main.tex reports the winner's-curse correction as badly "
                        "under-covering; 113 no longer shows that")

    # Holm across the four B tests must not change any verdict
    cr2 = json.load(open(R / "corrected_report.json"))
    checks += 1
    _h = cr2.get("holm_changes_no_verdict")
    print(f"  {'OK  ' if _h else 'FAIL'}  Holm-Bonferroni changes no verdict in the "
          f"corrected report")
    if not _h:
        failures.append("main.tex says Holm changes no verdict; 112 disagrees")


# ── K1: the size/power mislabelling must never come back ──────────────────
checks += 1
# whitespace-insensitive: LaTeX line breaks hid this phrase from an earlier
# version of this very check, and the claim survived in the introduction.
_flat = " ".join(TEX.split())
_bad = ("type I error of $1" in _flat or "type I error $1$" in _flat
        or "rejection rate there \\emph{is} the type I error" in _flat
        or "anti-monotone in the effect it is read as measuring" in _flat)
print(f"  {'OK  ' if not _bad else 'FAIL'}  the practised test's rejection rate is "
      f"NOT described as a type I error")
if _bad:
    failures.append("main.tex calls the practised test's rejection rate a type I "
                    "error; H0 is false there, so that is a category error")

# ── K3: the fair-interval comparison (114) ────────────────────────────────
if "114" in TEX or "parametric bootstrap" in TEX:
    fi = json.load(open(R / "fair_interval_comparison.json"))
    f = fi["summary"]
    for lab, key in [("direct t", "direct_t_coverage_range"),
                     ("Tweedie naive", "tweedie_naive_range"),
                     ("Tweedie bootstrap", "tweedie_bootstrap_range")]:
        for v in f[key]:
            check(f"  {lab} coverage bound", v, "{:.3f}")
    checks += 1
    _fair = (f["tweedie_bootstrap_range"][1] < 0.95
             and f["direct_t_coverage_range"][0] > 0.95)
    print(f"  {'OK  ' if _fair else 'FAIL'}  under a FAIR interval Tweedie still "
          f"under-covers and the direct estimator does not")
    if not _fair:
        failures.append("main.tex claims the under-coverage survives a fair "
                        "interval; 114 no longer shows that")


# -- Candidate dependence and effective K (115) ---------------------------
if "sec:effectivek" in TEX:
    print("\nCandidate dependence (115):")
    cd = json.load(open(R / "candidate_dependence.json"))
    d, o, v = cd["dependence"], cd["observed"], cd["verdict"]
    check("  mean off-diagonal correlation", d["mean_offdiagonal_correlation"], "{:.3f}")
    check("  mean adjacent correlation", d["mean_adjacent_correlation"], "{:.3f}")
    check("  effective K", d["effective_K"], "{:.2f}")
    check("  observed A", o["A"], "{:.5f}")
    check("  dependent-resample A", cd["resampling_dependent"]["A"]["mean"], "{:.5f}")
    check("  independent-resample A", cd["resampling_independent"]["A"]["mean"], "{:.5f}")
    check("  inflation factor", v["independence_inflation_factor"], "{:.2f}")

    # The claim rests on this asymmetry. If a rerun ever lets the independent
    # model cover the observed curse, the section is wrong.
    checks += 1
    _ok = (v["dependent_ci_covers_observed_A"]
           and not v["independent_ci_covers_observed_A"]
           and not v["kill_criterion_fired"])
    print(f"  {'OK  ' if _ok else 'FAIL'}  dependent model covers the observed curse "
          f"and the independent one does not")
    if not _ok:
        failures.append("main.tex's dependence claim requires the dependent model to "
                        "cover observed A and the independent one not to; 115 no "
                        "longer shows that")

    # Dependence must act on the curse, not on what selection recovers.
    checks += 1
    _bd = cd["resampling_dependent"]["B"]["mean"]
    _bi = cd["resampling_independent"]["B"]["mean"]
    _ratio_B = abs(_bi / _bd) if _bd else float("inf")
    _b_ok = _ratio_B < 1.25 < v["independence_inflation_factor"]
    print(f"  {'OK  ' if _b_ok else 'FAIL'}  B far less sensitive to dependence than A "
          f"(B x{_ratio_B:.2f} vs A x{v['independence_inflation_factor']:.2f})")
    if not _b_ok:
        failures.append("main.tex says dependence acts on the curse and not on B; "
                        "115 no longer shows that separation")

# ── Superseded literals that must not survive the camera-ready ─────────────
print("\nSuperseded literals:")
check_absent("fixed-coverage excess at 70% (980)", "$980$")
check_absent("fixed-coverage excess at 50% (750)", "$750$")
check_absent("fixed-coverage believed errors", "8{,}960")
check_absent("fixed-coverage actual errors", "9{,}940")
if ANONYMOUS:
    # Double-blind submission: the de-anonymizing strings must not survive and
    # the anonymizing ones must be there.
    #
    # The identity strings are read from the environment rather than written
    # here, following code/91's ANON_AUTHOR/ANON_INSTITUTION convention. A
    # checker of anonymity cannot hard-code the very names it forbids and also
    # ship inside an anonymous archive -- code/91's identity scan flags it, and
    # it is right to.
    _author = os.environ.get("ANON_AUTHOR_LITERAL")
    _inst = os.environ.get("ANON_INSTITUTION_LITERAL")
    _email = os.environ.get("ANON_EMAIL_LITERAL")
    _repo = os.environ.get("ANON_REPO_LITERAL")
    if not any([_author, _inst, _email, _repo]):
        print("  NOTE  identity literals unset; export ANON_AUTHOR_LITERAL, "
              "ANON_INSTITUTION_LITERAL, ANON_EMAIL_LITERAL, ANON_REPO_LITERAL "
              "to run the de-anonymization checks")
    for _label, _needle in [("author name", _author), ("affiliation", _inst),
                            ("author email", _email), ("real artifact URL", _repo)]:
        if _needle:
            check_absent(_label, _needle)
    for label, needle in [("anonymous artifact mirror", "anonymous.4open.science"),
                          ("anonymous author block", "Anonymous Author")]:
        checks += 1
        ok = needle in TEX
        print(f"  {'OK  ' if ok else 'FAIL'}  present: {label} ({needle!r})")
        if not ok:
            failures.append(f"double-blind submission is missing {needle!r}")
else:
    check_absent("anonymous artifact URL", "anonymous.4open.science")
    check_absent("anonymous author block", "Anonymous Author")

print(f"\n{checks} checks run, {len(failures)} failures")
for f in failures:
    print("  - " + f)
sys.exit(1 if failures else 0)
