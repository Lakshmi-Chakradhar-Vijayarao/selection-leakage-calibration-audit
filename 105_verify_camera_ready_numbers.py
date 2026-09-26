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

Usage:
    python 105_verify_camera_ready_numbers.py [path/to/main.tex]
Default path is the camera-ready staging directory.
"""
import json
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

if len(sys.argv) > 1:
    TEX_PATH = Path(sys.argv[1])
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

# The paper says "roughly 57% selection / 43% reuse"; guard the split so that
# sentence cannot drift away from the JSON it summarises.
checks += 1
_share = pc["reliability"]["share_of_gap_that_is_selection_specific"]
_share_ok = 0.50 <= _share <= 0.65 and "57" in TEX and "43" in TEX
print(f"  {'OK  ' if _share_ok else 'FAIL'}  reliability selection-specific share "
      f"{_share:.1%} consistent with the 57/43 split quoted in main.tex")
if not _share_ok:
    failures.append(f"selection-specific share is {_share:.1%}; the 57/43 split "
                    f"quoted in main.tex no longer describes it")

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

# ── Superseded literals that must not survive the camera-ready ─────────────
print("\nSuperseded literals:")
check_absent("fixed-coverage excess at 70% (980)", "$980$")
check_absent("fixed-coverage excess at 50% (750)", "$750$")
check_absent("fixed-coverage believed errors", "8{,}960")
check_absent("fixed-coverage actual errors", "9{,}940")
check_absent("anonymous artifact URL", "anonymous.4open.science")
check_absent("anonymous author block", "Anonymous Author")

print(f"\n{checks} checks run, {len(failures)} failures")
for f in failures:
    print("  - " + f)
sys.exit(1 if failures else 0)
