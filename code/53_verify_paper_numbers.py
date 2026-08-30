"""
trace every headline number in the abstract, §5 and the conclusion
back to the result JSON that produces it, and fail loudly on any mismatch.

This exists because the single most common failure mode in this project's
history has been a number surviving in prose after the run behind it was
corrected. Run this before every commit that touches main.tex.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
TEX = (ROOT / "draft" / "latex" / "main.tex").read_text()

# ── EVERY SHIPPED TEXT ARTIFACT, not just main.tex ──────────────────────────
# An independent review found a retracted "52x" surviving in SEVEN places after
# main.tex had been corrected: README.md (x2), draft/paper_draft.md (x3),
# draft/leakage_checklist.md (x1) and code/58's docstring (x2). Root cause: this
# script read ONLY main.tex, so it had zero coverage of the other files that go
# into the submitted supplementary zip. That is now fixed. `SHIPPED_TEXT` is the
# set of files a reader actually receives; the retracted-string guard runs over
# all of them, and `check_absent_everywhere` is the entry point for any claim
# that must not survive anywhere.
# A SECOND independent review found the SAME class of failure again, in the
# three case-study/worked-example markdown files main.tex explicitly directs
# readers to for "full detail" and which also ship inside the supplementary zip:
# draft/worked_examples.md still carried "L19" and "+0.034",
# draft/case_study_multihaludet.md still carried the withdrawn "48.6x", and
# draft/case_study_quantized_llm_paper.md still presented the retired rotation
# estimator's retired "+0.0076 (headline)" row as THE headline. All three are
# now in this list, so any future staleness in them fails the build.
SHIPPED_TEXT = {}
for _rel in ["draft/latex/main.tex", "README.md", "draft/paper_draft.md",
             "draft/leakage_checklist.md",
             "draft/worked_examples.md",
             "draft/case_study_multihaludet.md",
             "draft/case_study_quantized_llm_paper.md",
             # The two former appendices (Correction History for Case Study 3,
             # and for Case Studies 2/4, Mechanism 5, and the Severity Surface)
             # were compressed from dense prose into compact tables in main.tex
             # (a reviewer-driven page-count pass), with the full narrative
             # moved here verbatim. It ships in the supplementary zip alongside
             # REPRODUCE.md/ANONYMITY_AUDIT.md/DRY_RUN_REBUTTAL.md, so it is a
             # shipped artifact like the others and gets the same retracted-
             # string coverage.
             "PROVENANCE_LOG.md",
             # A second compression pass (79 printed pages to 34) relocated
             # technical detail -- the retired ratio diagnostic, the per-capacity
             # Part A/B controls, the marginal calibration sweeps' chronology,
             # the EVT/c_4(K) digression, and six former Appendix B/C/D
             # subsections -- out of main.tex and into this file, verbatim. It
             # ships in the supplementary zip like the others and gets the same
             # retracted-string coverage and the same TEX-membership coverage.
             "EXTENDED_TECHNICAL_DETAIL.md"]:
    _p = ROOT / _rel
    if _p.exists():
        SHIPPED_TEXT[_rel] = _p.read_text()
# Python docstrings ship too, and code/58's was stale relative to its own code.
# This script itself is excluded: it must be able to NAME the strings it forbids
# without matching them, and a verification script quoting a retracted claim in
# order to forbid it is the behaviour we want.
# code/91 is excluded for the same reason: it is the archive auditor, and its
# own pattern table necessarily spells out every retracted string.
_SELF = {Path(__file__).name, "91_build_supplementary_zip.py"}
for _p in sorted((ROOT / "code").glob("*.py")):
    if _p.name in _SELF:
        continue
    SHIPPED_TEXT[f"code/{_p.name}"] = _p.read_text()

# Every check below (`check()`, `check_absent()`, the ad hoc `... in TEX`
# assertions, and the reverse AUROC-literal/R^2/multiplier guards) tests
# membership in `TEX`. Most of what they guard lives in the main text (§4,
# §5, abstract, conclusion) and is untouched by the appendix-compression pass
# below. A smaller set of exact phrases and numbers lived only in the prose of
# the two correction-history appendices (Appendix A and Appendix E), which now
# carry a compact table in main.tex with the full narrative moved verbatim to
# PROVENANCE_LOG.md. Rather than hunt down each such check
# individually, `TEX` itself covers both documents from here on: content that
# used to live in appendix prose and now lives in the moved-out full record is
# still shipped and still traceable, which is what these checks are actually
# for (see the module docstring). This does not weaken coverage of main-text
# claims (they still have to appear in main.tex, which is part of the union),
# and it does not weaken the retracted-string guard `check_absent()` either:
# PROVENANCE_LOG.md is a byte-for-byte copy of what used to compile
# into the PDF, with the same ``...'' quoting around every retracted phrase it
# mentions for correction purposes, so the quote-span exemption logic applies
# identically to the moved text.
TEX = (TEX + "\n\n" + SHIPPED_TEXT.get("PROVENANCE_LOG.md", "")
           + "\n\n" + SHIPPED_TEXT.get("EXTENDED_TECHNICAL_DETAIL.md", ""))

failures = []
checks = 0


def load(name):
    return json.load(open(R / name))


def check(label, value, fmt="{:+.4f}", must_appear=True):
    """Assert the formatted value literally appears in main.tex or in
    PROVENANCE_LOG.md (see the TEX reassignment above).

    Caveat (found by an independent review, 2026-08-07): this is a
    document-wide substring search, not a located-claim match. A short
    formatted value (e.g. "-0.0008") can occur dozens of times across a
    3,800+ line, number-dense document for entirely unrelated quantities,
    so a passing check here does NOT confirm the value appears at the
    specific sentence it is nominally verifying -- only that it appears
    SOMEWHERE. This already produced one false-negative-for-mislocation
    (main.tex once quoted a different run's numbers at the K=200 sweep
    discussion; both quadruples of digits happened to occur elsewhere in
    the document, so `check()` reported OK for the wrong sentence). The
    `occurrences` count below surfaces this ambiguity for short/common
    strings without changing pass/fail semantics for the other 500+
    existing checks, whose call sites do not carry location context.
    """
    global checks
    checks += 1
    s = fmt.format(value)
    present = s in TEX or s.lstrip("+") in TEX
    ok = present if must_appear else not present
    occurrences = TEX.count(s) + (TEX.count(s.lstrip("+")) if s.startswith("+") else 0)
    ambiguous = ok and must_appear and occurrences > 1
    tag = f"  [{occurrences}x -- location not verified]" if ambiguous else ""
    print(f"  {'OK  ' if ok else 'FAIL'}  {label}: {s}{tag}")
    if not ok:
        failures.append(f"{label} ({s}) not found in main.tex or PROVENANCE_LOG.md")


def _quoted_spans(text):
    """Character spans of LaTeX ``...'' quotations."""
    spans, i = [], text.find("``")
    while i != -1:
        j = text.find("''", i + 2)
        if j == -1:
            break
        spans.append((i, j + 2))
        i = text.find("``", j + 2)
    return spans


_QUOTED = _quoted_spans(TEX)


def check_absent(label, s):
    """Fail if `s` appears in main.tex as an assertion.

    Occurrences INSIDE a LaTeX ``...'' quotation are allowed: those are this
    paper quoting its own retracted wording in order to correct it, which is
    the behaviour we want, not the behaviour we are guarding against. (This
    previously only exempted strings whose opening `` was immediately
    adjacent, which failed as soon as a retraction quoted a phrase from the
    middle of the retracted sentence -- e.g. ``the two axes interact
    multiplicatively rather than additively''.)"""
    global checks
    checks += 1
    asserted = 0
    i = TEX.find(s)
    while i != -1:
        inside_quote = any(a <= i and i + len(s) <= b for a, b in _QUOTED)
        if not inside_quote:
            asserted += 1
        i = TEX.find(s, i + 1)
    ok = asserted == 0
    print(f"  {'OK  ' if ok else 'FAIL'}  retracted string absent (or quoted-for-correction only): "
          f"{label!r}")
    if not ok:
        failures.append(f"retracted string {s!r} asserted {asserted}x in main.tex ({label})")


def _md_quoted_spans(text):
    """Character spans of markdown/plain quotations.

    code/52 renders LaTeX ``...'' as ASCII "..." in the markdown mirror, so a
    retraction that is correctly quoted in main.tex looks like a bare assertion
    in paper_draft.md unless this is handled. Curly quotes are covered too,
    since the mirror emits en-dashes and curly quotes for some sources."""
    spans = []
    for op, cl in [('"', '"'), ("\u201c", "\u201d")]:
        i = text.find(op)
        while i != -1:
            j = text.find(cl, i + 1)
            if j == -1:
                break
            spans.append((i, j + 1))
            i = text.find(op, j + 1)
    return spans


_QUOTED_ALL = {k: (_quoted_spans(v) + (_md_quoted_spans(v) if k.endswith(".md") else []))
               for k, v in SHIPPED_TEXT.items()}

# ── Formatting-insensitive matching ─────────────────────────────────────────
# A third independent review found retracted numbers surviving in shipped files
# for the third round running, and one reason the guard missed them is that the
# guard matched literal substrings. The SAME claim is spelled `$+0.0338$` in
# main.tex, `+0.0338` in the markdown mirror, `+0.0338` with different line
# wrapping in a docstring, and `$+0.0338$~AUROC` after a cosmetic edit --- and
# only the exact spelling listed was caught. Matching now runs on a normalized
# view in which LaTeX math delimiters, the inline-formatting macros, backslashes,
# braces and all whitespace runs are removed, with an index map back to the
# original so the quoted-for-correction exemption still works on real positions.
_STRIP_CHARS = set("$\\{}`*_~ \t\r\n")
_STRIP_WORDS = ("textbf", "emph", "textit", "texttt", "mathbf", "text", "mathrm")


def _normalize(text):
    """Return (normalized_text, index_map) with index_map[i] = original index."""
    out, idx = [], []
    i, n = 0, len(text)
    while i < n:
        if text[i] == "\\":
            for w in _STRIP_WORDS:
                if text.startswith("\\" + w, i):
                    i += 1 + len(w)
                    break
            else:
                i += 1
            continue
        c = text[i]
        if c in _STRIP_CHARS:
            i += 1
            continue
        out.append(c)
        idx.append(i)
        i += 1
    return "".join(out), idx


_NORM_ALL = {k: _normalize(v) for k, v in SHIPPED_TEXT.items()}


def check_absent_everywhere(label, *variants, allow_files=()):
    """Fail if ANY spelling of a retracted claim survives in ANY shipped file.

    `variants` are alternative renderings of the same claim (LaTeX, the
    markdown mirror's stripped-math form, a plain-prose docstring form), since
    the same number is spelled differently in main.tex, paper_draft.md and a
    Python docstring. Matching is now formatting-insensitive (see `_normalize`),
    so a variant written in any one of those spellings catches all of them and a
    cosmetic reformatting cannot let a retracted string slip past. Occurrences
    inside a LaTeX ``...'' quotation are exempt for the same reason as in
    check_absent. `allow_files` names files where the string is legitimately
    present (e.g. the script that withdraws it)."""
    global checks
    checks += 1
    hits = []
    for fname, text in SHIPPED_TEXT.items():
        if fname in allow_files:
            continue
        norm, imap = _NORM_ALL[fname]
        for s in variants:
            ns, _ = _normalize(s)
            if not ns:
                continue
            i = norm.find(ns)
            while i != -1:
                a0 = imap[i]
                a1 = imap[min(i + len(ns) - 1, len(imap) - 1)] + 1
                inside = any(a <= a0 and a1 <= b
                             for a, b in _QUOTED_ALL.get(fname, []))
                if not inside:
                    hits.append(f"{fname}:{text[:a0].count(chr(10)) + 1}:{s!r}")
                i = norm.find(ns, i + 1)
    ok = not hits
    print(f"  {'OK  ' if ok else 'FAIL'}  retracted across ALL shipped text: {label!r}"
          + ("" if ok else f"  -> {hits[:6]}"))
    if not ok:
        failures.append(f"retracted claim {label!r} survives in shipped text: {hits}")


print("== Mechanism 2 (GUARDIAN, code/48) ==")
_cs2 = load("case_study_2_layer_decomposition.json")
d = _cs2["randomized_stratified_splits"]
check("selection-specific component mean", d["selection_specific_component_mean"], "{:.4f}")
check("selection-specific component SD", d["selection_specific_component_sd"], "{:.4f}")
check("general gap mean", d["mean_gap_all_32_layers_mean"], "{:.4f}")
check("general gap SD", d["mean_gap_all_32_layers_sd"], "{:.4f}")
check("n_reps", d["n_reps"], "{:d}")
assert d["n_reps"] >= 50, "N_REPS must not silently shrink back"
assert "independence_caveat" in d, "the shared-dataset caveat must ship in the JSON"
assert "mechanical_positivity_caveat" in d
# The probe's convergence claim in SS4.2 must be measured, not assumed.
_cv = d["convergence"]
check("primary-probe max n_iter_", _cv["max_n_iter"], "{:d}")
assert _cv["n_convergence_warnings"] == 0 and _cv["max_n_iter"] < 1000
checks += 1
_rr = _cs2["regularization_robustness"]["by_C"]
_sel = [_rr[k]["selection_specific_component_mean"] for k in _rr]
_gen = [_rr[k]["general_gap_mean"] for k in _rr]
_ok = all(x > 0 for x in _sel) and all(abs(x) < 0.02 for x in _gen)
print(f"  {'OK  ' if _ok else 'FAIL'}  regularization robustness: sel-specific stays positive "
      f"({min(_sel):+.4f} to {max(_sel):+.4f}), general gap stays ~0 "
      f"({min(_gen):+.4f} to {max(_gen):+.4f})")
if not _ok:
    failures.append("regularization-robustness sweep no longer supports SS4.2's reading")
for k in ["0.01", "100.0"]:
    check(f"regularization sweep C={k} sel-specific", _rr[k]["selection_specific_component_mean"], "{:.4f}")

print("\n== Mechanism 3 isotropic sweep (code/02d via code/44) ==")
d = load("statistical_rigor_retrofit.json")["isotropic_sweep"]["by_capacity"]
for cap in ["16", "48", "128", "384"]:
    check(f"cap {cap} leaky-clean_matched", d[cap]["leaky_minus_clean_matched"]["gap_mean"])
# The seed-decoupling retrofit must not silently revert to the confounded scheme.
checks += 1
_02d = load("corrected_capacity_placebo_sweep.json")
_ok = _02d["config"].get("seed_scheme") == "decoupled"
print(f"  {'OK  ' if _ok else 'FAIL'}  code/02d ran with decoupled data/split/fold/init seeds")
if not _ok:
    failures.append("code/02d reverted to the coupled single-seed scheme (SS4.3 says it is fixed)")
checks += 1
# ...and the legacy coupled run must still ship, so the comparison SS4.3 makes is checkable.
_ok = (R / "corrected_capacity_placebo_sweep_coupled_seed_legacy.json").exists()
print(f"  {'OK  ' if _ok else 'FAIL'}  the superseded coupled-seed run ships for comparison")
if not _ok:
    failures.append("coupled-seed legacy JSON missing; SS4.3's before/after comparison is unverifiable")
checks += 1
# code/47's default cell must reproduce code/02d's cap-128 cell. This is a
# DETERMINISM check, not independent confirmation: code/47's docstring states
# its harness is "an exact port of code/02d", so the two share an
# implementation. It catches drift between them; it does not corroborate the
# result via a second implementation. (An adversarial review found the previous
# comment here, and the matching sentence in SS4.3, calling code/47's cell
# "independently-written". Both are withdrawn.)
_c47 = load("selection_multiplicity_sweep.json")["sweep_C_operating_point"]["0.8"]["gap_mean"]
_c02d = d["128"]["leaky_minus_clean_matched"]["gap_mean"]
_ok = abs(_c47 - _c02d) < 5e-5
print(f"  {'OK  ' if _ok else 'FAIL'}  code/47 default cell reproduces code/02d cap-128 "
      f"(determinism check, shared implementation) ({_c47:+.5f} vs {_c02d:+.5f})")
if not _ok:
    failures.append("code/47 and code/02d no longer agree on the shared configuration")

print("\n== Mechanism 3 fidelity extension (code/49) ==")
d = load("mechanism3_fidelity_extension.json")
g = d["leaky_plus_lrsched_minus_clean_matched_plus_lrsched"]
assert d["es_patience"] == 15, "reported run must use the repo-faithful ES patience of 15"
assert d["lr_patience"] == 3, "scheduler patience must remain 3"
assert d["calibration_method"] == "apply_calibration_label_free"
# Issue 14: the ported optimizer/data-pipeline settings must not silently revert.
import re as _re
_c49 = (ROOT / "code" / "49_mechanism3_fidelity_extension.py").read_text()
for _name, _want in [("LEARNING_RATE", "2e-4"), ("BATCH_SIZE", "28"),
                     ("WARMUP_EPOCHS", "5"), ("GRAD_CLIP", "0.5"),
                     ("MIN_LR", "1e-7"), ("MAX_EPOCHS", "45")]:
    checks += 1
    _ok = _re.search(rf"^{_name} = {_re.escape(_want)}\b", _c49, _re.M) is not None
    print(f"  {'OK  ' if _ok else 'FAIL'}  code/49 ports {_name} = {_want}")
    if not _ok:
        failures.append(f"code/49 no longer sets {_name} = {_want} (Appendix A issue 14)")
for _tok in ["AdamW", "RobustScaler", "clip_grad_norm_"]:
    checks += 1
    _ok = _tok in _c49
    print(f"  {'OK  ' if _ok else 'FAIL'}  code/49 uses {_tok}")
    if not _ok:
        failures.append(f"code/49 no longer uses {_tok} (Appendix A issue 14)")
check("fidelity-extension gap", g["gap_mean"])
check("fidelity-extension LEAKY operating point", d["means"]["leaky_plus_lrsched"], "{:.4f}")
# Appendix A item 10 quoted a stale CLEAN_MATCHED-vs-PLACEBO gap that did not
# survive the issue-14 fidelity port. Pin it to the shipped array.
check("fidelity-extension control-vs-placebo gap",
      d["clean_matched_plus_lrsched_minus_placebo_plus_lrsched"]["gap_mean"])
# Training-depth confound (5th correction) must be tracked, not silently dropped.
checks += 1
_be = d.get("best_epoch_stats")
_ok = _be is not None and "mean_best_epoch" in _be
print(f"  {'OK  ' if _ok else 'FAIL'}  code/49 tracks the training-depth confound (best_epoch_stats)")
if not _ok:
    failures.append("mechanism3_fidelity_extension.json has no best_epoch_stats (SS4.3 reports it)")
else:
    check("LEAKY mean kept-checkpoint epoch", _be["mean_best_epoch"]["leaky_plus_lrsched"], "{:.2f}")
    check("control mean kept-checkpoint epoch",
          _be["mean_best_epoch"]["clean_matched_plus_lrsched"], "{:.2f}")
    check("training-depth relative difference", _be["relative_difference_pct"], "{:.1f}")

print("\n== Mechanism 3 real-feature harness (code/43 via code/44) ==")
d = load("statistical_rigor_retrofit.json")["real_feature_train_only_calibrated"]["by_capacity"]
for cap in ["128", "384"]:
    check(f"cap {cap} leaky-clean_matched", d[cap]["leaky_minus_clean_matched"]["gap_mean"])
    check(f"cap {cap} tie count", d[cap]["leaky_minus_clean_matched"]["n_tied_abs_diffs"], "{:d}")

print("\n== Mechanism 4 (code/45) ==")
d = load("case_study_4_winners_curse.json")
# HEADLINE is the non-degenerate subset. The rotation estimator is
# algebraically 0 whenever all 3 rotations pick the same layer, so those cells
# measure nothing and must not be averaged in.
_nd = d["summary_non_degenerate"]
check("non-degenerate headline mean", _nd["mean"])
check("non-degenerate CI low", _nd["bca_ci_95"][0])
check("non-degenerate CI high", _nd["bca_ci_95"][1])
check("non-degenerate n_cells", _nd["n_cells"], "{:d}")
check("all-cells mean (contaminated, retained for continuity)", d["summary_all_cells"]["mean"])
check("all-cells CI low", d["summary_all_cells"]["bca_ci_95"][0])
check("all-cells CI high", d["summary_all_cells"]["bca_ci_95"][1])
check("non-saturated mean (contaminated)", d["summary_non_saturated"]["mean"])
check("non-saturated non-degenerate mean", d["summary_non_saturated_non_degenerate"]["mean"])
check("ceiling-saturated mean", d["summary_ceiling_saturated"]["mean"])
_da = d["degeneracy_audit"]
check("number of degenerate cells", _da["n_degenerate"], "{:d}")
check("degenerate cells' permutation-null mean",
      _da["permutation_null_mean_over_degenerate_cells"])
checks += 1
# Every degenerate cell must be EXACTLY zero, not a float residue: scipy's
# Wilcoxon drops exact zeros but ranks +-3.7e-17 as real observations, which
# is what moved the non-saturated subgroup's p from 0.047 to 0.0625.
_ok = all(v["winners_curse_estimate"] == 0.0 for v in d["per_cell"].values()
          if v["estimator_degenerate"])
print(f"  {'OK  ' if _ok else 'FAIL'}  degenerate cells are snapped to EXACT zero "
      f"(no float residue reaching the Wilcoxon)")
if not _ok:
    failures.append("code/45 is letting float residues through as signed observations again")
checks += 1
# The degeneracy must be DETECTED structurally, not inferred from the value.
_ok = all(("estimator_degenerate" in v and "selected_layers_across_rotations" in v)
          for v in d["per_cell"].values())
print(f"  {'OK  ' if _ok else 'FAIL'}  every cell records its per-rotation selected layers "
      f"and a degeneracy flag")
if not _ok:
    failures.append("code/45 no longer records the degeneracy diagnostic per cell")
checks += 1
# code/45's permutation null and bootstrap must be REPRODUCIBLE. An earlier
# version seeded them from Python's builtin hash() of the cell name, which is
# salted per interpreter process (PYTHONHASHSEED), so the shipped numbers could
# not be re-derived from the shipped code -- the exact failure mode this paper
# is about. It must use a stable hash.
_c45 = (ROOT / "code" / "45_case_study_4_winners_curse.py").read_text()
_ok = "zlib.crc32" in _c45 and "default_rng(abs(hash(" not in _c45
print(f"  {'OK  ' if _ok else 'FAIL'}  code/45 seeds its permutation null from a STABLE hash "
      f"(not Python's salted hash())")
if not _ok:
    failures.append("code/45's permutation null is seeded from a per-process-salted hash; "
                    "its reported null and bootstrap values are not reproducible")
checks += 1
# Per-model probed-layer counts: the paper said a flat "33 layers" for all 24
# cells; qwen2.5-7b probes 29.
_lc = d["per_model_layer_counts"]
_ok = _lc == {"llama3.1-8b": 33, "mistral-7b": 33, "qwen2.5-7b": 29} and "29" in TEX
print(f"  {'OK  ' if _ok else 'FAIL'}  per-model probed-layer counts shipped and stated: {_lc}")
if not _ok:
    failures.append("per-model layer counts missing from the JSON or from main.tex")

print("\n== Operating-point transport check (code/58) ==")
d = load("operating_point_transport_check.json")
_obs = d["matched_operating_point_comparison"]["observations"]
checks += 1
_ok = d["verdict"] == "HARNESS_SPECIFIC_NOT_A_TRANSPORTABLE_LAW"
print(f"  {'OK  ' if _ok else 'FAIL'}  verdict: {d['verdict']}")
if not _ok:
    failures.append("code/58's verdict changed; SS5's transport paragraph depends on it")
for o in _obs[1:]:
    check(f"ratio vs matched Sweep C cell ({o['harness'][:28]})",
          o["ratio_vs_sweep_C_matched_cell"], "{:.1f}")
    check(f"achieved operating point ({o['harness'][:28]})",
          o["achieved_operating_point"], "{:.4f}")
checks += 1
# The whole point is that these are matched on operating point.
_ok = d["matched_operating_point_comparison"]["operating_point_spread"] < 0.005
print(f"  {'OK  ' if _ok else 'FAIL'}  the three harnesses are matched on operating point to "
      f"{d['matched_operating_point_comparison']['operating_point_spread']:.4f} AUROC")
if not _ok:
    failures.append("code/58's three harnesses are no longer operating-point matched")

print("\n== Severity relationship: operating point (code/47 Sweep C) ==")
d = load("selection_multiplicity_sweep.json")["sweep_C_operating_point"]
check("AUROC0=0.70 gap", d["0.7"]["gap_mean"])
check("AUROC0=0.985 gap", d["0.985"]["gap_mean"])
ratio = d["0.7"]["gap_mean"] / d["0.985"]["gap_mean"]
check("operating-point decline factor", ratio, "{:.1f}")

print("\n== Severity relationship: K scaling refit (code/50) ==")
_evt = load("evt_scaling_refit.json")
d = _evt["fits"]
check("EVT per-cell-sigma R^2", d["evt_per_cell_sigma"]["r_squared"], "{:.3f}")
check("log two-parameter R^2", d["log_two_parameter"]["r_squared"], "{:.3f}")
checks += 1
# The EVT test is UNTESTABLE here, not falsified: sigma is a single training
# trajectory's per-epoch dispersion, not the sampling-noise SD EVT requires.
_ok = (d["evt_per_cell_sigma"].get("verdict") == "UNTESTABLE_IN_THIS_HARNESS_AS_INSTRUMENTED"
       and _evt["sigma_misspecification"]["verdict"] ==
       "UNTESTABLE_IN_THIS_HARNESS_AS_INSTRUMENTED")
print(f"  {'OK  ' if _ok else 'FAIL'}  the EVT fit ships labelled UNTESTABLE, not falsified")
if not _ok:
    failures.append("code/50 no longer records the sigma misspecification; SS5, Limitations "
                    "(2a)/(3) and Appendix A all depend on the downgraded verdict")
# M4: the K law must survive restriction to the flat-operating-point range.
_r = d["log_two_parameter_K_ge_10"]
check("K-law restricted to K>=10: b", _r["params"]["b"], "{:.5f}")
check("K-law restricted to K>=10: R^2", _r["r_squared"], "{:.3f}")
checks += 1
_full_b = d["log_two_parameter"]["params"]["b"]
_ok = abs(_r["params"]["b"] - _full_b) / abs(_full_b) < 0.10
print(f"  {'OK  ' if _ok else 'FAIL'}  K law survives restricting to the flat-operating-point "
      f"range (b {_full_b:.5f} -> {_r['params']['b']:.5f})")
if not _ok:
    failures.append("the K law no longer survives the K>=10 restriction; SS5 says it does")
checks += 1
# ...and Sweep A's achieved operating-point drift must be disclosed, not just its target.
_lo, _hi = _evt["achieved_operating_point_full_range"]
_ok = f"{_lo:.4f}" in TEX and f"{_hi:.4f}" in TEX or f"{_lo:.4f}" in TEX
print(f"  {'OK  ' if _ok else 'FAIL'}  Sweep A's achieved operating-point drift is disclosed "
      f"({_lo:.4f} to {_hi:.4f})")
if not _ok:
    failures.append("Sweep A's achieved operating-point drift is not disclosed in main.tex")

print("\n== K-law degeneracy gate (code/47 Sweep A -> code/50) ==")
_sw = load("selection_multiplicity_sweep.json")
checks += 1
# P0-3: the gate must actually be ENABLED at code/47's own call sites, not just
# defined. Every sweep must ship a per-cell degeneracy record.
_missing = [f"{name}[{k}]" for name in ["sweep_A_K", "sweep_B_sample_size",
                                        "sweep_C_operating_point", "sweep_D_n_val_isolated"]
            for k, v in _sw[name].items() if "degeneracy" not in v]
_ok = not _missing
print(f"  {'OK  ' if _ok else 'FAIL'}  every cell of Sweeps A/B/C/D carries a degeneracy record"
      + ("" if _ok else f" (missing: {_missing[:5]})"))
if not _ok:
    failures.append("code/47 no longer runs with degeneracy_check=True at every call site; "
                    "Appendix A.11 Correction 4 and SS5.3's K-law both depend on the records")
_ev = load("evt_scaling_refit.json")
_nd = _ev["fits"]["log_two_parameter_non_degenerate"]
_con = _ev["fits"]["log_two_parameter"]
check("corrected K-law slope b", _nd["params"]["b"], "{:.5f}")
check("corrected K-law R^2", _nd["r_squared"], "{:.3f}")
check("contaminated K-law R^2 (quoted as superseded)", _con["r_squared"], "{:.3f}")
checks += 1
# The threshold must be READ from code/57, not re-picked here after seeing which
# cells it removes. And no cell may sit near it.
_ok = (_nd["degeneracy_threshold_identical_state_dict_frac"] == 0.50
       and _nd["identical_frac_retained_max"] < 0.50 < _nd["identical_frac_excluded_min"]
       and "code/57" in _nd["threshold_source"])
print(f"  {'OK  ' if _ok else 'FAIL'}  gate threshold 0.50 read from code/57; retained cells "
      f"<= {_nd['identical_frac_retained_max']:.3f}, excluded >= "
      f"{_nd['identical_frac_excluded_min']:.3f} (no cell near the boundary)")
if not _ok:
    failures.append("the K-law degeneracy threshold is no longer inherited from code/57, or a "
                    "cell now sits near it; Appendix A.11 claims both")
checks += 1
# The correction's honest content: the slope survives, the dynamic range does not.
_slope_moved = abs(_nd["params"]["b"] - _con["params"]["b"]) / abs(_con["params"]["b"])
_ok = (_slope_moved < 0.10
       and _nd["dynamic_range_max_over_min"] < 0.5 * _con["dynamic_range_max_over_min"]
       and f"{_nd['dynamic_range_max_over_min']:.2f}" in TEX
       and f"{_con['dynamic_range_max_over_min']:.2f}" in TEX)
print(f"  {'OK  ' if _ok else 'FAIL'}  slope survives the gate ({_slope_moved * 100:.0f}% move) "
      f"but dynamic range falls {_con['dynamic_range_max_over_min']:.2f}x -> "
      f"{_nd['dynamic_range_max_over_min']:.2f}x, and both are in the tex")
if not _ok:
    failures.append("SS5.3 and A.11 state that the K-law's slope survives the degeneracy gate "
                    "while its dynamic range does not; that no longer holds or is not quoted")
checks += 1
_ok = _nd["K_values_excluded_as_degenerate"] == [3, 5, 10] and _nd["K_values_fit"] == [
    15, 25, 45, 75, 135, 225]
print(f"  {'OK  ' if _ok else 'FAIL'}  gate excludes K={_nd['K_values_excluded_as_degenerate']} "
      f"and fits K={_nd['K_values_fit']}")
if not _ok:
    failures.append("the set of degenerate K cells changed; SS5.3 names 1/3/5/10 explicitly")

print("\n== Adaptive-selection control power (code/22) ==")
import numpy as np
d = load("epoch_forcing_confound_control.json")["by_capacity"]
# P0-1 fix: the CORRECTED arm trains on tr2_idx, disjoint from its es_idx
# selection set (mirroring code/43). The superseded arm trained on the full
# tr_idx while selecting on a subset of it. Both retentions are checked, and
# both must appear in the tex -- the corrected one as the reported number, the
# contaminated one as the disclosed artifact.
for cap, want_fixed, want_contam in [("128", 94.2, 37.3), ("384", 79.9, 25.7)]:
    r = d[cap]["placebo_relative_retention_pct"]
    for arm, want in [("clean_matched_adaptive", want_fixed),
                      ("clean_matched_adaptive_contaminated", want_contam)]:
        checks += 1
        got = r[arm]
        ok = abs(got - want) < 0.15 and f"{got:.1f}" in TEX
        print(f"  {'OK  ' if ok else 'FAIL'}  cap {cap} placebo-relative retention "
              f"[{arm[:34]:34s}]: {got:.1f}%")
        if not ok:
            failures.append(f"retention cap {cap} {arm}: computed {got:.1f}%, "
                            f"wanted ~{want}% and present in tex")
    checks += 1
    # The diagnostic that exposes the defect: an arm selecting in-sample runs to
    # the end of the budget, an honest one does not.
    be = d[cap]["best_epoch_stats"]["mean_best_epoch"]
    _ok = (be["clean_matched_adaptive_contaminated"] > be["clean_matched_adaptive"] + 5
           and abs(be["clean_matched_adaptive"] - be["clean_matched"]) < 1.0)
    print(f"  {'OK  ' if _ok else 'FAIL'}  cap {cap} the contaminated arm runs deeper "
          f"({be['clean_matched_adaptive_contaminated']:.1f}) than the corrected arm "
          f"({be['clean_matched_adaptive']:.1f}), which tracks the honest arms "
          f"({be['clean_matched']:.1f})")
    if not _ok:
        failures.append(f"code/22 cap {cap}: the training-depth signature of the "
                        f"superseded control no longer reproduces; A.1 item 16 cites it")
checks += 1
# The corrected control must still be BEATEN by LEAKY -- that is the finding.
_ok = all(d[c]["gaps"]["leaky_minus_clean_matched_adaptive"]["mean"] > 0
          and d[c]["gaps"]["leaky_minus_clean_matched_adaptive"]["wilcoxon_p"] < 0.01
          for c in ["128", "384"])
print(f"  {'OK  ' if _ok else 'FAIL'}  LEAKY still beats the CORRECTED adaptive control at "
      f"both capacities (p<0.01): "
      + ", ".join(f"cap {c}: {d[c]['gaps']['leaky_minus_clean_matched_adaptive']['mean']:+.4f} "
                  f"(p={d[c]['gaps']['leaky_minus_clean_matched_adaptive']['wilcoxon_p']:.4f})"
                  for c in ["128", "384"]))
if not _ok:
    failures.append("code/22's corrected control no longer supports SS4.3's fold-reuse finding")

print("\n== Fidelity extension: budget matching and coupling isolation (code/49) ==")
d = load("mechanism3_fidelity_extension.json")
check("gap vs BUDGET-MATCHED control",
      d["leaky_plus_lrsched_minus_clean_matched_budget_matched"]["gap_mean"])
check("gap vs superseded es-fed control",
      d["leaky_plus_lrsched_minus_clean_matched_plus_lrsched"]["gap_mean"])
check("coupling continuity, isolated within one harness",
      d["coupling_continuity_isolation"]["gap_mean"])
checks += 1
# P0-2(d): the isolating arm must show that continuous coupling does NOT add
# severity over a one-shot argmax. The paper retracts the opposite claim.
_ok = d["coupling_continuity_isolation"]["gap_mean"] < 0
print(f"  {'OK  ' if _ok else 'FAIL'}  continuous coupling does NOT beat a one-shot argmax "
      f"({d['coupling_continuity_isolation']['gap_mean']:+.4f}); SS4.3 and the checklist "
      f"table retract the claim that it does")
if not _ok:
    failures.append("code/49's isolating arm no longer refutes the coupling-continuity claim, "
                    "which SS4.3 and Table checklist-scope both now state as retracted")
checks += 1
_r = d["sanity_ratios"]
_ok = _r["budget_matched_ratio_inside_comparator_band"] and not (
    _r["comparator_band"][0] <= _r["this_harness_vs_es_fed_control_superseded"]
    <= _r["comparator_band"][1])
print(f"  {'OK  ' if _ok else 'FAIL'}  (LEAKY-CM)/(CM-PLACEBO): budget-matched "
      f"{_r['this_harness_vs_budget_matched_control']:.2f} is INSIDE the comparator band "
      f"{[round(x, 2) for x in _r['comparator_band']]}, superseded "
      f"{_r['this_harness_vs_es_fed_control_superseded']:.2f} is outside")
if not _ok:
    failures.append("code/49's sanity-ratio story changed; SS4.3 states the budget-matched "
                    "ratio is inside the band and the superseded one outside")
checks += 1
_ratio = (d["leaky_plus_lrsched_minus_clean_matched_budget_matched"]["gap_mean"]
          / load("statistical_rigor_retrofit.json")["real_feature_train_only_calibrated"]
          ["by_capacity"]["128"]["leaky_minus_clean_matched"]["gap_mean"])
_ok = f"{_ratio:.1f}" in TEX
print(f"  {'OK  ' if _ok else 'FAIL'}  corrected like-for-like ratio {_ratio:.1f}x present in tex")
if not _ok:
    failures.append(f"corrected like-for-like ratio {_ratio:.1f}x missing from main.tex")

print("\n== Case Study 4 permutation null, reported in BOTH directions (code/45) ==")
d = load("case_study_4_winners_curse.json")["permutation_significance_audit"]
checks += 1
_ok = d["n_cells_p_below_05"] == 0 and "0 of the 24" in TEX.replace("$", "")
print(f"  {'OK  ' if _ok else 'FAIL'}  {d['n_cells_p_below_05']}/{d['n_cells_total']} cells reach "
      f"p<0.05 against their own permutation null, and the tex says so")
if not _ok:
    failures.append("main.tex must state plainly that 0 of the 24 Case Study 4 cells reach "
                    "p<0.05 against their own permutation null")
checks += 1
# The unfavourable direction: the non-degenerate cells' own null sits ABOVE
# their observed mean. Previously only the favourable direction was reported.
_ok = d["non_degenerate_observed_minus_null"] < 0 and \
    f"{d['non_degenerate_null_mean']:+.4f}".replace("+", "") in TEX
print(f"  {'OK  ' if _ok else 'FAIL'}  the non-degenerate cells' null ({d['non_degenerate_null_mean']:+.4f}) "
      f"sits above their observed mean ({d['non_degenerate_observed_mean']:+.4f}), and the tex "
      f"discloses it")
if not _ok:
    failures.append("main.tex must disclose that the non-degenerate cells' own permutation null "
                    "sits above their observed mean, not only the degenerate cells' favourable null")

print("\n== Fidelity-extension 2x2 ablation (code/54) ==")
d = load("fidelity_extension_2x2_ablation.json")
for c in d["cells"]:
    check(f"cell {c['calibration'][:14]}/ES{c['es_patience']}", c["gap_mean"])
checks += 1
# RETRACTED (Appendix A issues 12 and 14): under the ported training loop the
# patience correction no longer flips sign. This check now guards the retraction
# -- if the flip ever reappears, the appendix text must be revisited, not silently
# left stale in the other direction.
_ok = not d["sign_of_patience_effect_flips"]
print(f"  {'OK  ' if _ok else 'FAIL'}  patience-correction sign flip is ABSENT (retracted claim): "
      f"flips={d['sign_of_patience_effect_flips']}")
if not _ok:
    failures.append("2x2 ablation shows a sign flip again; Appendix A item 12 retracts it")
checks += 1
_cal = abs(d["calibration_effect_at_patience_15"])
_pat = abs(d["patience_effect_at_label_free_calibration"])
_ok = _cal > 5 * _pat
print(f"  {'OK  ' if _ok else 'FAIL'}  calibration effect dominates patience effect: "
      f"{_cal:+.4f} vs {_pat:+.4f}")
if not _ok:
    failures.append("Appendix A item 12's decomposition claim no longer holds")
# like-for-like ratio quoted in Appendix A issue 12
_rf = load("statistical_rigor_retrofit.json")["real_feature_train_only_calibrated"]
_ratio = (d["currently_reported_number"]
          / _rf["by_capacity"]["128"]["leaky_minus_clean_matched"]["gap_mean"])
check("like-for-like fidelity ratio at capacity 128", _ratio, "{:.1f}")

print("\n== Case Study 2 derived artifact must exist and replay ==")
checks += 1
_art = R / "case_study_2_probe_scores.npz"
_ok = _art.exists() and _art.stat().st_size < 5e6
print(f"  {'OK  ' if _ok else 'FAIL'}  shipped derived artifact present and small: "
      f"{_art.name} ({_art.stat().st_size / 1e6:.2f} MB)" if _art.exists() else "  FAIL  missing")
if not _ok:
    failures.append("case_study_2_probe_scores.npz missing or unexpectedly large")

print("\n== n_val isolation (code/47 Sweep D) and Sweep B relabelling ==")
d = load("selection_multiplicity_sweep.json")
assert "sweep_B_sample_size" in d and "sweep_B_n_val" not in d, \
    "Sweep B must be relabelled a sample-size sweep, not an n_val sweep"
assert "sweep_B_confound_note" in d
for k in ["2", "3", "5", "10"]:
    check(f"Sweep D K_CV={k} gap", d["sweep_D_n_val_isolated"][k]["gap_mean"])
checks += 1
_g = [d["sweep_D_n_val_isolated"][k]["gap_mean"] for k in ["2", "3", "5", "10"]]
# Non-increasing to within 1e-4: the K_CV=3 and K_CV=5 cells are tied at +0.0036
# (they differ by 2.7e-5), which SS5 and Appendix A both state rather than
# describing the sequence as strictly monotone.
_ok = all(_g[i] >= _g[i + 1] - 1e-4 for i in range(len(_g) - 1)) and _g[0] > _g[-1]
print(f"  {'OK  ' if _ok else 'FAIL'}  Sweep D gap is non-increasing as n_val falls "
      f"(opposite to the 1/sqrt(n_val) prediction): {[round(x, 4) for x in _g]}")
if not _ok:
    failures.append("Sweep D no longer shows the pattern SS5 and Appendix A describe")

print("\n== OOF-averaging alternative explanation (code/55) ==")
d = load("oof_averaging_control.json")["by_capacity"]
for cap in ["16", "48", "128", "384"]:
    check(f"cap {cap} un-averaged gap", d[cap]["single_model"]["leaky_minus_clean_matched"]["mean"])
    checks += 1
    _a = d[cap]["averaged"]["leaky_minus_clean_matched"]["mean"]
    _s = d[cap]["single_model"]["leaky_minus_clean_matched"]["mean"]
    _ok = _s > _a
    print(f"  {'OK  ' if _ok else 'FAIL'}  cap {cap}: un-averaged ({_s:+.4f}) exceeds averaged "
          f"({_a:+.4f}), ratio {_s / _a:.1f}x")
    if not _ok:
        failures.append(f"cap {cap}: averaging no longer attenuates the gap; SS4.3 says it does")
    # the averaged readout must still reproduce code/02d exactly
    checks += 1
    _ref = load("statistical_rigor_retrofit.json")["isotropic_sweep"]["by_capacity"][cap][
        "leaky_minus_clean_matched"]["gap_mean"]
    _ok = abs(_a - _ref) < 1e-9
    print(f"  {'OK  ' if _ok else 'FAIL'}  cap {cap}: averaged readout reproduces code/02d exactly")
    if not _ok:
        failures.append(f"cap {cap}: code/55's averaged readout diverged from code/02d")

print("\n== Fixed-d eigenspectrum sweep (code/56) ==")
d = load("eigenspectrum_sweep_fixed_dim.json")
assert d["feat_dim"] == 64, "the whole point is that d is held at the isotropic sweep's 64"
for name in ["powerlaw_beta0.0", "powerlaw_beta2.0", "real_top64"]:
    check(f"cap128 {name} gap", d["by_capacity"]["128"][name]["leaky_minus_clean_matched"]["mean"])
checks += 1
_c = d["by_capacity"]["128"]
_ok = all(abs(_c[n]["j_realized_population"] - d["j_target"]) < 1e-9 for n in _c)
print(f"  {'OK  ' if _ok else 'FAIL'}  every eigenspectrum cell is calibrated to the same "
      f"Mahalanobis J={d['j_target']:.4f}")
if not _ok:
    failures.append("code/56 cells are not all calibrated to the same J")

print("\n== Guardrail's identity-covariance assumption is documented ==")
checks += 1
_sc = (ROOT / "code" / "sanity_checks.py").read_text()
_ok = "Sigma = I" in _sc and "trace(Sigma)" in _sc and "NOT DIRECTLY APPLICABLE" in _sc
print(f"  {'OK  ' if _ok else 'FAIL'}  sanity_checks.py states the identity-covariance assumption")
if not _ok:
    failures.append("sanity_checks.py no longer documents its Sigma=I assumption")

print("\n== Mechanism 5 sample-size sensitivity (code/46, SWEEP=N) ==")
d = load("mechanism5_threshold_selection.json")
checks += 1
_ss = d.get("sample_size_sensitivity")
_ok = _ss is not None
print(f"  {'OK  ' if _ok else 'FAIL'}  Mechanism 5's n_test/n_val dependence is measured, not assumed")
if not _ok:
    failures.append("mechanism5_threshold_selection.json has no sample_size_sensitivity block")
else:
    for n in ["700", "1750", "3500", "10000"]:
        check(f"n={n} F1 gap", _ss["by_n_samples"][n]["f1_gap_mean"])
    check("F1-gap shrinkage factor 140->2000", _ss["shrinkage_factor_700_to_10000"], "{:.1f}")

print("\n== Joint severity surface (code/57) ==")
try:
    d = load("joint_severity_surface.json")
    checks += 1
    _ok = max(v["abs_diff"] for v in d["internal_consistency_vs_sweep_C"].values()) < 5e-5
    print(f"  {'OK  ' if _ok else 'FAIL'}  the grid's K=45 column reproduces code/47's Sweep C")
    if not _ok:
        failures.append("code/57's K=45 column diverged from code/47's Sweep C; harness drifted")
    for name in ["M1", "M3", "M4"]:
        check(f"{name} LOO R^2", d["fits"][name]["loo_r_squared"], "{:.3f}")
    check("interaction F-test p", d["interaction_f_test"]["p"], "{:.3f}")
    checks += 1
    _ok = "not a bound" in d["framing"]
    print(f"  {'OK  ' if _ok else 'FAIL'}  the joint fit ships labelled as an empirical fit, not a bound")
    if not _ok:
        failures.append("code/57's framing no longer disclaims being a bound")
    # ── the degeneracy gate: K=3 produced the old interaction on its own ──
    checks += 1
    _g = d["degeneracy_gate"]
    _ok = _g["passed"] and not _g["cells_exceeding"]
    print(f"  {'OK  ' if _ok else 'FAIL'}  degeneracy gate passed; worst fitted cell has "
          f"{_g['max_identical_frac_in_grid']:.2f} bitwise-identical LEAKY/CLEAN_MATCHED folds")
    if not _ok:
        failures.append(f"code/57's grid contains degenerate cells: {_g['cells_exceeding']}")
    checks += 1
    # K=3 must NOT be in the fitted grid, and its degeneracy must ship as evidence.
    _ok = (3 not in d["grid"]["K_values"] and _g["dropped_K3_column_identical_frac"]
           and min(_g["dropped_K3_column_identical_frac"].values()) > 0.5)
    print(f"  {'OK  ' if _ok else 'FAIL'}  K=3 is excluded from the fit and its degeneracy "
          f"ships as evidence "
          f"({min(_g['dropped_K3_column_identical_frac'].values()):.2f}-"
          f"{max(_g['dropped_K3_column_identical_frac'].values()):.2f} identical)")
    if not _ok:
        failures.append("code/57 refit K=3 or stopped shipping the evidence for dropping it")
    checks += 1
    # Every fitted cell must carry its own degeneracy record.
    _ok = all("degeneracy" in c for c in d["cells"].values())
    print(f"  {'OK  ' if _ok else 'FAIL'}  every fitted cell carries a per-cell degeneracy record")
    if not _ok:
        failures.append("code/57 cells no longer carry per-cell degeneracy records")
    checks += 1
    # The old, degenerate grid's F-test must ship for the record, and must NOT
    # be the number the paper reports.
    _old = d.get("interaction_f_test_on_old_degenerate_grid")
    _ok = _old is not None and abs(_old["p"] - 0.027) < 0.002
    print(f"  {'OK  ' if _ok else 'FAIL'}  the superseded degenerate-grid F-test ships for the "
          f"record (p={_old['p']:.4f} if present)" if _old else "  FAIL  missing")
    if not _ok:
        failures.append("code/57 no longer ships the old degenerate-grid F-test for comparison")
    # NOTE: F(1,16)=5.92 still APPEARS in SS5, but only inside the paragraph that
    # retracts it. It is deliberately not check_absent'd -- the retraction has to
    # be able to quote the number it is retracting. What is guarded instead is
    # the CLAIM the number was used to support (see the check_absent block below
    # for "interact multiplicatively rather than additively").
except FileNotFoundError:
    checks += 1
    print("  FAIL  joint_severity_surface.json missing (SS5 reports the joint fit)")
    failures.append("joint_severity_surface.json missing")

print("\n== Retracted claims must be gone ==")
check_absent("5.2x asserted as the current like-for-like ratio", "ratio of \\textbf{$5.2\\times$")
check_absent("severity differs sharply", "Severity differs sharply by mechanism")
check_absent("EVT described as confirmed", "the $K$-scaling prediction is now confirmed")
check_absent("Mechanisms 4/5 mapped to L1.1", "instances of L1.1")
check_absent("resists simple pattern-matching (unqualified)",
             "This class of leakage resists simple pattern-matching")
check_absent("verify all four case studies end to end",
             "independently verify all four case studies end to end")
check_absent("old alpha 0.1328 as the reported calibration",
             "converges to $\\alpha=0.1328$")
check_absent("2x2 sign flip asserted as a current finding",
             "The patience correction's sign flips with the operating")
check_absent("Sweep B still called an n_val sweep",
             "Sweeping $n_{\\text{val}}$ (via")
check_absent("stale fidelity-extension number", "$+0.0221$ (BCa 95\\% CI")
check_absent("stale like-for-like ratio", "is a ratio of $\\mathbf{2.4\\times}$")
check_absent("~30 layers", "~30 layers")
check_absent("stale 8-rep GUARDIAN count", "under $8$ randomized stratified")
check_absent("stale pre-fidelity-port control-vs-placebo gap", "$+0.0498$, $p=3.7")
check_absent("stale coupled-seed Holm verdict (48 and 128)", "at 48 and 128 units")
check_absent("stale coupled-seed CM-placebo range", "gaps $+0.016$ to $+0.033$")
check_absent("stale coupled-seed isotropic range in prose",
             "$+0.0009$ to $+0.0034$ AUROC on the synthetic sweep")
check_absent("pooled-family claim that nothing survives",
             "would leave nothing significant at")
# ── retractions from the fourth adversarial review ──────────────────────────
check_absent("operating point asserted to dominate mechanism differences",
             "That single relationship moves severity more than any")
check_absent("operating-point decline asserted as larger than any mechanism difference",
             "which is larger than any difference we\nmeasure \\emph{between} mechanisms")
check_absent("EVT asserted as falsified (downgraded to untestable)",
             "is falsified by this data once fit as stated")
check_absent("EVT falsification asserted in Limitations",
             "derivation would most naturally rest on is falsified here once fit as")
check_absent("EVT functional form asserted not to fit",
             "The extreme-value functional form, tested as stated,")
check_absent("interaction asserted as a finding in the contributions list",
             "interact multiplicatively rather than additively")
check_absent("code/47's cell called independently written",
             "independently-written default cell")
check_absent("flat 33-layer count for all 24 Case Study 4 cells",
             "argmax, over $33$ layers")
check_absent("Case Study 4's contaminated mean used as the headline CI",
             "$[+0.0032,+0.0100]$")
check_absent("stale non-saturated-vs-saturated contrast",
             "$+0.0042$ versus $+0.0065$ on the non-saturated")
check_absent("replay script claiming every Case Study 2 number",
             "every Case Study 2 number in the paper is")

print("\n== Retracted claims must be gone from EVERY shipped text artifact ==")
# These are the exact failure modes an independent review found surviving in
# README.md, draft/paper_draft.md, draft/leakage_checklist.md and code/58's
# docstring after main.tex had already been corrected. Each is checked in every
# spelling it takes across LaTeX, the markdown mirror and Python docstrings.
check_absent_everywhere(
    "superseded 52x transport ratio (corrected to 38x)",
    "14x and 52x", "14$\\times$ and 52$\\times$", "$14\\times$ and $52\\times$",
    "by 14x and 52x", "14-52x", "14–52x")
check_absent_everywhere(
    "48.6x operating-point multiplier (withdrawn: no finite CI, Fieller g>1)",
    "falls 48.6x", "falls by 48.6x", "falls by $48.6\\times$",
    "a 48.6x decline", "a $48.6\\times$ decline",
    # "**48.6x**" was dropped when matching became formatting-insensitive:
    # normalization makes it identical to a bare mention, and bare mentions
    # legitimately appear in the prose that withdraws the multiplier.
    "by a factor of 48.6x", "by a factor of\n$48.6\\times$",
    "severity falls by 48.6x", "whose 48.6x", "whose $48.6\\times$",
    "moves it by 48.6x", "moves it by $48.6\\times$",
    allow_files=("code/60_operating_point_ratio_fieller_check.py",))
check_absent_everywhere(
    "FP16-vs-AWQ reported as identical at 0.9600 (actual: 0.9460 vs 0.9392)",
    "identical AUROC ($0.9600$ both)", "identical AUROC (0.9600 both)")
check_absent_everywhere(
    "the +0.00065 transport reference cell called indistinguishable from zero "
    "(it is not: p=0.037, CI excludes zero)",
    "a synthetic cell of $+0.00065$\nwhose own gap is not distinguishable from zero",
    "a synthetic cell of +0.00065 whose own gap is not distinguishable from zero")
check_absent_everywhere(
    "Case Study 2 separability quoted at an uncomputed 0.734-0.776",
    "at AUROC $0.734$--$0.776$", "at AUROC 0.734-0.776", "at AUROC 0.734–0.776",
    "separable at AUROC 0.734", "AUROC 0.734-0.776, i.e.",
    allow_files=("code/48_case_study_2_layer_decomposition.py",
                 "code/61_case_study_2_half_membership_probe.py"))
check_absent_everywhere(
    "all-data argmax layer stated as L19 (actual: L21)",
    "all-data AUROC picks L19", "L19 under all-data AUROC",
    allow_files=("code/61_case_study_2_half_membership_probe.py",))
check_absent_everywhere(
    "Delta_wc's uniform positivity offered as evidence (it is guaranteed: Thm B.5)",
    "resting on the uniform\npositivity across $17$ independently-published files",
    "resting on the uniform positivity across 17 independently-published files")

print("\n== P0-1: Case Study 4's primary estimator (code/59) ==")
try:
    _ea = load("case_study_4_estimator_audit.json")
    checks += 1
    _nn = _ea["nonnegativity_verification"]
    _ok = _nn["theorem_holds_to_float_tolerance"] and _nn["n_matrices"] >= 100000
    print(f"  {'OK  ' if _ok else 'FAIL'}  rotation estimator verified non-negative over "
          f"{_nn['n_matrices']} random matrices (min {_nn['min_estimate_overall']:.2e})")
    if not _ok:
        failures.append("code/59's non-negativity verification no longer passes; SS4.4 and "
                        "Theorem B.5 depend on it")
    _bp = _ea["bootstrap_max_bias_primary"]
    check("primary estimate, all 24 cells", _bp["all_24_cells"]["mean"])
    check("primary estimate CI low", _bp["all_24_cells"]["bca_ci_95"][0])
    check("primary estimate CI high", _bp["all_24_cells"]["bca_ci_95"][1])
    check("primary estimate, 17 non-degenerate", _bp["non_degenerate_17"]["mean"])
    check("primary estimate, 7 rotation-degenerate", _bp["degenerate_7"]["mean"])
    checks += 1
    # The whole point of promoting this estimator: it CAN return a negative value.
    _ok = _bp["all_24_cells"]["n_negative"] > 0
    print(f"  {'OK  ' if _ok else 'FAIL'}  the primary estimator is not sign-constrained "
          f"({_bp['all_24_cells']['n_negative']} of 24 cells negative)")
    if not _ok:
        failures.append("the bootstrap max-bias estimator returned no negative cells; SS4.4's "
                        "argument for promoting it over Delta_wc rests on that property")
    checks += 1
    _ts = _ea["tie_break_sensitivity"]
    _ok = (_ts["degenerate_count_min_over_tie_resolutions"]
           <= _ts["degenerate_count_shipped_argmax_convention"]
           <= _ts["degenerate_count_max_over_tie_resolutions"]
           and f"{_ts['degenerate_count_min_over_tie_resolutions']}"
           and f"{_ts['all24_mean_band'][0]:.4f}" in TEX
           and f"{_ts['all24_mean_band'][1]:.4f}" in TEX)
    print(f"  {'OK  ' if _ok else 'FAIL'}  tie-break band disclosed: degenerate count "
          f"{_ts['degenerate_count_min_over_tie_resolutions']}-"
          f"{_ts['degenerate_count_max_over_tie_resolutions']} of 24, headline band "
          f"[{_ts['all24_mean_band'][0]:+.4f},{_ts['all24_mean_band'][1]:+.4f}] present in tex")
    if not _ok:
        failures.append("SS4.4/B.5's tie-break sensitivity band is missing from main.tex or no "
                        "longer brackets the shipped argmax convention")
    checks += 1
    _sm = _ea["seed_count_criterion_mismatch"]
    _ok = _sm["ratio_k2_over_k3_mean"] > 1.0 and f"{round(_sm['overstatement_pct_mean']):d}\\%" in TEX
    print(f"  {'OK  ' if _ok else 'FAIL'}  2-seed-vs-3-seed criterion mismatch disclosed "
          f"(~{_sm['overstatement_pct_mean']:.0f}% overstatement)")
    if not _ok:
        failures.append("the 2-seed vs 3-seed selection-criterion mismatch is not disclosed in "
                        "main.tex; SS4.4 and B.5 must state it")
except FileNotFoundError:
    checks += 1
    print("  FAIL  case_study_4_estimator_audit.json missing (SS4.4's primary estimate)")
    failures.append("case_study_4_estimator_audit.json missing")

print("\n== P0-2: the withdrawn operating-point multiplier (code/60) ==")
try:
    _fc = load("operating_point_ratio_fieller_check.json")
    checks += 1
    _f = _fc["fieller"]
    _ok = _f["is_unbounded"] and _f["g_regenerated"] >= 1.0 and _f["g_from_shipped_bca"] >= 1.0
    print(f"  {'OK  ' if _ok else 'FAIL'}  Fieller g >= 1 both ways "
          f"(regenerated {_f['g_regenerated']:.2f}, from shipped BCa "
          f"{_f['g_from_shipped_bca']:.2f}) -> the ratio has no finite CI")
    if not _ok:
        failures.append("code/60 no longer shows the 48.6x ratio to be unbounded; SS5.3's "
                        "withdrawal depends on it")
    check("Fieller g, regenerated", _f["g_regenerated"], "{:.2f}")
    check("Fieller g, from shipped BCa halfwidth", _f["g_from_shipped_bca"], "{:.2f}")
    check("denominator-cell sign test p", _fc["denominator_cell"]["sign_test_p"], "{:.3f}")
    check("denominator-cell SD over mean", _fc["denominator_cell"]["sd_over_mean"], "{:.1f}")
    for _k in ["n_positive", "n_negative", "n_zero"]:
        check(f"denominator cell {_k}", _fc["denominator_cell"][_k], "{:d}")
    checks += 1
    _pb = _fc["paired_bootstrap"]
    _ok = (_pb["fraction_of_resamples_with_denominator_le_zero"] > 0.0
           and _pb["percentile_interval_95_of_finite_ratios"][0] < 0
           < _pb["percentile_interval_95_of_finite_ratios"][1])
    print(f"  {'OK  ' if _ok else 'FAIL'}  paired bootstrap: "
          f"{100 * _pb['fraction_of_resamples_with_denominator_le_zero']:.1f}% of resamples have "
          f"a non-positive denominator; interval "
          f"[{_pb['percentile_interval_95_of_finite_ratios'][0]:.0f},"
          f"{_pb['percentile_interval_95_of_finite_ratios'][1]:.0f}] spans zero")
    if not _ok:
        failures.append("code/60's paired bootstrap no longer shows the ratio spanning zero")
    checks += 1
    # The regeneration must reproduce the shipped cell means, or the check is worthless.
    _ok = (abs(_fc["denominator_cell"]["mean"] - _fc["denominator_cell"]["shipped_mean"]) < 1e-12
           and abs(_fc["numerator_cell"]["mean"] - _fc["numerator_cell"]["shipped_mean"]) < 1e-12)
    print(f"  {'OK  ' if _ok else 'FAIL'}  code/60's regenerated Sweep C endpoint cells reproduce "
          f"the shipped selection_multiplicity_sweep.json exactly")
    if not _ok:
        failures.append("code/60's regeneration diverged from the shipped Sweep C cells")
except FileNotFoundError:
    checks += 1
    print("  FAIL  operating_point_ratio_fieller_check.json missing (SS5.3's withdrawal)")
    failures.append("operating_point_ratio_fieller_check.json missing")

print("\n== P0-4: Case Study 2's half-membership probe has real provenance (code/61) ==")
try:
    _hm = load("case_study_2_half_membership_probe.json")
    checks += 1
    _pr = _hm["half_membership_probe"]
    _ok = _pr.get("status") == "COMPUTED" and len(_pr.get("per_layer", [])) == 32
    print(f"  {'OK  ' if _ok else 'FAIL'}  the half-membership probe is actually FITTED "
          f"({len(_pr.get('per_layer', []))} layers), not asserted in a docstring")
    if not _ok:
        failures.append("SS4.2's separability claim still has no computed backing")
    else:
        check("half-membership probe max AUROC", _pr["max_auroc"], "{:.4f}")
        check("half-membership probe min AUROC", _pr["min_auroc"], "{:.4f}")
        check("half-membership probe mean AUROC", _pr["mean_auroc"], "{:.4f}")
    checks += 1
    _al = _hm["all_data_argmax_layer"]
    _ok = _al.get("status") == "COMPUTED" and f"L{_al['argmax_layer']}" in TEX
    print(f"  {'OK  ' if _ok else 'FAIL'}  the all-data argmax layer is computed "
          f"(L{_al.get('argmax_layer')}) and stated in main.tex")
    if not _ok:
        failures.append("SS4.2's all-data argmax layer is uncomputed or disagrees with main.tex")
    _br = _hm["label_base_rates"]
    check("sequential selection-half positive rate",
          _br["sequential_selection_half_positive_rate"], "{:.4f}")
    check("sequential held-out-half positive rate",
          _br["sequential_heldout_half_positive_rate"], "{:.4f}")
    checks += 1
    _ok = (abs(_br["sequential_base_rate_difference_pp"]) > 5.0
           and abs(_br["randomized_base_rate_difference_pp_max_abs"]) < 1.0)
    print(f"  {'OK  ' if _ok else 'FAIL'}  the sequential halves differ in base rate by "
          f"{_br['sequential_base_rate_difference_pp']:+.1f} pp while the randomized reps agree "
          f"to {_br['randomized_base_rate_difference_pp_max_abs']:.2f} pp")
    if not _ok:
        failures.append("the base-rate confound SS4.2 now reports no longer reproduces")
    _mc = _hm["base_rate_matched_sequential_control"]
    check("base-rate-matched sel-specific component",
          _mc["selection_specific_component_mean"], "{:.4f}")
    check("base-rate-matched sel-specific SD", _mc["selection_specific_component_sd"], "{:.4f}")
except FileNotFoundError:
    checks += 1
    print("  FAIL  case_study_2_half_membership_probe.json missing (SS4.2's provenance)")
    failures.append("case_study_2_half_membership_probe.json missing")


# ═══════════════════════════════════════════════════════════════════════════
# follow-up REMEDIATION ROUND: every number the P1 items introduce
# ═══════════════════════════════════════════════════════════════════════════

print("\n== P1-1: Mechanism 5 measured at the threshold the repo reports at (code/62) ==")
_m5 = load("mechanism5_youden_threshold.json")
checks += 1
_ok = _m5["summary"]["reproduces_code46_f1argmax_column"]
print(f"  {'OK  ' if _ok else 'FAIL'}  code/62's F1-argmax column reproduces code/46's "
      f"shipped cells exactly (so the two scripts cannot drift)")
if not _ok:
    failures.append("code/62's F1-argmax column no longer reproduces code/46")
for _a0 in ["0.7", "0.8", "0.9", "0.95", "0.985"]:
    check(f"F1 gap at Youden, AUROC0={_a0}", _m5["cells"][_a0]["f1_gap_youden_mean"])
    check(f"negative reps at Youden, AUROC0={_a0}",
          _m5["cells"][_a0]["f1_gap_youden_n_negative"], "{:d}")
check("max Youden/F1-argmax ratio", _m5["summary"]["max_ratio_youden_over_f1argmax"], "{:.2f}")
checks += 1
_ok = _m5["summary"]["f1argmax_total_negative_reps"] == 0
print(f"  {'OK  ' if _ok else 'FAIL'}  the F1-argmax convention is algebraically non-negative "
      f"(0 negative reps over all 5 cells) while the Youden one is not")
if not _ok:
    failures.append("the F1-argmax non-negativity claim no longer holds")
checks += 1
_ok = _m5["summary"]["max_youden_wilcoxon_p"] < 5e-8
print(f"  {'OK  ' if _ok else 'FAIL'}  every Youden cell significant at p<5e-8 "
      f"(max p={_m5['summary']['max_youden_wilcoxon_p']:.3g})")
if not _ok:
    failures.append("a Youden cell no longer clears p<5e-8")
_ssy = _m5["sample_size_sensitivity_at_youden"]["by_n_samples"]
for _n in ["700", "1750", "3500", "10000"]:
    check(f"Youden F1 gap at N={_n}", _ssy[_n]["f1_gap_youden_mean"])
check("Youden sample-size shrinkage 700->10000",
      _m5["sample_size_sensitivity_at_youden"]["shrinkage_factor_700_to_10000"], "{:.1f}")

print("\n== P1-2: Delta_sel against its mechanical null (code/64) ==")
_sn = load("case_study_2_selection_null.json")
check("observed Delta_sel (replayed)", _sn["observed_delta_sel"])
check("layer-permutation null mean", _sn["layer_permutation_null"]["mean"])
check("null 95% CI low", _sn["layer_permutation_null"]["ci_95"][0])
check("null 95% CI high", _sn["layer_permutation_null"]["ci_95"][1])
check("null SD", _sn["layer_permutation_null"]["sd"], "{:.4f}")
check("A (winner's curse on the selection criterion)",
      _sn["decomposition"]["A_winners_curse_on_selection_criterion"], "{:.6f}")
check("B (transferred layer quality)",
      _sn["decomposition"]["B_transferred_layer_quality"])
check("B's negative-rep count", _sn["decomposition"]["B_n_negative_reps"], "{:d}")
check("B offsets this fraction of A",
      100 * _sn["decomposition"]["B_over_A_fraction_offset"], "{:.1f}")
checks += 1
_ok = (_sn["layer_permutation_null"]["observed_below_entire_ci"]
       and abs(_sn["decomposition"]["identity_check_A_minus_B_equals_delta"]) < 1e-12
       and _sn["decomposition"]["A_equals_null_mean_abs_diff"] < 1e-3)
print(f"  {'OK  ' if _ok else 'FAIL'}  Delta_sel = A - B exactly, A equals the simulated null "
      f"mean, and the observed value sits below the null's entire 95% interval")
if not _ok:
    failures.append("the Delta_sel null/decomposition identities no longer hold")

print("\n== P1-3: the K-law is monotone and concave, not identified as ln K (code/63, 68) ==")
_man = load("sweep_per_seed_manifest.json")
checks += 1
_ok = _man["reproduces_shipped_cell_means"]
print(f"  {'OK  ' if _ok else 'FAIL'}  the shipped per-seed artifact reproduces every Sweep A "
      f"and Sweep C cell mean to <1e-12 (max |d|={_man['max_abs_diff_vs_shipped']:.1e})")
if not _ok:
    failures.append("results/sweep_per_seed.npz no longer reproduces the shipped cell means")
_kl = load("k_law_functional_form.json")
check("mean per-cell Monte-Carlo SE",
      _kl["identification"]["mean_cell_monte_carlo_se"] * 1e4, "{:.2f}")
check("RMS residual of the a+b lnK fit",
      _kl["identification"]["rms_residual_of_paper_form"] * 1e4, "{:.2f}")
check("MC-noise-over-residual ratio",
      _kl["identification"]["mc_noise_over_residual_ratio"], "{:.1f}")
for _form, _lbl in [("saturating_exp", "saturating exponential"),
                    ("a_plus_b_over_K", "a+b/K"),
                    ("a_plus_b_lnlnK", "a+b lnlnK"),
                    ("a_plus_b_sqrtlnK", "a+b sqrt(lnK)"),
                    ("a_plus_b_lnK", "a+b lnK"),
                    ("power_law", "a K^b"),
                    ("linear_in_K", "a+bK")]:
    check(f"{_lbl} R^2", _kl["fits"][_form]["r_squared"], "{:.4f}")
    check(f"{_lbl} AICc", _kl["fits"][_form]["aicc"], "{:.2f}")
    check(f"{_lbl} LOO R^2", _kl["fits"][_form]["loo_r_squared"], "{:.3f}")
check("P(ln K is the best-fitting form)",
      _kl["parametric_bootstrap"]["p_lnK_best_on_aicc"], "{:.3f}")
check("P(saturating exponential best)",
      _kl["parametric_bootstrap"]["p_best_on_aicc"]["saturating_exp"], "{:.3f}")
check("P(a+b/K best)",
      _kl["parametric_bootstrap"]["p_best_on_aicc"]["a_plus_b_over_K"], "{:.3f}")
_ps = _kl["properties_the_data_support"]
check("ln K slope", _ps["lnK_slope"], "{:.5f}")
check("ln K slope CI low", _ps["lnK_slope_bootstrap_ci_95"][0], "{:.5f}")
check("ln K slope CI high", _ps["lnK_slope_bootstrap_ci_95"][1], "{:.5f}")
check("lnK-concavity worst violation magnitude",
      _ps["lnK_concavity_worst_violation_magnitude"] * 1e4, "{:.2f}")
checks += 1
_ok = _ps["monotone_nondecreasing_in_K"] and _ps["concave_in_K"] and not _ps["concave_in_lnK"]
print(f"  {'OK  ' if _ok else 'FAIL'}  the six fitted cells are monotone and concave in K, and "
      f"NOT strictly concave in ln K (which is why SS5.3 claims the former, not the latter)")
if not _ok:
    failures.append("the monotone/concave-in-K properties SS5.3 now claims no longer hold")
checks += 1
_nbeat = sum(1 for f in ["saturating_exp", "a_plus_b_over_K", "a_plus_b_lnlnK",
                         "a_plus_b_sqrtlnK"]
             if (_kl["fits"][f]["r_squared"] > _kl["fits"]["a_plus_b_lnK"]["r_squared"]
                 and _kl["fits"][f]["aicc"] < _kl["fits"]["a_plus_b_lnK"]["aicc"]
                 and _kl["fits"][f]["loo_r_squared"] > _kl["fits"]["a_plus_b_lnK"]["loo_r_squared"]))
_ok = _nbeat == 4
print(f"  {'OK  ' if _ok else 'FAIL'}  exactly 4 alternative forms beat a+b lnK simultaneously "
      f"on R^2, AICc and LOO (found {_nbeat})")
if not _ok:
    failures.append(f"the '4 alternatives beat ln K' claim no longer holds ({_nbeat} do)")

print("\n== P1-4: the variance-compression control for the operating-point axis (code/69) ==")
_vc = load("operating_point_variance_control.json")
for _a0 in ["0.7", "0.8", "0.9", "0.95", "0.985"]:
    check(f"probit gap at AUROC0={_a0}", _vc["cells"][_a0]["probit_gap_mean"])
    check(f"Cohen's d at AUROC0={_a0}", _vc["cells"][_a0]["cohens_d_paired"], "{:.3f}")
    check(f"rank-biserial at AUROC0={_a0}", _vc["cells"][_a0]["rank_biserial_r"], "{:.3f}")
check("probit decline ratio", _vc["declines"]["probit_gap"]["ratio"], "{:.1f}")
check("Cohen's d decline ratio", _vc["declines"]["cohens_d_paired"]["ratio"], "{:.1f}")
check("rank-biserial decline ratio", _vc["declines"]["rank_biserial_r"]["ratio"], "{:.1f}")
for _k, _lbl in [("probit_gap", "probit"), ("cohens_d_paired", "standardized"),
                 ("rank_biserial_r", "rank")]:
    check(f"surviving log-share of the raw decline ({_lbl})",
          _vc["surviving_log_share_of_raw_decline"][_k], "{:.2f}")
checks += 1
_ok = (_vc["verdict"] == "DECLINE_SURVIVES_IN_MAGNITUDE_BUT_NOT_IN_STRICT_MONOTONICITY"
       and _vc["compression_diagnostic"]["mean_falls_faster_than_sd"])
print(f"  {'OK  ' if _ok else 'FAIL'}  split verdict: the decline survives all three "
      f"stabilizers in magnitude but not in strict monotonicity, and the mean falls faster "
      f"than the SD (so pure compression is refuted)")
if not _ok:
    failures.append(f"the variance-compression control's verdict changed to {_vc['verdict']!r}")
checks += 1
_mean_over_sd = (_vc["compression_diagnostic"]["gap_mean_ratio_low_over_high"]
                 / _vc["compression_diagnostic"]["gap_sd_ratio_low_over_high"])
# Whitespace-insensitive: the 34-page compression pass rewrapped SS5.3 from an
# itemize into headed prose, so the literal line break this used to require is
# gone while the claim is unchanged.
_ok = abs(_mean_over_sd - 2.7) < 0.1 and "outruns the SD's by a factor of $2.7$" in " ".join(TEX.split())
print(f"  {'OK  ' if _ok else 'FAIL'}  the mean's decline outruns the SD's by 2.7x "
      f"(computed {_mean_over_sd:.2f}) and SS5.3 says so")
if not _ok:
    failures.append("the mean-vs-SD decline-rate comparison in SS5.3 no longer traces")

print("\n== P1-7: the adaptivity control's budget deficit, measured (code/67) ==")
_bd = load("adaptivity_control_budget_deficit.json")
for _cap in ["128", "384"]:
    _c = _bd["by_capacity"][_cap]
    check(f"cap {_cap} budget deficit", _c["budget_deficit_mean"])
    check(f"cap {_cap} budget-deficit CI low", _c["budget_deficit_bca_ci_95"][0])
    check(f"cap {_cap} budget-deficit CI high", _c["budget_deficit_bca_ci_95"][1])
    check(f"cap {_cap} budget deficit as a fraction of the fold-reuse increment",
          100 * _c["budget_deficit_over_fold_reuse_effect"], "{:.0f}")
    checks += 1
    _ok = _c["reproduces_code22_clean_matched_per_seed"]
    print(f"  {'OK  ' if _ok else 'FAIL'}  cap {_cap}: code/67's recomputed CLEAN_MATCHED arm "
          f"reproduces code/22's shipped per-seed values exactly")
    if not _ok:
        failures.append(f"code/67's CLEAN_MATCHED arm drifted from code/22 at capacity {_cap}")
checks += 1
_c384 = _bd["by_capacity"]["384"]
_ok = _c384["budget_deficit_mean"] > _c384["leaky_minus_clean_matched"]
print(f"  {'OK  ' if _ok else 'FAIL'}  at capacity 384 the budget deficit "
      f"({_c384['budget_deficit_mean']:+.4f}) exceeds the LEAKY-minus-CLEAN_MATCHED gap "
      f"({_c384['leaky_minus_clean_matched']:+.4f}), which is why SS4.3 now reads the "
      f"fold-reuse increments as upper bounds")
if not _ok:
    failures.append("SS4.3's upper-bound qualifier no longer follows from code/67")

print("\n== P1-9: the joint surface's exponent under a second estimator (code/65) ==")
_es = load("joint_surface_estimator_sensitivity.json")["summary"]
check("b under Gauss-Newton NLS, all 20 cells", _es["b_nls_all_cells"], "{:.3f}")
check("b under log-space OLS, all 20 cells", _es["b_log_ols_all_cells"], "{:.3f}")
check("c under Gauss-Newton NLS, all 20 cells", _es["c_nls_all_cells"], "{:.2f}")
check("c under log-space OLS, all 20 cells", _es["c_log_ols_all_cells"], "{:.2f}")
check("b's estimator shift", _es["b_estimator_shift_all_cells"], "{:.3f}")
check("b's column-deletion span within NLS", _es["b_column_deletion_span_within_nls"], "{:.3f}")
check("c's full range low", _es["c_range_over_both_estimators_and_all_deletions"][0], "{:.2f}")
check("c's full range high", _es["c_range_over_both_estimators_and_all_deletions"][1], "{:.2f}")
check("c's relative span over both estimators",
      100 * _es["c_relative_span_over_both_estimators"], "{:.0f}")
check("b's relative span over both estimators",
      100 * _es["b_relative_span_over_both_estimators"], "{:.0f}")
checks += 1
_ok = (_es["b_estimator_shift_exceeds_deletion_span"]
       and _es["b_log_ols_outside_nls_deletion_range"])
print(f"  {'OK  ' if _ok else 'FAIL'}  b moves further under a change of estimator than under "
      f"column deletion, and the log-OLS value falls outside the quoted deletion range")
if not _ok:
    failures.append("Appendix C.2's estimator-sensitivity disclosure no longer follows")

print("\n== P1-10: Case Study 4's permutation null in both tails (code/66) ==")
_ts = load("case_study_4_two_sided_null.json")
_pe = _ts["primary_estimator_bootstrap_max_bias"]
check("cells above their own null (primary estimator)",
      _pe["n_significantly_above_own_null_p05"], "{:d}")
check("cells below their own null (primary estimator)",
      _pe["n_significantly_below_own_null_p05"], "{:d}")
check("non-degenerate cells below their own null",
      _pe["n_significantly_below_own_null_p05_nondegenerate"], "{:d}")
check("retired rotation diagnostic: cells below their null",
      _ts["retired_rotation_diagnostic"]["n_significantly_below_own_null_p05"], "{:d}")
check("retired rotation diagnostic: non-degenerate cells below their null",
      _ts["retired_rotation_diagnostic"]["n_significantly_below_own_null_p05_nondegenerate"],
      "{:d}")
check("exact-enumeration all-24 bootstrap max-bias",
      _ts["mean_bootstrap_max_bias_exact_all_24"], "{:.6f}")
checks += 1
_ok = (abs(_ts["mean_bootstrap_max_bias_exact_all_24"]
           - _ts["mean_bootstrap_max_bias_superseded_mc_all_24"]) < 5e-5
       and _pe["n_significantly_above_own_null_p05"] == 0)
print(f"  {'OK  ' if _ok else 'FAIL'}  the exact 27-resample enumeration agrees with the "
      f"superseded 2,000-draw estimate to <5e-5 IN THE POOLED MEAN (so SS4.4's +0.0021 "
      f"does not depend on the bootstrap's own Monte-Carlo error), and 0 of 24 cells sit "
      f"above their null")
if not _ok:
    failures.append("Case Study 4's exact-vs-MC agreement or its upper-tail count changed")



# ══════════════════════════════════════════════════════════════════════════
# ROUND 3 -- the fourth independent review. Every numeric claim introduced by
# code/70 (exact enumeration + Theorem 2), code/71 (calibration against a known
# ground truth), code/72 (the magnitude triangle), code/73 (exchangeable
# candidates) and code/74 (Mechanism 5 size-matching) is guarded here.
# ══════════════════════════════════════════════════════════════════════════

print("\n== P0-1/P0-3: Case Study 4 by EXACT ENUMERATION (code/70) ==")
_ee = load("case_study_4_exact_enumeration.json")
_g = _ee["groups"]["implemented"]
check("all-24 bootstrap max-bias, exact", _g["all_24"]["mean"], "{:+.4f}")
check("all-24 BCa CI low, exact", _g["all_24"]["bca_ci_95"][0], "{:+.4f}")
check("all-24 BCa CI high, exact", _g["all_24"]["bca_ci_95"][1], "{:+.4f}")
check("ceiling-saturated under the reported estimator", _g["ceiling_saturated"]["mean"], "{:+.4f}")
check("non-saturated under the reported estimator", _g["non_saturated"]["mean"], "{:+.4f}")
check("rotation-degenerate cells under the reported estimator",
      _g["rotation_degenerate"]["mean"], "{:+.4f}")
check("rotation-non-degenerate cells under the reported estimator",
      _g["rotation_non_degenerate"]["mean"], "{:+.4f}")
check("SD across the 24 cells under the reported estimator",
      _g["sd_across_24_cells"], "{:.4f}")
check("MDE under the reported estimator", _g["mde_alpha05_power80_n24"], "{:+.4f}")
check("dataset-clustered CI low", _g["dataset_clustered_bootstrap_ci_95"][0], "{:+.4f}")
check("dataset-clustered CI high", _g["dataset_clustered_bootstrap_ci_95"][1], "{:+.4f}")
check("largest single cell under the reported estimator", _g["all_24"]["max"], "{:+.4f}")
check("standard bootstrap variant, all 24", _ee["groups"]["standard"]["all_24"]["mean"], "{:+.4f}")
_ma = _ee["monte_carlo_artifact_audit"]
checks += 1
_ok = (_ma["n_cells_negative_under_exact_enumeration"] == 0
       and _ma["n_cells_negative_under_shipped_monte_carlo"] == 2
       and _ma["n_cells_bootstrap_argmax_stable"] == 2
       and set(_ma["cells_negative_under_shipped_monte_carlo"])
           == set(_ma["cells_bootstrap_argmax_stable"]))
print(f"  {'OK  ' if _ok else 'FAIL'}  the 2 cells that read NEGATIVE under the superseded "
      f"Monte-Carlo bootstrap are EXACTLY the 2 whose grand-mean argmax is bootstrap-stable, "
      f"and 0 of 24 are negative under exact enumeration (Theorem 2's equality case)")
if not _ok:
    failures.append("Case Study 4's Monte-Carlo-artifact audit changed")
checks += 1
_tc = _ee["theorem2_numerical_check"]
_ok = all(v["min_implemented"] > -1e-9 and v["min_standard"] > -1e-9 for v in _tc.values())
print(f"  {'OK  ' if _ok else 'FAIL'}  Theorem 2 verified numerically: no random matrix "
      f"produces a negative value beyond float residue "
      f"(min {min(v['min_implemented'] for v in _tc.values()):.2e})")
if not _ok:
    failures.append("Theorem 2's numerical verification produced a genuinely negative value")
# The ceiling multiplier must now be quoted as an estimator-labelled RANGE.
checks += 1
_r_impl = _ee["groups"]["implemented"]["ceiling_ratio_nonsat_over_sat"]
_r_std = _ee["groups"]["standard"]["ceiling_ratio_nonsat_over_sat"]
_r_rot = _ee["groups"]["rotation_delta_wc"]["ceiling_ratio_nonsat_over_sat"]
_ok = (f"${_r_impl:.1f}\\times$" in TEX and f"${_r_std:.1f}\\times$" in TEX
       and f"${_r_rot:.1f}\\times$" in TEX)
print(f"  {'OK  ' if _ok else 'FAIL'}  the ceiling-composition multiplier is quoted as the "
      f"estimator-labelled range {_r_impl:.1f}/{_r_rot:.1f}/{_r_std:.1f}x rather than as one "
      f"unlabelled figure")
if not _ok:
    failures.append("the ceiling-composition multiplier range is not fully quoted in main.tex")

print("\n== P0-2: calibration against a KNOWN ground truth (code/71) ==")
_cal = load("cs4_estimator_calibration.json")
_rr = _cal["real_data_matching"]["calibration_ratio_ranges"]
for _name, _lbl in (("implemented", "the reported estimator"),
                    ("standard", "the textbook bootstrap variant"),
                    ("rotation_delta_wc", "the retired rotation diagnostic")):
    _lo, _hi = _rr[_name]["over_rho_grid"]
    check(f"calibration ratio low, {_lbl}", _lo, "{:.2f}")
    check(f"calibration ratio high, {_lbl}", _hi, "{:.2f}")
    check(f"calibration-corrected true bias via {_lbl}",
          _cal["real_data_matching"]["calibrated"][_name]["calibration_corrected_true_bias"],
          "{:+.4f}")
checks += 1
_cov = _cal["coverage"]
_ok = (all(c["coverage"]["implemented"] == 0.0 for c in _cov)
       and all(c["coverage"]["standard"] == 0.0 for c in _cov))
print(f"  {'OK  ' if _ok else 'FAIL'}  the 24-cell BCa interval covers the TRUE bias in 0% of "
      f"simulated studies for both bootstrap estimators, which is why SS4.4 no longer offers "
      f"it as a confidence interval for the severity")
if not _ok:
    failures.append("Case Study 4's simulated CI coverage is no longer 0% for the bootstrap estimators")
checks += 1
_z = _cal["controlled_sweep"][-1]
_ok = (abs(_z["true_bias"]) < 1e-5
       and all(_z[k]["frac_negative"] == 0.0
               for k in ("implemented", "standard", "rotation_delta_wc")))
print(f"  {'OK  ' if _ok else 'FAIL'}  at the known-ZERO-bias condition (with seed noise still "
      f"present) all three estimators return ~0 and NONE is ever negative, as Theorems 1 and 2 "
      f"require")
if not _ok:
    failures.append("the known-zero-bias null no longer behaves as the theorems predict")
checks += 1
_ok = abs(_ee["ratio_implemented_over_standard_all24"] - 2.35) < 0.02
print(f"  {'OK  ' if _ok else 'FAIL'}  the implemented estimator is "
      f"{_ee['ratio_implemented_over_standard_all24']:.2f}x the textbook variant on these 24 "
      f"cells (they are different estimands, not two computations of one)")
if not _ok:
    failures.append("the implemented/standard bootstrap ratio moved")

print("\n== P0-5: the three severity axes on one scale (code/72) ==")
_mt = load("magnitude_triangle.json")
check("operating-point axis, raw span low", _mt["operating_point_axis"]["min"], "{:.1f}")
check("operating-point axis, raw span high", _mt["operating_point_axis"]["max"], "{:.1f}")
check("candidate-count axis, raw span low", _mt["candidate_count_axis"]["min"], "{:.1f}")
check("candidate-count axis, raw span high", _mt["candidate_count_axis"]["max"], "{:.1f}")
check("candidate-count axis excluding K=15, low",
      _mt["candidate_count_axis"]["excluding_K15_column"]["min"], "{:.1f}")
check("candidate-count axis excluding K=15, high",
      _mt["candidate_count_axis"]["excluding_K15_column"]["max"], "{:.1f}")
_dsc = _mt["denominator_sound_comparison"]
check("denominator-sound operating-point low", _dsc["operating_point"][0], "{:.1f}")
check("denominator-sound operating-point high", _dsc["operating_point"][1], "{:.1f}")
check("denominator-sound candidate-count low", _dsc["candidate_count"][0], "{:.1f}")
check("denominator-sound candidate-count high", _dsc["candidate_count"][1], "{:.1f}")
checks += 1
_ok = (_dsc["mechanism_and_harness"][1] > _dsc["operating_point"][1]
       and _dsc["mechanism_and_harness"][0] <= _dsc["operating_point"][1]
       and _dsc["mechanism_and_harness"][0] > _dsc["candidate_count"][1])
print(f"  {'OK  ' if _ok else 'FAIL'}  restricted to denominators whose intervals exclude zero, "
      f"the mechanism-and-harness axis ({_dsc['mechanism_and_harness'][0]:.1f}-"
      f"{_dsc['mechanism_and_harness'][1]:.1f}x) is AT LEAST COMPARABLE TO, AND PLAUSIBLY "
      f"LARGER THAN, the operating-point axis ({_dsc['operating_point'][0]:.1f}-"
      f"{_dsc['operating_point'][1]:.1f}x) -- its own lower end sits at or below the "
      f"operating-point axis's upper end, so the two ranges overlap and this is NOT a claim "
      f"that mechanism-and-harness is strictly the largest of the three -- which is what "
      f"retired the title's 'rather than mechanism'")
if not _ok:
    failures.append("the denominator-sound axis comparability/plausible-dominance relationship changed; SS5.2's conclusion needs revisiting")
# The withdrawn comparative claim must not survive anywhere.
check_absent_everywhere(
    "the title/abstract claim that severity is governed by two axes RATHER THAN mechanism",
    "Rather Than Mechanism", "rather than mechanism but by",
    "governed not by \\emph{which} mechanism is responsible but by two continuous quantities")

print("\n== P1-1: does the K-law transfer to EXCHANGEABLE candidates? (code/73) ==")
_ex = load("exchangeable_candidate_sweep.json")
for _k, _c in _ex["cells"].items():
    check(f"exchangeable-candidate gap at K={_k}", _c["gap_mean"], "{:+.4f}")
_cmp = _ex["comparison_to_code47_sweep_A"]
check("exchangeable-candidate lnK slope", _cmp["exchangeable_lnK_slope_b"], "{:+.5f}")
checks += 1
_ok = (_ex["verdict"] == "DOES_NOT_TRANSFER"
       and not _cmp["b_ci_excludes_zero"]
       and _cmp["ci_overlaps_code47"])
print(f"  {'OK  ' if _ok else 'FAIL'}  the exchangeable-candidate slope's interval contains zero "
      f"AND overlaps code/47's, so SS5.3 must report this as a failure to reproduce at low "
      f"power rather than as a demonstrated contradiction")
if not _ok:
    failures.append("the exchangeable-candidate verdict or its power caveat changed")
checks += 1
_hi_k = _ex["cells"][max(_ex["cells"], key=lambda k: int(k))]
_ok = _hi_k["degeneracy"]["same_winning_init_seed_frac"] < 0.05
print(f"  {'OK  ' if _ok else 'FAIL'}  at the largest K the two honest selection rules agree on "
      f"the same candidate in only {_hi_k['degeneracy']['same_winning_init_seed_frac']*100:.1f}% "
      f"of folds, so the null there is not a degeneracy artifact")
if not _ok:
    failures.append("the exchangeable sweep's top-K degeneracy fraction is no longer negligible")
# The power limitation must be quoted, because the MDE EXCEEDS the effect the
# design is testing for -- a null this underpowered must not read as a
# contradiction of Sweep A.
_pw = _ex["power"]
check("exchangeable sweep MDE at 80% power",
      _pw["min_detectable_gap_80pct_power_two_sided_05"], "{:+.4f}")
checks += 1
_ok = (_pw["min_detectable_gap_80pct_power_two_sided_05"]
       > _pw["code47_gap_at_K45_for_scale"])
print(f"  {'OK  ' if _ok else 'FAIL'}  the exchangeable design's MDE "
      f"({_pw['min_detectable_gap_80pct_power_two_sided_05']:+.4f}) EXCEEDS code/47's K=45 gap "
      f"({_pw['code47_gap_at_K45_for_scale']:+.4f}), so SS5.3 must report the null as "
      f"underpowered rather than as a contradiction")
if not _ok:
    failures.append("the exchangeable sweep's power disclosure no longer applies")
# The selection-stage decomposition: EVT holds where it applies.
_ssc = _ex["evt_test_A11_could_not_run"]["selection_stage_check"]["per_cell"]
checks += 1
_ratios = [c["observed_over_exact_gaussian"] for c in _ssc.values()]
_ok = all(abs(r - 1.0) < 0.05 for r in _ratios)
print(f"  {'OK  ' if _ok else 'FAIL'}  at the SELECTION stage the observed (max-mean)/sigma "
      f"matches the exact Gaussian E[max of K] to within "
      f"{max(abs(r-1) for r in _ratios)*100:.1f}% at every K -- the winner's curse is present "
      f"and correctly scaled there; what fails is its propagation downstream")
if not _ok:
    failures.append("the selection-stage EVT agreement changed")
for _c in _ssc.values():
    check(f"selection-stage EVT ratio", _c["observed_over_exact_gaussian"], "{:.3f}")

print("\n== P1-2: Mechanism 5's undisclosed selection-set asymmetry (code/74) ==")
_m5 = load("mechanism5_size_matched.json")
_c985 = _m5["cells"]["0.985"]
check("size-matched F1 gap at 0.985", _c985["matched"]["f1_gap_youden_mean"], "{:+.4f}")
check("size-matched accuracy gap at 0.985", _c985["matched"]["acc_gap_mean"], "{:+.4f}")
check("matched minus shipped, F1",
      _c985["differences"]["f1_youden"]["matched_minus_unmatched_mean"], "{:+.4f}")
check("matched minus shipped, accuracy",
      _c985["differences"]["acc"]["matched_minus_unmatched_mean"], "{:+.4f}")
check("size effect isolated from the training-budget cost, F1",
      _c985["differences"]["f1_youden"]["matched_minus_budget_control_mean"], "{:+.4f}")
check("selection-set size ratio", _m5["design"]["selection_set_size_ratio_shipped"], "{:.2f}")
_an = _m5["accuracy_nonnegativity"]
checks += 1
_ok = (_an["n_balanced_negative"] == 0 and _an["n_negative"] == 5
       and _an["f1_youden_n_negative_for_contrast"] > 100)
print(f"  {'OK  ' if _ok else 'FAIL'}  the accuracy gap is negative in {_an['n_negative']}/"
      f"{_an['n_replicates']} replicates and in {_an['n_balanced_negative']}/"
      f"{_an['n_balanced_replicates']} exactly-balanced ones, against "
      f"{_an['f1_youden_n_negative_for_contrast']} negatives for F1 -- so SS4.5's withdrawn "
      f"non-negativity caveat still applies to ACCURACY")
if not _ok:
    failures.append("Mechanism 5's accuracy non-negativity pattern changed")
for _v in (_an["n_negative"], _an["n_replicates"], _an["n_balanced_replicates"],
           _an["f1_youden_n_negative_for_contrast"]):
    check(f"accuracy-non-negativity count {_v}", _v, "{:d}")

print("\n== P1-4: Mechanism 3's second (selection-run) budget asymmetry (code/75) ==")
_bc = load("mechanism3_selection_budget_controls.json")
_sm = _bc["summary"]
for _cap in sorted(_sm["es_0.25_fold_matched_gaps_by_capacity"], key=int):
    check(f"fold-matched (ES=0.25) gap at capacity {_cap}",
          _sm["es_0.25_fold_matched_gaps_by_capacity"][_cap], "{:+.4f}")
    check(f"out-of-fold gap at capacity {_cap}",
          _sm["out_of_fold_gaps_by_capacity"][_cap], "{:+.4f}")
    check(f"selection-run asymmetry at capacity {_cap}",
          _sm["asymmetry_2_size_by_capacity"][_cap], "{:+.4f}")
check("out-of-fold mean gap", _sm["out_of_fold_mean_gap"], "{:+.4f}")
check("primary mean gap", _sm["primary_mean_gap"], "{:+.4f}")
checks += 1
_ok = (_sm["n_capacities_with_bca_ci_excluding_zero_from_below"] == 0
       and _sm["out_of_fold_mean_gap"] < _sm["primary_mean_gap"])
print(f"  {'OK  ' if _ok else 'FAIL'}  with the selection-run asymmetry removed, "
      f"{_sm['n_capacities_with_bca_ci_excluding_zero_from_below']} of 4 BCa intervals exclude "
      f"zero and the mean gap falls {_sm['primary_mean_gap']:+.4f} -> "
      f"{_sm['out_of_fold_mean_gap']:+.4f}; SS4.3 must report this attenuation next to the band")
if not _ok:
    failures.append("Mechanism 3's out-of-fold attenuation changed")
checks += 1
_pa = _bc["part_A_es_hold_fraction"]
_ok = (_pa["geometry"]["exact_fold_matched_es_fraction"] == 0.25
       and all(v["reproduces_code02d_per_seed"] for v in _pa["by_capacity"].values()))
print(f"  {'OK  ' if _ok else 'FAIL'}  the fold-matched carve-out fraction is 1/(K-1)=0.25 (not "
      f"1/K=0.20), and every recomputed ES=0.15 arm reproduces code/02d's shipped per-seed "
      f"values bit-for-bit")
if not _ok:
    failures.append("code/75's geometry derivation or its bit-exactness assertion changed")
checks += 1
_oofp = [v["gap_cm_oof_minus_placebo"]["gap_mean"] if "gap_cm_oof_minus_placebo" in v
         else v["gap_cm_oof_minus_placebo_r"]["gap_mean"]
         for v in _bc["part_B_out_of_fold_carveout"]["by_capacity"].values()]
_ok = min(_oofp) > 0.01
print(f"  {'OK  ' if _ok else 'FAIL'}  honest checkpoint selection still beats the placebo by "
      f"{min(_oofp):+.4f} to {max(_oofp):+.4f} at every capacity, so what attenuates is the "
      f"fold-reuse residual and not the mechanism's whole signal")
if not _ok:
    failures.append("the out-of-fold control's placebo-relative benefit changed")

print("\n== Additional round-3 findings ==")
# Finding 8/9: the sanity-ratio band is a restriction, and code/27's cells violate it.
_an27 = load("anisotropic_covariance_capacity_sweep.json")["capacities"]
_ratios = {k: (v["leaky_minus_clean_matched"]["mean"]
               / v["clean_matched_minus_placebo"]["mean"]) for k, v in _an27.items()}
checks += 1
_outside = {k: r for k, r in _ratios.items() if not (0.06 <= r <= 0.85)}
_ok = (len(_outside) == 3 and any(r < 0 for r in _ratios.values())
       and "restriction, not a census" in TEX)
print(f"  {'OK  ' if _ok else 'FAIL'}  {len(_outside)} of code/27's 4 anisotropic cells fall "
      f"OUTSIDE the 0.06-0.85 sanity band ({ {k: round(r,3) for k,r in _outside.items()} }), one "
      f"with a negative denominator, and main.tex scopes the 8-cell comparator set as a "
      f"restriction rather than a census")
if not _ok:
    failures.append("the sanity-ratio scope disclosure is missing or the anisotropic ratios moved")
check("anisotropic capacity-384 sanity ratio (negative denominator)", _ratios["384"], "{:+.2f}")
check("anisotropic capacity-384 negative denominator",
      _an27["384"]["clean_matched_minus_placebo"]["mean"], "{:+.4f}")
# Finding 10: the 2x2 ablation's currently_reported_number must track code/49.
checks += 1
_2x2 = load("fidelity_extension_2x2_ablation.json")
_fx = load("mechanism3_fidelity_extension.json")
_ok = abs(_2x2["currently_reported_number"]
          - _fx["leaky_plus_lrsched_minus_clean_matched_budget_matched"]["gap_mean"]) < 1e-12
print(f"  {'OK  ' if _ok else 'FAIL'}  fidelity_extension_2x2_ablation.json's "
      f"`currently_reported_number` ({_2x2['currently_reported_number']:+.4f}) tracks code/49's "
      f"BUDGET-MATCHED arm rather than this grid's own non-budget-matched cell")
if not _ok:
    failures.append("the 2x2 ablation's currently_reported_number is stale again")
# Finding 5: the Fieller check must use one quantile family throughout.
checks += 1
_fc = load("operating_point_ratio_fieller_check.json")
def _find_g(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "g_from_bca_legacy_mixed_quantiles":
                return o
            r = _find_g(v)
            if r:
                return r
    return None
_gg = _find_g(_fc)
_ok = _gg is not None and _gg["g_from_bca_legacy_mixed_quantiles"] > 1
print(f"  {'OK  ' if _ok else 'FAIL'}  the Fieller check uses the t quantile on both sides; the "
      f"legacy mixed-quantile value is retained and both exceed 1")
if not _ok:
    failures.append("the Fieller quantile-consistency fields are missing")
# Finding 2: the sign statistic must not be called a rank-biserial correlation.
check_absent_everywhere(
    "the sign statistic mislabelled as a matched-pairs rank-biserial correlation",
    "matched-pairs rank-biserial correlation",
    "matched-pairs \\emph{rank-biserial correlation}",
    # code/69 is the script that makes the correction, so its docstring must be
    # able to name the retired label in order to retire it -- the same exemption
    # this script and code/91 already carry.
    allow_files=("code/69_operating_point_variance_control.py",))

print("\n== P1 round: superseded values must not survive in ANY shipped artifact ==")
# The P1 round changed three numbers that were quoted OUTSIDE main.tex -- in
# README.md and in draft/leakage_checklist.md, which main.tex explicitly directs
# readers to. That is the same class of failure the retracted "52x" produced, so
# each gets a guard here rather than a one-time hand-fix.
check_absent_everywhere(
    "Mechanism 5's F1 range measured at the discarded F1-argmax threshold",
    "+0.021 to +0.034", "$+0.021$ to $+0.034$", "+0.021--+0.034",
    "$+0.021$--$+0.034$", "+0.021\u2013+0.034", "$+0.021$\u2013$+0.034$")
check_absent_everywhere(
    "the a+b lnK law asserted as an established functional form",
    "gap grows as `a + b ln K`", "well fit by $a+b\\ln K$, $R^2=0.939$",
    "the gap grows as $a+b\\ln K$ ($R^2=0.939$")
check_absent_everywhere(
    "the pre-P0 severity band upper endpoint (from the retired Delta_wc diagnostic)",
    "roughly 0.000 to 0.034 AUROC", "roughly $0.000$ and $0.034$ AUROC",
    "between roughly $0.000$ and $0.034$,")
# Forward direction: the three shipped text artifacts must carry the CURRENT
# Mechanism 5 range and sample-size table, not just fail to carry the old one.
for _fname, _needles in [
    ("draft/leakage_checklist.md",
     ["+0.0225", "+0.0520", "6.3", "+0.022", "+0.052"]),
    ("README.md", ["0.000 to 0.026", "monotonically and\nconcavely with K"]),
]:
    checks += 1
    _txt = SHIPPED_TEXT.get(_fname, "")
    _missing = [n for n in _needles if n not in _txt]
    _ok = not _missing
    print(f"  {'OK  ' if _ok else 'FAIL'}  {_fname} carries this round's corrected values")
    if not _ok:
        failures.append(f"{_fname} is missing corrected values {_missing}")

print("\n== Reverse guard: every AUROC-shaped literal in main.tex traces to a JSON ==")
# WHY THIS EXISTS. Every check above runs in ONE direction: it takes a number
# from a result JSON and asserts it appears in main.tex. That cannot catch a
# number that is in main.tex and in NO json -- e.g. a stale value, or the same
# cell rounded two different ways in two paragraphs. An adversarial review
# found exactly that: the (K=45, AUROC_0=0.80) cell (0.003649) was rendered
# "+0.0037" in one place and "+0.0036"/"+0.00365" elsewhere. This guard closes
# the loop for the class of literal that bug lived in: 4-and-5-decimal
# AUROC/gap-shaped numbers.
import glob as _glob

_json_floats = set()


def _collect(o):
    if isinstance(o, dict):
        for v in o.values():
            _collect(v)
    elif isinstance(o, list):
        for v in o:
            _collect(v)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        _json_floats.add(abs(float(o)))


for _f in _glob.glob(str(R / "*.json")):
    _collect(json.load(open(_f)))
_reprs = {f"{v:.{p}f}" for v in _json_floats for p in (4, 5)}

# Numbers that legitimately do NOT come from any shipped JSON. Each is here for
# a stated reason, not to silence the check.
ALLOWED_NON_JSON_LITERALS = {
    # Case Study 1 (HaRP): reported from prior work, explicitly NOT
    # re-verifiable from this artifact, and excluded from every comparison (SS4.1).
    "0.1906", "0.9620", "0.7583",
    # Numbers this paper quotes in order to RETRACT or supersede them. The
    # runs behind them are superseded and their JSONs no longer ship.
    "0.0498",   # pre-fidelity-port control-vs-placebo gap (Appendix A)
    "0.00064",  # Wilcoxon p under the superseded label-conditional calibration
    "0.9176",   # operating point of the superseded fidelity-extension run
    "0.00198", "0.00154",  # per-seed MDEs under the superseded label-conditional
                           # calibration, quoted in Appendix A's power check
    # Analytic, not measured: the Bayes-optimal AUROC the calibration bug
    # actually realized for a cell labelled 0.985 (Appendix A, issue 11).
    "0.99994",
}

# Quantiles of Case Study 2's per-rep Delta_sel are DERIVED from a shipped array
# rather than stored as scalars, so the substring trace above cannot see them.
# Recomputing them here is a stronger guarantee than allowlisting: if the array
# changes, this fails rather than silently passing.
_ssc = np.array(load("case_study_2_layer_decomposition.json")
                ["randomized_stratified_splits"]["selection_specific_component_per_rep"])
_q = np.percentile(_ssc, [0, 25, 50, 75, 100])
for _v in _q:
    ALLOWED_NON_JSON_LITERALS.add(f"{abs(_v):.4f}")
checks += 1
_ok = (all(f"{abs(v):.4f}" in TEX for v in _q)
       and f"{int((_ssc > 0).sum())} of {len(_ssc)}" in TEX.replace("$", ""))
print(f"  {'OK  ' if _ok else 'FAIL'}  Case Study 2's per-rep Delta_sel quantiles "
      f"({', '.join(f'{v:+.4f}' for v in _q)}) and its "
      f"{int((_ssc > 0).sum())}/{len(_ssc)} positive-rep count are in main.tex")
if not _ok:
    failures.append("Appendix B.2's per-rep Delta_sel disaggregation does not match the "
                    "shipped per-rep array")

_lits = set(re.findall(r"(?<![\d.])[+-]?0\.\d{4,5}(?![\d])", TEX))
_untraced = sorted(L for L in _lits
                   if L.lstrip("+-") not in _reprs
                   and L.lstrip("+-") not in ALLOWED_NON_JSON_LITERALS)
checks += 1
_ok = not _untraced
print(f"  {'OK  ' if _ok else 'FAIL'}  {len(_lits)} AUROC-shaped literals in main.tex; "
      f"{len(_untraced)} trace to no shipped JSON and are not allowlisted")
for L in _untraced:
    print(f"        untraceable: {L}")
if not _ok:
    failures.append(f"main.tex contains AUROC-shaped literals with no JSON source: {_untraced}")

# ── WIDENED REVERSE GUARD (2-3 decimals, and ratio multipliers) ─────────────
# The guard above covers only 4-and-5-decimal literals. That is exactly the
# wrong coverage for the class of number this round of corrections was about:
# every R^2 value is 3 decimals and every severity multiplier is written "38x"
# or "$38\times$", so neither was checked. A retracted "52x" therefore survived
# in four shipped files and two docstrings while this script reported a clean
# pass. Both shapes are now scanned, in every shipped text artifact.
_reprs_23 = {f"{v:.{q}f}" for v in _json_floats for q in (2, 3)}
_reprs_int = {f"{v:.0f}" for v in _json_floats}
_reprs_1 = {f"{v:.1f}" for v in _json_floats}

# 2-3 decimal literals are far more likely to be legitimately non-JSON (page
# fractions, p-value thresholds, grid coordinates, prose quantities), so this
# arm is a REPORT rather than a hard failure -- except for R^2 values, which are
# always fitted quantities and must trace.
_r2_lits = set(re.findall(r"R\^\{?2\}?\s*=\s*\$?([01]\.\d{2,3})", TEX))
_r2_untraced = sorted(L for L in _r2_lits if L not in _reprs_23 and L not in _reprs)
checks += 1
_ok = not _r2_untraced
print(f"  {'OK  ' if _ok else 'FAIL'}  {len(_r2_lits)} R^2 literals in main.tex; "
      f"{len(_r2_untraced)} trace to no shipped JSON"
      + ("" if _ok else f"  -> {_r2_untraced}"))
if not _ok:
    failures.append(f"main.tex quotes R^2 values with no JSON source: {_r2_untraced}")

# Ratio multipliers, in every rendering, across every shipped text file. Any
# multiplier the paper asserts must equal some shipped number to 0-1 decimals.
_MULT_RE = re.compile(r"(?<![\d.])(\d{1,4}(?:\.\d)?)\s*(?:\\times|x)\b")
# Multipliers that are legitimately not a shipped JSON scalar. Stated, not silent.
ALLOWED_MULTIPLIERS = {
    "2",     # "1.5x the held-out seed's deviation", "2x" in prose/algebra
    "1.5",
    "3",     # "3 seeds", "3x" in algebraic asides
    "23",    # the +0.0011-to-+0.0255 ratio-scale span, computed in prose
    "7.2",   # SD/mean of the denominator cell (2 s.f. of a shipped value)
    "2.8",   # retired: ratio of Delta_wc to bootstrap max-bias
    "1000",  # "1000x" style prose, and max_iter=1000
    "10",
    "100",
    # Ratios COMPUTED IN PROSE from two shipped values. Each is verified below
    # against its sources rather than merely allowlisted, so a change in either
    # source fails the check instead of silently passing.
    "3.7",   # CS4 non-saturated-non-degenerate / ceiling-saturated
    "3.3",   # Sweep C AUROC_0=0.80 gap / 0.90 gap
    "2.6",   # Sweep C AUROC_0=0.70 gap / 0.80 gap
    "6.5",   # feature-dimension ratio 414/64 (not an AUROC quantity)
    "4.3",   # joint-grid K-span at AUROC_0=0.95 (column max/min, Table in SS5.5)
    "4.2", "7.1", "1.6", "1.8",  # the other four K-spans in that same column
    "2.3", "5.8", "3.9",  # OOF-averaging attenuation range/mean (SS4.3)
}

# Verify the derived multipliers rather than trusting the allowlist above.
for _lbl, _num, _den, _want in [
    ("CS4 non-saturated-non-degenerate / ceiling-saturated",
     load("case_study_4_winners_curse.json")["summary_non_saturated_non_degenerate"]["mean"],
     load("case_study_4_winners_curse.json")["summary_ceiling_saturated"]["mean"], 3.7),
    ("Sweep C 0.80 gap / 0.90 gap",
     load("selection_multiplicity_sweep.json")["sweep_C_operating_point"]["0.8"]["gap_mean"],
     load("selection_multiplicity_sweep.json")["sweep_C_operating_point"]["0.9"]["gap_mean"], 3.3),
    ("Sweep C 0.70 gap / 0.80 gap",
     load("selection_multiplicity_sweep.json")["sweep_C_operating_point"]["0.7"]["gap_mean"],
     load("selection_multiplicity_sweep.json")["sweep_C_operating_point"]["0.8"]["gap_mean"], 2.6),
]:
    checks += 1
    _got = _num / _den
    # Tolerance 0.1: these appear in prose as "roughly Nx" / "a Nx difference",
    # so a value rounding to an adjacent tenth is a wording choice, not a stale
    # number. What must hold is that the multiplier still follows from its two
    # shipped sources at all, and still appears in main.tex.
    _ok = abs(_got - _want) < 0.1 and f"${_want}\\times$" in TEX
    print(f"  {'OK  ' if _ok else 'FAIL'}  in-prose multiplier {_want}x traces to its sources "
          f"({_lbl}: {_got:.3f})")
    if not _ok:
        failures.append(f"in-prose multiplier {_want}x no longer follows from {_lbl} "
                        f"(computed {_got:.3f}) or is missing from main.tex")
_mult_hits = {}
for _fname, _text in SHIPPED_TEXT.items():
    if not (_fname.endswith(".tex") or _fname.endswith(".md")):
        continue
    for _m in _MULT_RE.finditer(_text):
        _v = _m.group(1)
        if _v in ALLOWED_MULTIPLIERS:
            continue
        if _v in _reprs_int or _v in _reprs_1 or _v in _reprs_23 or _v in _reprs:
            continue
        # allow a multiplier quoted inside a retraction
        _i = _m.start()
        if any(a <= _i <= b for a, b in _QUOTED_ALL.get(_fname, [])):
            continue
        _mult_hits.setdefault(_v, []).append(f"{_fname}:{_text[:_i].count(chr(10)) + 1}")
checks += 1
_ok = not _mult_hits
print(f"  {'OK  ' if _ok else 'FAIL'}  ratio multipliers across all shipped text trace to a "
      f"shipped JSON value ({len(_mult_hits)} untraceable)")
for _v, _where in sorted(_mult_hits.items()):
    print(f"        untraceable multiplier {_v}x at {_where[:4]}")
if not _ok:
    failures.append(f"untraceable ratio multipliers in shipped text: "
                    f"{ {k: v[:3] for k, v in _mult_hits.items()} }")

# ── The markdown mirror must be a mechanical function of main.tex ───────────
# draft/paper_draft.md is generated by code/52. If it is hand-edited or left
# unregenerated it drifts, which is how three stale "52x" occurrences survived.
checks += 1
import subprocess as _sp
_r = _sp.run([sys.executable, str(ROOT / "code" / "52_sync_paper_draft_md.py"), "--check"],
             capture_output=True, text=True)
_ok = _r.returncode == 0
print(f"  {'OK  ' if _ok else 'FAIL'}  draft/paper_draft.md is in sync with main.tex "
      f"(code/52 --check)")
if not _ok:
    failures.append("draft/paper_draft.md has drifted from main.tex; re-run "
                    "code/52_sync_paper_draft_md.py")


# ── Round 4: the corrected Mechanism 3 controls, and the checklist generator ─
print("\n== Round 4: corrected selection controls (F1, F4-F10) ==")

_fac = load("mechanism3_factorial_selection_controls.json")
_fc = _fac["factorial_cells"]["fold_matched__oof__matched"]["gap_leaky_minus_arm"]
_sh = _fac["factorial_cells"]["small__infold__free"]["gap_leaky_minus_arm"]
check("factorial shipped-construction cell (capacity 128)", _sh["gap_mean"])
check("factorial fully-corrected cell", _fc["gap_mean"])
check("factorial fully-corrected CI low", _fc["gap_bca_ci_95"][0])
check("factorial fully-corrected CI high", _fc["gap_bca_ci_95"][1])
_me = _fac["decomposition"]["main_effects_average_over_other_factors"]
check("factorial selection-set-size main effect",
      _me["selection_set_size_small_minus_fold_matched"]["value"])
check("factorial selection-run-budget main effect",
      _me["selection_run_budget_infold_minus_oof"]["value"])
check("factorial training-depth main effect",
      _me["training_depth_free_minus_matched"]["value"])
checks += 1
_ok = _fac["verdict"] == "NOT_ESTABLISHED"
print(f"  {'OK  ' if _ok else 'FAIL'}  factorial verdict is NOT_ESTABLISHED "
      f"(got {_fac['verdict']!r}); SS4.3's conclusion depends on it")
if not _ok:
    failures.append(f"code/77 verdict changed to {_fac['verdict']!r}; SS4.3 says NOT_ESTABLISHED")
# The share attributed to selection-set size is quoted in the abstract as 75%.
checks += 1
_share = _fac["decomposition"]["share_of_total_movement"][
    "selection_set_size_small_minus_fold_matched"]
_ok = 0.70 <= _share <= 0.80
print(f"  {'OK  ' if _ok else 'FAIL'}  selection-set size accounts for ~75% of the "
      f"movement (got {100*_share:.1f}%), as the abstract states")
if not _ok:
    failures.append(f"selection-set-size share is {100*_share:.1f}%, abstract says ~75%")

_rfc = load("real_feature_corrected_selection_controls.json")
for _h in ("128", "384"):
    _A = _rfc["part_A_in_fold_es_sweep"]["by_capacity"][_h]["by_es_fraction"]
    check(f"real-feature shipped gap, capacity {_h}", _A["0.15"]["gap_mean"])
    check(f"real-feature fold-matched gap, capacity {_h}", _A["0.25"]["gap_mean"])
    _B = _rfc["part_B_factorial"]["by_capacity"][_h]
    check(f"real-feature fully-corrected gap, capacity {_h}",
          _B["decomposition"]["fully_corrected_gap"])
# The shipped arm must still reproduce code/43 exactly, or the comparison is void.
checks += 1
_ok = all(_rfc["part_A_in_fold_es_sweep"]["by_capacity"][_h]["reproduces_code43_per_seed"]
          for _h in ("128", "384"))
print(f"  {'OK  ' if _ok else 'FAIL'}  code/78's ES=0.15 arm reproduces code/43 per seed")
if not _ok:
    failures.append("code/78 no longer reproduces code/43's shipped per-seed values")

_tr = load("transport_check_arm_matching_sensitivity.json")
_am = _tr["arm_match_quality"]
check("transport check: LEAKY-arm spread across harnesses",
      _am["leaky"]["max_pairwise_spread"], "{:.4f}")
check("transport check: CONTROL-arm spread across harnesses",
      _am["control"]["max_pairwise_spread"], "{:.4f}")
check("transport check: PLACEBO-arm spread across harnesses",
      _am["placebo"]["max_pairwise_spread"], "{:.4f}")
_cm = _tr["headline"]["control_matched"]["interpolated"]
check("mechanism axis under control-arm matching, low", _cm[0], "{:.1f}")
check("mechanism axis under control-arm matching, high", _cm[1], "{:.1f}")
# F8: the two axes abut because they SHARE a denominator cell, not because two
# independent quantities meet. If that ever stops being true the disclosure in
# SS5.2 becomes wrong in the other direction.
_js = load("joint_severity_surface.json")["cells"]
_swC = load("selection_multiplicity_sweep.json")["sweep_C_operating_point"]
checks += 1
_ok = _js["45|0.95"]["gap_mean"] == _swC["0.95"]["gap_mean"]
print(f"  {'OK  ' if _ok else 'FAIL'}  the operating-point axis's sound-maximum "
      f"denominator and the mechanism axis's denominator are the SAME cell "
      f"(SS5.2 discloses this)")
if not _ok:
    failures.append("SS5.2 says the two axes share a denominator cell; they no longer do")

# F9: the control-health ratio applied to Sweep C, which SS4.3 now reports.
for _t, _lbl in (("0.7", "0.70"), ("0.8", "0.80"), ("0.9", "0.90"),
                 ("0.95", "0.95"), ("0.985", "0.985")):
    _c = _swC[_t]
    check(f"Sweep C control-health ratio at AUROC_0={_lbl}",
          (_c["leaky_mean"] - _c["clean_matched_mean"])
          / (_c["clean_matched_mean"] - _c["placebo_mean"]), "{:.3f}")

# F3/F10: the per-cell null counts SS4.4 and Table 8 now state.
_two = load("case_study_4_two_sided_null.json")
_pc = _two["per_cell"]
checks += 1
_n2 = sum(1 for v in _pc.values() if v["p_two_sided"] < 0.05)
_nw2 = sum(1 for v in _pc.values() if v["rotation_p_two_sided"] < 0.05)
_nup = sum(1 for v in _pc.values() if v["p_upper"] < 0.05)
_nwup = sum(1 for v in _pc.values() if v["rotation_p_upper"] < 0.05)
_ok = (_n2 == 16 and _nw2 == 11 and _nup == 0 and _nwup == 0)
print(f"  {'OK  ' if _ok else 'FAIL'}  CS4 per-cell null counts as SS4.4 states them: "
      f"two-sided {_n2}/24 (Delta_boot) and {_nw2}/24 (Delta_wc), upper-tail "
      f"{_nup} and {_nwup}")
if not _ok:
    failures.append(f"CS4 null counts moved: two-sided {_n2}/{_nw2}, upper {_nup}/{_nwup}")
# Table 8 (the full 24-cell table, relocated in the 30-35pp compression pass
# to EXTENDED_TECHNICAL_DETAIL.md, Appendix B.3) must show each estimator's OWN
# null. Spot-check the row the third review used, where the two differ most
# visibly. Searched against the combined TEX (main.tex + both supplementary
# .md files, per the TEX reassignment above) since the table itself no longer
# lives in main.tex.
checks += 1
_row = _pc["llama3.1-8b__fever__linear"]
_want = f"${_row['bootstrap_max_bias_exact']:+.4f}$ & ${_row['null_mean']:+.4f}$"
_ok = _want in TEX
print(f"  {'OK  ' if _ok else 'FAIL'}  Table 8 (relocated to "
      f"EXTENDED_TECHNICAL_DETAIL.md) shows Delta_boot's own null beside "
      f"Delta_boot (llama3.1-8b/fever/linear: {_want})")
if not _ok:
    failures.append("Table 8's null column is not Delta_boot's own null for the "
                    "llama3.1-8b/fever/linear row (checked main.tex + "
                    "PROVENANCE_LOG.md + EXTENDED_TECHNICAL_DETAIL.md)")

_oa = load("oof_averaging_control_corrected_arms.json")
for _r in ("averaged", "single_model", "oof_apparent"):
    check(f"OOF-averaging readout {_r}: gap vs shipped control",
          _oa["by_readout"][_r]["gap_vs_shipped_control"]["gap_mean"])
    check(f"OOF-averaging readout {_r}: gap vs corrected control",
          _oa["by_readout"][_r]["gap_vs_corrected_control"]["gap_mean"])

# F11: the checklist is now generated, and must match its template + the JSONs.
checks += 1
_r = _sp.run([sys.executable, str(ROOT / "code" / "80_generate_leakage_checklist.py"),
              "--check"], capture_output=True, text=True)
_ok = _r.returncode == 0
print(f"  {'OK  ' if _ok else 'FAIL'}  draft/leakage_checklist.md is in sync with its "
      f"template and the result JSONs (code/80 --check)")
if not _ok:
    failures.append("draft/leakage_checklist.md has drifted; re-run "
                    "code/80_generate_leakage_checklist.py. " + _r.stdout.strip()[:300])



# ── Round 5: P0 revision pass (Weak Accept conditional fixes) ───────────────
print("\n== Round 5: P0 revision pass -- corrected transport ratios, factorial "
      "share bootstrap, Mechanism 1 reconstruction ==")

# P0-1: Table 7's real-feature capacity-384 fully-corrected interval, printed
# to 5 decimals so it visibly excludes zero.
_rf = load("real_feature_corrected_selection_controls.json")["headline_comparison"]["384"]
check("real-feature cap.384 fully-corrected CI low (5dp)",
      _rf["fully_corrected_ci"][0], fmt="{:+.5f}")
check("real-feature cap.384 fully-corrected CI high (5dp)",
      _rf["fully_corrected_ci"][1], fmt="{:+.5f}")

# P0-2: code/84's rerun of the transport-check arm-matching sensitivity against
# the CORRECTED (fold-matched) numerators, replacing code/82's superseded ones.
_tc = load("transport_check_arm_matching_sensitivity_corrected.json")
_shipped_leaky = _tc["shipped_numerators"]["leaky_matched"]["range"]
_shipped_ctrl = _tc["shipped_numerators"]["control_matched"]["range"]
_corr_leaky = _tc["corrected_numerators"]["leaky_matched"]["range"]
_corr_ctrl = _tc["corrected_numerators"]["control_matched"]["range"]
checks += 1
_ok = (abs(_shipped_leaky[0] - 15.1) < 0.05 and abs(_shipped_leaky[1] - 42.0) < 0.05
       and abs(_shipped_ctrl[0] - 13.1) < 0.05 and abs(_shipped_ctrl[1] - 28.7) < 0.05)
print(f"  {'OK  ' if _ok else 'FAIL'}  code/84 reproduces code/82's shipped-numerator "
      f"ratios (15.1/42.0 leaky-matched, 13.1/28.7 control-matched)")
if not _ok:
    failures.append("code/84's shipped-numerator reproduction has drifted from code/82")
check("corrected leaky-matched (probit interp.) ratio, low", _corr_leaky[0], fmt="{:.1f}")
check("corrected leaky-matched (probit interp.) ratio, high", _corr_leaky[1], fmt="{:.1f}")
check("corrected control-matched ratio, low", _corr_ctrl[0], fmt="{:.1f}")
check("corrected control-matched ratio, high", _corr_ctrl[1], fmt="{:.1f}")

# P0-5: code/85's seed-level bootstrap over the factorial's three main-effect
# shares. The point estimates (75.5/28.1/-6.2%) were already checked in Round 4
# as raw AUROC-scale effects; here we check the BCa intervals on the SHARES
# newly attached to them, and that the budget/depth shares (unlike size) cover
# zero -- the honesty check this round exists for.
_bs = load("mechanism3_factorial_share_bootstrap.json")
_ci = _bs["bca_intervals"]
check("share_size BCa low (%)", _ci["share_size"]["ci_95"][0] * 100, fmt="{:+.1f}")
check("share_size BCa high (%)", _ci["share_size"]["ci_95"][1] * 100, fmt="{:+.1f}")
check("share_budget BCa low (%)", _ci["share_budget"]["ci_95"][0] * 100, fmt="{:+.1f}")
check("share_budget BCa high (%)", _ci["share_budget"]["ci_95"][1] * 100, fmt="{:+.1f}")
check("share_depth BCa low (%)", _ci["share_depth"]["ci_95"][0] * 100, fmt="{:+.1f}")
check("share_depth BCa high (%)", _ci["share_depth"]["ci_95"][1] * 100, fmt="{:+.1f}")
checks += 1
_ok = (not _bs["size_share_distinguishable_from_zero"]["distinguishable_from_zero_bca"] is False
       and _bs["size_share_distinguishable_from_zero"]["distinguishable_from_zero_bca"]
       and not _bs["budget_share_distinguishable_from_zero"]["distinguishable_from_zero_bca"]
       and not _bs["depth_share_distinguishable_from_zero"]["distinguishable_from_zero_bca"])
print(f"  {'OK  ' if _ok else 'FAIL'}  only the selection-set-size share is BCa-"
      f"distinguishable from zero; budget and depth shares are not")
if not _ok:
    failures.append("factorial share bootstrap's zero-distinguishability pattern changed "
                    "(expected: size yes, budget no, depth no)")

# Optional Mechanism 1 stretch measurement (code/86): all three cells must be
# BCa-established and must appear in main.tex's new §4.1 paragraph and Table 8.
_m1 = load("mechanism1_severity_probe.json")
_rng = _m1["range_over_grid"]
check("Mechanism 1 reconstruction gap range, low", _rng[0], fmt="{:+.4f}")
check("Mechanism 1 reconstruction gap range, high", _rng[1], fmt="{:+.4f}")
checks += 1
_ok = all(_m1["by_auroc0"][k]["verdict"] == "ESTABLISHED" for k in _m1["by_auroc0"])
print(f"  {'OK  ' if _ok else 'FAIL'}  Mechanism 1 reconstruction (code/86) is "
      f"BCa-established at all {len(_m1['by_auroc0'])} operating points")
if not _ok:
    failures.append("code/86's Mechanism 1 reconstruction is no longer established at "
                    "every operating point")
checks += 1
_ok = ("+0.13" in TEX and "+0.29" in TEX)
print(f"  {'OK  ' if _ok else 'FAIL'}  Mechanism 1's +0.13 to +0.29 range is quoted in main.tex")
if not _ok:
    failures.append("Mechanism 1's reconstructed severity range is not quoted in main.tex")


# ── Round 6: MAXIMUM-RIGOR PASS (items 1-9, 11 of the second remediation) ───
print("\n== Round 6: MAXIMUM-RIGOR PASS -- oracle baseline, K-decoupling, fix "
      "validation, ADA bound, non-Gaussian sweep, composition test, Mechanism 1 "
      "real-feature check, CS4 cutpoint sensitivity, denser capacity grid, "
      "second model family ==")

# Item 1: Mechanism 3 oracle baseline (code/87).
_m3o = load("mechanism3_oracle_baseline.json")
_fc = _m3o["by_condition"]["fully_corrected_oof"]
checks += 1
_ok = abs(_fc["true_bias"]) < 1e-3 and _fc["coverage"] > 0.85
print(f"  {'OK  ' if _ok else 'FAIL'}  fully-corrected control's true bias is ~0 "
      f"({_fc['true_bias']:+.6f}) with coverage {_fc['coverage']*100:.1f}%")
if not _ok:
    failures.append("Mechanism 3 oracle baseline's fully-corrected arm is no longer "
                    "calibrated near zero / adequate coverage")
checks += 1
_ok = "sec:m3-oracle" in TEX or "oracle baseline" in TEX.lower()
print(f"  {'OK  ' if _ok else 'FAIL'}  oracle-baseline subsection present in main.tex")
if not _ok:
    failures.append("Mechanism 3 oracle baseline section missing from main.tex")

# Item 2: K decoupled from training length (code/88).
_kd = load("k_decoupled_from_training_length.json")
checks += 1
_ok = bool(_kd["decoupled_monotone_nondecreasing_in_k"])
print(f"  {'OK  ' if _ok else 'FAIL'}  K-decoupling: decoupled gaps remain monotone "
      f"non-decreasing in K")
if not _ok:
    failures.append("K-decoupling result (code/88) is no longer monotone non-decreasing")
check("K-decoupling ln(K) slope", _kd["decoupled_ln_k_slope"], fmt="{:+.4f}")

# Item 3: validate prescribed fixes (code/89).
_vf = load("validate_prescribed_fixes.json")
_gap_removed = _vf["part_a_mechanism2_nested_cv"]["gap_removed_by_nesting"]["mean"]
check("Mechanism 2 nested-CV: optimism removed by nesting", _gap_removed, fmt="{:+.4f}")
_fit_count = _vf["part_a_mechanism2_nested_cv"]["cost"]["artifact_probe_fit_count"]
_fit_count_tex = f"{_fit_count:,}".replace(",", "{,}")  # main.tex renders thousands as 9{,}984
checks += 1
_ok = str(_fit_count) in TEX or _fit_count_tex in TEX
print(f"  {'OK  ' if _ok else 'FAIL'}  GUARDIAN artifact probe-fit count ({_fit_count}) "
      f"quoted in main.tex")
if not _ok:
    failures.append(f"Probe-fit count {_fit_count} not found in main.tex")
_m3_synth_runtime = _vf["part_b_mechanism3_disjoint_holdout"]["cost"]["synthetic_factorial_runtime_seconds"]
_m3_real_runtime = _vf["part_b_mechanism3_disjoint_holdout"]["cost"]["real_feature_corrected_controls_runtime_seconds"]
checks += 1
_ok = "288" in TEX and "3{,}166" in TEX
print(f"  {'OK  ' if _ok else 'FAIL'}  fix-cost runtimes (288s, 3166s) quoted in main.tex")
if not _ok:
    failures.append("Fix-cost runtime figures not found in main.tex as expected")

# Item 4: ADA bound numeric check (code/90).
_ada = load("ada_bound_numeric_check.json")
checks += 1
_ok = all(2.0 <= r <= 2000.0 for r in
         [row["ratio_bound_over_measured_severity"] for row in _ada["rows"]])
print(f"  {'OK  ' if _ok else 'FAIL'}  ADA bound / measured-severity ratios all in "
      f"a sane [2, 2000] range (table reproduces)")
if not _ok:
    failures.append("ADA bound numeric check (code/90) ratios drifted outside expected range")
_min_ratio = min(row["ratio_bound_over_measured_severity"] for row in _ada["rows"])
checks += 1
_ok = abs(_min_ratio - 3.7) < 0.5
print(f"  {'OK  ' if _ok else 'FAIL'}  smallest bound/severity ratio ~3.7x (GUARDIAN row)")
if not _ok:
    failures.append(f"ADA bound's smallest ratio drifted: {_min_ratio}")

# Item 5: non-Gaussian operating-point sweep (code/92).
_ng = load("nongaussian_operating_point_sweep.json")
checks += 1
_ok = ("nongaussian" in TEX.lower() or "non-Gaussian" in TEX)
print(f"  {'OK  ' if _ok else 'FAIL'}  non-Gaussian operating-point section present in main.tex")
if not _ok:
    failures.append("Non-Gaussian operating-point section missing from main.tex")
_mix_gaps = [_ng["mixture_nongaussian"][str(t)]["gap_mean"] for t in
            [0.70, 0.80, 0.90, 0.95, 0.985]]
checks += 1
_ok = _mix_gaps[0] > _mix_gaps[-1]   # net decline from lowest to highest operating point
print(f"  {'OK  ' if _ok else 'FAIL'}  non-Gaussian mixture shows net decline from "
      f"AUROC_0=0.70 ({_mix_gaps[0]:+.4f}) to 0.985 ({_mix_gaps[-1]:+.4f})")
if not _ok:
    failures.append("Non-Gaussian mixture no longer shows a net decline across the "
                    "operating-point grid")

# Item 6: composition test (code/93).
_ct = load("composition_test_m3_m5.json")
checks += 1
_ok = not bool(_ct["interaction_bca_excludes_zero"])
print(f"  {'OK  ' if _ok else 'FAIL'}  M3+M5 composition test: interaction term's BCa "
      f"CI covers zero (additive verdict)")
if not _ok:
    failures.append("Composition test (code/93) no longer supports an additive verdict")
check("Composition test: M5 effect (F1)", _ct["m5_effect_f1"]["mean"], fmt="{:+.4f}")

# Item 7: Mechanism 1 real-feature analog (code/94).
_m1r = load("mechanism1_real_feature_analog.json")
check("Mechanism 1 real-feature gap (standardized, C=1.0)",
      _m1r["primary_standardized_C1"]["gap_mean"], fmt="{:+.4f}")
checks += 1
_ok = _m1r["primary_standardized_C1"]["gap_mean"] < 0.05   # an order of magnitude below +0.13-+0.29
print(f"  {'OK  ' if _ok else 'FAIL'}  Mechanism 1 real-feature gap is an order of "
      f"magnitude below the synthetic reconstruction's +0.13-+0.29")
if not _ok:
    failures.append("Mechanism 1 real-feature gap (code/94) is no longer far below the "
                    "synthetic reconstruction's range")

# Item 8: CS4 ceiling-cutpoint sensitivity (code/95).
_cp = load("cs4_ceiling_cutpoint_sensitivity.json")
checks += 1
_stable_splits = [r for r in _cp["cutpoint_sweep"] if r["n_saturated"] == 12]
_ok = len(_stable_splits) >= 5
print(f"  {'OK  ' if _ok else 'FAIL'}  the 12/12 ceiling split is stable across "
      f"{len(_stable_splits)}/{len(_cp['cutpoint_sweep'])} cutpoints in the grid")
if not _ok:
    failures.append("CS4 cutpoint sensitivity (code/95): 12/12 split is no longer stable "
                    "across most of the cutpoint grid")
_r_nd = _cp["continuous_regression_excluding_degenerate_cells"]["pearson_r"]
checks += 1
_ok = _r_nd < -0.5 and _cp["continuous_regression_excluding_degenerate_cells"]["pearson_p"] < 0.05
print(f"  {'OK  ' if _ok else 'FAIL'}  continuous regression (degenerate cells excluded) "
      f"gives a significant negative association (r={_r_nd:+.3f})")
if not _ok:
    failures.append("CS4 cutpoint-free regression no longer significant once degenerate "
                    "cells are excluded")

# Item 9: denser capacity grid, real features (code/96).
_dc = load("denser_capacity_grid_real_features.json")
checks += 1
_ok = not bool(_dc["strictly_decreasing_shipped"])
print(f"  {'OK  ' if _ok else 'FAIL'}  denser capacity grid confirms NON-monotone "
      f"shipped-control decline (as reported in main.tex)")
if not _ok:
    failures.append("Denser capacity grid (code/96) now shows a strictly monotone "
                    "decline, contradicting main.tex's non-monotone framing")

# Item 11: second model family, Qwen2.5-7B (Kaggle kernel output, also copied
# into results/ so the reverse AUROC-literal guard above can trace its numbers).
_qwen_path = R / "real_feature_leakage_test_result_qwen.json"
if _qwen_path.exists():
    _qwen = json.load(open(_qwen_path))
    _qwen_gap = _qwen["gaps"]["leaky_minus_clean_matched"]["mean_gap"]
    checks += 1
    _ok = abs(_qwen_gap) < 0.005 and _qwen["gaps"]["leaky_minus_clean_matched"]["wilcoxon_p"] > 0.05
    print(f"  {'OK  ' if _ok else 'FAIL'}  Qwen2.5-7B second-model-family gap "
          f"({_qwen_gap:+.5f}) remains small and non-significant")
    if not _ok:
        failures.append("Qwen2.5-7B second-model-family result "
                        "(results/real_feature_leakage_test_result_qwen.json) "
                        "drifted from what main.tex reports")
else:
    checks += 1
    print("  FAIL  Qwen2.5-7B kernel output JSON not found")
    failures.append("results/real_feature_leakage_test_result_qwen.json is missing")


print("\n== Matched head-to-head across all five mechanisms (code/97, §5.9) ==")
_h2h = load("matched_head_to_head_five_mechanisms.json")
_pc = _h2h["pooled_auroc_scale_model_comparison"]
_ma = _pc["model_a_pooled"]
_mb = _pc["model_b_mechanism_aware"]
_ft = _pc["nested_f_test"]
_mc = _h2h["matched_cell_comparison"]["summary"]

checks += 1
_grid_ok = (_h2h["grid"]["K_values"] == [5, 10, 20, 40, 80]
            and _h2h["grid"]["auroc0_values"] == [0.7, 0.8, 0.95])
print(f"  {'OK  ' if _grid_ok else 'FAIL'}  H2H grid matches main.tex (K in [5,10,20,40,80], "
      f"AUROC0 in [0.70,0.80,0.95])")
if not _grid_ok:
    failures.append("code/97's grid no longer matches what main.tex describes")

check("H2H pooled model R^2", _ma["r_squared"], fmt="{:.3f}")
check("H2H pooled model LOO-R^2", _ma["loo_r_squared"], fmt="{:.3f}")
check("H2H pooled model beta (wrong-signed)", _ma["coefficients"]["beta"], fmt="{:.2f}")
check("H2H mechanism-aware model R^2", _mb["r_squared"], fmt="{:.3f}")
check("H2H mechanism-aware model LOO-R^2", _mb["loo_r_squared"], fmt="{:.3f}")
check("H2H mechanism-aware shared beta (correctly-signed)", _mb["shared_beta_lnK"], fmt="{:.3f}")
check("H2H mechanism-aware shared c", _mb["shared_c_z0"], fmt="{:.3f}")
check("H2H intercept spread, multiplicative", _mb["intercept_spread"]["exp_range_multiplicative"], fmt="{:.0f}")
check("H2H nested F-stat", _ft["f_stat"], fmt="{:.1f}")
check("H2H matched-cell ratio range, low", _mc["max_over_min_ratio_range"][0], fmt="{:.1f}")
check("H2H matched-cell ratio range, high", _mc["max_over_min_ratio_range"][1], fmt="{:.0f}")
check("H2H matched-cell ratio median", _mc["max_over_min_ratio_median"], fmt="{:.1f}")
check("H2H runtime in hours", _h2h["runtime_seconds"] / 3600, fmt="{:.1f}")
check("H2H Mechanism 3 N_SEEDS", _h2h["mechanism_3"]["n_seeds"], fmt="{:.0f}")

checks += 1
_p_ok = "1.1" in TEX and "10^{-16}" in TEX
print(f"  {'OK  ' if _p_ok else 'FAIL'}  H2H nested F-test p-value (1.1e-16) quoted in main.tex")
if not _p_ok:
    failures.append("H2H nested F-test p-value no longer quoted in main.tex")

checks += 1
_f_ok = "F(3,42)" in TEX
print(f"  {'OK  ' if _f_ok else 'FAIL'}  H2H nested F-test df (3,42) quoted in main.tex")
if not _f_ok:
    failures.append("H2H nested F-test degrees of freedom no longer quoted in main.tex")

checks += 1
_floor_ok = _pc["n_cells_floored_to_positive"] == 9 and _pc["n_cells"] == 48
print(f"  {'OK  ' if _floor_ok else 'FAIL'}  H2H: 9 of 48 cells floored to a positive value "
      f"(as reported in §5.9)")
if not _floor_ok:
    failures.append("H2H floored-cell count or total cell count drifted from what main.tex reports")

print("\n== Properly-powered exchangeable-candidate K-sweep, N=200 (code/98, §5.3) ==")
_n200 = load("exchangeable_candidate_sweep_n200.json")
checks += 1
_ok = _n200["n_seeds"] == 200
print(f"  {'OK  ' if _ok else 'FAIL'}  code/98 ran at N_SEEDS=200")
if not _ok:
    failures.append("results/exchangeable_candidate_sweep_n200.json no longer shows N_SEEDS=200")

_pw = _n200["power"]
check("N200 sweep: MDE at 80% power", _pw["min_detectable_gap_80pct_power_two_sided_05"], fmt="{:.5f}")
check("N200 sweep: target effect (code/47 K=45 gap)", _pw["code47_gap_at_K45_for_scale"], fmt="{:.5f}")
checks += 1
_gap_closed = _pw["min_detectable_gap_80pct_power_two_sided_05"] <= _pw["code47_gap_at_K45_for_scale"]
print(f"  {'OK  ' if _gap_closed else 'FAIL'}  power gap is closed at N_SEEDS=200 (MDE <= target effect)")
if not _gap_closed:
    failures.append("N=200 exchangeable sweep no longer closes the power gap main.tex claims it does")

for _K in ["2", "5", "15", "45"]:
    _cell = _n200["cells"][_K]
    check(f"N200 sweep K={_K} gap", _cell["gap_mean"], fmt="{:.4f}")
    checks += 1
    _covers_zero = _cell["gap_bca_ci_95"][0] < 0 < _cell["gap_bca_ci_95"][1]
    print(f"  {'OK  ' if _covers_zero else 'FAIL'}  K={_K} BCa interval covers zero")
    if not _covers_zero:
        failures.append(f"N=200 exchangeable sweep K={_K} BCa interval no longer covers zero")

_cmp = _n200["_n200_power_summary"]
check("N200 sweep: exchangeable ln K slope b", _cmp["exchangeable_lnK_slope_b"], fmt="{:.5f}")
check("N200 sweep: exchangeable slope CI low", _cmp["exchangeable_lnK_slope_b_ci_95"][0], fmt="{:.5f}")
check("N200 sweep: exchangeable slope CI high", _cmp["exchangeable_lnK_slope_b_ci_95"][1], fmt="{:.5f}")
checks += 1
_wrong_signed = _cmp["exchangeable_lnK_slope_b"] < 0 < _cmp["code47_epoch_lnK_slope_b"]
print(f"  {'OK  ' if _wrong_signed else 'FAIL'}  exchangeable slope is wrong-signed against code/47's epoch-indexed slope")
if not _wrong_signed:
    failures.append("N=200 exchangeable slope sign relationship to code/47's slope has changed")

_evt = _n200["evt_test_A11_could_not_run"]["selection_stage_check"]["per_cell"]
for _K in ["2", "5", "15", "45"]:
    check(f"N200 sweep K={_K} selection-stage observed/exact-Gaussian ratio",
          _evt[_K]["observed_over_exact_gaussian"], fmt="{:.3f}")

_wilcoxon_ps = [_n200["cells"][_K]["wilcoxon_p"] for _K in ["2", "5", "15", "45"]]
check("N200 sweep: wilcoxon_p range low", min(_wilcoxon_ps), fmt="{:.3f}")
check("N200 sweep: wilcoxon_p range high", max(_wilcoxon_ps), fmt="{:.3f}")

print("\n== follow-up review pass: Mechanism 2 second construction (code/99, §5.9) ==")
_m2b = load("mechanism2_construction_b_correlated_latent.json")
_sa = _m2b["stability_assessment"]
_corr = _m2b["a_vs_b_correlation"]
_sub = _m2b["pooled_model_comparison_with_construction_b_substituted_for_mechanism_2"]
_subcmp = _sub["comparison"]

checks += 1
_ok = _sa["verdict"] == "STABLE"
print(f"  {'OK  ' if _ok else 'FAIL'}  M2 construction-B stability verdict is STABLE "
      f"(got {_sa['verdict']!r})")
if not _ok:
    failures.append(f"code/99's stability verdict changed to {_sa['verdict']!r}; "
                    f"main.tex §5.9 states STABLE")

check("M2 construction-B own-fit intercept a_B", _sa["a_B_own_fit"], fmt="{:.3f}")
check("M2 construction-A intercept a_2 (reference)", _sa["a_2_construction_A"], fmt="{:.3f}")
check("M2 A-vs-B intercept delta", _sa["intercept_delta_log_scale"], fmt="{:.3f}")
check("M2 A-vs-B per-cell Pearson r", _corr["pearson_r"], fmt="{:.3f}")

checks += 1
_p_ok = "9.3" in TEX and "10^{-11}" in TEX
print(f"  {'OK  ' if _p_ok else 'FAIL'}  M2 A-vs-B Pearson p-value (9.3e-11) quoted in main.tex")
if not _p_ok:
    failures.append("M2 A-vs-B Pearson p-value no longer quoted in main.tex")

check("M2 substituted intercept (construction B in place of A)",
      _sub["mechanism_2_intercept_construction_b_substituted"], fmt="{:.3f}")
check("M2 substituted intercept shift", _sub["mechanism_2_intercept_shift"], fmt="{:+.3f}")
check("M2 substituted pooled model R^2", _subcmp["model_a_pooled"]["r_squared"], fmt="{:.3f}")
check("M2 substituted mechanism-aware model R^2", _subcmp["model_b_mechanism_aware"]["r_squared"], fmt="{:.3f}")
check("M2 substituted nested F-stat", _subcmp["nested_f_test"]["f_stat"], fmt="{:.1f}")
check("M2 substituted intercept spread, multiplicative",
      _subcmp["model_b_mechanism_aware"]["intercept_spread"]["exp_range_multiplicative"], fmt="{:.0f}")

checks += 1
_rho_ok = _m2b["construction_b"]["rho"] == 0.5
print(f"  {'OK  ' if _rho_ok else 'FAIL'}  M2 construction B used RHO=0.5 (as main.tex states)")
if not _rho_ok:
    failures.append("code/99's RHO constant no longer matches what main.tex states (0.5)")


print("\n== follow-up review pass: ES_HOLD_FRACTION sensitivity sweep on CS3 "
      "factorial (code/100, §4.3 / Appendix A) ==")
_esf = load("es_hold_fraction_sensitivity_cs3_factorial.json")
_esf_summary = _esf["summary"]

checks += 1
_grid_ok = _esf["config"]["n_seeds"] == 100 and _esf_summary["f_small_grid_run"] == [0.05, 0.10, 0.15, 0.20]
print(f"  {'OK  ' if _grid_ok else 'FAIL'}  ES_HOLD_FRACTION sweep ran at N_SEEDS=100 over "
      f"F_SMALL in [0.05,0.10,0.15,0.20]")
if not _grid_ok:
    failures.append("code/100's sweep grid or N_SEEDS no longer matches what main.tex describes")

checks += 1
_anchor_ok = _esf_summary["anchor_check_at_0.15"]["ok"]
print(f"  {'OK  ' if _anchor_ok else 'FAIL'}  ES_HOLD_FRACTION sweep's F_SMALL=0.15 point "
      f"reproduces code/77's shipped factorial")
if not _anchor_ok:
    failures.append("code/100's F_SMALL=0.15 anchor no longer reproduces code/77's shipped values")

check("ES_HOLD_FRACTION sweep: selection-set-size share range, low",
      _esf_summary["selection_set_size_share_range"][0] * 100, fmt="{:.1f}")
check("ES_HOLD_FRACTION sweep: selection-set-size share range, high",
      _esf_summary["selection_set_size_share_range"][1] * 100, fmt="{:.1f}")
check("ES_HOLD_FRACTION sweep: share at shipped F_SMALL=0.15",
      _esf_summary["selection_set_size_share_at_shipped_0.15"] * 100, fmt="{:.1f}")

_by_f = _esf["by_f_small"]
for _f in ["0.05", "0.1", "0.2"]:
    check(f"ES_HOLD_FRACTION sweep F_SMALL={_f}: shipped gap",
          _by_f[_f]["shipped_uncorrected_gap"], fmt="{:.4f}")
    check(f"ES_HOLD_FRACTION sweep F_SMALL={_f}: selection-set-size share (%)",
          _by_f[_f]["share_of_total_movement"]["selection_set_size_small_minus_fold_matched"] * 100,
          fmt="{:.1f}")
    check(f"ES_HOLD_FRACTION sweep F_SMALL={_f}: selection-run-budget share (%)",
          _by_f[_f]["share_of_total_movement"]["selection_run_budget_infold_minus_oof"] * 100,
          fmt="{:.1f}")

checks += 1
_omit_ok = _esf_summary["f_small_omitted"] == [0.25, 0.30]
print(f"  {'OK  ' if _omit_ok else 'FAIL'}  ES_HOLD_FRACTION sweep documents both F_SMALL=0.25 "
      f"and 0.30 as infeasible under the factorial's shared out-of-fold pool geometry")
if not _omit_ok:
    failures.append("code/100's documented omitted F_SMALL value no longer matches main.tex")

print(f"\n{checks} checks run, {len(failures)} failures")
for f in failures:
    print("  - " + f)
sys.exit(1 if failures else 0)
