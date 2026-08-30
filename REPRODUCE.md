# Reproducing this paper

This file is both a copy-pasteable command sequence (sections 0-5) and the
full, per-claim **script-to-result map** (section 6): which script produces
which `results/*.json`, and what a from-scratch rerun needs beyond that JSON.
The map previously lived as an appendix of `draft/latex/main.tex`; it was
moved here in a page-budget compression pass, since it duplicated this file
and contains no scientific content the main text does not state.

**Two facts bear on every entry below.** First, this paper's only
GPU-dependent computation is the AWQ-quantized Mistral-7B feature extraction
that produced `results/real_features_mistral7b_halueval.npz`, run once on
Kaggle (`kaggle_runs/real-feature-v2/`) and shipped as a cached
array; every script that consumes it (`code/25`, `27`, `31`, `33`, `43`)
reruns from that cache on CPU in seconds to minutes and needs no GPU itself.
Second, Case Study 2's raw 171 MB hidden-state cache is **not** shipped;
`code/48` emits a 4.0 MB derived artifact instead, from which `code/51`
recomputes Case Study 2's primary numbers, and where a number depends on the
raw cache directly that is stated below.

## 0. Setup

```bash
cd paper2-leakage-audit
pip install -r requirements.txt
```

`requirements.txt`'s header states which pins are locally-verified and which
(`gptqmodel`) are Kaggle-only.

## 1. Cheap path: re-verify every shipped result against its cached JSON

No GPU, no external repository, seconds. This is the recommended default —
it checks that every headline number in `main.tex` still matches the
`results/*.json` file behind it, and fails loudly on any mismatch.

```bash
python3 code/53_verify_paper_numbers.py
```

## 2. From-scratch: regenerate every `results/*.json`

Everything below runs on CPU except step 2c. Script numbering is
chronological-by-writing-order, **not** a strict dependency order: most
scripts only need files already shipped in `results/`, but a few read another
script's output. Run in ascending numeric order first; if a script raises
`FileNotFoundError` on a `results/*.json`, run the producing script named in
this map or in the failing script's own docstring, then retry. One
already-identified exception: `code/72_magnitude_triangle.py` reads the
outputs of `code/57`, `code/78` and `code/79`, all numbered *after* it — run
it last, not in its numeric slot.

```bash
# 2a. Case Study 3 (MultiHaluDet) core pipeline and corrections
for f in code/02_synthetic_leakage_ablation.py \
         code/02b_capacity_sweep_ablation.py \
         code/02c_placebo_and_power_check.py \
         code/02d_corrected_capacity_placebo_sweep.py \
         code/19_real_feature_leakage_diagnostics.py \
         code/22_epoch_forcing_confound_control.py \
         code/25_real_feature_leakage_test_corrected_architecture.py \
         code/27_anisotropic_covariance_capacity_sweep.py \
         code/31_real_feature_test_calibrated.py \
         code/33_real_feature_test_cv_calibrated.py \
         code/43_calibration_leakage_diagnostic.py \
         code/44_statistical_rigor_retrofit.py \
         code/49_mechanism3_fidelity_extension.py \
         code/50_evt_scaling_refit.py \
         code/55_oof_averaging_control.py \
         code/56_eigenspectrum_sweep_fixed_dim.py \
         code/67_adaptivity_control_budget_deficit.py \
         code/68_k_law_functional_form.py \
         code/73_exchangeable_candidate_sweep.py \
         code/75_mechanism3_selection_budget_controls.py \
         code/77_mechanism3_factorial_selection_controls.py \
         code/78_real_feature_corrected_selection_controls.py \
         code/79_fidelity_extension_corrected_selection_controls.py \
         code/83_oof_averaging_control_corrected_arms.py; do
  python3 "$f"
done

# 2b. The Mechanism-3 fidelity-extension 2x2 ablation needs code/49 run under
#     four env-var configurations before it can aggregate them (recipe is in
#     code/54's own docstring):
for ES in 3 15; do
  USE_SUPERSEDED_CALIBRATION=1 ALPHA_OVERRIDE=0.1328 ES_PATIENCE_OVERRIDE=$ES \
    OUT_NAME=abl_oldcal_es$ES.json python3 code/49_mechanism3_fidelity_extension.py
done
ES_PATIENCE_OVERRIDE=3 OUT_NAME=mechanism3_fidelity_extension_espatience3_ablation.json \
  python3 code/49_mechanism3_fidelity_extension.py
python3 code/49_mechanism3_fidelity_extension.py   # the reported cell
python3 code/54_fidelity_extension_2x2_ablation.py

# 2c. GPU step (optional -- only needed to regenerate the real Mistral-7B
#     feature cache from scratch; the cache is already shipped at
#     results/real_features_mistral7b_halueval.npz, so 2a's real-feature
#     scripts (25, 27, 31, 33, 43) run on CPU against the shipped cache
#     without this step).
#     Push kaggle_runs/real-feature-v2/ to Kaggle (GPU runtime,
#     TheBloke/Mistral-7B-Instruct-v0.2-AWQ) and copy its output
#     real_features_mistral7b_halueval.npz into results/.

# 2d. Case Study 2 (GUARDIAN)
python3 code/48_case_study_2_layer_decomposition.py
python3 code/61_case_study_2_half_membership_probe.py
python3 code/64_case_study_2_selection_null.py
python3 code/51_case_study_2_replay_from_artifact.py   # replays 48's primary numbers from the shipped derived .npz

# 2e. Case Study 4 (quantized-LLM paper) -- vendored probe results already in
#     code/external/HallucinationPatternDetection/results/probes/*.json
python3 code/45_case_study_4_winners_curse.py
python3 code/59_case_study_4_nonnegativity_and_sensitivity.py
python3 code/66_case_study_4_two_sided_null.py
python3 code/70_cs4_exact_enumeration.py
python3 code/81_cs4_joint_calibration_sweep.py
python3 code/71_cs4_estimator_calibration.py

# 2f. Mechanism 5 (MultiHaluDet, threshold selection)
python3 code/46_mechanism5_threshold_selection.py
python3 code/62_mechanism5_youden_threshold.py
python3 code/74_mechanism5_size_matched_control.py

# 2g. Severity surface (Sec. 5) -- needs 2a and 2f above
python3 code/47_selection_multiplicity_sweep.py
python3 code/63_sweep_per_seed_artifact.py
python3 code/57_joint_severity_surface.py
python3 code/65_joint_surface_estimator_sensitivity.py
python3 code/58_operating_point_transport_check.py
python3 code/82_transport_check_arm_matching_sensitivity.py
python3 code/60_operating_point_ratio_fieller_check.py
python3 code/69_operating_point_variance_control.py
python3 code/72_magnitude_triangle.py    # run last: needs 57, 78, 79 above

# 2h. Checklist and scanner (Sec. 6)
python3 code/04_leakage_linter.py

# 2i. Figures
python3 code/05_generate_capacity_sweep_figure.py
python3 code/76_generate_severity_surface_figure.py
```

## 3. Regenerate the paper's companion documents

```bash
python3 code/52_sync_paper_draft_md.py            # draft/paper_draft.md from main.tex
python3 code/52_sync_paper_draft_md.py --check     # verify it's current instead of writing
python3 code/80_generate_leakage_checklist.py      # draft/leakage_checklist.md from results/*.json
python3 code/80_generate_leakage_checklist.py --check
```

## 4. Rebuild the PDF

```bash
cd draft/latex
rm -f main.aux main.out main.log main.pdf
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex   # second pass for cross-references
/usr/bin/grep -n "^! [A-Za-z]" main.log      # expect no output (0 errors)
cd ../..
```

## 5. Final check

```bash
python3 code/53_verify_paper_numbers.py      # re-run after any regeneration above
```

## 6. Script-to-claim map

`code/53_verify_paper_numbers.py` is a narrower, complementary instrument: it
asserts that a specific formatted number appears in `main.tex` and matches its
JSON. This map instead answers "which script do I run to obtain that JSON in
the first place."

### Case Study 1 (HaRP) — §4.1

**Not reproducible from this artifact, by design.** The +0.1906 AUROC figure
and the `probe_conf` 1.0000-vs-0.7573 comparison have no script, log or data
file behind them anywhere in this repository; `draft/worked_examples.md` gives
the prose account and `main.tex` states the non-reproducibility directly (§4.1,
and again in §3). This is the one entry with no script/JSON pair, which is
exactly why the figure is excluded from the abstract's severity range and from
every cross-mechanism comparison in the paper.

### Case Study 2 (GUARDIAN) — §4.2

- **Headline Δ_sel = +0.0255 (SD 0.0186) under randomized stratified splits, and
  the sequential-split +0.192/+0.188 figures it replaces**:
  `code/48_case_study_2_layer_decomposition.py` →
  `results/case_study_2_layer_decomposition.json`.
- **The mechanical null (+0.0429) and the winner's-curse decomposition
  (A = +0.042865, B = +0.0173, 40.4% cancellation)**:
  `code/64_case_study_2_selection_null.py` →
  `results/case_study_2_selection_null.json`.
- **Half-membership separability (0.5214–0.7885, mean 0.7259), the 0.7300/0.8100
  base-rate gap, all-data argmax at L21, and the base-rate-matched control
  (−0.0050 vs −0.0038)**: `code/61_case_study_2_half_membership_probe.py` →
  `results/case_study_2_half_membership_probe.json`.
- **Reproducible from a shipped derived artifact.** `code/48` also emits
  `results/case_study_2_probe_scores.npz` (4.0 MB);
  `code/51_case_study_2_replay_from_artifact.py` recomputes every number in the
  first two bullets from that npz alone — no access to the 171 MB raw
  hidden-state cache — and asserts bitwise agreement with the committed JSON.
  Cheap: seconds, no GPU.
- **Not re-verifiable from the shipped artifact**, and stated as such by
  `code/51` itself: the twelve numbers of §4.2's `StandardScaler`×C
  regularization-robustness sweep (five Δ_sel values, five general-gap values,
  two like-for-like unscaled-vs-scaled values). `code/48` computes that sweep by
  refitting probes on the raw 171 MB cache and never writes the resulting scores
  into the shipped `.npz`. A from-scratch rerun of this one sub-result needs
  GUARDIAN's original hidden-state cache, which this release does not include.

### Case Study 3 (MultiHaluDet) — §4.3, Appendix A

- **The primary 2×2×2 factorial** (fully corrected control at −0.0001,
  decomposition 75.5%/28.1%/−6.2%):
  `code/77_mechanism3_factorial_selection_controls.py` →
  `results/mechanism3_factorial_selection_controls.json`.
- **Headline fold-matched severities, Δ_real = +0.0060/+0.0067 and
  fidelity-extension +0.0159** (the shipped JSONs'
  `headline_comparison.*.fold_matched_in_fold_gap` fields):
  `code/78_real_feature_corrected_selection_controls.py` →
  `results/real_feature_corrected_selection_controls.json`;
  `code/79_fidelity_extension_corrected_selection_controls.py` →
  `results/fidelity_extension_corrected_selection_controls.json`. The same two
  scripts' `part_B_factorial` block gives the fully-corrected out-of-fold
  variants (+0.0030/+0.0044 at capacity 384).
- **Un-averaged OOF readout, corrected arms**:
  `code/83_oof_averaging_control_corrected_arms.py` →
  `results/oof_averaging_control_corrected_arms.json`.
- **Underlying pipeline components**: primary isotropic synthetic sweep,
  `code/02d_corrected_capacity_placebo_sweep.py` →
  `results/corrected_capacity_placebo_sweep.json` (its superseded coupled-seed
  run ships alongside as
  `results/corrected_capacity_placebo_sweep_coupled_seed_legacy.json`);
  statistical-rigor wrapper adding BCa CIs and paired permutation tests,
  `code/44_statistical_rigor_retrofit.py` →
  `results/statistical_rigor_retrofit.json`; real-feature harness,
  `code/43_calibration_leakage_diagnostic.py` →
  `results/calibration_leakage_diagnostic.json` and
  `results/real_feature_test_train_only_calibrated.json`; fidelity extension
  (raw), `code/49_mechanism3_fidelity_extension.py` →
  `results/mechanism3_fidelity_extension.json` — run under four
  environment-variable configurations (see §2b above) to also produce
  `results/abl_oldcal_es3.json`, `results/abl_oldcal_es15.json` and
  `results/mechanism3_fidelity_extension_espatience3_ablation.json`, which
  `code/54_fidelity_extension_2x2_ablation.py` reads together into
  `results/fidelity_extension_2x2_ablation.json`; un-averaged control,
  `code/55_oof_averaging_control.py` → `results/oof_averaging_control.json`;
  anisotropic covariance sweep,
  `code/27_anisotropic_covariance_capacity_sweep.py` →
  `results/anisotropic_covariance_capacity_sweep.json`; fixed-d eigenspectrum
  sweep, `code/56_eigenspectrum_sweep_fixed_dim.py` →
  `results/eigenspectrum_sweep_fixed_dim.json`; the adaptivity and
  budget-deficit controls, `code/22_epoch_forcing_confound_control.py` →
  `results/epoch_forcing_confound_control.json` and
  `code/67_adaptivity_control_budget_deficit.py` →
  `results/adaptivity_control_budget_deficit.json` (the two are bitwise the same
  arm); corrected-architecture real-feature test,
  `code/25_real_feature_leakage_test_corrected_architecture.py` →
  `results/real_feature_leakage_test_corrected_architecture.json`; EVT/K-form
  refit, `code/50_evt_scaling_refit.py` → `results/evt_scaling_refit.json`;
  selection-budget controls,
  `code/75_mechanism3_selection_budget_controls.py` →
  `results/mechanism3_selection_budget_controls.json`.
- **Reproducible from this repo, cheap.** MultiHaluDet is vendored at its pinned
  commit (`code/external/MultiHaluDet`, `c7597518`) and the real
  Mistral-7B/HaluEval feature cache is shipped
  (`results/real_features_mistral7b_halueval.npz`); every script above runs on
  CPU from shipped inputs in well under a minute. A from-scratch rerun of the
  feature cache itself needs the Kaggle GPU kernel; reusing the shipped cache
  needs none.
- **Superseded scripts, retained for provenance** rather than for reproducing a
  currently-reported number (each still ships its own result JSON):
  `code/02_synthetic_leakage_ablation.py`,
  `code/02b_capacity_sweep_ablation.py`,
  `code/02c_placebo_and_power_check.py`,
  `code/03_real_feature_leakage_test.py`,
  `code/19_real_feature_leakage_diagnostics.py`,
  `code/31_real_feature_test_calibrated.py`,
  `code/33_real_feature_test_cv_calibrated.py`.

### Case Study 4 (quantized-LLM paper) — §4.4

- **Headline Δ_boot = +0.0021 over all 24 cells, the retired Δ_wc diagnostic,
  and every per-cell value**: `code/45_case_study_4_winners_curse.py` →
  `results/case_study_4_winners_curse.json`.
- **The two non-negativity theorems, verified numerically over random
  matrices**: `code/59_case_study_4_nonnegativity_and_sensitivity.py` →
  `results/case_study_4_estimator_audit.json`.
- **Exact enumeration of all 27 bootstrap resamples**, replacing the 2,000-draw
  Monte Carlo estimate: `code/70_cs4_exact_enumeration.py` →
  `results/case_study_4_exact_enumeration.json`.
- **Calibration against a known ground truth** (the 21% average overstatement
  figure): `code/71_cs4_estimator_calibration.py` →
  `results/cs4_estimator_calibration.json`; joint calibration sweep,
  `code/81_cs4_joint_calibration_sweep.py` →
  `results/cs4_joint_calibration_sweep.json`.
- **Permutation null, reported in both tails**:
  `code/66_case_study_4_two_sided_null.py` →
  `results/case_study_4_two_sided_null.json`.
- **Reproducible from this repo, cheap.** HallucinationPatternDetection is
  vendored at its pinned commit (`code/external/HallucinationPatternDetection`,
  `ea0b9678`), and its own published per-layer, per-seed probe results ship in
  full (24 files under
  `code/external/HallucinationPatternDetection/results/probes/*.json`). Every
  script above recomputes directly from those 24 files; no model inference and
  no GPU is needed to reproduce any Case Study 4 number.

### Mechanism 5 (MultiHaluDet, threshold selection) — §4.5

- **Superseded measurement** (F1 scored at the discarded F1-argmax threshold
  rather than the Youden one the pipeline actually reports at):
  `code/46_mechanism5_threshold_selection.py` →
  `results/mechanism5_threshold_selection.json` (this JSON's
  `sample_size_sensitivity` block also gives the 6.3× decline from n_test = 140
  to 2000).
- **Headline, repo-faithful Youden threshold** (F1 gaps +0.0225 to +0.0520;
  reproduces `code/46`'s F1-argmax column exactly as an internal check):
  `code/62_mechanism5_youden_threshold.py` →
  `results/mechanism5_youden_threshold.json`.
- **Size-matched n_val/n_test control**:
  `code/74_mechanism5_size_matched_control.py` →
  `results/mechanism5_size_matched.json`.
- **Reproducible from this repo, cheap.** Same synthetic isotropic-Gaussian
  generator as Case Study 3 plus a logistic-regression classifier; no external
  dependency beyond the calibration guardrail in `code/sanity_checks.py`.

### Severity surface — §5

- **Candidate-count sweep (A), operating-point sweep (C), sample-size sweep (B),
  and n_val-isolated sweep (D)**, all four in one file:
  `code/47_selection_multiplicity_sweep.py` →
  `results/selection_multiplicity_sweep.json`.
- **Per-seed artifact for Sweeps A and C** (needed for the Monte-Carlo-error
  argument in §5.3 and for any variance-stabilized re-expression):
  `code/63_sweep_per_seed_artifact.py` → `results/sweep_per_seed.npz`,
  `results/sweep_per_seed_manifest.json`.
- **K-regularity functional-form comparison** (seven two-parameter forms,
  degeneracy-gated refit): `code/50_evt_scaling_refit.py` →
  `results/evt_scaling_refit.json`; `code/68_k_law_functional_form.py` →
  `results/k_law_functional_form.json`.
- **Exchangeable-candidate control** (K indexes candidates rather than epochs):
  `code/73_exchangeable_candidate_sweep.py` →
  `results/exchangeable_candidate_sweep.json`.
- **Fieller's-condition endpoint check**:
  `code/60_operating_point_ratio_fieller_check.py` →
  `results/operating_point_ratio_fieller_check.json`.
- **Variance-stabilized re-expression of the operating-point axis**:
  `code/69_operating_point_variance_control.py` →
  `results/operating_point_variance_control.json`.
- **Joint K×AUROC_0 factorial**: `code/57_joint_severity_surface.py` →
  `results/joint_severity_surface.json`; figure rendered directly from that JSON
  by `code/76_generate_severity_surface_figure.py` →
  `draft/latex/figures/severity-surface.pdf`.
- **Joint-fit estimator sensitivity** (drop-one-column refits):
  `code/65_joint_surface_estimator_sensitivity.py` →
  `results/joint_surface_estimator_sensitivity.json`.
- **Transport check** (§5.4): `code/58_operating_point_transport_check.py` →
  `results/operating_point_transport_check.json`; arm-matching-convention
  sensitivity, `code/82_transport_check_arm_matching_sensitivity.py` →
  `results/transport_check_arm_matching_sensitivity.json`.
- **Magnitude triangle** (§5.2): `code/72_magnitude_triangle.py` →
  `results/magnitude_triangle.json` (reads its real-feature and
  fidelity-extension numerators from
  `results/real_feature_corrected_selection_controls.json` and
  `results/fidelity_extension_corrected_selection_controls.json`).
- **Reproducible from this repo, cheap.** Every script in this subsection runs
  on the paper's own synthetic isotropic-Gaussian harness; none needs a GPU or
  an external repository.
- **Known staleness, disclosed inline in the paper and repeated here.**
  `code/82`'s table and the probit-space interpolation it and §5.4 report
  (13.1–28.7×, 13.0–24.0×, 15.1–42.0×) have not been re-run against the
  corrected +0.0060/+0.0159 real-feature and fidelity-extension numerators (§5.2
  and §5.4 both say so); they still read the superseded shipped +0.0093/+0.0250
  gaps. The primary leaky-arm-matched comparison (9.2–24.3×) does not depend on
  them. No other script/JSON mismatch or mis-cited script was found while
  building this map.

### Checklist and automated scanner — §6

- **Regex leakage linter** (7 raw hits over 7 repositories, 0 true positives,
  both known bugs missed): `code/04_leakage_linter.py` →
  `results/leakage_linter_report.json`.
- **The checklist document itself**, whose numbers are resolved mechanically
  from result JSONs rather than typed as literals:
  `code/80_generate_leakage_checklist.py` generates
  `draft/leakage_checklist.md` from `draft/leakage_checklist.md.in`; run with
  `--check` to verify the checked-in copy is current.
- **Not script-reproducible.** The blinded second-rater protocol (§6.2) — an
  independent language-model reading of the same 7 flagged sites, plus the two
  case-study repositories, under a protocol withholding this paper's draft — is
  a qualitative process with no result JSON behind it. There is nothing to rerun
  beyond re-reading the cited source lines against the pinned commits named in
  §6.2 and in `EXTENDED_TECHNICAL_DETAIL.md`.

### Meta and build tooling

- `code/52_sync_paper_draft_md.py` regenerates `draft/paper_draft.md` from
  `main.tex` (`--check` verifies it is current).
- `code/53_verify_paper_numbers.py` traces every headline figure in the
  abstract, §5 and the conclusion back to its result JSON and fails loudly on
  any mismatch; run it before every commit that touches `main.tex`.
- `code/91_build_supplementary_zip.py` builds and audits the supplementary
  archive from an explicit manifest (`--check` audits only, without writing).

See `README.md` for the reproducibility status of each of the four case
studies.
