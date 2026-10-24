# Supplementary artifact

[![Preprint DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23624980.svg)](https://doi.org/10.5281/zenodo.23624980) [![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23624982.svg)](https://doi.org/10.5281/zenodo.23624982)

Paper: *Data Reuse, Not Selection: A Code-Verified Audit of Optimism in Hidden-State Hallucination-Detection Pipelines* — accepted at the NeurIPS 2026 Workshop on Trustworthy AI Evaluation. Preprint: https://doi.org/10.5281/zenodo.23624980


Code, result files, and reproduction map for the accompanying submission.

Entry points:
- `REPRODUCE.md` — copy-pasteable command sequence and per-claim script-to-result map.
- `code/53_verify_paper_numbers.py` — re-checks the severity, mechanism and
  severity-relationship numbers against the shipped result JSONs (558 checks).
  It does **not** cover the calibration-bridge or selective-prediction tables;
  those are produced by `95`–`97` and are checkable directly against their JSONs.
- `PROVENANCE_LOG.md` — per-number correction history.
- `EXTENDED_TECHNICAL_DETAIL.md` — derivations and factorial tables.

## Which paper this archive accompanies

`draft/latex/main.tex` in this archive is **not** the submitted paper. It is the
earlier full-length draft, titled *Severity Is a Property of Mechanism, Metric,
and Control*, which covers five mechanisms across roughly 3{,}800 lines. The
submission compresses that material to one mechanism and retitles it *Data
Reuse, Not Selection*. Where the two differ, **the submitted paper is
authoritative**; the draft is included because it carries the per-mechanism
detail the submission had to cut, and because the result files here were
produced for it.

The draft is consistent with the corrected analysis -- it reports
`\Delta_{\text{sel}}=+0.0255` together with the mechanical null that places
that value *below* the null interval, which is the reading the submission
keeps. It does not contain any of the claims later withdrawn.

The only GPU-dependent step is shipped as a cached array; every check runs on CPU.
