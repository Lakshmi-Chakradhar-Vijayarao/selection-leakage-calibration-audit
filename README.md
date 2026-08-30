# Supplementary artifact

Code, result files, and reproduction map for the accompanying submission.

Entry points:
- `REPRODUCE.md` — copy-pasteable command sequence and per-claim script-to-result map.
- `code/53_verify_paper_numbers.py` — re-checks the severity, mechanism and
  severity-relationship numbers against the shipped result JSONs (558 checks).
  It does **not** cover the calibration-bridge or selective-prediction tables;
  those are produced by `95`–`97` and are checkable directly against their JSONs.
- `PROVENANCE_LOG.md` — per-number correction history.
- `EXTENDED_TECHNICAL_DETAIL.md` — derivations and factorial tables.

The only GPU-dependent step is shipped as a cached array; every check runs on CPU.
