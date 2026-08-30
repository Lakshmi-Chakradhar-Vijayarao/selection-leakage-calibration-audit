"""
build (and audit) draft/latex/supplementary_material.zip.

This script exists because an independent review found a retracted "52x"
surviving in SEVEN places inside the SHIPPED supplementary archive after
main.tex had already been corrected -- README.md (x2), draft/paper_draft.md
(x3), draft/leakage_checklist.md (x1) and code/58's docstring (x2). The archive
had been assembled by hand, so nothing checked it.

It now (a) rebuilds the archive from an explicit manifest, (b) refuses to write
one whose archive comment is non-empty, (c) scans every text member for
identity-leak strings, and (d) scans every text member for retracted numbers.
Any failure aborts the build; a bad archive is never written.

Run:  python3 code/91_build_supplementary_zip.py
      python3 code/91_build_supplementary_zip.py --check   # audit only
"""
import os
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "draft" / "latex" / "supplementary_material.zip"

# ── manifest ────────────────────────────────────────────────────────────────
ROOT_FILES = [".gitignore", "README.md", "requirements.txt", "REPRODUCE.md",
              "ANONYMITY_AUDIT.md", "PROVENANCE_LOG.md",
              "EXTENDED_TECHNICAL_DETAIL.md"]
DIRS = ["code", "results", "kaggle_kernels", "draft"]
# Excluded: LaTeX build products, caches, the archive itself, and the reviewer
# document (which is not part of the submission).
EXCLUDE_SUFFIX = {".pyc", ".aux", ".log", ".out", ".synctex.gz", ".zip", ".fls",
                  ".fdb_latexmk", ".bbl", ".blg", ".DS_Store"}
EXCLUDE_NAMES = {"VERSION_2.0_MEGA_REVIEW.md", "main.pdf"}
EXCLUDE_PARTS = {"__pycache__", ".git", ".ipynb_checkpoints"}
TEXT_SUFFIX = {".py", ".md", ".tex", ".txt", ".json", ".yaml", ".yml", ".sty",
               ".bst", ".cfg", ".toml", ".sh", ".gitignore"}

# ── identity-leak strings (double-blind review) ─────────────────────────────
# Absolute local paths are the usual leak: they carry a username. Author names,
# emails and institution strings are checked too.
def _required(var):
    """Return an identity regex from the environment, or abort.

    Fails loudly rather than returning a never-matching pattern: a silent
    no-match would make this scanner report a leaking archive as clean.
    """
    v = os.environ.get(var)
    if not v:
        sys.exit(
            f"{var} is unset. This shipped copy is redacted for anonymous "
            f"review; export ANON_INSTITUTION and ANON_AUTHOR (regexes) to run "
            f"the identity scan."
        )
    return v


IDENTITY_PATTERNS = [
    (r"/Users/[A-Za-z0-9._-]+", "absolute macOS home path (leaks a username)"),
    (r"/home/[A-Za-z0-9._-]+", "absolute Linux home path (leaks a username)"),
    (r"C:\\\\Users\\\\[A-Za-z0-9._-]+", "absolute Windows home path"),
    (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(?:edu|com|org|net|ac\.[a-z]{2})",
     "email address"),
    # REDACTED FOR ANONYMOUS REVIEW. The institution- and author-name literals
    # previously sat here as plain regexes -- which meant this scanner leaked,
    # in its own source, the identity it exists to strip. They are now read from
    # the environment. _required() aborts if they are unset, so a reviewer who
    # runs this cannot get a false "clean" result from the redaction itself.
    (_required("ANON_INSTITUTION"), "institution name"),
    (_required("ANON_AUTHOR"), "author name"),
    (r"(?i)\bCo-Authored-By\b", "commit trailer"),
]
# Third-party vendored repositories carry their OWN authors' names and URLs,
# which are cited in the paper and are not an anonymity leak. Likewise the
# LaTeX class/style files, which carry their upstream maintainers' contacts.
IDENTITY_EXEMPT_PREFIXES = ("code/external/",)
IDENTITY_EXEMPT_FILES = {
    "draft/latex/fancyhdr.sty", "draft/latex/tmlr.sty", "draft/latex/tmlr.bst",
    "draft/latex/math_commands.tex",
    # This script has to be able to NAME the patterns it forbids.
    "code/91_build_supplementary_zip.py",
    # ...as does the number-verification script.
    "code/53_verify_paper_numbers.py",
}

# main.tex carries a \author block that tmlr.sty suppresses in the compiled PDF
# but which is plainly readable in the SOURCE that ships in this archive. The
# archive therefore receives an anonymized copy: the block is replaced, the file
# on disk is untouched, and the substitution is asserted to have happened.
ANON_AUTHOR_BLOCK = (
    # The placeholder deliberately uses the reserved .invalid TLD so it cannot
    # itself trip the email detector above, and cannot resolve to anyone.
    "\\author{\\name Anonymous Author(s) \\email anonymous@anon.invalid \\\\\n"
    "      \\addr Affiliation withheld for double-blind review}")
AUTHOR_BLOCK_RE = re.compile(r"\\author\{[^}]*\}[^\n]*\n(?:\s+\\addr[^\n]*\n)?")

# ── retracted numbers that must not survive anywhere in the archive ─────────
RETRACTED = [
    (r"14x and 52x|14\$\\times\$ and 52\$\\times\$|\$14\\times\$ and \$52\\times\$",
     "superseded 52x transport ratio (corrected to 38x)"),
    (r"falls (?:by )?48\.6x|a 48\.6x decline|\*\*48\.6x\*\*|by a factor of\s*\$?48\.6"
     r"|whose \$?48\.6|moves it by \$?48\.6|severity falls by 48\.6x",
     "48.6x operating-point multiplier (withdrawn: no finite CI)"),
    (r"identical AUROC \(\$?0\.9600\$? both\)",
     "FP16-vs-AWQ reported as identical at 0.9600"),
    (r"at AUROC \$?0\.734\$?(?:--|-|\u2013)\$?0\.776",
     "Case Study 2 separability quoted at an uncomputed 0.734-0.776"),
    (r"all-data AUROC picks L19|L19 under all-data AUROC",
     "all-data argmax layer stated as L19 (actual: L21)"),
    # ── Round 3 (the fourth independent review) ────────────────────────────
    # These are tagged ROUND3 below: the prose files that WITHDRAW them must be
    # able to quote them, so they are exempt for these patterns ONLY and stay
    # under the scan for every round-1/2 pattern. code/53's
    # check_absent_everywhere covers the same strings in those files with a
    # LaTeX-quotation-aware exemption, so coverage is not lost.
    (r"Rather Than Mechanism"
     r"|governed not by \\?\*?\*?emph\{?which\}?\*?\*? mechanism is responsible but by two"
     r"|rather than by which mechanism is responsible",
     "the withdrawn 'rather than mechanism' comparative claim (title/abstract; E18)"),
    (r"not sign-constrained|which is the property \$?\\Delta_\{?\\?text\{wc\}\}?\$? structurally cannot"
     r"|2 of the 24 cells return negative values|\$2\$ of \$24\$ cells negative",
     "the claim that the bootstrap max-bias estimator is not sign-constrained (Theorem 2; E15)"),
    (r"needs no degeneracy exclusion",
     "the claim that the promoted estimator needs no degeneracy exclusion (E15)"),
    (r"Wilcoxon \$?p=6\.0\\?times10\^\{?-7\}?|Wilcoxon p = 6\.0e-07|p=6\.0\\times10\^\{-7\}",
     "the withdrawn signed-rank p-value for a sign-constrained estimator (E15)"),
    (r"a 3\.7x difference|a \$3\.7\\times\$ difference",
     "the ceiling-composition multiplier computed on the retired estimator (E17)"),
    (r"24 independent published result files",
     "the '24 independent published result files' description of one crossed design (E19)"),
    (r"matched-pairs rank-biserial correlation",
     "the sign statistic mislabelled as a rank-biserial correlation"),
]
# Files allowed to name a retracted claim: the scripts and sections that
# withdraw it, and this build script itself.
RETRACTION_EXEMPT = {
    "code/60_operating_point_ratio_fieller_check.py",
    "code/61_case_study_2_half_membership_probe.py",
    "code/48_case_study_2_layer_decomposition.py",
    "code/53_verify_paper_numbers.py",
    "code/91_build_supplementary_zip.py",
    # The standalone correction-history record (moved out of the appendix to
    # cut page count): its entire purpose is to quote what was claimed and
    # explain the withdrawal, so every retracted string is legitimately
    # present here in a withdrawal context.
    "PROVENANCE_LOG.md",
}


# The last 7 RETRACTED entries are Round 3. The files that perform those
# withdrawals must be able to name the withdrawn claim; they remain under the
# scan for every earlier pattern.
ROUND3_PATTERN_COUNT = 7
ROUND3_EXEMPT = {
    "code/45_case_study_4_winners_curse.py",
    "code/69_operating_point_variance_control.py",
    "code/70_cs4_exact_enumeration.py",
    "code/71_cs4_estimator_calibration.py",
    "code/72_magnitude_triangle.py",
    "draft/latex/main.tex",
    "draft/paper_draft.md",
    "draft/case_study_quantized_llm_paper.md",
    "draft/case_study_multihaludet.md",
    "draft/worked_examples.md",
    "README.md",
}


def retracted_patterns_for(arcname):
    """RETRACTED, minus the Round-3 entries for files that withdraw them."""
    if arcname in ROUND3_EXEMPT:
        return RETRACTED[:-ROUND3_PATTERN_COUNT]
    return RETRACTED


def members():
    """Every file that goes into the archive, as (abs_path, arcname)."""
    out = []
    for name in ROOT_FILES:
        p = ROOT / name
        if p.exists():
            out.append((p, name))
    for d in DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            if any(part in EXCLUDE_PARTS for part in p.parts):
                continue
            if p.suffix in EXCLUDE_SUFFIX or p.name in EXCLUDE_NAMES:
                continue
            out.append((p, str(p.relative_to(ROOT))))
    return out


def read_text(p):
    try:
        return p.read_text(errors="replace")
    except Exception:
        return None


def anonymize(arc, text):
    """Text as it should appear INSIDE the archive."""
    if arc == "draft/latex/main.tex":
        # lambda, not a replacement string: the replacement is full of
        # backslashes and re would try to interpret them as group escapes.
        new, n = AUTHOR_BLOCK_RE.subn(lambda _m: ANON_AUTHOR_BLOCK + "\n", text, count=1)
        if n != 1:
            raise SystemExit("BUILD ABORTED: could not locate main.tex's \\author block to "
                             "anonymize. Check AUTHOR_BLOCK_RE against the current main.tex "
                             "rather than shipping a de-anonymized source.")
        return new
    return text


def _quoted_spans(text, latex=True, markdown=True):
    """Character spans of quotations, in which a retracted claim may be NAMED.

    A paper that withdraws its own number has to be able to quote the number it
    is withdrawing. LaTeX renders that as ``...''; code/52's markdown mirror
    renders the same thing as "..." with ASCII or curly quotes."""
    spans = []
    pairs = []
    if latex:
        pairs.append(("``", "''"))
    if markdown:
        pairs += [('"', '"'), ("\u201c", "\u201d")]
    for op, cl in pairs:
        i = text.find(op)
        while i != -1:
            j = text.find(cl, i + len(op))
            if j == -1:
                break
            spans.append((i, j + len(cl)))
            i = text.find(op, j + len(cl))
    return spans


def audit(files):
    """Returns a list of failure strings. Empty means the archive is clean."""
    failures = []

    # (a) identity leaks
    for p, arc in files:
        if arc.startswith(IDENTITY_EXEMPT_PREFIXES) or arc in IDENTITY_EXEMPT_FILES:
            continue
        if p.suffix not in TEXT_SUFFIX and p.name not in ROOT_FILES:
            continue
        text = read_text(p)
        if text is None:
            continue
        text = anonymize(arc, text)   # audit what SHIPS, not what is on disk
        for pat, why in IDENTITY_PATTERNS:
            for m in re.finditer(pat, text):
                line = text[:m.start()].count("\n") + 1
                failures.append(f"IDENTITY LEAK  {arc}:{line}  {why}: {m.group(0)!r}")

    # (b) retracted numbers
    for p, arc in files:
        if arc in RETRACTION_EXEMPT or arc.startswith(IDENTITY_EXEMPT_PREFIXES):
            continue
        if p.suffix not in TEXT_SUFFIX and p.name not in ROOT_FILES:
            continue
        text = read_text(p)
        if text is None:
            continue
        spans = _quoted_spans(text)
        for pat, why in retracted_patterns_for(arc):
            for m in re.finditer(pat, text):
                if any(a <= m.start() and m.end() <= b for a, b in spans):
                    continue   # quoted in order to be withdrawn
                line = text[:m.start()].count("\n") + 1
                failures.append(f"STALE NUMBER   {arc}:{line}  {why}: {m.group(0)!r}")

    return failures


def audit_existing_archive(path):
    """Audit an already-written archive: comment must be empty, members clean."""
    failures = []
    with zipfile.ZipFile(path) as z:
        if z.comment:
            failures.append(f"ARCHIVE COMMENT is non-empty: {z.comment!r}")
        for info in z.infolist():
            if info.comment:
                failures.append(f"MEMBER COMMENT on {info.filename}: {info.comment!r}")
            if (info.filename.startswith(IDENTITY_EXEMPT_PREFIXES)
                    or info.filename in IDENTITY_EXEMPT_FILES):
                continue
            if not any(info.filename.endswith(s) for s in TEXT_SUFFIX):
                continue
            try:
                text = z.read(info.filename).decode(errors="replace")
            except Exception:
                continue
            for pat, why in IDENTITY_PATTERNS:
                for m in re.finditer(pat, text):
                    failures.append(f"IN-ARCHIVE IDENTITY LEAK  {info.filename}  "
                                    f"{why}: {m.group(0)!r}")
            if info.filename in RETRACTION_EXEMPT:
                continue
            _spans = _quoted_spans(text)
            for pat, why in retracted_patterns_for(info.filename):
                for m in re.finditer(pat, text):
                    if any(a <= m.start() and m.end() <= b for a, b in _spans):
                        continue
                    failures.append(f"IN-ARCHIVE STALE NUMBER   {info.filename}  "
                                    f"{why}: {m.group(0)!r}")
    return failures


def main():
    check_only = "--check" in sys.argv
    files = members()
    print(f"manifest: {len(files)} files "
          f"({sum(1 for _, a in files if a.startswith('code/external/'))} vendored)")

    failures = audit(files)
    if failures:
        print(f"\nAUDIT FAILED ({len(failures)} problems); archive NOT written:")
        for f in failures[:60]:
            print("  " + f)
        if len(failures) > 60:
            print(f"  ... and {len(failures) - 60} more")
        sys.exit(1)
    print("source audit clean: no identity leaks, no retracted numbers")

    if check_only:
        if OUT.exists():
            post = audit_existing_archive(OUT)
            if post:
                print(f"\nEXISTING ARCHIVE FAILED ({len(post)} problems):")
                for f in post[:40]:
                    print("  " + f)
                sys.exit(1)
            print(f"existing archive clean: {OUT.name}")
        return

    tmp = OUT.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        # Explicitly empty archive comment. A zip comment is a classic
        # de-anonymization channel because no viewer shows it by default.
        z.comment = b""
        for p, arc in files:
            if arc == "draft/latex/main.tex":
                z.writestr(arc, anonymize(arc, p.read_text()))
            else:
                z.write(p, arc)

    post = audit_existing_archive(tmp)
    if post:
        tmp.unlink()
        print(f"\nPOST-BUILD AUDIT FAILED ({len(post)} problems); archive discarded:")
        for f in post[:40]:
            print("  " + f)
        sys.exit(1)

    tmp.replace(OUT)
    size = OUT.stat().st_size
    with zipfile.ZipFile(OUT) as z:
        n = len(z.namelist())
    print(f"post-build audit clean")
    print(f"wrote {OUT}  ({n} entries, {size / 1e6:.2f} MB, comment empty)")


if __name__ == "__main__":
    main()
