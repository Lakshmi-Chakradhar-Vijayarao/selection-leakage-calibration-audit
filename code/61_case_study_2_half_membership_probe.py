"""
computational provenance for Case Study 2's central correction.

An independent review grepped every code/*.py and results/*.json in this
repository and found that the claim §4.2's retraction rests on --

    "the two halves of GUARDIAN's sequential split are separable by a
     hidden-state probe at AUROC 0.734-0.776"

-- existed ONLY as a hardcoded string in `code/48`'s docstring (line 18) and in
a hardcoded JSON `note` field (`code/48` line ~350). No half-membership probe
was fitted anywhere. The same was true of "argmax over all-data AUROC picks
L19". Both were prose. This script computes them, for the first time, and ships
the result.

It also measures a SIMPLER confound the paper had not reported, which the same
review identified and which is directly checkable in the already-shipped
`results/case_study_2_probe_scores.npz`: the two halves differ in LABEL BASE
RATE by 8 percentage points. That is sufficient on its own to make the halves
separable and to make their held-out AUROCs non-comparable, by a shorter route
than a general population difference.

Four computations, all shipped to
`results/case_study_2_half_membership_probe.json`:

  (1) HALF-MEMBERSHIP PROBE. Per layer, fit a logistic probe on the raw
      hidden states to predict "is this sample in H[:400] (the CV-selection
      half) or in H[400:] (the held-out half)". Report stratified 5-fold
      out-of-fold AUROC per layer, plus the range across layers. This is the
      number §4.2 quotes; it has never before been computed.

  (2) ALL-DATA ARGMAX LAYER. Per layer, 5-fold CV AUROC for the hallucination
      label over all 700 samples; report the argmax. This is the "argmax
      all-data AUROC picks L19" claim.

  (3) LABEL BASE RATES. Positive rate in each half of the sequential split,
      and in the randomized comparison reps, read directly from the shipped
      replay artifact -- so this part is verifiable without the 171 MB raw
      cache.

  (4) A BASE-RATE-MATCHED SEQUENTIAL CONTROL. Re-run the sequential split
      after subsampling both halves to a common positive rate, and re-measure
      the selection-specific component. If the sequential-split anomaly is
      driven by the base-rate difference, matching should move it.

The raw cache (700, 32, 4096) is required for (1), (2) and (4); (3) needs only
the shipped artifact. If the cache is absent the script still runs (3) and
records the others as unavailable, rather than failing.
"""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
GUARDIAN_ROOT = Path(os.path.expanduser(os.environ.get("GUARDIAN_ROOT", "~/Desktop/guardian")))
GUARDIAN_NPZ = GUARDIAN_ROOT / "results" / "hidden_states" / "mistral_7b_tqa_hidden_states.npz"
ARTIFACT_PATH = ROOT / "results" / "case_study_2_probe_scores.npz"
OUT_PATH = ROOT / "results" / "case_study_2_half_membership_probe.json"

N_TRAIN_SELECT = 400
RANDOM_STATE = 42
N_MATCH_DRAWS = 20


def _load_48():
    spec = importlib.util.spec_from_file_location(
        "s48", ROOT / "code" / "48_case_study_2_layer_decomposition.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def oof_auroc(X, target, n_splits=5, seed=RANDOM_STATE, C=1.0):
    """Out-of-fold AUROC of a logistic probe, matching code/48's probe config
    (unscaled raw hidden states, max_iter=1000, C=1.0)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = np.zeros(len(target), dtype=float)
    for tr, te in cv.split(X, target):
        clf = LogisticRegression(max_iter=1000, C=C).fit(X[tr], target[tr])
        scores[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(target, scores)), scores


def base_rates_from_artifact():
    a = np.load(ARTIFACT_PATH)
    seq_sel = a["sequential__y_sel"].astype(float)
    seq_ho = a["sequential__y_ho"].astype(float)
    rand_sel, rand_ho = [], []
    r = 0
    while f"rand{r}__y_sel" in a.files:
        rand_sel.append(float(a[f"rand{r}__y_sel"].mean()))
        rand_ho.append(float(a[f"rand{r}__y_ho"].mean()))
        r += 1
    return {
        "sequential_selection_half_positive_rate": float(seq_sel.mean()),
        "sequential_selection_half_n": int(seq_sel.size),
        "sequential_heldout_half_positive_rate": float(seq_ho.mean()),
        "sequential_heldout_half_n": int(seq_ho.size),
        "sequential_base_rate_difference_pp": float(
            100 * (seq_ho.mean() - seq_sel.mean())),
        "n_randomized_reps": r,
        "randomized_selection_half_positive_rate_mean": float(np.mean(rand_sel)),
        "randomized_heldout_half_positive_rate_mean": float(np.mean(rand_ho)),
        "randomized_base_rate_difference_pp_mean": float(
            100 * (np.mean(rand_ho) - np.mean(rand_sel))),
        "randomized_base_rate_difference_pp_max_abs": float(
            100 * np.max(np.abs(np.array(rand_ho) - np.array(rand_sel)))),
        "note": ("The randomized comparison reps are stratified, so their halves match "
                 "in base rate to within a fraction of a point. The sequential split, "
                 "which is what GUARDIAN actually ran, does not."),
    }


def base_rate_matched_sequential(m, H, y, rng):
    """Sequential split, subsampled so both halves carry the same positive rate.

    Matching is done downward to the lower of the two rates: positives are
    dropped at random from whichever half is positive-enriched, so no sample is
    reweighted or duplicated. Repeated over N_MATCH_DRAWS draws because which
    positives are dropped is arbitrary."""
    sel_idx_all = np.arange(N_TRAIN_SELECT)
    ho_idx_all = np.arange(N_TRAIN_SELECT, len(y))
    p_sel, p_ho = y[sel_idx_all].mean(), y[ho_idx_all].mean()
    target = float(min(p_sel, p_ho))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    comps, l_stars, achieved = [], [], []
    for d in range(N_MATCH_DRAWS):
        keep = []
        for idx in (sel_idx_all, ho_idx_all):
            pos = idx[y[idx] == 1]
            neg = idx[y[idx] == 0]
            # keep all negatives; keep n_pos so that n_pos/(n_pos+n_neg) == target
            n_pos_keep = int(round(target * len(neg) / (1.0 - target)))
            n_pos_keep = min(n_pos_keep, len(pos))
            keep.append(np.sort(np.concatenate([
                rng.choice(pos, size=n_pos_keep, replace=False), neg])))
        s_idx, h_idx = keep
        achieved.append([float(y[s_idx].mean()), float(y[h_idx].mean())])
        per_layer = m.per_layer_gaps(H[s_idx], y[s_idx], H[h_idx], y[h_idx],
                                     H.shape[1], cv, variant=None)
        l_star = int(np.argmax([r["cv_auroc_selectpool"] for r in per_layer]))
        comp, _ = m.selection_specific_component(per_layer, l_star)
        comps.append(comp)
        l_stars.append(l_star)
        print(f"    matched draw {d+1}/{N_MATCH_DRAWS}: L*={l_star:2d} "
              f"sel-specific={comp:+.4f}", flush=True)
    return {
        "n_draws": N_MATCH_DRAWS,
        "target_positive_rate": target,
        "achieved_positive_rates_per_draw": achieved,
        "selection_specific_component_per_draw": [float(c) for c in comps],
        "selection_specific_component_mean": float(np.mean(comps)),
        "selection_specific_component_sd": float(np.std(comps, ddof=1)),
        "selected_layer_per_draw": l_stars,
    }


def main():
    # Record the cache location RELATIVE TO $HOME. An absolute path leaks the
    # local username into a shipped artifact, which is a double-blind leak (and
    # exactly the class of thing code/91 now refuses to ship).
    try:
        _rel = "~/" + str(GUARDIAN_NPZ.relative_to(Path.home()))
    except ValueError:
        _rel = GUARDIAN_NPZ.name
    out = {"raw_cache_path_relative_to_home": _rel,
           "raw_cache_env_var": "GUARDIAN_ROOT",
           "raw_cache_available": GUARDIAN_NPZ.exists()}

    print("(3) label base rates, from the shipped replay artifact ...", flush=True)
    out["label_base_rates"] = base_rates_from_artifact()
    br = out["label_base_rates"]
    print(f"    sequential: selection half {br['sequential_selection_half_positive_rate']:.4f} "
          f"(n={br['sequential_selection_half_n']}) vs held-out "
          f"{br['sequential_heldout_half_positive_rate']:.4f} "
          f"(n={br['sequential_heldout_half_n']})  -> "
          f"{br['sequential_base_rate_difference_pp']:+.1f} pp", flush=True)
    print(f"    randomized reps: {br['randomized_selection_half_positive_rate_mean']:.4f} vs "
          f"{br['randomized_heldout_half_positive_rate_mean']:.4f} "
          f"(max |diff| {br['randomized_base_rate_difference_pp_max_abs']:.2f} pp)", flush=True)

    if not GUARDIAN_NPZ.exists():
        out["half_membership_probe"] = {"status": "UNAVAILABLE: raw hidden-state cache not found"}
        out["all_data_argmax_layer"] = {"status": "UNAVAILABLE: raw hidden-state cache not found"}
        json.dump(out, open(OUT_PATH, "w"), indent=2)
        print(f"\nwrote {OUT_PATH} (base rates only)")
        return

    m = _load_48()
    d = np.load(GUARDIAN_NPZ)
    H, y = d["hidden_states"], d["labels"]
    valid = y >= 0
    H, y = H[valid], y[valid]
    n_layers = H.shape[1]
    print(f"\nLoaded raw cache: H={H.shape}, n={len(y)}, hall_rate={y.mean():.4f}", flush=True)

    # ── (1) half-membership probe ───────────────────────────────────────────
    print("\n(1) half-membership probe (is this sample in H[:400] or H[400:]?) ...", flush=True)
    half = np.zeros(len(y), dtype=int)
    half[N_TRAIN_SELECT:] = 1
    per_layer_half = []
    for l in range(n_layers):
        auc, _ = oof_auroc(H[:, l, :], half)
        per_layer_half.append({"layer": l, "half_membership_auroc": auc})
        print(f"    L{l:02d}: half-membership AUROC = {auc:.4f}", flush=True)
    aucs = np.array([r["half_membership_auroc"] for r in per_layer_half])
    out["half_membership_probe"] = {
        "status": "COMPUTED",
        "target": "1 if sample index >= 400 (held-out half), else 0",
        "probe": "LogisticRegression(max_iter=1000, C=1.0) on unscaled raw hidden states",
        "evaluation": "stratified 5-fold out-of-fold AUROC, random_state=42",
        "per_layer": per_layer_half,
        "min_auroc": float(aucs.min()), "max_auroc": float(aucs.max()),
        "mean_auroc": float(aucs.mean()), "median_auroc": float(np.median(aucs)),
        "argmin_layer": int(aucs.argmin()), "argmax_layer": int(aucs.argmax()),
        "n_layers_above_0_70": int((aucs > 0.70).sum()),
        "note": ("This is the quantity §4.2 previously asserted without computing it. "
                 "An AUROC materially above 0.5 means the two halves are systematically "
                 "different populations, so the held-out half is not an exchangeable "
                 "draw and the sequential-split optimism gap conflates selection "
                 "optimism with a population difference."),
    }

    # ── (2) all-data argmax layer ───────────────────────────────────────────
    print("\n(2) all-data (n=700) per-layer CV AUROC for the hallucination label ...", flush=True)
    per_layer_all = []
    for l in range(n_layers):
        auc, _ = oof_auroc(H[:, l, :], y.astype(int))
        per_layer_all.append({"layer": l, "all_data_cv_auroc": auc})
        print(f"    L{l:02d}: all-data CV AUROC = {auc:.4f}", flush=True)
    a_all = np.array([r["all_data_cv_auroc"] for r in per_layer_all])
    out["all_data_argmax_layer"] = {
        "status": "COMPUTED",
        "n_samples": int(len(y)),
        "per_layer": per_layer_all,
        "argmax_layer": int(a_all.argmax()),
        "argmax_auroc": float(a_all.max()),
        "note": ("The layer an argmax over all 700 samples would select. §4.2 previously "
                 "asserted a value for this in prose with no computation behind it."),
    }

    # ── (4) base-rate-matched sequential control ────────────────────────────
    print("\n(4) base-rate-matched sequential control ...", flush=True)
    rng = np.random.default_rng(20260803)
    out["base_rate_matched_sequential_control"] = base_rate_matched_sequential(m, H, y, rng)

    # unmatched sequential reference, recomputed here so the comparison is like-for-like
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pl = m.per_layer_gaps(H[:N_TRAIN_SELECT], y[:N_TRAIN_SELECT],
                          H[N_TRAIN_SELECT:], y[N_TRAIN_SELECT:], n_layers, cv, variant=None)
    l_star = int(np.argmax([r["cv_auroc_selectpool"] for r in pl]))
    comp, _ = m.selection_specific_component(pl, l_star)
    out["unmatched_sequential_reference"] = {
        "selected_layer": l_star, "selection_specific_component": float(comp)}
    print(f"    unmatched sequential: L*={l_star}, sel-specific={comp:+.4f}", flush=True)

    json.dump(out, open(OUT_PATH, "w"), indent=2)
    print(f"\nwrote {OUT_PATH}")
    print(f"\nSUMMARY  half-membership AUROC range = "
          f"[{aucs.min():.4f}, {aucs.max():.4f}] (mean {aucs.mean():.4f})")
    print(f"         all-data argmax layer = L{a_all.argmax()} ({a_all.max():.4f})")
    print(f"         base-rate gap = {br['sequential_base_rate_difference_pp']:+.1f} pp")
    mc = out["base_rate_matched_sequential_control"]
    print(f"         sel-specific: unmatched {comp:+.4f} -> base-rate-matched "
          f"{mc['selection_specific_component_mean']:+.4f} "
          f"(SD {mc['selection_specific_component_sd']:.4f})")


if __name__ == "__main__":
    main()
