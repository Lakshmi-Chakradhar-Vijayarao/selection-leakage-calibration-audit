"""
how much of Mechanism 5's reported LEAKY-minus-HONEST gap is
THRESHOLD LEAKAGE, and how much is an UNDISCLOSED SELECTION-SET SIZE
ASYMMETRY between the two arms?

WHY THIS SCRIPT EXISTS. An independent review found that the Mechanism 5
comparison (code/46, and code/62 at the repo-faithful Youden rule) is not a
clean A/B on threshold leakage alone. The two arms select their thresholds on
selection sets of DIFFERENT SIZE:

    LEAKY   selects on the n_test = 140 points it is then scored on
    HONEST  selects on a disjoint n_val  = 112 validation split,
            and is then scored on the same 140 test points

(N_SAMPLES=700, TEST_SIZE=0.20 -> n_test=140; VAL_SIZE_OF_TRAIN=0.20 applied to
the remaining 560 -> n_val=112; verified against code/46's constants and
code/62's split path, not assumed.) That is a 140/112 = 1.25x difference in
selection-set size. A threshold chosen by argmax over a SMALLER, hence NOISIER,
criterion is a worse threshold in expectation -- the same winner's-curse-over-
noise logic §4.5 already invokes to explain why the gap shrinks with N. So the
asymmetry INFLATES the measured gap ON TOP OF the leakage it is meant to
isolate, and it inflates it in the direction of the reported finding.

The paper discloses the exactly analogous asymmetry for Mechanism 3 -- §4.3 and
A.3 name the "1.67x difference in selection-set size whose extra noise would
inflate the reported gap rather than merely accompany it" for the adaptivity
control -- but says nothing about the 1.25x one sitting inside Mechanism 5. The
disclosure standard the paper already holds itself to is not being applied
evenly, which is what this script exists to fix.

THE ISOLATION. Three arms, all on code/62's exact data/split/fit path, at
MultiHaluDet's own operating point (0.985) and at the four other cells, 200
seeds each:

    A  UNMATCHED (shipped)   test = 140   val = 112   train = 448
    B  MATCHED               test = 140   val = 140   train = 420
    C  BUDGET CONTROL        test = 140   val = 112   train = 420, where the
                             112 are the first 112 of B's own 140-point val

Arm A is asserted, per seed and per cell, to reproduce code/62's shipped
run_one_seed output field-for-field, so the pairing cannot silently drift.

Arms B and C share the SAME classifier, the SAME test split, the SAME LEAKY
threshold, and a NESTED validation split, so B - C is an exactly paired
isolation of selection-set size alone (140 vs 112) with nothing else moving.
That is the primary estimate. Size-matching inside a fixed N=700 costs 28
training samples (448 -> 420, a 6.25% budget deficit), which is what arm C
measures on its own; B - A is the total move and decomposes exactly as
(B - A) = (C - A) + (B - C). The budget deficit degrades the classifier
slightly and therefore pushes the matched gap UP (lower effective operating
point -> larger gaps, per §4.5's own AUROC trend), so the raw B - A comparison
UNDERSTATES how much the asymmetry contributes; the paired B - C figure is the
one to read.

SECOND, CHEAPER CHECK: IS THE ACCURACY GAP STRUCTURALLY NON-NEGATIVE TOO?
§4.5's non-negativity caveat is scoped only to F1 ("An earlier draft
additionally warned that the F1 gap is structurally guaranteed non-negative"),
and is withdrawn there because the Youden threshold is not an F1 argmax. But it
sits in a paragraph that also quotes accuracy gaps, and the withdrawal does not
transfer to accuracy: Youden J = tpr - fpr = 2*balanced_accuracy - 1, so
LEAKY's threshold IS the exact maximizer of balanced accuracy over the test
set's own ROC threshold set. On an exactly class-balanced test split raw
accuracy equals balanced accuracy, and any threshold HONEST could pick induces
the same predictions as some member of that set -- so the accuracy gap is
non-negative by algebra there, and can only go negative when the test split is
class-IMBALANCED. This script counts the negative accuracy gaps over all
5 x 200 = 1000 shipped replicates and splits that count by whether the test
split is exactly 70/70.

Output: results/mechanism5_size_matched.json
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
OUT_PATH = ROOT / "results" / "mechanism5_size_matched.json"

_spec = importlib.util.spec_from_file_location(
    "m62", CODE / "62_mechanism5_youden_threshold.py")
m62 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m62)
m46 = m62.m46

TARGET_AUROCS = m62.TARGET_AUROCS          # [0.70, 0.80, 0.90, 0.95, 0.985]
PRIMARY_TARGET = 0.985                     # MultiHaluDet's own reported regime
N_SEEDS = m62.N_SEEDS                      # 200
RNG_GLOBAL = np.random.default_rng(2026)

# Derived from code/46's constants, not hard-coded.
N_SAMPLES = m46.N_SAMPLES
N_TEST = int(N_SAMPLES * m46.TEST_SIZE)                              # 140
N_VAL_SHIPPED = int((N_SAMPLES - N_TEST) * m46.VAL_SIZE_OF_TRAIN)    # 112
N_VAL_MATCHED = N_TEST                                               # 140


def _scored(probs_test, yte, probs_sel, y_sel):
    """LEAKY (threshold on the test labels) and HONEST (threshold on the
    selection split) at code/62's metric conventions: F1 and accuracy at the
    repo-faithful Youden threshold, plus the F1-argmax column for the record."""
    leaky_t = m46.find_best_thresholds(probs_test, yte)
    honest_t = m46.find_best_thresholds(probs_sel, y_sel)

    def f1_at(t):
        return f1_score(yte, (probs_test >= t).astype(int), zero_division=0)

    def acc_at(t):
        return accuracy_score(yte, (probs_test >= t).astype(int))

    return {
        "leaky_f1_youden": f1_at(leaky_t["youden"]),
        "honest_f1_youden": f1_at(honest_t["youden"]),
        "leaky_f1_argmax": f1_at(leaky_t["f1"]),
        "honest_f1_argmax": f1_at(honest_t["f1"]),
        "leaky_acc": acc_at(leaky_t["youden"]),
        "honest_acc": acc_at(honest_t["youden"]),
    }


def run_one_seed(seed, target_auroc):
    """All three arms on one seed, on code/62's exact data/split/fit path."""
    X, y = m46.make_synthetic_data(seed, target_auroc)
    n = len(y)
    assert n == N_SAMPLES
    rng = np.random.default_rng(seed + 50000)          # identical to code/62
    idx = rng.permutation(n)
    test_idx = idx[:N_TEST]
    yte = y[test_idx]

    # --- arm A: shipped, unmatched (n_val=112, n_train=448) ----------------
    val_a = idx[N_TEST:N_TEST + N_VAL_SHIPPED]
    train_a = idx[N_TEST + N_VAL_SHIPPED:]
    clf_a = LogisticRegression(max_iter=2000).fit(X[train_a], y[train_a])
    pt_a = clf_a.predict_proba(X[test_idx])[:, 1]
    arm_a = _scored(pt_a, yte, clf_a.predict_proba(X[val_a])[:, 1], y[val_a])

    # --- arms B and C: one classifier, nested selection splits -------------
    val_b = idx[N_TEST:N_TEST + N_VAL_MATCHED]         # 140
    val_c = val_b[:N_VAL_SHIPPED]                      # the first 112 of val_b
    train_b = idx[N_TEST + N_VAL_MATCHED:]             # 420
    clf_b = LogisticRegression(max_iter=2000).fit(X[train_b], y[train_b])
    pt_b = clf_b.predict_proba(X[test_idx])[:, 1]
    arm_b = _scored(pt_b, yte, clf_b.predict_proba(X[val_b])[:, 1], y[val_b])
    arm_c = _scored(pt_b, yte, clf_b.predict_proba(X[val_c])[:, 1], y[val_c])

    return {
        "unmatched": arm_a, "matched": arm_b, "budget_control": arm_c,
        "n_pos_test": int(yte.sum()),
        "n_train": {"unmatched": len(train_a), "matched": len(train_b),
                    "budget_control": len(train_b)},
    }


def bca(x, n_resamples=10000):
    x = np.asarray(x)
    if x.std(ddof=1) == 0:      # degenerate: BCa is undefined, the CI is the point
        return [float(x.mean()), float(x.mean())]
    res = bootstrap((x,), np.mean, confidence_level=0.95,
                    n_resamples=n_resamples, method="BCa", random_state=RNG_GLOBAL)
    return [float(res.confidence_interval.low), float(res.confidence_interval.high)]


def wilcoxon_p(d):
    """Wilcoxon signed-rank p, with the all-zero case reported as 1.0 rather
    than raising (it carries no evidence either way)."""
    d = np.asarray(d)
    if np.all(d == 0):
        return 1.0
    return float(wilcoxon(d).pvalue)


def bca_paired_ratio(num_pair, den, n_resamples=10000):
    """BCa CI on mean(num_pair[0] - num_pair[1]) / mean(den), resampling seeds
    jointly so the ratio's numerator and denominator stay paired."""
    a, b = np.asarray(num_pair[0]), np.asarray(num_pair[1])
    den = np.asarray(den)

    def stat(a_, b_, d_, axis=-1):
        return (a_.mean(axis=axis) - b_.mean(axis=axis)) / d_.mean(axis=axis)

    res = bootstrap((a, b, den), stat, paired=True, confidence_level=0.95,
                    n_resamples=n_resamples, method="BCa", vectorized=True,
                    random_state=RNG_GLOBAL)
    return [float(res.confidence_interval.low), float(res.confidence_interval.high)]


def gap_block(g, label):
    g = np.asarray(g)
    p = wilcoxon_p(g)
    return {
        f"{label}_mean": float(g.mean()),
        f"{label}_sd": float(g.std(ddof=1)),
        f"{label}_bca_ci_95": bca(g),
        f"{label}_wilcoxon_p": float(p),
        f"{label}_n_negative": int((g < 0).sum()),
        f"{label}_n_zero": int((g == 0).sum()),
        f"{label}_n_positive": int((g > 0).sum()),
    }


def main():
    t0 = time.time()
    out = {
        "design": {
            "n_samples": N_SAMPLES,
            "n_test": N_TEST,
            "n_val_shipped": N_VAL_SHIPPED,
            "n_val_matched": N_VAL_MATCHED,
            "selection_set_size_ratio_shipped": N_TEST / N_VAL_SHIPPED,
            "n_train_shipped": N_SAMPLES - N_TEST - N_VAL_SHIPPED,
            "n_train_matched": N_SAMPLES - N_TEST - N_VAL_MATCHED,
            "n_seeds": N_SEEDS,
            "target_aurocs": TARGET_AUROCS,
            "primary_target_auroc": PRIMARY_TARGET,
            "arms": {
                "unmatched": ("shipped code/46+code/62 configuration: LEAKY selects on the "
                              f"{N_TEST} test points it is scored on, HONEST selects on a "
                              f"disjoint {N_VAL_SHIPPED}-point validation split. Classifier "
                              f"trained on {N_SAMPLES - N_TEST - N_VAL_SHIPPED}."),
                "matched": (f"size-matched: HONEST's selection split grown to {N_VAL_MATCHED} "
                            f"so both arms select on {N_TEST} points. Inside a fixed "
                            f"N={N_SAMPLES} this costs "
                            f"{N_SAMPLES - N_TEST - N_VAL_SHIPPED - (N_SAMPLES - N_TEST - N_VAL_MATCHED)} "
                            "training samples."),
                "budget_control": ("the matched arm's classifier and test split exactly, but "
                                   f"HONEST selects on the first {N_VAL_SHIPPED} of its own "
                                   f"{N_VAL_MATCHED}-point validation split. Isolates the "
                                   "training-budget cost of size-matching, so that "
                                   "matched - budget_control is a paired isolation of "
                                   "selection-set size alone."),
            },
        },
        "cells": {},
    }

    acc_reps = []   # (target, seed, acc_gap, n_pos_test) over the SHIPPED arm
    f1_reps = []

    for target in TARGET_AUROCS:
        rows = [run_one_seed(s, target) for s in range(N_SEEDS)]

        # --- drift guard: arm A must reproduce code/62 field-for-field ------
        for s, r in enumerate(rows):
            ref = m62.run_one_seed(s, target)
            for k, v in ref.items():
                assert abs(r["unmatched"][k] - v) < 1e-12, (
                    f"unmatched arm drifted from code/62 at target={target} seed={s} "
                    f"field={k}: {r['unmatched'][k]} vs {v}")

        cell = {"n_seeds": N_SEEDS}
        gaps = {}
        for arm in ("unmatched", "matched", "budget_control"):
            gy = np.array([r[arm]["leaky_f1_youden"] - r[arm]["honest_f1_youden"] for r in rows])
            ga = np.array([r[arm]["leaky_acc"] - r[arm]["honest_acc"] for r in rows])
            gx = np.array([r[arm]["leaky_f1_argmax"] - r[arm]["honest_f1_argmax"] for r in rows])
            gaps[arm] = {"f1_youden": gy, "acc": ga, "f1_argmax": gx}
            block = {}
            block.update(gap_block(gy, "f1_gap_youden"))
            block.update(gap_block(ga, "acc_gap"))
            block.update(gap_block(gx, "f1_gap_f1argmax"))
            block["leaky_f1_youden_mean"] = float(np.mean([r[arm]["leaky_f1_youden"] for r in rows]))
            block["honest_f1_youden_mean"] = float(np.mean([r[arm]["honest_f1_youden"] for r in rows]))
            block["leaky_acc_mean"] = float(np.mean([r[arm]["leaky_acc"] for r in rows]))
            block["honest_acc_mean"] = float(np.mean([r[arm]["honest_acc"] for r in rows]))
            block["n_train"] = rows[0]["n_train"][arm]
            block["n_val"] = N_VAL_MATCHED if arm == "matched" else N_VAL_SHIPPED
            cell[arm] = block

        # --- differences, all paired on seed ------------------------------
        diffs = {}
        for metric in ("f1_youden", "acc"):
            u, m, c = gaps["unmatched"][metric], gaps["matched"][metric], gaps["budget_control"][metric]
            d_total = m - u          # matched minus shipped
            d_size = m - c           # exactly paired: selection-set size only
            d_budget = c - u         # training-budget cost of size-matching
            p_total = wilcoxon_p(d_total)
            p_size = wilcoxon_p(d_size)
            diffs[metric] = {
                "gap_unmatched_mean": float(u.mean()),
                "gap_matched_mean": float(m.mean()),
                "gap_budget_control_mean": float(c.mean()),
                "matched_minus_unmatched_mean": float(d_total.mean()),
                "matched_minus_unmatched_bca_ci_95": bca(d_total),
                "matched_minus_unmatched_wilcoxon_p": float(p_total),
                "matched_minus_budget_control_mean": float(d_size.mean()),
                "matched_minus_budget_control_bca_ci_95": bca(d_size),
                "matched_minus_budget_control_wilcoxon_p": float(p_size),
                "budget_control_minus_unmatched_mean": float(d_budget.mean()),
                "budget_control_minus_unmatched_bca_ci_95": bca(d_budget),
                # fraction of the SHIPPED gap attributable to the size asymmetry
                "frac_attributable_to_size_paired": float((c.mean() - m.mean()) / u.mean()),
                "frac_attributable_to_size_paired_bca_ci_95":
                    bca_paired_ratio((c, m), u),
                "frac_attributable_raw_matched_vs_shipped": float((u.mean() - m.mean()) / u.mean()),
                "frac_attributable_raw_bca_ci_95": bca_paired_ratio((u, m), u),
                "paired": True,
                "pairing_note": ("All arms share the seed, the generated data, the permutation, "
                                 "and the test split, so every difference here is paired on seed. "
                                 "matched vs budget_control additionally share the classifier and "
                                 "the LEAKY threshold, and their selection splits are nested."),
            }
        cell["differences"] = diffs
        out["cells"][str(target)] = cell

        for s, r in enumerate(rows):
            acc_reps.append((target, s, r["unmatched"]["leaky_acc"] - r["unmatched"]["honest_acc"],
                             r["n_pos_test"]))
            f1_reps.append((target, s, r["unmatched"]["leaky_f1_youden"] - r["unmatched"]["honest_f1_youden"],
                            r["n_pos_test"]))

        d = diffs["f1_youden"]
        print(f"AUROC0={target}: F1@Youden gap  shipped={d['gap_unmatched_mean']:+.4f}  "
              f"matched={d['gap_matched_mean']:+.4f}  "
              f"(paired size effect {d['matched_minus_budget_control_mean']:+.4f}, "
              f"{100*d['frac_attributable_to_size_paired']:.1f}% of shipped)  "
              f"[{time.time()-t0:.0f}s]", flush=True)
        da = diffs["acc"]
        print(f"           acc gap        shipped={da['gap_unmatched_mean']:+.4f}  "
              f"matched={da['gap_matched_mean']:+.4f}  "
              f"({100*da['frac_attributable_to_size_paired']:.1f}% of shipped)", flush=True)

    # ── accuracy non-negativity over the shipped replicates ────────────────
    acc_g = np.array([r[2] for r in acc_reps])
    npos = np.array([r[3] for r in acc_reps])
    balanced = (2 * npos == N_TEST)
    f1_g = np.array([r[2] for r in f1_reps])
    neg_by_cell = {str(t): int(sum(1 for r in acc_reps if r[0] == t and r[2] < 0))
                   for t in TARGET_AUROCS}
    out["accuracy_nonnegativity"] = {
        "n_replicates": int(len(acc_g)),
        "n_negative": int((acc_g < 0).sum()),
        "n_zero": int((acc_g == 0).sum()),
        "n_positive": int((acc_g > 0).sum()),
        "n_negative_by_target_auroc": neg_by_cell,
        "n_balanced_replicates": int(balanced.sum()),
        "n_balanced_negative": int((acc_g[balanced] < 0).sum()),
        "min_acc_gap_balanced": float(acc_g[balanced].min()),
        "n_imbalanced_replicates": int((~balanced).sum()),
        "n_imbalanced_negative": int((acc_g[~balanced] < 0).sum()),
        "min_acc_gap_imbalanced": float(acc_g[~balanced].min()),
        "n_distinct_balanced_seeds": int(len({r[1] for r, b in zip(acc_reps, balanced) if b})),
        "test_pos_count_range": [int(npos.min()), int(npos.max())],
        "f1_youden_n_negative_for_contrast": int((f1_g < 0).sum()),
        "argument": (
            "Youden J = tpr - fpr = 2*balanced_accuracy - 1, so LEAKY's threshold is the exact "
            "maximizer of BALANCED accuracy over the test set's own ROC threshold set. On an "
            "exactly class-balanced test split raw accuracy equals balanced accuracy, and any "
            "threshold HONEST could select induces the same test predictions as some member of "
            "that set, so the accuracy gap is non-negative there by algebra. Negatives can only "
            "arise on class-imbalanced test splits, where raw accuracy weights the two error "
            "types unequally and Youden's objective is no longer the one being scored."),
        "note_on_replicate_vs_seed_counting": (
            "The test split is identical across the five operating points for a given seed "
            "(make_synthetic_data's permutation and the split RNG both depend only on the seed, "
            "not on target_auroc), so balanced replicates come in blocks of five. An independent "
            "check reporting '0 of 21 exactly-balanced' was counting distinct SEEDS; the "
            "replicate-level count is five times that."),
    }

    # ── verdict ────────────────────────────────────────────────────────────
    p = out["cells"][str(PRIMARY_TARGET)]["differences"]
    fr_f1 = p["f1_youden"]["frac_attributable_to_size_paired"]
    fr_acc = p["acc"]["frac_attributable_to_size_paired"]
    fr_all = [out["cells"][str(t)]["differences"]["f1_youden"]["frac_attributable_to_size_paired"]
              for t in TARGET_AUROCS]
    worst = max(fr_all)
    if worst < 0.10:
        verdict = "SIZE_ASYMMETRY_NEGLIGIBLE"
    elif worst < 0.33:
        verdict = "SIZE_ASYMMETRY_MINOR"
    else:
        verdict = "SIZE_ASYMMETRY_MATERIAL"
    acc_nn = out["accuracy_nonnegativity"]
    acc_verdict = ("ACCURACY_GAP_NONNEGATIVE_ON_BALANCED_SPLITS"
                   if acc_nn["n_balanced_negative"] == 0
                   else "ACCURACY_NONNEGATIVITY_ARGUMENT_FALSIFIED")
    out["verdict"] = f"{verdict}; {acc_verdict}"
    out["statement"] = (
        f"Mechanism 5's two arms select their decision thresholds on sets of different size: "
        f"LEAKY on the {N_TEST} test points it is scored on, HONEST on {N_VAL_SHIPPED} validation "
        f"points ({N_TEST/N_VAL_SHIPPED:.2f}x). Growing HONEST's selection split to {N_VAL_MATCHED} "
        f"so the two match, holding the classifier, the test split and LEAKY's threshold exactly "
        f"fixed, changes the F1-at-Youden gap at the audited pipeline's own operating point "
        f"({PRIMARY_TARGET}) from {p['f1_youden']['gap_budget_control_mean']:+.4f} to "
        f"{p['f1_youden']['gap_matched_mean']:+.4f}, i.e. {100*fr_f1:.1f}% of the shipped "
        f"{p['f1_youden']['gap_unmatched_mean']:+.4f} gap is attributable to the size asymmetry "
        f"rather than to threshold leakage; the accuracy gap moves by {100*fr_acc:.1f}% of its "
        f"shipped value. Across all five operating points the largest attributable share is "
        f"{100*worst:.1f}%. "
        + ("The asymmetry is real and should be disclosed the way §4.3 discloses Mechanism 3's "
           "1.67x one, but it does not carry the finding: the great majority of the gap survives "
           "size-matching."
           if worst < 0.33 else
           "This is a large share of the reported effect and the shipped Mechanism 5 numbers "
           "should be read as inflated by it.")
        + f" Separately, {acc_nn['n_negative']} of {acc_nn['n_replicates']} shipped replicates "
        f"have a negative accuracy gap, of which {acc_nn['n_balanced_negative']} of "
        f"{acc_nn['n_balanced_replicates']} exactly class-balanced replicates do -- consistent "
        f"with the algebraic argument that at the Youden threshold the accuracy gap is "
        f"non-negative whenever the test split is balanced, so §4.5's withdrawn non-negativity "
        f"caveat still applies to ACCURACY in the balanced case even though it no longer applies "
        f"to F1.")
    out["runtime_seconds"] = time.time() - t0

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print()
    print(f"verdict: {out['verdict']}")
    print(f"accuracy gaps negative: {acc_nn['n_negative']}/{acc_nn['n_replicates']}  "
          f"(balanced: {acc_nn['n_balanced_negative']}/{acc_nn['n_balanced_replicates']}, "
          f"imbalanced: {acc_nn['n_imbalanced_negative']}/{acc_nn['n_imbalanced_replicates']})")
    print(f"Saved: {OUT_PATH}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
