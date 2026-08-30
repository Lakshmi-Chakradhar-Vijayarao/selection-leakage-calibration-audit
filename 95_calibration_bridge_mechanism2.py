"""
UncertaiNLP calibration bridge, Mechanism 2 (GUARDIAN).

WHY THIS SCRIPT EXISTS. The paper measures selection-induced optimism only on
AUROC (a discrimination metric). It never asks whether the same CV-argmax
layer-selection mechanism also degrades CALIBRATION -- whether the selected
layer's probability outputs are honestly reliable, not just well-discriminating.
This is a directly new question, not a re-analysis of an existing number: does
selection-induced optimism inflate *confidence reliability* the same way it
inflates AUROC?

METHOD. Reuses results/case_study_2_probe_scores.npz (the same shipped
derived artifact as code/64), and the identical layer-selection rule
(l_star = argmax_l CV AUROC per rep, fold-averaged, exactly matching code/64).
For each of the 50 randomized-split reps:
  LEAKY:         cv_scores[l_star] scored against y_sel  (the reused fold's
                 own predictions -- the same data used to pick l_star)
  CLEAN_MATCHED: ho_scores[l_star] scored against y_ho   (genuinely held out,
                 played no role in selecting l_star)
Both Brier score (primary; well-defined at n=300-400, no binning) and 10-bin
ECE (secondary, standard in the calibration literature) are computed for each
condition. The severity of the calibration gap is BRIER_leaky - BRIER_clean
(lower Brier is better, so a positive gap means LEAKY looks *better calibrated*
than it honestly is), reported with the same BCa 95% bootstrap the rest of
this paper uses (scipy.stats.bootstrap, method="BCa", matching code/44's
bca_ci helper exactly).
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "results" / "case_study_2_probe_scores.npz"
REF_JSON = ROOT / "results" / "case_study_2_layer_decomposition.json"
OUT_PATH = ROOT / "results" / "calibration_bridge_mechanism2.json"

RNG = np.random.default_rng(20260811)
N_BINS = 10


def per_layer_cv_ho_scores(a, variant, n_layers):
    y_sel = a[f"{variant}__y_sel"]
    y_ho = a[f"{variant}__y_ho"]
    fold_id = a[f"{variant}__fold_id"]
    cv_scores = a[f"{variant}__cv_scores"]
    ho_scores = a[f"{variant}__ho_scores"]
    folds = sorted(set(fold_id.tolist()))
    cv_auroc = np.array([
        np.mean([roc_auc_score(y_sel[fold_id == f], cv_scores[l][fold_id == f])
                 for f in folds]) for l in range(n_layers)])
    return cv_auroc, y_sel, y_ho, cv_scores, ho_scores


def ece_binned(scores, labels, n_bins=N_BINS):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(scores)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        if mask.sum() == 0:
            continue
        conf = scores[mask].mean()
        acc = labels[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def bca_ci(a, b):
    diff = np.asarray(a) - np.asarray(b)
    res = bootstrap((diff,), np.mean, confidence_level=0.95, n_resamples=10000,
                     method="BCa", random_state=RNG)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def main():
    a = np.load(ARTIFACT)
    n_layers = int(a["n_layers"])
    n_reps = int(a["n_reps"])
    ref = json.load(open(REF_JSON))["randomized_stratified_splits"]

    brier_leaky, brier_clean = [], []
    ece_leaky, ece_clean = [], []
    auroc_leaky, auroc_clean = [], []
    l_stars = []

    for r in range(n_reps):
        cv_auroc, y_sel, y_ho, cv_scores, ho_scores = per_layer_cv_ho_scores(a, f"rand{r}", n_layers)
        l_star = int(cv_auroc.argmax())
        l_stars.append(l_star)

        s_leaky = cv_scores[l_star]
        s_clean = ho_scores[l_star]

        brier_leaky.append(brier_score_loss(y_sel, s_leaky))
        brier_clean.append(brier_score_loss(y_ho, s_clean))
        ece_leaky.append(ece_binned(s_leaky, y_sel))
        ece_clean.append(ece_binned(s_clean, y_ho))
        auroc_leaky.append(roc_auc_score(y_sel, s_leaky))
        auroc_clean.append(roc_auc_score(y_ho, s_clean))

    assert l_stars == list(ref["selected_layer_per_rep"]), "selected layers differ from code/64's reference -- layer-selection replication failed"
    print(f"Layer-selection replication OK: {n_reps} reps match code/64's reference selected layers.\n")

    brier_leaky, brier_clean = np.array(brier_leaky), np.array(brier_clean)
    ece_leaky, ece_clean = np.array(ece_leaky), np.array(ece_clean)
    auroc_leaky, auroc_clean = np.array(auroc_leaky), np.array(auroc_clean)

    # Positive gap = LEAKY looks better-calibrated than it honestly is
    # (lower Brier/ECE is better, so leaky-minus-clean POSITIVE means clean's
    # error is higher -- i.e. LEAKY's own-fold score looks artificially good).
    brier_gap = brier_clean - brier_leaky
    ece_gap = ece_clean - ece_leaky
    auroc_gap = auroc_leaky - auroc_clean

    brier_lo, brier_hi = bca_ci(brier_clean, brier_leaky)
    ece_lo, ece_hi = bca_ci(ece_clean, ece_leaky)
    auroc_lo, auroc_hi = bca_ci(auroc_leaky, auroc_clean)

    out = {
        "n_reps": n_reps,
        "mechanism": "Mechanism 2 (GUARDIAN, CV-argmax layer selection) -- calibration bridge",
        "method": "Brier score and 10-bin ECE at the CV-selected layer l_star per rep, "
                  "LEAKY = own-fold scores (cv_scores, y_sel), CLEAN_MATCHED = genuinely "
                  "held-out scores (ho_scores, y_ho), same l_star for both.",
        "brier_leaky_mean": float(brier_leaky.mean()),
        "brier_clean_mean": float(brier_clean.mean()),
        "brier_gap_mean": float(brier_gap.mean()),
        "brier_gap_bca_95ci": [brier_lo, brier_hi],
        "ece_leaky_mean": float(ece_leaky.mean()),
        "ece_clean_mean": float(ece_clean.mean()),
        "ece_gap_mean": float(ece_gap.mean()),
        "ece_gap_bca_95ci": [ece_lo, ece_hi],
        "auroc_leaky_mean": float(auroc_leaky.mean()),
        "auroc_clean_mean": float(auroc_clean.mean()),
        "auroc_gap_mean": float(auroc_gap.mean()),
        "auroc_gap_bca_95ci": [auroc_lo, auroc_hi],
        "auroc_gap_matches_paper_delta_sel": float(auroc_gap.mean()),
    }

    print(f"AUROC gap (sanity check vs paper's Delta_sel=+0.0255): {auroc_gap.mean():+.4f}  BCa 95% [{auroc_lo:+.4f},{auroc_hi:+.4f}]")
    print(f"Brier gap (clean - leaky):  {brier_gap.mean():+.4f}  BCa 95% [{brier_lo:+.4f},{brier_hi:+.4f}]")
    print(f"  Brier LEAKY mean={brier_leaky.mean():.4f}  CLEAN mean={brier_clean.mean():.4f}")
    print(f"ECE gap (clean - leaky):    {ece_gap.mean():+.4f}  BCa 95% [{ece_lo:+.4f},{ece_hi:+.4f}]")
    print(f"  ECE LEAKY mean={ece_leaky.mean():.4f}  CLEAN mean={ece_clean.mean():.4f}")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
