"""
Reliability diagrams for Mechanism 2  [Reviewer DUxq, concern 1].

DUxq asked for "a Brier decomposition, calibration slope/intercept, reliability
diagrams, or an additional calibration-specific estimator." 103 supplies the
first, second and fourth. This supplies the third, which is the only one of the
four that shows WHERE in the probability range the reused-fold scores are
optimistic rather than only that they are.

Pooled across the 50 randomized splits at each split's own CV-argmax layer,
using the same cached per-sample scores as 103/104 (no new inference). Equal-
mass bins, so every point carries the same number of predictions and the visual
weight of a bin matches its statistical weight; equal-width bins are written to
the JSON as well for anyone who prefers them.

Emits figures/fig_reliability.pdf (and .png) plus
results/reliability_diagram.json.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "results" / "case_study_2_probe_scores.npz"
OUT_JSON = ROOT / "results" / "reliability_diagram.json"
FIG_DIR = ROOT / "figures"
N_BINS = 10


def pooled_scores(a, n_layers, n_reps):
    """Every split's predictions at that split's own selected layer, pooled."""
    leaky_s, leaky_y, clean_s, clean_y = [], [], [], []
    for rep in range(n_reps):
        v = f"rand{rep}"
        y_sel, y_ho = a[f"{v}__y_sel"], a[f"{v}__y_ho"]
        fold_id = a[f"{v}__fold_id"]
        cv, ho = a[f"{v}__cv_scores"], a[f"{v}__ho_scores"]
        folds = sorted(set(fold_id.tolist()))
        cv_auroc = np.array([
            np.mean([roc_auc_score(y_sel[fold_id == f], cv[l][fold_id == f])
                     for f in folds]) for l in range(n_layers)])
        l = int(cv_auroc.argmax())
        leaky_s.append(cv[l]); leaky_y.append(y_sel)
        clean_s.append(ho[l]); clean_y.append(y_ho)
    return (np.concatenate(leaky_s), np.concatenate(leaky_y),
            np.concatenate(clean_s), np.concatenate(clean_y))


def curve(scores, labels, n_bins, scheme):
    s = np.clip(np.asarray(scores, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(labels, dtype=float)
    if scheme == "mass":
        edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    pts = []
    for i in range(len(edges) - 1):
        m = (s >= edges[i]) & (s < edges[i + 1]) if i < len(edges) - 2 \
            else (s >= edges[i]) & (s <= edges[i + 1])
        if not m.any():
            continue
        pts.append({"bin": i, "n": int(m.sum()),
                    "mean_predicted": float(s[m].mean()),
                    "observed_frequency": float(y[m].mean())})
    return pts


def main():
    a = np.load(ARTIFACT)
    n_layers, n_reps = int(a["n_layers"]), int(a["n_reps"])
    ls, ly, cs, cy = pooled_scores(a, n_layers, n_reps)
    print(f"pooled: LEAKY n={len(ls):,}  CLEAN n={len(cs):,}  "
          f"base rates {ly.mean():.4f} / {cy.mean():.4f}")

    out = {"n_reps": n_reps, "n_bins": N_BINS,
           "n_pooled_leaky": int(len(ls)), "n_pooled_clean": int(len(cs)),
           "base_rate_leaky": float(ly.mean()), "base_rate_clean": float(cy.mean()),
           "note": "Pooled across splits at each split's own CV-argmax layer. "
                   "Pooling mixes splits, so these curves are descriptive; every "
                   "inferential statement in the paper is computed per split and "
                   "aggregated with BCa over splits (see 103).",
           "curves": {}}
    for scheme in ("mass", "width"):
        out["curves"][scheme] = {
            "leaky": curve(ls, ly, N_BINS, scheme),
            "clean": curve(cs, cy, N_BINS, scheme)}

    L = out["curves"]["mass"]["leaky"]
    C = out["curves"]["mass"]["clean"]

    # Where does the miscalibration sit? A selective-prediction threshold only
    # ever operates in the high-confidence region, so the deviation there is
    # the part with operational consequences -- this is what connects
    # the calibration result to the risk-violation result.
    def region(pts, lo):
        sel = [p for p in pts if p["mean_predicted"] >= lo]
        dev = [p["observed_frequency"] - p["mean_predicted"] for p in sel]
        return {"threshold": lo, "n_bins": len(sel),
                "n_predictions": int(sum(p["n"] for p in sel)),
                "mean_deviation": float(np.mean(dev)) if dev else None}

    out["high_confidence_region"] = {
        "leaky": region(L, 0.8), "clean": region(C, 0.8),
        "excess_overconfidence_clean_minus_leaky": float(
            region(C, 0.8)["mean_deviation"] - region(L, 0.8)["mean_deviation"]),
        "note": "Negative mean_deviation = overconfident (observed below "
                "predicted). A more negative value for CLEAN means the "
                "genuinely held-out predictions are more overconfident than "
                "the reused-fold ones precisely where an abstention threshold "
                "operates.",
    }
    hc = out["high_confidence_region"]
    print(f"\nhigh-confidence region (mean predicted >= 0.8):")
    print(f"  LEAKY mean deviation {hc['leaky']['mean_deviation']:+.4f} "
          f"over {hc['leaky']['n_predictions']:,} predictions")
    print(f"  CLEAN mean deviation {hc['clean']['mean_deviation']:+.4f} "
          f"over {hc['clean']['n_predictions']:,} predictions")
    print(f"  excess overconfidence {hc['excess_overconfidence_clean_minus_leaky']:+.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.9))

    ax[0].plot([0, 1], [0, 1], ls="--", lw=1, color="0.6", label="perfect")
    ax[0].plot([p["mean_predicted"] for p in L], [p["observed_frequency"] for p in L],
               "o-", color="0.35", lw=1.6, ms=5, label="LEAKY (reused fold)")
    ax[0].plot([p["mean_predicted"] for p in C], [p["observed_frequency"] for p in C],
               "s-", color="#c0392b", lw=1.6, ms=5, label="CLEAN (held out)")
    ax[0].set_xlabel("mean predicted probability")
    ax[0].set_ylabel("observed frequency")
    ax[0].set_title("Reliability, equal-mass bins", fontsize=10)
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1)

    # Deviation view: the gap is small on an absolute scale and is easy to
    # miss in the left panel, so plot observed-minus-predicted directly.
    ax[1].axhline(0, ls="--", lw=1, color="0.6")
    ax[1].plot([p["mean_predicted"] for p in L],
               [p["observed_frequency"] - p["mean_predicted"] for p in L],
               "o-", color="0.35", lw=1.6, ms=5, label="LEAKY")
    ax[1].plot([p["mean_predicted"] for p in C],
               [p["observed_frequency"] - p["mean_predicted"] for p in C],
               "s-", color="#c0392b", lw=1.6, ms=5, label="CLEAN")
    ax[1].set_xlabel("mean predicted probability")
    ax[1].set_ylabel("observed $-$ predicted")
    ax[1].set_title("Deviation from calibration", fontsize=10)
    ax[1].legend(fontsize=8)
    ax[1].set_xlim(0, 1)

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_reliability.{ext}", dpi=200, bbox_inches="tight")

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print("\nequal-mass bins (mean predicted -> observed):")
    for tag, pts in (("LEAKY", L), ("CLEAN", C)):
        print(f"  {tag}: " + "  ".join(
            f"{p['mean_predicted']:.2f}->{p['observed_frequency']:.2f}" for p in pts))
    print(f"\nSaved: {OUT_JSON} and {FIG_DIR}/fig_reliability.pdf")


if __name__ == "__main__":
    main()
