"""
MAXIMUM-RIGOR PASS, Item 5. §5's entire operating-point axis
(z_0 = Phi^{-1}(AUROC_0)) is calibrated via the closed-form BINORMAL identity,
AUROC_0 = Phi(delta/sqrt(2)), for isotropic (and, in the anisotropic sweep,
Gaussian) features. Normality itself is never relaxed. This script adds one
non-Gaussian generative process -- a two-component mixture of Gaussians per
class (a bimodal sub-population structure within each class, e.g. "easy" and
"hard" hallucination cases with different feature geometry) -- calibrated to
the SAME target AUROC_0 values via Monte Carlo bisection on the class-mean
separation, rather than the closed-form binormal identity (which does not
apply to a mixture), and reruns Sweep C (the operating-point sweep) under it.

GENERATIVE PROCESS. Per class y in {0,1}, features are a 50/50 mixture of two
Gaussian components, both isotropic (sigma=1) in a FEAT_DIM-dim space:
  y=0: component means at -delta/2 +/- skew*e2 (e2 orthogonal to the signal axis)
  y=1: component means at +delta/2 +/- skew*e2
The class-conditional marginal is then bimodal (a genuine departure from
normality -- a QQ-plot against a Gaussian on the signal axis is close to
normal by projection, but the joint density is not unimodal, and a linear
probe's residual is heavy-tailed/bimodal in the direction e2), while the
MEAN separation on the signal axis is still delta, controllable exactly as
before. delta is calibrated by bisection so a large Monte Carlo sample's
empirical AUROC (of the same linear direction e1 code/47 uses, not a
re-derived Bayes-optimal rule for the mixture) matches each target AUROC_0,
closing the loop code/47's docstring left open ("what if the data are not
Gaussian").

SIMPLIFICATION, DISCLOSED (same spirit as code/88's item-2 script): a single
reused validation fold and a single train/test split, reading each trained
MLP's own sigmoid AUROC directly, rather than code/47's full 5-fold-OOF-plus-
downstream-logistic-regression pipeline. This targets the same question --
does the monotone decline in gap with operating point survive under a
non-Gaussian generator -- at lower engineering and compute cost. An isotropic-
Gaussian run under the IDENTICAL simplified harness is included as an internal
control, so the comparison is non-Gaussian-vs-Gaussian under one script, not
against Sweep C's own absolute numbers.

Output: results/nongaussian_operating_point_sweep.json
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import bootstrap, norm, wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "nongaussian_operating_point_sweep.json"

CAPACITY = 128
FEAT_DIM = 64
EPOCHS = 45
N_SAMPLES = 700
ES_HOLD_FRACTION = 0.15
TEST_SIZE = 0.20
VAL_FRACTION = 0.20
N_SEEDS = 80
TARGET_GRID = [0.70, 0.80, 0.90, 0.95, 0.985]
SKEW = 2.5   # mixture-component offset on an orthogonal axis; bimodality strength


class SweepMLP(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_isotropic_gaussian(data_seed, n_samples, target_auroc):
    rng = np.random.default_rng(data_seed)
    j_target = 2 * (norm.ppf(target_auroc)) ** 2
    class_sep = np.sqrt(j_target / FEAT_DIM)
    n_pos = n_samples // 2; n_neg = n_samples - n_pos
    X_pos = class_sep / 2 + rng.standard_normal((n_pos, FEAT_DIM))
    X_neg = -class_sep / 2 + rng.standard_normal((n_neg, FEAT_DIM))
    X = np.vstack([X_pos, X_neg]).astype(np.float32)
    y = np.array([1] * n_pos + [0] * n_neg)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def _draw_mixture(rng, n, delta, dim, sign, skew):
    """n draws of the bimodal mixture for one class (sign=+1 or -1)."""
    comp = rng.integers(0, 2, size=n)
    mu = np.zeros((n, dim))
    mu[:, 0] = sign * delta / 2
    mu[:, 1] += np.where(comp == 0, skew, -skew)
    return mu + rng.standard_normal((n, dim))


def _achieved_auroc_mixture(delta, dim, skew, n=60_000, seed=0):
    rng = np.random.default_rng(seed)
    Xp = _draw_mixture(rng, n // 2, delta, dim, +1, skew)
    Xn = _draw_mixture(rng, n // 2, delta, dim, -1, skew)
    scores = np.concatenate([Xp[:, 0], Xn[:, 0]])   # code/47's convention: signal on axis 0
    y = np.array([1] * (n // 2) + [0] * (n // 2))
    return roc_auc_score(y, scores)


def calibrate_delta_mixture(target_auroc, dim=FEAT_DIM, skew=SKEW, tol=1e-3, max_iter=40):
    """Monte Carlo bisection -- NOT the closed-form binormal identity -- since
    a mixture's achieved AUROC under a linear score has no closed form here."""
    lo, hi = 0.0, 12.0
    for it in range(max_iter):
        mid = (lo + hi) / 2
        achieved = _achieved_auroc_mixture(mid, dim, skew, seed=1000 + it)
        if achieved < target_auroc:
            lo = mid
        else:
            hi = mid
        if abs(achieved - target_auroc) < tol:
            break
    return mid, achieved


def make_mixture_data(data_seed, n_samples, delta, skew=SKEW):
    rng = np.random.default_rng(data_seed)
    n_pos = n_samples // 2; n_neg = n_samples - n_pos
    Xp = _draw_mixture(rng, n_pos, delta, FEAT_DIM, +1, skew)
    Xn = _draw_mixture(rng, n_neg, delta, FEAT_DIM, -1, skew)
    X = np.vstack([Xp, Xn]).astype(np.float32)
    y = np.array([1] * n_pos + [0] * n_neg)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def roc_auc(y, scores):
    try:
        return float(roc_auc_score(y, scores))
    except ValueError:
        return 0.5


def train_to_best_checkpoint(X_tr, y_tr, X_sel, y_sel, hidden, epochs, init_seed):
    torch.manual_seed(init_seed)
    model = SweepMLP(X_tr.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=6e-5)
    crit = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32); yt = torch.tensor(y_tr, dtype=torch.float32)
    Xs = torch.tensor(X_sel, dtype=torch.float32)
    best_auc, best_state, best_epoch = -1.0, None, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = crit(model(Xt), yt); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            auc = roc_auc(y_sel, torch.sigmoid(model(Xs)).numpy())
        if auc > best_auc:
            best_auc, best_state, best_epoch = auc, {k: v.clone() for k, v in model.state_dict().items()}, ep + 1
    model.load_state_dict(best_state)
    return model, best_epoch


def train_fixed_epochs(X_tr, y_tr, hidden, n_epochs, init_seed):
    torch.manual_seed(init_seed)
    model = SweepMLP(X_tr.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=6e-5)
    crit = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32); yt = torch.tensor(y_tr, dtype=torch.float32)
    for _ in range(max(n_epochs, 1)):
        model.train(); opt.zero_grad()
        loss = crit(model(Xt), yt); loss.backward(); opt.step()
    model.eval()
    return model


def eval_auc(model, X, y):
    model.eval()
    with torch.no_grad():
        return roc_auc(y, torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy())


def run_one_seed(X, y, seed):
    split_seed, fold_seed, init_seed = seed + 100000, seed + 200000, seed + 300000
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=split_seed)
    scaler = StandardScaler(); X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)
    fit_idx, val_idx = train_test_split(np.arange(len(y_train)), test_size=VAL_FRACTION, stratify=y_train, random_state=fold_seed)
    tr2_idx, es_idx = train_test_split(fit_idx, test_size=ES_HOLD_FRACTION, stratify=y_train[fit_idx], random_state=fold_seed + 1)

    model_leaky, _ = train_to_best_checkpoint(X_train[fit_idx], y_train[fit_idx], X_train[val_idx], y_train[val_idx], CAPACITY, EPOCHS, init_seed)
    leaky_auc = eval_auc(model_leaky, X_test, y_test)

    model_clean, best_epoch = train_to_best_checkpoint(X_train[tr2_idx], y_train[tr2_idx], X_train[es_idx], y_train[es_idx], CAPACITY, EPOCHS, init_seed)
    model_clean_matched = train_fixed_epochs(X_train[fit_idx], y_train[fit_idx], CAPACITY, best_epoch, init_seed)
    clean_matched_auc = eval_auc(model_clean_matched, X_test, y_test)

    rng = np.random.default_rng(fold_seed + 2)
    y_val_permuted = rng.permutation(y_train[val_idx])
    model_placebo, _ = train_to_best_checkpoint(X_train[fit_idx], y_train[fit_idx], X_train[val_idx], y_val_permuted, CAPACITY, EPOCHS, init_seed)
    placebo_auc = eval_auc(model_placebo, X_test, y_test)

    return leaky_auc, clean_matched_auc, placebo_auc


def sweep_cell(gen_fn, n_seeds=N_SEEDS):
    leaky, clean_matched, placebo = [], [], []
    for seed in range(n_seeds):
        X, y = gen_fn(seed)
        l, c, p = run_one_seed(X, y, seed)
        leaky.append(l); clean_matched.append(c); placebo.append(p)
    leaky, clean_matched, placebo = map(np.array, (leaky, clean_matched, placebo))
    gap = leaky - clean_matched
    _, p_val = wilcoxon(gap) if not np.allclose(gap, 0) else (None, 1.0)
    res = bootstrap((gap,), np.mean, confidence_level=0.95, n_resamples=5000, method="BCa",
                    random_state=np.random.default_rng(9999))
    return {"leaky_mean": float(leaky.mean()), "clean_matched_mean": float(clean_matched.mean()),
            "placebo_mean": float(placebo.mean()), "gap_mean": float(gap.mean()),
            "gap_bca_ci_95": [float(res.confidence_interval.low), float(res.confidence_interval.high)],
            "wilcoxon_p": float(p_val), "n_positive": int((gap > 0).sum()), "n_seeds": n_seeds}


def main():
    t0 = time.time()
    print("=== Calibrating mixture delta via Monte Carlo bisection (not the binormal identity) ===")
    deltas = {}
    for target in TARGET_GRID:
        delta, achieved = calibrate_delta_mixture(target)
        deltas[target] = delta
        print(f"  target AUROC_0={target}: calibrated delta={delta:.4f} (achieved MC AUROC={achieved:.4f})")

    print("\n=== Non-Gaussian (bimodal mixture) operating-point sweep ===")
    mixture_out = {}
    for target in TARGET_GRID:
        delta = deltas[target]
        cell = sweep_cell(lambda seed, d=delta: make_mixture_data(seed, N_SAMPLES, d))
        mixture_out[str(target)] = cell
        print(f"  AUROC_0={target}: gap={cell['gap_mean']:+.4f} CI={cell['gap_bca_ci_95']} "
              f"p={cell['wilcoxon_p']:.4g} achieved_leaky={cell['leaky_mean']:.4f}  "
              f"elapsed={time.time()-t0:.0f}s", flush=True)

    print("\n=== Isotropic-Gaussian control, IDENTICAL simplified harness ===")
    gaussian_out = {}
    for target in TARGET_GRID:
        cell = sweep_cell(lambda seed, t=target: make_isotropic_gaussian(seed, N_SAMPLES, t))
        gaussian_out[str(target)] = cell
        print(f"  AUROC_0={target}: gap={cell['gap_mean']:+.4f} CI={cell['gap_bca_ci_95']} "
              f"p={cell['wilcoxon_p']:.4g}  elapsed={time.time()-t0:.0f}s", flush=True)

    gaps_mix = np.array([mixture_out[str(t)]["gap_mean"] for t in TARGET_GRID])
    gaps_gauss = np.array([gaussian_out[str(t)]["gap_mean"] for t in TARGET_GRID])
    monotone_mix = bool(np.all(np.diff(gaps_mix) <= 1e-9))
    monotone_gauss = bool(np.all(np.diff(gaps_gauss) <= 1e-9))

    out = {
        "design": {"capacity": CAPACITY, "epochs": EPOCHS, "n_samples": N_SAMPLES, "n_seeds": N_SEEDS,
                  "target_grid": TARGET_GRID, "skew": SKEW,
                  "calibrated_deltas": deltas,
                  "simplification_note": ("Single reused validation fold + single train/test split, "
                                          "direct sigmoid-AUROC readout (same simplification as "
                                          "code/88's item-2 script). Internal mixture-vs-Gaussian "
                                          "comparison only, not compared to Sweep C's absolute numbers.")},
        "mixture_nongaussian": mixture_out,
        "isotropic_gaussian_control": gaussian_out,
        "mixture_monotone_nonincreasing_in_auroc0": monotone_mix,
        "gaussian_control_monotone_nonincreasing_in_auroc0": monotone_gauss,
        "runtime_seconds": time.time() - t0,
    }
    verdict = (
        f"Non-Gaussian (bimodal-mixture) operating-point gaps across AUROC_0={TARGET_GRID} are "
        f"{[round(g, 5) for g in gaps_mix]}, "
        f"{'monotone non-increasing' if monotone_mix else 'NOT monotone'}. The identically-harnessed "
        f"isotropic-Gaussian control gives {[round(g, 5) for g in gaps_gauss]}, "
        f"{'monotone' if monotone_gauss else 'NOT monotone'}. "
        f"{'The monotone decline SURVIVES the non-Gaussian generative process.' if monotone_mix else 'The monotone decline DOES NOT cleanly survive the non-Gaussian generative process.'}"
    )
    out["verdict"] = verdict
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n{verdict}")
    print(f"\nSaved: {OUT_PATH}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
