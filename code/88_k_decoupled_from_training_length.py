"""
MAXIMUM-RIGOR PASS, Item 2. In Sweep A (§5.3, code/47), K literally
IS EPOCHS: each K cell trains a fresh model to exactly K total epochs, so
"more candidates" and "more training" are the same manipulation. This script
decouples them: train ONE model to a FIXED total budget (225 epochs, Sweep
A's own maximum) per (seed, fold-analogue), then vary ONLY which of those 225
checkpoints are ELIGIBLE for the argmax selection rule, by subsampling the
candidate set at strides {1, 3, 15, 45} (giving effective candidate counts of
225, 75, 15, 5 -- matching four of Sweep A's own K values exactly, so the two
designs are directly comparable cell-for-cell). If the K-regularity survives
this decoupling it is a property of candidate COUNT; if it vanishes or
reshuffles, §5.3's headline is at least partly a training-LENGTH effect
masquerading as a candidate-count effect.

TRAINING COST. Per seed, exactly two 225-epoch trajectories are trained (one
for LEAKY's reused-fold selection statistic, one for the honest arm's disjoint
carve-out selection statistic) -- NOT four separate runs per stride -- plus one
cheap fixed-epoch retrain per stride for the honest arm (unavoidable, since a
different stride selects a different checkpoint depth to retrain to; this is
the same per-cell cost Sweep A already pays for each K value it reports). No
run trains beyond 225 epochs at any stride.

SIMPLIFICATION, DISCLOSED. Unlike code/47's full 5-fold-OOF-plus-downstream-
logistic-regression pipeline, this script reads each trained checkpoint's own
sigmoid output AUROC directly on a single held-out test split (one reused
validation fold, one train/test split, no cross-fold feature averaging). This
is a simpler design chosen to make full-trajectory recording and per-stride
retraining tractable without re-deriving the OOF-assembly step, and it targets
the same question -- does the K-regularity survive decoupling K from training
length -- at lower engineering risk. It is not a drop-in replacement for
Sweep A's own numbers and is not compared to them at face value; the
comparison drawn below is internal, between this script's own coupled-vs-
decoupled cells at the same nominal K.

A COUPLED BASELINE IS INCLUDED for direct comparison: the same generator,
architecture and calibration, but training to K total epochs (K = the nominal
candidate count itself), exactly mirroring what Sweep A does structurally
(simplified to the same single-fold readout above, so the coupled-vs-decoupled
contrast is apples-to-apples within this script, not against Sweep A's
absolute numbers).

Output: results/k_decoupled_from_training_length.json
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import bootstrap, norm, wilcoxon
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "k_decoupled_from_training_length.json"

CAPACITY = 128
FEAT_DIM = 64
TOTAL_EPOCHS = 225        # fixed training budget for the decoupled design
DEFAULT_N_SAMPLES = 700
DEFAULT_TARGET_AUROC = 0.80
ES_HOLD_FRACTION = 0.15
TEST_SIZE = 0.20
VAL_FRACTION = 0.20       # of the training pool, matching n_val~112 (Sweep D's K_CV=5 default)
STRIDES = {1: 225, 3: 75, 15: 15, 45: 5}   # stride -> nominal effective K
N_SEEDS = 100


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


def make_synthetic_data(data_seed, n_samples, target_auroc):
    """Identical convention to code/47's make_synthetic_data."""
    rng = np.random.default_rng(data_seed)
    j_target = 2 * (norm.ppf(target_auroc)) ** 2
    class_sep = np.sqrt(j_target / FEAT_DIM)
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos
    X_pos = class_sep / 2 + rng.standard_normal((n_pos, FEAT_DIM))
    X_neg = -class_sep / 2 + rng.standard_normal((n_neg, FEAT_DIM))
    X = np.vstack([X_pos, X_neg]).astype(np.float32)
    y = np.array([1] * n_pos + [0] * n_neg)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def roc_auc(y, scores):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, scores))
    except ValueError:
        return 0.5


def train_full_trajectory(X_tr, y_tr, X_val, y_val, X_test, y_test, hidden, total_epochs, init_seed):
    """Train ONE model to `total_epochs`, recording val AUROC and test AUROC
    at EVERY epoch. Returns (val_traj, test_traj), each length total_epochs."""
    torch.manual_seed(init_seed)
    model = SweepMLP(X_tr.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=6e-5)
    crit = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    Xte = torch.tensor(X_test, dtype=torch.float32)
    val_traj, test_traj = np.zeros(total_epochs), np.zeros(total_epochs)
    for ep in range(total_epochs):
        model.train(); opt.zero_grad()
        loss = crit(model(Xt), yt); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val_traj[ep] = roc_auc(y_val, torch.sigmoid(model(Xv)).numpy())
            test_traj[ep] = roc_auc(y_test, torch.sigmoid(model(Xte)).numpy())
    return val_traj, test_traj


def train_fixed_epochs_readout(X_tr, y_tr, X_test, y_test, hidden, n_epochs, init_seed):
    torch.manual_seed(init_seed)
    model = SweepMLP(X_tr.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=6e-5)
    crit = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    Xte = torch.tensor(X_test, dtype=torch.float32)
    for _ in range(max(n_epochs, 1)):
        model.train(); opt.zero_grad()
        loss = crit(model(Xt), yt); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return roc_auc(y_test, torch.sigmoid(model(Xte)).numpy())


def eligible_epochs(stride, total_epochs=TOTAL_EPOCHS):
    """1-indexed epoch numbers eligible for the argmax, every `stride`-th one,
    always including the final epoch so the honest arm always has a candidate
    at the training budget's end."""
    idx = list(range(stride, total_epochs + 1, stride))
    if idx[-1] != total_epochs:
        idx.append(total_epochs)
    return np.array(idx)


def run_one_seed_decoupled(seed):
    data_seed, split_seed, fold_seed, init_seed_leaky, init_seed_clean = (
        seed, seed + 100000, seed + 200000, seed + 300000, seed + 400000)
    X, y = make_synthetic_data(data_seed, DEFAULT_N_SAMPLES, DEFAULT_TARGET_AUROC)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=split_seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)

    # reused fold (LEAKY's own validation set, reported on downstream)
    fit_idx, val_idx = train_test_split(
        np.arange(len(y_train)), test_size=VAL_FRACTION, stratify=y_train, random_state=fold_seed)
    # disjoint carve-out for the honest arm's selection (matches ES_HOLD_FRACTION)
    tr2_idx, es_idx = train_test_split(
        fit_idx, test_size=ES_HOLD_FRACTION, stratify=y_train[fit_idx], random_state=fold_seed + 1)

    # ONE 225-epoch trajectory for LEAKY: trained on fit_idx, val AUROC on
    # val_idx (reused), test AUROC on X_test at every epoch.
    leaky_val_traj, leaky_test_traj = train_full_trajectory(
        X_train[fit_idx], y_train[fit_idx], X_train[val_idx], y_train[val_idx],
        X_test, y_test, CAPACITY, TOTAL_EPOCHS, init_seed_leaky)

    # ONE 225-epoch trajectory for the honest arm's DISJOINT selection stat:
    # trained on tr2_idx, val AUROC on es_idx (disjoint from val_idx).
    clean_sel_val_traj, _ = train_full_trajectory(
        X_train[tr2_idx], y_train[tr2_idx], X_train[es_idx], y_train[es_idx],
        X_test, y_test, CAPACITY, TOTAL_EPOCHS, init_seed_clean)

    out = {}
    for stride, k_nominal in STRIDES.items():
        elig = eligible_epochs(stride)
        # LEAKY: argmax over eligible epochs of its OWN reused-fold trajectory;
        # report the TEST AUROC recorded at that same epoch (zero extra cost).
        e_hat_leaky = elig[np.argmax(leaky_val_traj[elig - 1])]
        leaky_auc = leaky_test_traj[e_hat_leaky - 1]

        # Honest arm: argmax over eligible epochs of the disjoint carve-out
        # trajectory, then ONE fresh retrain on fit_idx (the full pool) for
        # that many epochs -- the only "extra" training this script does, and
        # it is the same per-cell cost Sweep A already pays for every K.
        e_hat_clean = elig[np.argmax(clean_sel_val_traj[elig - 1])]
        clean_matched_auc = train_fixed_epochs_readout(
            X_train[fit_idx], y_train[fit_idx], X_test, y_test, CAPACITY,
            int(e_hat_clean), init_seed_leaky)

        out[stride] = {"k_nominal": k_nominal, "e_hat_leaky": int(e_hat_leaky),
                       "e_hat_clean": int(e_hat_clean),
                       "leaky_auc": leaky_auc, "clean_matched_auc": clean_matched_auc,
                       "gap": leaky_auc - clean_matched_auc}
    return out


def run_one_seed_coupled(seed, k):
    """Baseline: train to exactly K epochs (K = nominal candidate count), the
    structurally-coupled design, under this script's OWN simplified single-
    fold readout (for an apples-to-apples internal comparison, not compared to
    Sweep A's absolute numbers)."""
    data_seed, split_seed, fold_seed, init_seed_leaky, init_seed_clean = (
        seed, seed + 100000, seed + 200000, seed + 300000, seed + 400000)
    X, y = make_synthetic_data(data_seed, DEFAULT_N_SAMPLES, DEFAULT_TARGET_AUROC)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=split_seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)
    fit_idx, val_idx = train_test_split(
        np.arange(len(y_train)), test_size=VAL_FRACTION, stratify=y_train, random_state=fold_seed)
    tr2_idx, es_idx = train_test_split(
        fit_idx, test_size=ES_HOLD_FRACTION, stratify=y_train[fit_idx], random_state=fold_seed + 1)

    leaky_val_traj, leaky_test_traj = train_full_trajectory(
        X_train[fit_idx], y_train[fit_idx], X_train[val_idx], y_train[val_idx],
        X_test, y_test, CAPACITY, k, init_seed_leaky)
    clean_sel_val_traj, _ = train_full_trajectory(
        X_train[tr2_idx], y_train[tr2_idx], X_train[es_idx], y_train[es_idx],
        X_test, y_test, CAPACITY, k, init_seed_clean)

    e_hat_leaky = 1 + int(np.argmax(leaky_val_traj))
    leaky_auc = leaky_test_traj[e_hat_leaky - 1]
    e_hat_clean = 1 + int(np.argmax(clean_sel_val_traj))
    clean_matched_auc = train_fixed_epochs_readout(
        X_train[fit_idx], y_train[fit_idx], X_test, y_test, CAPACITY, e_hat_clean, init_seed_leaky)
    return leaky_auc - clean_matched_auc


def gap_summary(gaps, boot_seed):
    gaps = np.asarray(gaps, float)
    _, p = wilcoxon(gaps) if not np.allclose(gaps, 0) else (None, 1.0)
    res = bootstrap((gaps,), np.mean, confidence_level=0.95, n_resamples=5000,
                    method="BCa", random_state=np.random.default_rng(boot_seed))
    return {"gap_mean": float(gaps.mean()), "gap_bca_ci_95": [float(res.confidence_interval.low),
            float(res.confidence_interval.high)], "wilcoxon_p": float(p),
            "n_positive": int((gaps > 0).sum()), "n_seeds": int(len(gaps))}


def main():
    t0 = time.time()
    print(f"=== Decoupled design: train once to {TOTAL_EPOCHS} epochs, vary eligible-checkpoint stride ===")
    decoupled = {s: [] for s in STRIDES}
    for seed in range(N_SEEDS):
        res = run_one_seed_decoupled(seed)
        for stride in STRIDES:
            decoupled[stride].append(res[stride]["gap"])
        if (seed + 1) % 20 == 0:
            print(f"  seed {seed+1}/{N_SEEDS}  elapsed={time.time()-t0:.0f}s", flush=True)

    decoupled_out = {}
    for stride, k_nominal in STRIDES.items():
        s = gap_summary(decoupled[stride], boot_seed=7000 + stride)
        s["k_nominal"] = k_nominal
        decoupled_out[str(stride)] = s
        print(f"  stride={stride:3d} (K_eff={k_nominal:3d}): gap={s['gap_mean']:+.4f} "
              f"CI={s['gap_bca_ci_95']} p={s['wilcoxon_p']:.4g}")

    print(f"\n=== Coupled baseline (this script's own simplified readout): train to K epochs directly ===")
    coupled_out = {}
    for stride, k_nominal in STRIDES.items():
        gaps = [run_one_seed_coupled(seed, k_nominal) for seed in range(N_SEEDS)]
        s = gap_summary(gaps, boot_seed=8000 + stride)
        s["k"] = k_nominal
        coupled_out[str(k_nominal)] = s
        print(f"  K={k_nominal:3d}: gap={s['gap_mean']:+.4f} CI={s['gap_bca_ci_95']} "
              f"p={s['wilcoxon_p']:.4g}  elapsed={time.time()-t0:.0f}s", flush=True)

    # Fit monotonicity / concavity check on the decoupled design
    ks = np.array(sorted(STRIDES.values()))
    gaps_by_k = np.array([decoupled_out[str([s for s in STRIDES if STRIDES[s] == k][0])]["gap_mean"] for k in ks])
    monotone = bool(np.all(np.diff(gaps_by_k) >= -1e-9))
    ln_k = np.log(ks)
    if len(ks) >= 2:
        b_fit = float(np.polyfit(ln_k, gaps_by_k, 1)[0])
    else:
        b_fit = None

    ks_c = np.array(sorted(int(k) for k in coupled_out))
    gaps_c = np.array([coupled_out[str(k)]["gap_mean"] for k in ks_c])
    monotone_c = bool(np.all(np.diff(gaps_c) >= -1e-9))

    out = {
        "design": {"total_epochs": TOTAL_EPOCHS, "n_seeds": N_SEEDS, "capacity": CAPACITY,
                  "n_samples": DEFAULT_N_SAMPLES, "target_auroc": DEFAULT_TARGET_AUROC,
                  "strides": STRIDES, "es_hold_fraction": ES_HOLD_FRACTION,
                  "simplification_note": (
                      "Single reused validation fold + single train/test split, reading each "
                      "checkpoint's own sigmoid AUROC directly, rather than code/47's 5-fold "
                      "OOF-plus-downstream-logistic-regression pipeline (disclosed in the "
                      "module docstring). Internal coupled-vs-decoupled comparison only.")},
        "decoupled": decoupled_out,
        "coupled_baseline": coupled_out,
        "decoupled_monotone_nondecreasing_in_k": monotone,
        "decoupled_ln_k_slope": b_fit,
        "coupled_monotone_nondecreasing_in_k": monotone_c,
        "runtime_seconds": time.time() - t0,
    }

    verdict = (
        "SURVIVES" if monotone and b_fit is not None and b_fit > 0 else
        "DOES_NOT_SURVIVE_CLEANLY"
    )
    out["verdict"] = {
        "label": verdict,
        "statement": (
            f"Decoupled design (fixed {TOTAL_EPOCHS}-epoch training budget, only eligible-"
            f"checkpoint count varies): gaps at K_eff={list(STRIDES.values())} are "
            f"{[round(decoupled_out[str(s)]['gap_mean'], 5) for s in STRIDES]}, "
            f"{'monotone non-decreasing' if monotone else 'NOT monotone'} in K, "
            f"ln(K) slope {b_fit:+.6f}. Coupled baseline (this script's own simplified "
            f"train-to-K-epochs readout) at the same nominal K values gives "
            f"{[round(coupled_out[str(k)]['gap_mean'], 5) for k in sorted(int(k) for k in coupled_out)]}, "
            f"{'monotone' if monotone_c else 'NOT monotone'}."
        ),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n{out['verdict']['statement']}")
    print(f"\nSaved: {OUT_PATH}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
