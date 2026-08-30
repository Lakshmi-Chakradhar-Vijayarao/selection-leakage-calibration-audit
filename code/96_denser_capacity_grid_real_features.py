"""
MAXIMUM-RIGOR PASS, Item 9. The real-feature Mechanism-3 harness
(code/43/78) only tests probe capacities {128, 384} (the fidelity extension's
capacity-dependent decline, 4.8x shipped / 3.6x fold-matched from 128 to 384,
§4.3). Capacity here is the MLP hidden-width hyperparameter (SweepMLP's
`hidden` argument), a property of the PROBE, not of feature extraction --
confirmed by reading code/78 (CAPACITIES = s43.CAPACITIES = [128, 384], used
only to instantiate SweepMLP(in_dim, hidden)). Re-fitting at additional
capacities therefore needs NO new GPU feature extraction: this script adds
capacities 64 and 256 on the SAME already-cached real Mistral-7B/HaluEval
features (results/real_features_mistral7b_halueval.npz) and checks whether
the capacity-dependent decline is monotone across a denser {64,128,256,384}
grid or a two-point-grid artifact.

SIMPLIFICATION, DISCLOSED (same as items 2/5/6): single reused validation
fold, single train/test split, direct sigmoid-output readout, rather than
code/78's full 5-fold-OOF-plus-downstream-logistic-regression pipeline. Both
the shipped-style (in-fold ES=0.15) and fully-corrected (disjoint OOF, same
size) controls are included, matching Table tab:m3-real's two most different
columns.

Output: results/denser_capacity_grid_real_features.json
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import bootstrap, wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
FEATS_PATH = ROOT / "results" / "real_features_mistral7b_halueval.npz"
OUT_PATH = ROOT / "results" / "denser_capacity_grid_real_features.json"

CAPACITIES = [64, 128, 256, 384]
EPOCHS = 45
ES_HOLD_FRACTION = 0.15
TEST_SIZE = 0.20
VAL_FRACTION = 0.20
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


def load_features():
    d = np.load(FEATS_PATH)
    X_seq, X_glob, y = d["X_seq"], d["X_glob"], d["y"]
    X = np.hstack([X_seq.reshape(X_seq.shape[0], -1), X_glob]).astype(np.float32)
    return X, y


def roc_auc(y, s):
    try:
        return float(roc_auc_score(y, s))
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


def run_one_seed(X, y, hidden, seed, disjoint_sel):
    rng_split = seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=rng_split)
    scaler = StandardScaler(); X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)
    fit_idx, val_idx = train_test_split(np.arange(len(y_train)), test_size=VAL_FRACTION, stratify=y_train, random_state=seed + 1)

    model_leaky, _ = train_to_best_checkpoint(X_train[fit_idx], y_train[fit_idx], X_train[val_idx], y_train[val_idx], hidden, EPOCHS, seed)
    leaky_auc = eval_auc(model_leaky, X_test, y_test)

    if disjoint_sel:
        # fully-corrected: select on a disjoint carve-out the SAME size as
        # val_idx, excluded from the fit pool entirely (out-of-fold).
        pool = fit_idx
        sel_idx, fit2_idx = train_test_split(pool, train_size=len(val_idx), stratify=y_train[pool], random_state=seed + 2)
        model_clean, best_epoch = train_to_best_checkpoint(X_train[fit2_idx], y_train[fit2_idx], X_train[sel_idx], y_train[sel_idx], hidden, EPOCHS, seed)
        model_clean_matched = train_fixed_epochs(X_train[fit2_idx], y_train[fit2_idx], hidden, best_epoch, seed)
    else:
        # shipped-style: in-fold ES carve-out from the SAME fit pool LEAKY uses.
        tr2_idx, es_idx = train_test_split(fit_idx, test_size=ES_HOLD_FRACTION, stratify=y_train[fit_idx], random_state=seed + 3)
        model_clean, best_epoch = train_to_best_checkpoint(X_train[tr2_idx], y_train[tr2_idx], X_train[es_idx], y_train[es_idx], hidden, EPOCHS, seed)
        model_clean_matched = train_fixed_epochs(X_train[fit_idx], y_train[fit_idx], hidden, best_epoch, seed)
    clean_matched_auc = eval_auc(model_clean_matched, X_test, y_test)
    return leaky_auc, clean_matched_auc


def sweep(X, y, hidden, disjoint_sel, n_seeds=N_SEEDS):
    leaky, clean_matched = [], []
    for seed in range(n_seeds):
        l, c = run_one_seed(X, y, hidden, seed, disjoint_sel)
        leaky.append(l); clean_matched.append(c)
    leaky, clean_matched = np.array(leaky), np.array(clean_matched)
    gap = leaky - clean_matched
    _, p = wilcoxon(gap) if not np.allclose(gap, 0) else (None, 1.0)
    res = bootstrap((gap,), np.mean, confidence_level=0.95, n_resamples=5000, method="BCa",
                    random_state=np.random.default_rng(hidden))
    return {"leaky_mean": float(leaky.mean()), "clean_matched_mean": float(clean_matched.mean()),
            "gap_mean": float(gap.mean()), "gap_bca_ci_95": [float(res.confidence_interval.low),
            float(res.confidence_interval.high)], "wilcoxon_p": float(p), "n_positive": int((gap > 0).sum())}


def main():
    t0 = time.time()
    X, y = load_features()
    print(f"Loaded real features: X={X.shape}, hall_rate={1 - y.mean():.3f}\n")

    out = {"shipped_style_infold": {}, "fully_corrected_oof": {}}
    print("=== Shipped-style (in-fold ES=0.15) control, denser capacity grid ===")
    for cap in CAPACITIES:
        r = sweep(X, y, cap, disjoint_sel=False)
        out["shipped_style_infold"][str(cap)] = r
        print(f"  capacity={cap:4d}: gap={r['gap_mean']:+.4f} CI={r['gap_bca_ci_95']} "
              f"p={r['wilcoxon_p']:.4g}  elapsed={time.time()-t0:.0f}s", flush=True)

    print("\n=== Fully-corrected (disjoint OOF) control, denser capacity grid ===")
    for cap in CAPACITIES:
        r = sweep(X, y, cap, disjoint_sel=True)
        out["fully_corrected_oof"][str(cap)] = r
        print(f"  capacity={cap:4d}: gap={r['gap_mean']:+.4f} CI={r['gap_bca_ci_95']} "
              f"p={r['wilcoxon_p']:.4g}  elapsed={time.time()-t0:.0f}s", flush=True)

    gaps_shipped = [out["shipped_style_infold"][str(c)]["gap_mean"] for c in CAPACITIES]
    gaps_corrected = [out["fully_corrected_oof"][str(c)]["gap_mean"] for c in CAPACITIES]
    monotone_shipped = bool(np.all(np.diff(gaps_shipped) <= 1e-9) or np.all(np.diff(gaps_shipped) >= -1e-9))
    strictly_decreasing_shipped = bool(np.all(np.diff(gaps_shipped) < 0))
    strictly_decreasing_corrected = bool(np.all(np.diff(gaps_corrected) < 0))

    ratio_128_384_shipped = (out["shipped_style_infold"]["128"]["gap_mean"] /
                             out["shipped_style_infold"]["384"]["gap_mean"]
                             if out["shipped_style_infold"]["384"]["gap_mean"] > 1e-9 else None)

    out["design"] = {"capacities": CAPACITIES, "epochs": EPOCHS, "n_seeds": N_SEEDS,
                     "es_hold_fraction": ES_HOLD_FRACTION}
    out["gaps_by_capacity_shipped"] = dict(zip(CAPACITIES, gaps_shipped))
    out["gaps_by_capacity_fully_corrected"] = dict(zip(CAPACITIES, gaps_corrected))
    out["strictly_decreasing_shipped"] = strictly_decreasing_shipped
    out["strictly_decreasing_fully_corrected"] = strictly_decreasing_corrected
    out["ratio_128_over_384_shipped_this_harness"] = ratio_128_384_shipped
    out["runtime_seconds"] = time.time() - t0

    verdict = (
        f"Shipped-style control across capacities {CAPACITIES}: gaps = "
        f"{[round(g, 5) for g in gaps_shipped]}, "
        f"{'strictly decreasing' if strictly_decreasing_shipped else 'NOT strictly decreasing'} "
        f"(this simplified harness's own 128-vs-384 ratio: "
        f"{ratio_128_384_shipped:.2f}x, for comparison against the paper's fidelity-extension "
        "4.8x/3.6x figures measured on a different, more faithful harness -- not directly "
        "comparable in magnitude, only in whether the qualitative decline persists on a denser "
        "grid). Fully-corrected control: gaps = "
        f"{[round(g, 5) for g in gaps_corrected]}, "
        f"{'strictly decreasing' if strictly_decreasing_corrected else 'NOT strictly decreasing'}. "
        f"{'The capacity-dependent decline is monotone across the denser 4-point grid under both controls.' if (strictly_decreasing_shipped and strictly_decreasing_corrected) else 'The capacity-dependent decline is NOT cleanly monotone across the denser 4-point grid -- the 128-vs-384 comparison in the original 2-point grid understates how much non-monotonicity the intermediate capacities reveal.'}"
    )
    out["verdict"] = verdict
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{verdict}")
    print(f"\nSaved: {OUT_PATH}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
