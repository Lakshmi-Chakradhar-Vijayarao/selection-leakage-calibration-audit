"""
MAXIMUM-RIGOR PASS, Item 6. Mechanisms 3 (per-fold checkpoint
selection) and 5 (test-set threshold selection) co-occur in a single audited
file (MultiHaluDet's run_pipeline.py) -- this paper found Mechanism 5 while
auditing Mechanism 3's own repository (§4.5). Every severity number in this
paper is measured for one mechanism in isolation. This script builds a
synthetic setting where BOTH are present at once and asks the practically
relevant question: do their severities on a shared metric (F1) ADD, combine
SUB-additively, or INTERACT, relative to each measured alone?

DESIGN: a 2x2 factorial, paired on identical data/seeds/folds throughout.
  Axis 1 (Mechanism 3): checkpoint selection is either LEAKY (argmax on the
    reused validation fold) or CLEAN_MATCHED (argmax on a disjoint carve-out,
    retrained to the matched epoch count) -- identical construction to
    code/47/92.
  Axis 2 (Mechanism 5): the decision threshold is either TEST-SELECTED
    (argmax F1 over an 81-point grid using the TEST labels themselves --
    exactly run_pipeline.py's own pattern, §4.5) or FIXED at 0.5 (no
    test-label information used to pick it).

Four cells: (LEAKY, test-thresh) = both mechanisms present, the realistic
composition; (LEAKY, fixed) = Mechanism 3 alone; (CLEAN_MATCHED, test-thresh)
= Mechanism 5 alone; (CLEAN_MATCHED, fixed) = neither (the honest baseline).
All four report F1 (Mechanism 5's own metric of interest, since AUROC is
invariant to threshold choice by construction and so cannot show Mechanism
5's effect at all).

  M3_effect       = F1(LEAKY, fixed)        - F1(CLEAN_MATCHED, fixed)
  M5_effect       = F1(CLEAN_MATCHED, test) - F1(CLEAN_MATCHED, fixed)
  Combined_effect = F1(LEAKY, test)         - F1(CLEAN_MATCHED, fixed)
  Additive_prediction = M3_effect + M5_effect
  Interaction_term    = Combined_effect - Additive_prediction

SIMPLIFICATION, DISCLOSED (same as code/88, code/92): single reused
validation fold, single train/test split, direct sigmoid-output readout.

Output: results/composition_test_m3_m5.json
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import bootstrap, norm, wilcoxon
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "composition_test_m3_m5.json"

CAPACITY = 128
FEAT_DIM = 64
EPOCHS = 45
N_SAMPLES = 700
TARGET_AUROC0 = 0.80
ES_HOLD_FRACTION = 0.15
TEST_SIZE = 0.20
VAL_FRACTION = 0.20
N_SEEDS = 100
N_THRESH_GRID = 81


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


def probs_of(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()


def f1_test_selected(probs, y):
    """Mechanism 5: argmax F1 over an 81-point threshold grid using the TEST
    labels themselves -- the leak."""
    best_f1 = -1.0
    for thresh in np.linspace(0.01, 0.99, N_THRESH_GRID):
        pred = (probs >= thresh).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
    return float(best_f1)


def f1_fixed(probs, y, thresh=0.5):
    pred = (probs >= thresh).astype(int)
    return float(f1_score(y, pred, zero_division=0))


def run_one_seed(seed):
    split_seed, fold_seed, init_seed = seed + 100000, seed + 200000, seed + 300000
    X, y = make_synthetic_data(seed, N_SAMPLES, TARGET_AUROC0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=split_seed)
    scaler = StandardScaler(); X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)
    fit_idx, val_idx = train_test_split(np.arange(len(y_train)), test_size=VAL_FRACTION, stratify=y_train, random_state=fold_seed)
    tr2_idx, es_idx = train_test_split(fit_idx, test_size=ES_HOLD_FRACTION, stratify=y_train[fit_idx], random_state=fold_seed + 1)

    model_leaky, _ = train_to_best_checkpoint(X_train[fit_idx], y_train[fit_idx], X_train[val_idx], y_train[val_idx], CAPACITY, EPOCHS, init_seed)
    probs_leaky = probs_of(model_leaky, X_test)

    model_clean, best_epoch = train_to_best_checkpoint(X_train[tr2_idx], y_train[tr2_idx], X_train[es_idx], y_train[es_idx], CAPACITY, EPOCHS, init_seed)
    model_clean_matched = train_fixed_epochs(X_train[fit_idx], y_train[fit_idx], CAPACITY, best_epoch, init_seed)
    probs_clean_matched = probs_of(model_clean_matched, X_test)

    return {
        "leaky_test_thresh": f1_test_selected(probs_leaky, y_test),
        "leaky_fixed": f1_fixed(probs_leaky, y_test),
        "clean_test_thresh": f1_test_selected(probs_clean_matched, y_test),
        "clean_fixed": f1_fixed(probs_clean_matched, y_test),
        "leaky_auroc": roc_auc(y_test, probs_leaky),
        "clean_auroc": roc_auc(y_test, probs_clean_matched),
    }


def summarize(arr, boot_seed):
    arr = np.asarray(arr, float)
    _, p = wilcoxon(arr) if not np.allclose(arr, 0) else (None, 1.0)
    res = bootstrap((arr,), np.mean, confidence_level=0.95, n_resamples=5000, method="BCa",
                    random_state=np.random.default_rng(boot_seed))
    return {"mean": float(arr.mean()), "bca_ci_95": [float(res.confidence_interval.low),
            float(res.confidence_interval.high)], "wilcoxon_p": float(p), "sd": float(arr.std(ddof=1))}


def main():
    t0 = time.time()
    cells = {k: [] for k in ["leaky_test_thresh", "leaky_fixed", "clean_test_thresh",
                             "clean_fixed", "leaky_auroc", "clean_auroc"]}
    for seed in range(N_SEEDS):
        r = run_one_seed(seed)
        for k in cells:
            cells[k].append(r[k])
        if (seed + 1) % 25 == 0:
            print(f"  seed {seed+1}/{N_SEEDS}  elapsed={time.time()-t0:.0f}s", flush=True)

    cells = {k: np.array(v) for k, v in cells.items()}
    m3_effect = cells["leaky_fixed"] - cells["clean_fixed"]
    m5_effect = cells["clean_test_thresh"] - cells["clean_fixed"]
    combined_effect = cells["leaky_test_thresh"] - cells["clean_fixed"]
    additive_prediction = m3_effect + m5_effect
    interaction = combined_effect - additive_prediction

    m3_s = summarize(m3_effect, 1); m5_s = summarize(m5_effect, 2)
    combined_s = summarize(combined_effect, 3); additive_s = summarize(additive_prediction, 4)
    interaction_s = summarize(interaction, 5)

    print(f"\nM3 effect (F1, fixed threshold):        {m3_s['mean']:+.4f} CI={m3_s['bca_ci_95']}")
    print(f"M5 effect (F1, CLEAN_MATCHED checkpoint): {m5_s['mean']:+.4f} CI={m5_s['bca_ci_95']}")
    print(f"Combined (both present):                  {combined_s['mean']:+.4f} CI={combined_s['bca_ci_95']}")
    print(f"Additive prediction (M3+M5):               {additive_s['mean']:+.4f} CI={additive_s['bca_ci_95']}")
    print(f"Interaction (combined - additive):        {interaction_s['mean']:+.4f} CI={interaction_s['bca_ci_95']} p={interaction_s['wilcoxon_p']:.4g}")

    interaction_excludes_zero = (interaction_s['bca_ci_95'][0] > 0) or (interaction_s['bca_ci_95'][1] < 0)
    ratio_combined_over_additive = combined_s['mean'] / additive_s['mean'] if abs(additive_s['mean']) > 1e-9 else None
    verdict = (
        "ADDITIVE" if not interaction_excludes_zero else
        ("SUPER_ADDITIVE" if interaction_s['mean'] > 0 else "SUB_ADDITIVE")
    )

    out = {
        "design": {"capacity": CAPACITY, "epochs": EPOCHS, "n_samples": N_SAMPLES,
                  "target_auroc0": TARGET_AUROC0, "n_seeds": N_SEEDS, "n_thresh_grid": N_THRESH_GRID},
        "mean_auroc": {"leaky": float(cells["leaky_auroc"].mean()), "clean_matched": float(cells["clean_auroc"].mean())},
        "m3_effect_f1": m3_s, "m5_effect_f1": m5_s,
        "combined_effect_f1": combined_s, "additive_prediction_f1": additive_s,
        "interaction_term_f1": interaction_s,
        "interaction_bca_excludes_zero": bool(interaction_excludes_zero),
        "ratio_combined_over_additive": ratio_combined_over_additive,
        "verdict": verdict,
        "statement": (
            f"With both mechanisms present, F1 inflation is {combined_s['mean']:+.4f} "
            f"(CI {combined_s['bca_ci_95']}), against an additive prediction from each "
            f"measured alone of {additive_s['mean']:+.4f} (CI {additive_s['bca_ci_95']}). "
            f"The interaction term is {interaction_s['mean']:+.4f} "
            f"(CI {interaction_s['bca_ci_95']}, p={interaction_s['wilcoxon_p']:.3g}), which "
            f"{'excludes' if interaction_excludes_zero else 'does not exclude'} zero, so the "
            f"two mechanisms combine {verdict.replace('_', '-').lower()} in this synthetic "
            "reconstruction. This is directly relevant to real pipelines like MultiHaluDet's "
            "own run_pipeline.py, which carries both mechanisms at once: a practitioner who "
            "fixes them independently and expects severities to simply subtract should read "
            "this result as a caution rather than an assumption."
        ),
        "runtime_seconds": time.time() - t0,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{out['statement']}")
    print(f"\nSaved: {OUT_PATH}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
