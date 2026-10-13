"""
Population-level answers from independent samples.

    python3 123_population_from_independent_samples.py <name_prefix> <n_shards>

Input: results/audit_<prefix>_<k>.npz for k = 0..n_shards-1, one per
independent 700-item sample, each produced by 122 (the unchanged Mechanism-2
audit, with its re-splits inside the sample).

Within one sample, every quantity the paper reports -- the reuse gap at the
selected layer, B, the risk violation -- is a mean over re-splits of THAT
sample, and its interval is split-conditional. Across samples the means are
independent draws, so a t-interval over them is a valid population interval
with no re-split correction at all. That is the point of this design.

Reported:
  1. Population estimate and 95% interval, across samples, of the reuse gap at
     the selected layer (AUROC, Brier, reliability, ECE), of B on the same four
     metrics, and of the risk violation at 30/50/70% coverage.
  2. For each quantity, how often a single-sample analysis declares it
     established -- naively, and with the Nadeau-Bengio correction -- and how
     often that verdict matches the population answer from (1). This is the
     direct check on the correction that 119 Part B made in simulation.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
LOSSES = {"brier", "reliability", "ece"}


def per_sample(path):
    d = np.load(path)
    cv, ho, ls, viol = d["cv"], d["ho"], d["l_star"], d["risk_violation"]
    metrics = [str(m) for m in d["metrics"]]
    n_sel, n = int(d["n_sel"]), int(d["n_items"])
    rows = np.arange(len(ls))
    q = {}
    for k, m in enumerate(metrics):
        c, h = cv[:, k], ho[:, k]
        # reuse gap signed so that positive = the reused number looks better
        gap = (h[rows, ls] - c[rows, ls]) if m in LOSSES else (c[rows, ls] - h[rows, ls])
        # B signed so that positive = the selected layer is better held out
        B = (h.mean(axis=1) - h[rows, ls]) if m in LOSSES else (h[rows, ls] - h.mean(axis=1))
        q[f"gap_{m}"], q[f"B_{m}"] = gap, B
    for j, c in enumerate(d["coverages"]):
        q[f"risk_{int(round(float(c) * 100))}"] = viol[:, j]
    return q, (n - n_sel) / n_sel


def within_verdict(x, ratio):
    J = len(x); m, s = x.mean(), x.std(ddof=1)
    crit = stats.t.ppf(0.975, J - 1)
    naive = abs(m) > crit * np.sqrt(s ** 2 / J)
    nb = abs(m) > crit * np.sqrt((1 / J + ratio) * s ** 2)
    return bool(naive), bool(nb), float(np.sign(m))


def main():
    prefix, n_shards = sys.argv[1], int(sys.argv[2])
    samples = [per_sample(ROOT / "results" / f"audit_{prefix}_{k}.npz") for k in range(n_shards)]
    names = list(samples[0][0].keys())
    out = {"n_samples": n_shards, "quantities": {}}
    print(f"{n_shards} independent samples\n")
    print(f"{'quantity':<20} {'pop. mean':>10} {'95% CI (across samples)':>26} {'est.':>5} | "
          f"{'naive says est.':>15} {'corrected says est.':>19} | {'naive right':>11} {'corr. right':>11}")
    for q in names:
        means = np.array([s[0][q].mean() for s in samples])
        m, se = means.mean(), means.std(ddof=1) / np.sqrt(n_shards)
        crit = stats.t.ppf(0.975, n_shards - 1)
        lo, hi = m - crit * se, m + crit * se
        pop_est = bool(lo > 0 or hi < 0)
        pop_sign = float(np.sign(m))
        v = [within_verdict(s[0][q], s[1]) for s in samples]
        naive_est = np.mean([a for a, _, _ in v]); nb_est = np.mean([b for _, b, _ in v])
        # a single-sample verdict is "right" if it matches the population verdict
        # (established with the same sign, or not established)
        def right(est, sign):
            return (est and pop_est and sign == pop_sign) or (not est and not pop_est)
        naive_right = np.mean([right(a, sg) for a, _, sg in v])
        nb_right = np.mean([right(b, sg) for _, b, sg in v])
        out["quantities"][q] = {
            "population_mean": float(m), "population_ci_95": [float(lo), float(hi)],
            "population_established": pop_est,
            "sample_means": means.tolist(),
            "share_samples_naive_established": float(naive_est),
            "share_samples_corrected_established": float(nb_est),
            "share_naive_matches_population": float(naive_right),
            "share_corrected_matches_population": float(nb_right),
        }
        print(f"{q:<20} {m:>+10.5f} [{lo:>+11.5f},{hi:>+11.5f}] {'yes' if pop_est else 'no':>5} | "
              f"{naive_est:>15.2f} {nb_est:>19.2f} | {naive_right:>11.2f} {nb_right:>11.2f}")
    path = ROOT / "results" / f"population_{prefix}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
