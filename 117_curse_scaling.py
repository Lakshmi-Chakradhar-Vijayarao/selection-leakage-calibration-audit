"""
How the winner's curse scales with candidate count and candidate dependence.
[Phase 1: replaces the sqrt(1-rho) heuristic with a measured relationship.]

Section "How many candidates are there, really?" quotes sqrt(1 - rho_bar) as
an anticipated shrinkage factor for the curse, and notes that the measured
factor (1.65x) differs from what that heuristic suggests (about 2x). A
heuristic quoted and then contradicted is worse than no heuristic, so this
script measures the relationship directly over a grid of dependence and
candidate counts.

MODEL. K candidates with equicorrelated criterion noise:

    cv_l = q_l + sqrt(rho) * z0 + sqrt(1 - rho) * z_l,   z0, z_l ~ N(0,1)

so Corr(cv_i, cv_j) = rho for i != j. The shared component z0 shifts every
candidate together and therefore cannot be exploited by an argmax; only the
idiosyncratic part drives the curse. The prediction is

    E[A] = sigma * sqrt(1 - rho) * e_K,     e_K = E[max of K standard normals]

exactly, under equicorrelation and equal quality. We check that, then ask how
far it survives when quality varies (tau > 0) and when the correlation matrix
is NOT equicorrelated but has the banded structure real layers exhibit.

WHY THE BANDED CASE MATTERS. The measured correlation matrix for layers is far
from equicorrelated: adjacent layers correlate at 0.957 while distant ones
correlate far less. The equicorrelation formula uses only the mean, so it is
expected to be approximate here, and this script quantifies how approximate.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "results" / "curse_scaling.json"

N_TRIALS = 40000
SIGMA = 0.03
SEED = 20261004
K_GRID = [4, 8, 16, 32, 64, 128]
RHO_GRID = [0.0, 0.25, 0.5, 0.755, 0.9]


def e_K(K, rng, n=400000):
    """E[max of K standard normals], by simulation."""
    return float(rng.standard_normal((n // K + 1, K)).max(axis=1).mean())


def curse_equicorrelated(K, rho, tau, rng):
    z0 = rng.standard_normal((N_TRIALS, 1))
    z = rng.standard_normal((N_TRIALS, K))
    q = rng.normal(0, tau, size=(N_TRIALS, K)) if tau > 0 else 0.0
    cv = q + SIGMA * (np.sqrt(rho) * z0 + np.sqrt(1 - rho) * z)
    return float((cv.max(axis=1) - cv.mean(axis=1)).mean())


def curse_banded(K, decay, rng):
    """Correlation decaying with |i-j|, the structure layers actually show."""
    idx = np.arange(K)
    C = decay ** np.abs(idx[:, None] - idx[None, :])
    L = np.linalg.cholesky(C + 1e-10 * np.eye(K))
    cv = SIGMA * (rng.standard_normal((N_TRIALS, K)) @ L.T)
    off = C[~np.eye(K, dtype=bool)]
    w = np.linalg.eigvalsh(C); w = w[w > 1e-12]
    return (float((cv.max(axis=1) - cv.mean(axis=1)).mean()),
            float(off.mean()), float((w.sum() ** 2) / (w ** 2).sum()))


def main():
    rng = np.random.default_rng(SEED)
    eK = {K: e_K(K, rng) for K in K_GRID}
    out = {"n_trials": N_TRIALS, "sigma": SIGMA, "seed": SEED,
           "expected_max_standard_normal": eK, "equicorrelated": {}, "banded": {}}

    print("=" * 96)
    print("WINNER'S CURSE vs CANDIDATE COUNT AND DEPENDENCE")
    print("=" * 96)
    print("\nEquicorrelated, equal quality (tau=0). "
          "Prediction: E[A] = sigma*sqrt(1-rho)*e_K\n")
    print(f"{'K':>5} {'rho':>7} {'measured':>11} {'predicted':>11} {'ratio':>8}")
    print("-" * 46)
    worst = 0.0
    for K in K_GRID:
        for rho in RHO_GRID:
            m = curse_equicorrelated(K, rho, 0.0, rng)
            pred = SIGMA * np.sqrt(1 - rho) * eK[K]
            ratio = m / pred if pred else float("nan")
            worst = max(worst, abs(ratio - 1))
            out["equicorrelated"][f"K={K},rho={rho}"] = {
                "K": K, "rho": rho, "measured": m, "predicted": pred,
                "ratio": float(ratio)}
            print(f"{K:>5} {rho:>7.3f} {m:>11.5f} {pred:>11.5f} {ratio:>8.4f}")
        print("-" * 46)
    out["equicorrelated_max_relative_error"] = float(worst)
    print(f"\nworst relative error of the formula under equicorrelation: {worst:.4f}")

    print("\n\nBanded correlation (decay^|i-j|), K=32 -- the structure layers show\n")
    print(f"{'decay':>7} {'rho_bar':>9} {'K_eff':>8} {'measured':>11} "
          f"{'equicorr pred':>14} {'ratio':>8}")
    print("-" * 62)
    for decay in [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]:
        m, rb, ke = curse_banded(32, decay, rng)
        pred = SIGMA * np.sqrt(max(0.0, 1 - rb)) * eK[32]
        out["banded"][f"decay={decay}"] = {
            "decay": decay, "rho_bar": rb, "effective_K": ke,
            "measured": m, "equicorrelated_prediction": pred,
            "ratio": float(m / pred) if pred else None}
        print(f"{decay:>7.2f} {rb:>9.3f} {ke:>8.2f} {m:>11.5f} {pred:>14.5f} "
              f"{m/pred:>8.3f}")
    print("-" * 62)

    b = out["banded"]
    ratios = [v["ratio"] for v in b.values() if v["ratio"]]
    out["banded_ratio_range"] = [min(ratios), max(ratios)]
    print(f"\nUnder banded dependence the equicorrelation formula is off by "
          f"{min(ratios):.2f}x to {max(ratios):.2f}x,")
    print("so rho_bar alone does not determine the curse and we quote the "
          "measurement, not the formula.")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
