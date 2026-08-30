"""
figure for the K x AUROC_0 severity surface (Section 5.5).

WHY THIS SCRIPT EXISTS. Two independent reviews flagged the same presentation
gap: a 70-page paper whose central quantitative result is a two-dimensional
severity surface contained exactly ONE figure, and that surface appeared only
as a table. This renders it.

Two panels, both read directly from results/joint_severity_surface.json so the
figure cannot drift from the table:

  (a) heatmap of the mean gap over the 4 x 5 grid, annotated with each cell's
      value and marked where the cell's own BCa interval includes zero -- which
      is most of the low-severity half of the grid, and is the fact the table
      makes hardest to see;
  (b) the same cells as curves against the probit operating point, one line per
      K, on a log severity axis. The near-parallel lines are what "separable /
      multiplicative" means, and the fanning at the left is the weak
      candidate-count effect.

Output: draft/latex/figures/severity-surface.pdf
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "joint_severity_surface.json"
OUT = ROOT / "draft" / "latex" / "figures" / "severity-surface.pdf"


def main():
    cells = json.load(open(SRC))["cells"]
    Ks = sorted({v["K"] for v in cells.values()})
    As = sorted({v["target_auroc"] for v in cells.values()})
    G = {(v["K"], v["target_auroc"]): v for v in cells.values()}
    M = np.array([[G[(k, a)]["gap_mean"] for a in As] for k in Ks])
    Z = np.array([[G[(k, a)]["gap_bca_ci_95"][0] <= 0 for a in As] for k in Ks])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.9))

    im = ax1.imshow(M * 1000, cmap="YlOrRd", aspect="auto", origin="lower")
    ax1.set_xticks(range(len(As)))
    ax1.set_xticklabels([f"{a:g}" for a in As])
    ax1.set_yticks(range(len(Ks)))
    ax1.set_yticklabels([str(k) for k in Ks])
    ax1.set_xlabel(r"operating point AUROC$_0$")
    ax1.set_ylabel(r"candidate count $K$")
    ax1.set_title("(a) severity gap (AUROC $\\times 10^{-3}$)", fontsize=10)
    for i in range(len(Ks)):
        for j in range(len(As)):
            v = M[i, j] * 1000
            ax1.text(j, i, f"{v:.2f}" + ("$^\\circ$" if Z[i, j] else ""),
                     ha="center", va="center", fontsize=8,
                     color="white" if v > M.max() * 1000 * 0.6 else "black")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    z0 = norm.ppf(As)
    markers = ["o", "s", "^", "D"]
    for i, k in enumerate(Ks):
        ax2.plot(z0, M[i] * 1000, marker=markers[i], ms=4.5, lw=1.3, label=f"$K={k}$")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$\Phi^{-1}(\mathrm{AUROC}_0)$")
    ax2.set_ylabel(r"severity gap (AUROC $\times 10^{-3}$)")
    ax2.set_title("(b) the same surface, log severity vs probit operating point",
                  fontsize=10)
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(alpha=0.25, lw=0.5)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"grid {M.shape}, min {M.min():.5f}, max {M.max():.5f}, "
          f"{int(Z.sum())} of {Z.size} cells' BCa intervals include zero")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
