"""
UncertaiNLP: reliability/risk-coverage figure for Mechanism 1's
selective-prediction result (Table 5 in main.tex).

Plots the exact numbers already reported and BCa-established in the paper --
no new computation, no new data. Mechanism 1's believed risk is exactly zero
at every coverage level tested (near-exact in-sample separation at the
probe's default regularization); the bars are the actual risk achieved on
genuinely held-out (CLEAN_MATCHED) data at the identical threshold, with the
paper's own BCa 95% intervals (which, since believed=0, equal the reported
risk-violation intervals).

Source numbers (Table 5, Mechanism 1 rows):
  30% coverage: actual 0.002, BCa [0.001, 0.004]
  50% coverage: actual 0.008, BCa [0.006, 0.008]  (violation point est. 0.007;
                the actual-risk point estimate and the violation point
                estimate are independently rounded means across reps, not
                required to be identical -- see note below)
  70% coverage: actual 0.009, BCa [0.008, 0.010]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "submissions" / "uncertainlp2026" / "figures" / "fig1_risk_coverage.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

coverage = np.array([30, 50, 70])
actual_risk = np.array([0.002, 0.008, 0.009])
ci_lo = np.array([0.001, 0.006, 0.008])
ci_hi = np.array([0.004, 0.008, 0.010])
yerr = np.vstack([actual_risk - ci_lo, ci_hi - actual_risk])

fig, ax = plt.subplots(figsize=(3.3, 1.35))

ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.3, zorder=1)
ax.text(0.02, 0.93, "believed risk: exactly zero", transform=ax.transAxes,
        fontsize=6.5, color="#444444", ha="left", va="top")

bars = ax.bar(coverage, actual_risk, width=8, color="#b23b3b", zorder=2,
              yerr=yerr, capsize=3, error_kw={"elinewidth": 1.0, "capthick": 1.0})

ax.set_xticks(coverage)
ax.set_xticklabels([f"{c}%" for c in coverage], fontsize=8)
ax.set_xlabel("Target coverage", fontsize=8)
ax.set_ylabel("Actual risk\n(held-out data)", fontsize=8)
ax.set_ylim(-0.0015, 0.0165)
ax.tick_params(axis="y", labelsize=7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for x, y in zip(coverage, actual_risk):
    ax.text(x, y + 0.0018, f"{y:.3f}", ha="center", fontsize=7)

fig.tight_layout(pad=0.3)
fig.savefig(OUT, bbox_inches="tight")
print(f"Saved: {OUT}")
