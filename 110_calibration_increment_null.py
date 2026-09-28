"""
Mechanical null for the selection-specific CALIBRATION increment.
The calibration analogue of code/64, using code/64's null construction.

WHY THIS EXISTS. code/64 establishes that Delta_sel (the AUROC selection-
specific component) has a mechanical, non-zero null: l* = argmax_l cv_l while
gap_l = cv_l - ho_l contains the same +cv_l term, so testing H0: Delta_sel = 0
tests a hypothesis nobody holds. Its answer is striking -- the observed
+0.0255 lies BELOW the null's entire 95% interval, because CV-argmax recovers
a genuinely better-than-average layer and that transferable quality cancels
40.4% of the raw winner's curse.

The calibration increment reported in the calibration-bridge section has the
identical structure and has never been tested against the identical null. This
script closes that, and deliberately reuses code/64's construction rather than
inventing a second one: using two different nulls for two versions of the same
statistic would itself be a finding against us.

NULL CONSTRUCTION (code/64's, applied to calibration). Per-layer LEAKY and
CLEAN calibration values are computed ONCE per rep. The null then permutes
which layer's CLEAN value is paired with which layer's LEAKY value:

    gap_l       = clean_l        - leaky_l          (observed)
    gap_l^pi    = clean_{pi(l)}  - leaky_l          (null draw)

This leaves both marginal vectors, the argmax rule, the layer count and the
selection criterion untouched, and removes only the layer correspondence --
i.e. exactly the transferable component. Because the metric values are
precomputed, 20,000 draws is array shuffling rather than 20,000 re-evaluations
of the calibration metrics, which is what makes code/64 affordable and makes
this affordable too.

CLOSED FORM AND EXACT DECOMPOSITION. Writing A = leaky_{l*} - mean(leaky) and
B = clean_{l*} - mean(clean),

    increment = gap_{l*} - mean(gap) = B - A                     (exact)

and since E_pi[clean_{pi(l*)}] = mean(clean), the null has mean -A in closed
form. We report the simulated null against that closed form as a correctness
check, exactly as code/64 does.

Reuses results/case_study_2_probe_scores.npz and 103's metric functions.
No refitting. Runs in seconds.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "results" / "case_study_2_probe_scores.npz"
OUT_PATH = ROOT / "results" / "calibration_increment_null.json"

N_DRAWS = 20000
SEED = 20260928
METRICS = ("reliability", "ece", "brier")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rr = _load(ROOT / "103_reviewer_response_mechanism2.py", "reviewer_response")


def per_layer_values(a, rep, n_layers):
    """Per-layer LEAKY and CLEAN calibration values, and the selected layer."""
    v = f"rand{rep}"
    y_sel, y_ho = a[f"{v}__y_sel"], a[f"{v}__y_ho"]
    fold_id = a[f"{v}__fold_id"]
    cv, ho = a[f"{v}__cv_scores"].astype(np.float64), a[f"{v}__ho_scores"].astype(np.float64)
    folds = sorted(set(fold_id.tolist()))

    cv_auroc = np.array([
        np.mean([roc_auc_score(y_sel[fold_id == f], cv[l][fold_id == f])
                 for f in folds]) for l in range(n_layers)])
    l_star = int(cv_auroc.argmax())

    leaky = {m: np.empty(n_layers) for m in METRICS}
    clean = {m: np.empty(n_layers) for m in METRICS}
    for l in range(n_layers):
        rel_l, _, _ = rr.murphy_decomposition(cv[l], y_sel, 10, "width")
        rel_c, _, _ = rr.murphy_decomposition(ho[l], y_ho, 10, "width")
        leaky["reliability"][l], clean["reliability"][l] = rel_l, rel_c
        leaky["ece"][l] = rr.ece_binned(cv[l], y_sel, 10, "width")
        clean["ece"][l] = rr.ece_binned(ho[l], y_ho, 10, "width")
        leaky["brier"][l] = np.mean((cv[l] - y_sel) ** 2)
        clean["brier"][l] = np.mean((ho[l] - y_ho) ** 2)
    return leaky, clean, l_star


def main():
    a = np.load(ARTIFACT)
    n_layers, n_reps = int(a["n_layers"]), int(a["n_reps"])
    print(f"precomputing per-layer calibration values: {n_reps} reps x {n_layers} layers",
          flush=True)

    LEAK = {m: np.empty((n_reps, n_layers)) for m in METRICS}
    CLEAN = {m: np.empty((n_reps, n_layers)) for m in METRICS}
    lstar = np.empty(n_reps, dtype=int)
    for r in range(n_reps):
        lk, cl, ls = per_layer_values(a, r, n_layers)
        for m in METRICS:
            LEAK[m][r] = lk[m]; CLEAN[m][r] = cl[m]
        lstar[r] = ls
    print("done; running the layer-permutation null", flush=True)

    rng = np.random.default_rng(SEED)
    rows = np.arange(n_reps)
    out = {"n_draws": N_DRAWS, "n_reps": n_reps, "n_layers": n_layers, "seed": SEED,
           "null_construction": "code/64's: permute which layer's CLEAN calibration "
                                "value is paired with which layer's LEAKY value; "
                                "marginals, argmax rule and layer count untouched",
           "selected_layer_per_rep": lstar.tolist(), "metrics": {}}

    print("\n" + "=" * 100)
    print("MECHANICAL NULL FOR THE SELECTION-SPECIFIC CALIBRATION INCREMENT")
    print("(same construction as code/64's Delta_sel null)")
    print("=" * 100)

    for m in METRICS:
        lk, cl = LEAK[m], CLEAN[m]
        A = lk[rows, lstar] - lk.mean(axis=1)      # curse on the leaky arm
        B = cl[rows, lstar] - cl.mean(axis=1)      # transferred, on data not used to select
        observed_per_rep = B - A
        observed = float(observed_per_rep.mean())

        # simulated null: permute CLEAN across layers, independently per rep per draw
        null = np.empty(N_DRAWS)
        for d in range(N_DRAWS):
            perm = rng.permuted(np.tile(np.arange(n_layers), (n_reps, 1)), axis=1)
            cl_perm_at_lstar = cl[rows, perm[rows, lstar]]
            null[d] = float(((cl_perm_at_lstar - cl.mean(axis=1)) - A).mean())

        closed_form = float((-A).mean())
        lo, hi = np.quantile(null, [0.025, 0.975])
        p_le = float(np.mean(null <= observed))

        e = {
            "observed_increment": observed,
            "null_mean_simulated": float(null.mean()),
            "null_mean_closed_form": closed_form,
            "null_closed_form_abs_diff": abs(float(null.mean()) - closed_form),
            "null_sd": float(null.std(ddof=1)),
            "null_ci_95": [float(lo), float(hi)],
            "p_null_le_observed": p_le,
            "observed_below_entire_null_ci": bool(observed < lo),
            "decomposition": {
                "A_curse_on_leaky_arm": float(A.mean()),
                "B_transferred_on_clean_arm": float(B.mean()),
                "identity_check_B_minus_A_equals_increment":
                    abs(float((B - A).mean()) - observed),
                # B is the properly identified estimand here. It is built ONLY
                # from CLEAN values -- data that played no part in choosing
                # l* -- so unlike the increment it has a genuine zero null and
                # can be tested directly. B < 0 says the CV-argmax-selected
                # layer is WORSE calibrated on fresh data than an average
                # layer, which is a selection effect on calibration that
                # survives identification.
                "B_bca_95ci": list(rr.bca_ci(B)),
                "B_excludes_zero": bool(rr.summarise(B)["excludes_zero"]),
                "B_sd_over_reps": float(B.std(ddof=1)),
                "B_n_negative_reps": int((B < 0).sum()),
            },
        }
        out["metrics"][m] = e
        verdict = ("BELOW the null's entire 95% CI" if e["observed_below_entire_null_ci"]
                   else "inside/above the null CI")
        print(f"\n  {m}")
        print(f"    observed increment      {observed:+.5f}")
        print(f"    null mean (simulated)   {null.mean():+.5f}  "
              f"(closed form {closed_form:+.5f}, diff {e['null_closed_form_abs_diff']:.2e})")
        print(f"    null 95% CI             [{lo:+.5f},{hi:+.5f}]   SD {null.std(ddof=1):.5f}")
        print(f"    P(null <= observed)     {p_le:.4f}")
        print(f"    verdict                 observed is {verdict}")
        print(f"    decomposition           A(curse)={A.mean():+.5f}  "
              f"B(transferred)={B.mean():+.5f}")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
