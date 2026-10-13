"""
What survives once the 50 re-splits are not treated as independent?

Every interval in the paper resamples, or t-tests over, 50 random re-splits of
ONE sample (400 selection / 300 held-out from 700 items). The re-splits share
data, so a t-test or bootstrap over them answers a split-conditional question
about this sample. The paper says so ("those estimands are split-conditional
and nothing here extrapolates"), but a deployment claim is a population
claim, and 119 Part B measures how far the split-conditional test can overstate
it when no candidate is better in the population.

This script applies the standard correction for re-split dependence --
Nadeau & Bengio (2003), Machine Learning 52:239-281 -- to the paper's own
numbers, and reports each quantity both ways:

    naive:     se^2 = s^2 / J
    corrected: se^2 = (1/J + n_test/n_train) s^2      (J = 50 re-splits)

For the main harness n_test/n_train = 300/400; for the second model family
171/229. The correction is approximate (it is derived for a held-out mean
under a simple variance model) and is usually described as conservative.

Quantities, per harness:
  - B (the corrected selection test) on AUROC, reliability, ECE, Brier
  - the reuse gap at the selected layer on the same four metrics
  - the selective-prediction risk violation at 30/50/70% coverage (main only)
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "results" / "resplit_dependence_correction.json"
J = 50


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cr = _load(ROOT / "112_corrected_report.py", "corrected_report")
sp = _load(ROOT / "97_selective_prediction_consequence.py", "selective_prediction")
sp.ROOT = ROOT   # 97 resolves its root one level up; point it at this repository


def both(x, ratio):
    x = np.asarray(x, float)
    m, s = x.mean(), x.std(ddof=1)
    tcrit = stats.t.ppf(0.975, len(x) - 1)
    out = {}
    for name, f in (("naive", 1.0 / len(x)), ("nadeau_bengio", 1.0 / len(x) + ratio)):
        se = np.sqrt(f * s ** 2)
        lo, hi = m - tcrit * se, m + tcrit * se
        t = m / se
        out[name] = {"mean": float(m), "t": float(t),
                     "p": float(2 * stats.t.sf(abs(t), len(x) - 1)),
                     "ci_95": [float(lo), float(hi)],
                     "excludes_zero": bool(lo > 0 or hi < 0)}
    return out


def main_harness():
    a = np.load(ROOT / "results" / "case_study_2_probe_scores.npz")
    L, R = int(a["n_layers"]), int(a["n_reps"])
    CV = {m: np.empty((R, L)) for m, _ in cr.METRICS}
    HO = {m: np.empty((R, L)) for m, _ in cr.METRICS}
    ls = np.empty(R, int)
    for r in range(R):
        c, h, l = cr.per_layer(a, r, L)
        for m, _ in cr.METRICS:
            CV[m][r], HO[m][r] = c[m], h[m]
        ls[r] = l
    rows = np.arange(R)
    ratio = 300 / 400
    res = {"ratio_test_train": ratio, "B": {}, "gap_at_selected": {}, "risk_violation": {}}
    for m, _ in cr.METRICS:
        res["B"][m] = both(HO[m][rows, ls] - HO[m].mean(axis=1), ratio)
        res["gap_at_selected"][m] = both(CV[m][rows, ls] - HO[m][rows, ls], ratio)
    reps = sp.mechanism2_reps()
    for cov in sp.COVERAGE_TARGETS:
        viol = []
        for s_l, y_l, s_c, y_c in reps:
            tau, _, believed = sp.risk_at_coverage(s_l, y_l, cov)
            _, actual = sp.apply_threshold(s_c, y_c, tau)
            viol.append(actual - believed)
        res["risk_violation"][f"{int(cov*100)}%"] = both(viol, ratio)
    return res


def second_family():
    d = json.load(open(ROOT / "results" / "second_family_corrected_report.json"))
    ratio = 171 / 229
    out = {"ratio_test_train": ratio, "by_model": {}}
    for name, v in d["by_model"].items():
        out["by_model"][name] = {"B": {}, "gap_at_selected": {}}
        for m, e in v["metrics"].items():
            out["by_model"][name]["B"][m] = both(e["B_per_rep"], ratio)
            out["by_model"][name]["gap_at_selected"][m] = both(e["gap_per_rep"], ratio)
    return out


def show(title, block):
    print(f"\n{title}")
    for m, r in block.items():
        n, c = r["naive"], r["nadeau_bengio"]
        print(f"  {m:<12} mean {n['mean']:+.5f} | naive t {n['t']:>7.2f} "
              f"{'EST' if n['excludes_zero'] else 'n.s.':>4} | corrected t {c['t']:>6.2f} "
              f"[{c['ci_95'][0]:+.5f},{c['ci_95'][1]:+.5f}] {'EST' if c['excludes_zero'] else 'n.s.'}")


def main():
    mh = main_harness()
    show("MAIN HARNESS -- B (selected layer transfers?)", mh["B"])
    show("MAIN HARNESS -- reuse gap at the selected layer", mh["gap_at_selected"])
    show("MAIN HARNESS -- selective-prediction risk violation", mh["risk_violation"])
    sf = second_family()
    for name, v in sf["by_model"].items():
        show(f"SECOND FAMILY {name} -- B", v["B"])
        show(f"SECOND FAMILY {name} -- reuse gap at the selected layer", v["gap_at_selected"])
    with open(OUT_PATH, "w") as f:
        json.dump({"J": J, "reference": "Nadeau & Bengio (2003), Machine Learning 52:239-281",
                   "main_harness": mh, "second_family": sf}, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
