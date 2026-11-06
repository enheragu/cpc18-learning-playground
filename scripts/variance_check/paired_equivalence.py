#!/usr/bin/env python3
"""Seed-paired cross-machine comparison for the paper's cross-machine section.

For each (condition, machine) vs the reference:
  - paired per-seed differences d_i = MSE_tag(seed i) - MSE_ref(seed i), in MSE*100
  - mean diff, median diff, sd of diffs, SE
  - paired t-test p, Wilcoxon signed-rank p (two-sided)
  - 90% t-CI of the mean diff (the TOST interval)
  - TOST verdict at delta = 0.05 and 0.10 (the manuscript's margins)
  - shift as % of reference mean and % of reference seed sigma
For stable models: max |diff| and max relative diff.
"""
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "cad", os.path.join(_HERE, "compare_and_decompose.py"))
cad = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cad)

import numpy as np
from scipy import stats

CONDS = {c[0]: (c[1], c[2]) for c in cad.CONDITIONS}
TAGS = [("M03", "AVX2"), ("C02", "AMD Zen 2"), ("M06", "ARM DEFAULT"), ("M03-forced-DEFAULT", "x86 DEFAULT forced")]
PAIRED_CONDS = ["BourginMLP_B32_L0_001", "BourginMLP_B16_L0_0001"]
STABLE_CONDS = ["SparseMLP_E20_L0_001", "SparseMLP_E40_L0_0001", "ContextDepNet", "BEAST_GB"]

out = {}
for cond in PAIRED_CONDS + STABLE_CONDS:
    csv_name, yaml_name = CONDS[cond]
    ref = cad.load_reference(csv_name, cond.split("_")[0] if cond.startswith("BEAST") else cond)
    if not ref:  # model column name differs for regime2 models
        ref = cad.load_reference(csv_name, cond)
    out[cond] = {}
    for tag, tier in TAGS:
        try:
            ours = cad.load_ours(cond, tag, yaml_name)
        except FileNotFoundError:
            out[cond][tag] = "missing"
            continue
        seeds = sorted(set(ref) & set(ours))
        r = np.array([ref[s] for s in seeds]) * 100.0
        o = np.array([ours[s] for s in seeds]) * 100.0
        d = o - r
        n = len(d)
        res = {"tier": tier, "n": n,
               "ref_mean": round(float(r.mean()), 4), "tag_mean": round(float(o.mean()), 4),
               "mean_diff": round(float(d.mean()), 4), "median_diff": round(float(np.median(d)), 4),
               "sd_diff": round(float(d.std(ddof=1)), 4)}
        if cond in PAIRED_CONDS:
            se = d.std(ddof=1) / np.sqrt(n)
            tcrit = stats.t.ppf(0.95, n - 1)
            ci90 = (float(d.mean() - tcrit * se), float(d.mean() + tcrit * se))
            t_p = float(stats.ttest_rel(o, r).pvalue)
            w_p = float(stats.wilcoxon(o, r).pvalue) if np.any(d != 0) else 1.0
            sigma_ref = r.std(ddof=1)
            res.update({
                "se": round(float(se), 4), "ci90": [round(ci90[0], 4), round(ci90[1], 4)],
                "paired_t_p": f"{t_p:.2e}", "wilcoxon_p": f"{w_p:.2e}",
                "tost_pass_005": bool(ci90[0] > -0.05 and ci90[1] < 0.05),
                "tost_pass_010": bool(ci90[0] > -0.10 and ci90[1] < 0.10),
                "shift_pct_of_mean": round(float(100 * d.mean() / r.mean()), 2),
                "shift_pct_of_sigma": round(float(100 * d.mean() / sigma_ref), 2),
                "sd_diff_pct_of_sigma": round(float(100 * d.std(ddof=1) / sigma_ref), 1),
            })
        else:
            res.update({"max_abs_diff": f"{float(np.max(np.abs(d))):.2e}",
                        "max_rel_diff": f"{float(np.max(np.abs(d) / r)):.2e}",
                        "bit_exact_frac": round(float(np.mean(d == 0)), 4)})
        out[cond][tag] = res

print(json.dumps(out, indent=1))
