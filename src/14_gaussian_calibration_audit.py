#!/usr/bin/env python3
"""Audit of the Gaussian (parametric Monte Carlo) assumption on CPC18.

1. Per-model lower-tail calibration: how much mass the fitted Gaussian
   puts at or below the empirical P5/P10/P25 (nominal 0.05/0.10/0.25),
   and the implied expected trials-to-P10 (Gaussian vs the empirical 10).
2. Per-pair single-run swap probability P(X_A < X_B):
   Gaussian closed form  Phi((mu_B-mu_A)/sqrt(sA^2+sB^2))
   vs the empirical independent-pairs estimate (all 800x800 pairs).
   Reports the error for all 15 pairs.
Data: analysis_results/csv/results_sharedpool_r2.csv (main run, N=800).
"""
import csv
import itertools
import os

import numpy as np
from scipy import stats
from scipy.stats import norm

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "analysis_results", "csv", "results_sharedpool_r2.csv")

vals = {}
with open(CSV, newline="") as f:
    for row in csv.DictReader(f):
        vals.setdefault(row["model"], []).append(float(row["mse"]) * 100.0)
vals = {m: np.asarray(v) for m, v in vals.items()}

print("== Per-model Gaussian lower-tail calibration (N=%d each) ==" % len(next(iter(vals.values()))))
print(f"{'model':<15} {'skew':>6} {'kurt':>6} | {'G@P5':>6} {'G@P10':>6} {'G@P25':>6} | {'E[trials to P10]':>16}")
for m, v in sorted(vals.items()):
    mu, sd = v.mean(), v.std(ddof=1)
    sk, ku = stats.skew(v), stats.kurtosis(v)
    row = []
    for p in (5, 10, 25):
        q = np.percentile(v, p)
        row.append(norm.cdf(q, mu, sd))
    trials_g = 1.0 / row[1] if row[1] > 0 else float("inf")
    print(f"{m:<15} {sk:6.2f} {ku:6.1f} | {row[0]:6.3f} {row[1]:6.3f} {row[2]:6.3f} | "
          f"gauss {trials_g:5.1f} vs emp 10.0")

print()
print("== Per-pair single-run swap probability: Gaussian vs empirical (independent pairs) ==")
print(f"{'pair':<32} {'gauss':>7} {'emp':>7} {'abs err':>8}")
errs = []
for a, b in itertools.combinations(sorted(vals), 2):
    va, vb = vals[a], vals[b]
    mua, sda = va.mean(), va.std(ddof=1)
    mub, sdb = vb.mean(), vb.std(ddof=1)
    # P(A beats B) = P(X_A < X_B), lower is better
    g = norm.cdf(0, mua - mub, np.sqrt(sda**2 + sdb**2))
    emp = float((va[:, None] < vb[None, :]).mean())
    # swap prob relative to the mean order: P(worse-on-average model wins)
    swap_g, swap_e = (1 - g, 1 - emp) if mua < mub else (g, emp)
    errs.append((abs(swap_g - swap_e), a, b, swap_g, swap_e))
    print(f"{a+'-'+b:<32} {swap_g:7.3f} {swap_e:7.3f} {abs(swap_g-swap_e):8.3f}")

errs.sort(reverse=True)
print()
print("Worst 5 pairs by |Gaussian - empirical|:")
for e, a, b, g, emp in errs[:5]:
    print(f"  {a}-{b}: gauss {g:.3f} vs emp {emp:.3f} (err {e:.3f})")
med = float(np.median([e[0] for e in errs]))
print(f"Median abs error across 15 pairs: {med:.3f}")

# Pairs among the three near-Gaussian models only
NEAR = ("BEAST_GB", "PsychForest", "LinearLogistic")
sub = [e for e in errs if e[1] in NEAR and e[2] in NEAR]
print(f"Max abs error among near-Gaussian trio pairs: {max(s[0] for s in sub):.4f}")
bmlp = [e for e in errs if "BourginMLP" in (e[1], e[2])]
print(f"Max abs error among pairs involving BourginMLP: {max(s[0] for s in bmlp):.3f}")
