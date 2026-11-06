#!/usr/bin/env python3
# encoding: utf-8
"""Random-effects variance components (EMS, sigma^2_eps = 0 by determinism)
for the 35x35 init x order factorial, with row/col bootstrap CIs and the
capping robustness — the sigma^2 presentation used by the manuscript (variance-decomposition table).
Run:  python src/13_variance_components.py

Balanced a x b, n=1, random model with sigma2_eps = 0:
  MS_A  = SS_A/(a-1);  MS_B = SS_B/(b-1);  MS_AB = SS_AB/((a-1)(b-1))
  s2_AB = MS_AB
  s2_A  = (MS_A - MS_AB)/b     (truncated at 0)
  s2_B  = (MS_B - MS_AB)/a     (truncated at 0)
Shares = each / (s2_A + s2_B + s2_AB).
"""
import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260710)
_CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "analysis_results", "csv")
dec = pd.read_csv(os.path.join(_CSV_DIR, "results_factorial_decomp.csv"), float_precision="round_trip")


def components(gv):
    a, b = gv.shape
    grand = gv.mean()
    ssa = b * ((gv.mean(axis=1) - grand) ** 2).sum()
    ssb = a * ((gv.mean(axis=0) - grand) ** 2).sum()
    sst = ((gv - grand) ** 2).sum()
    ssab = sst - ssa - ssb
    msa, msb = ssa / (a - 1), ssb / (b - 1)
    msab = ssab / ((a - 1) * (b - 1))
    s2ab = msab
    s2a = max(0.0, (msa - msab) / b)
    s2b = max(0.0, (msb - msab) / a)
    tot = s2a + s2b + s2ab
    return np.array([s2a, s2b, s2ab]), np.array([s2a, s2b, s2ab]) / tot * 100


for m in sorted(dec.model.unique()):
    gv = dec[dec.model == m].pivot_table(
        index="init_seed", columns="order_seed", values="mse_x100").values
    comp, pct = components(gv)
    boots = []
    for _ in range(3000):
        sub = gv[np.ix_(RNG.integers(0, 35, 35), RNG.integers(0, 35, 35))]
        boots.append(components(sub)[1])
    lo, hi = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
    print(f"{m:15s} s2: init {comp[0]:.4f} order {comp[1]:.4f} "
          f"inter {comp[2]:.4f}")
    print(f"{'':15s} %:  init {pct[0]:5.1f} [{lo[0]:4.1f},{hi[0]:5.1f}]  "
          f"order {pct[1]:4.1f} [{lo[1]:4.1f},{hi[1]:4.1f}]  "
          f"inter {pct[2]:5.1f} [{lo[2]:4.1f},{hi[2]:5.1f}]")
    for q in (95, 90):
        cp = components(np.minimum(gv, np.percentile(gv, q)))[1]
        print(f"{'':15s} cap@P{q}: init {cp[0]:.1f} order {cp[1]:.1f} "
              f"inter {cp[2]:.1f}")
