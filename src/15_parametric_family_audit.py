#!/usr/bin/env python3
"""Can parametric Monte Carlo be rescued by a better family than the normal?

Part 1 (large N): fit normal / lognormal / gamma / two-component mixture
  (split at the manuscript's antimode threshold MSE*100 = 1.5) to the
  BourginMLP main-run sample and measure
   - lower-tail calibration: family mass at/below empirical P5/P10/P25
   - the 5 BourginMLP pair win probabilities vs the empirical ones
     (partner modelled as Gaussian: calibrated to <=0.021 there).
Part 2 (small N): family identifiability at N in {5, 10, 20}: fraction of
  subsamples where the large-N-best family also wins by log-likelihood.
"""
import csv
import os

import numpy as np
from scipy.stats import norm, lognorm, gamma as gammad

rng = np.random.default_rng(20260522)
CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "analysis_results", "csv", "results_sharedpool_r2.csv")
vals = {}
with open(CSV, newline="") as f:
    for row in csv.DictReader(f):
        vals.setdefault(row["model"], []).append(float(row["mse"]) * 100.0)
vals = {m: np.asarray(v) for m, v in vals.items()}
bm = vals["BourginMLP"]

M = 2_000_000  # MC sample size per family

def sample_normal(x, size):
    return rng.normal(x.mean(), x.std(ddof=1), size)

def sample_lognorm(x, size):
    s, loc, sc = lognorm.fit(x, floc=0)
    return lognorm.rvs(s, loc=loc, scale=sc, size=size, random_state=rng)

def sample_gamma(x, size):
    a, loc, sc = gammad.fit(x, floc=0)
    return gammad.rvs(a, loc=loc, scale=sc, size=size, random_state=rng)

def sample_mixture(x, size, thr=1.5):
    """Two-component mixture at the manuscript's antimode threshold:
    lognormal for the converged mode, lognormal for the failed mode."""
    lo, hi = x[x <= thr], x[x > thr]
    w = len(lo) / len(x)
    pick = rng.random(size) < w
    out = np.empty(size)
    s1, l1, c1 = lognorm.fit(lo, floc=0)
    s2, l2, c2 = lognorm.fit(hi, floc=0)
    out[pick] = lognorm.rvs(s1, loc=l1, scale=c1, size=int(pick.sum()), random_state=rng)
    out[~pick] = lognorm.rvs(s2, loc=l2, scale=c2, size=int((~pick).sum()), random_state=rng)
    return out

FAMS = [("normal", sample_normal), ("lognormal", sample_lognorm),
        ("gamma", sample_gamma), ("mixture@1.5", sample_mixture)]

print("== Part 1a: BourginMLP lower-tail calibration (mass at/below empirical P5/P10/P25) ==")
qs = {p: np.percentile(bm, p) for p in (5, 10, 25)}
print(f"{'family':<12} {'@P5(0.05)':>10} {'@P10(0.10)':>11} {'@P25(0.25)':>11}")
samples = {}
for name, fn in FAMS:
    s = fn(bm, M)
    samples[name] = s
    print(f"{name:<12} {np.mean(s <= qs[5]):10.3f} {np.mean(s <= qs[10]):11.3f} {np.mean(s <= qs[25]):11.3f}")

print()
print("== Part 1b: BourginMLP pairs, P(worse-on-mean wins), family-for-BMLP vs empirical ==")
partners = ["BEAST_GB", "PsychForest", "LinearLogistic", "ContextDepNet", "SparseMLP"]
hdr = f"{'pair':<26} {'emp':>6}" + "".join(f"{n:>12}" for n, _ in FAMS)
print(hdr)
maxerr = {n: 0.0 for n, _ in FAMS}
for p in partners:
    vp = vals[p]
    emp = float((bm[:, None] < vp[None, :]).mean())          # P(BMLP wins)
    # swap prob = P(worse-on-mean model wins)
    flip = bm.mean() < vp.mean()
    emp_swap = 1 - emp if flip else emp
    row = f"{'BMLP-'+p:<26} {emp_swap:6.3f}"
    partner_s = rng.normal(vp.mean(), vp.std(ddof=1), M)
    for name, _ in FAMS:
        w = float(np.mean(samples[name] < partner_s))
        w_swap = 1 - w if flip else w
        row += f"{w_swap:12.3f}"
        maxerr[name] = max(maxerr[name], abs(w_swap - emp_swap))
    print(row)
print("max |err| per family:", {k: round(v, 3) for k, v in maxerr.items()})

print()
print("== Part 2: family identifiability at small N (BourginMLP subsamples, 2000 reps) ==")
print("fraction of subsamples where each family has the best log-likelihood")
for N in (5, 10, 20):
    wins = {"normal": 0, "lognormal": 0, "gamma": 0}
    for _ in range(2000):
        x = rng.choice(bm, N, replace=False)
        ll = {}
        ll["normal"] = norm.logpdf(x, x.mean(), max(x.std(ddof=1), 1e-9)).sum()
        try:
            s, l, c = lognorm.fit(x, floc=0)
            ll["lognormal"] = lognorm.logpdf(x, s, loc=l, scale=c).sum()
        except Exception:
            ll["lognormal"] = -np.inf
        try:
            a, l, c = gammad.fit(x, floc=0)
            ll["gamma"] = gammad.logpdf(x, a, loc=l, scale=c).sum()
        except Exception:
            ll["gamma"] = -np.inf
        wins[max(ll, key=ll.get)] += 1
    tot = sum(wins.values())
    print(f"N={N:>3}: " + "  ".join(f"{k} {v/tot:.2f}" for k, v in wins.items()))

# Large-N reference: which single family fits best on the full sample?
llf = {}
llf["normal"] = norm.logpdf(bm, bm.mean(), bm.std(ddof=1)).sum()
s, l, c = lognorm.fit(bm, floc=0); llf["lognormal"] = lognorm.logpdf(bm, s, loc=l, scale=c).sum()
a, l, c = gammad.fit(bm, floc=0); llf["gamma"] = gammad.logpdf(bm, a, loc=l, scale=c).sum()
print("full-sample log-likelihoods:", {k: round(v, 1) for k, v in llf.items()})
