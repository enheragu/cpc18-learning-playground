#!/usr/bin/env python3
# encoding: utf-8

"""
Does the DISTRIBUTION survive a change of hardware, even when individual runs
do not?

Bit-exactness across arithmetic paths is lost, and per-seed ranking with it
for the sensitive conditions (Appendix D of the manuscript). The narrower
question is whether the summary statistics still hold when someone replicates
outside the controlled environment.

Runs the distribution comparison of Appendix D for any tag, so a new machine
can be tested the same way. Reports, per condition:

  - centre / spread / shape agreement (Mann-Whitney, Levene, KS)
  - percentile deltas vs the reference, p50 through p99
  - bootstrap CI of the empirical P90 (upper tail), to say whether an
    observed shift is distinguishable from sampling noise at all

Usage:  distribution_robustness.py <TAG> [<TAG> ...]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import path_setup  # noqa: F401

import importlib.util
import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("cd_", os.path.join(_HERE, "compare_and_decompose.py"))
_cd = importlib.util.module_from_spec(_spec)
_argv, sys.argv = sys.argv, ["compare_and_decompose.py"]
_spec.loader.exec_module(_cd)
sys.argv = _argv

PCTS = [50, 75, 90, 95, 99]
N_BOOT = 5000
BOOT_SEED = 20260522


def boot_ci(x, pct, n=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    vals = np.percentile(x[idx], pct, axis=1)
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def main():
    tags = sys.argv[1:]
    if not tags:
        raise SystemExit(__doc__)

    for model_name, csv_name, yaml_name in _cd.CONDITIONS:
        ref_map = _cd.load_reference(csv_name, model_name)
        rows = []
        for tag in tags:
            try:
                ours = _cd.load_ours(model_name, tag, yaml_name)
            except FileNotFoundError:
                continue
            common = sorted(set(ref_map) & set(ours))
            if not common:
                continue
            rows.append((tag, np.array([ref_map[s] for s in common]),
                         np.array([ours[s] for s in common])))
        if not rows:
            continue

        print(f"\n{'='*78}\n{model_name}\n{'='*78}")
        ref = rows[0][1]
        print(f"  reference: n={len(ref)}  mean={ref.mean():.6f}  "
              f"median={np.median(ref):.6f}  P90={np.percentile(ref,90):.6f}")
        lo, hi = boot_ci(ref, 90)
        print(f"              bootstrap 95% CI of P90: [{lo:.6f}, {hi:.6f}]")

        for tag, r, o in rows:
            mw = stats.mannwhitneyu(r, o).pvalue
            lev = stats.levene(r, o).pvalue
            ks = stats.ks_2samp(r, o).pvalue
            d = {p: (np.percentile(o, p) / np.percentile(r, p) - 1) * 100 for p in PCTS}
            olo, ohi = boot_ci(o, 90)
            rlo, rhi = boot_ci(r, 90)
            overlap = not (ohi < rlo or olo > rhi)
            print(f"\n  --- {tag} ---")
            print(f"    centre (Mann-Whitney) p={mw:.3f} | spread (Levene) p={lev:.3f} "
                  f"| shape (KS) p={ks:.3f}")
            print("    delta percentiles: " +
                  "  ".join(f"p{p}={d[p]:+.1f}%" for p in PCTS))
            print(f"    P90={np.percentile(o,90):.6f}  95% CI=[{olo:.6f}, {ohi:.6f}]  "
                  f"-> {'overlaps the reference' if overlap else '*** NO OVERLAP ***'}")


if __name__ == "__main__":
    main()
