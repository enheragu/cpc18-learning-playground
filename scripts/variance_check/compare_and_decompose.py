#!/usr/bin/env python3
# encoding: utf-8
"""
Reusable cross-CPU comparison. Reads the reference CSVs (machine M00)
and a tagged output_data_<TAG>/ tree (this repo's own trainer output) for
the 6 replicated conditions (four ablation cells and two main-run models,
see CONDITIONS), and reports:

  - per-seed exact-match rate and absolute-difference stats
  - KS test (distribution-shape agreement)
  - Spearman rank correlation + misranking probability (overall and for
    the closest 10% of pairs) -- same "closest-pair misranking" framing
    the repo's own B1/B3 analysis (07_analyze_ablation.py) uses.
  - two-way variance decomposition (seed x machine), using the exact SS
    formula from src/06_analyze_decomp.py's twoWayDecomposition(), just
    relabelling "init x order" as "seed x machine". Answers directly:
    what fraction of total MSE variance is seed, what fraction is a
    systematic hardware shift, what fraction is seed*hardware
    interaction (hardware noise that depends on which seed it hits --
    this is what drives misranking).

Usage: python compare_and_decompose.py <TAG>
  e.g. python compare_and_decompose.py M03
       python compare_and_decompose.py M06
"""
import os
import sys
import csv

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
import path_setup  # noqa: F401
from cpc18_utils.training_logs import loadTrainingLog

import numpy as np
from scipy import stats

_ABLATION_YAML = "sharedpool_r2_ablation_training_metrics.yaml"
_REGIME2_YAML = "sharedpool_r2_training_metrics.yaml"
CONDITIONS = [
    ("BourginMLP_B32_L0_001", "results_ablation_b1.csv", _ABLATION_YAML),
    ("BourginMLP_B16_L0_0001", "results_ablation_b1.csv", _ABLATION_YAML),
    ("SparseMLP_E20_L0_001", "results_ablation_b3.csv", _ABLATION_YAML),
    ("SparseMLP_E40_L0_0001", "results_ablation_b3.csv", _ABLATION_YAML),
    ("ContextDepNet", "results_sharedpool_r2.csv", _REGIME2_YAML),
    ("BEAST_GB", "results_sharedpool_r2.csv", _REGIME2_YAML),
]


def load_reference(csv_name, model_name):
    path = os.path.join(REPO, "analysis_results", "csv", csv_name)
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] == model_name:
                out[int(row["seed"])] = float(row["mse"])
    return out


def load_ours(model_name, tag, yaml_name):
    path = os.path.join(REPO, f"output_data_{tag}", model_name, yaml_name)
    return {int(entry["seed"]): float(entry["mse"])
            for entry in loadTrainingLog(path).values() if "seed" in entry}


def pairwise_misranking(ref_vals, our_vals, close_frac=0.10):
    n = len(ref_vals)
    ref_diff = ref_vals[:, None] - ref_vals[None, :]
    our_diff = our_vals[:, None] - our_vals[None, :]
    iu = np.triu_indices(n, k=1)
    rd, od = ref_diff[iu], our_diff[iu]
    misranked = rd * od < 0  # a strict inversion: a tie is not a reversal
    overall_rate = misranked.mean()
    thresh = np.quantile(np.abs(rd), close_frac)
    close_rate = misranked[np.abs(rd) <= thresh].mean()
    return overall_rate, close_rate


def _varianceComponentShares(SS_a, SS_b, SS_resid, n_a, n_b):
    """Verbatim from src/06_analyze_decomp.py::_varianceComponentShares,
    relabelled generically (a=seed, b=machine)."""
    if n_a < 2 or n_b < 2:
        return float("nan"), float("nan"), float("nan")
    MS_a = SS_a / (n_a - 1)
    MS_b = SS_b / (n_b - 1)
    MS_inter = SS_resid / ((n_a - 1) * (n_b - 1))
    s2_inter = MS_inter
    s2_a = max(0.0, MS_a - MS_inter) / n_b
    s2_b = max(0.0, MS_b - MS_inter) / n_a
    tot = s2_a + s2_b + s2_inter
    if tot <= 0:
        return float("nan"), float("nan"), float("nan")
    return 100 * s2_a / tot, 100 * s2_b / tot, 100 * s2_inter / tot


def twoWayDecomposition(M):
    """Verbatim SS formula from 06_analyze_decomp.py::twoWayDecomposition.
    M: (n_seeds, n_machines=2) matrix. Rows=seed effect, cols=machine effect."""
    grand = M.mean()
    row_means = M.mean(axis=1, keepdims=True)
    col_means = M.mean(axis=0, keepdims=True)
    SS_total = float(np.sum((M - grand) ** 2))
    SS_seed = float(M.shape[1] * np.sum((row_means - grand) ** 2))
    SS_machine = float(M.shape[0] * np.sum((col_means - grand) ** 2))
    SS_resid = SS_total - SS_seed - SS_machine
    n_seed, n_machine = M.shape
    pct_seed = 100 * SS_seed / SS_total if SS_total > 0 else float("nan")
    pct_machine = 100 * SS_machine / SS_total if SS_total > 0 else float("nan")
    pct_resid = 100 * SS_resid / SS_total if SS_total > 0 else float("nan")
    vc_seed, vc_machine, vc_resid = _varianceComponentShares(
        SS_seed, SS_machine, SS_resid, n_seed, n_machine)
    return dict(pct_seed=pct_seed, pct_machine=pct_machine, pct_resid=pct_resid,
                vc_seed=vc_seed, vc_machine=vc_machine, vc_resid=vc_resid)


def main(tag):
    print(f"=== {tag} vs reference (M00) ===\n")
    hdr = (f"{'condition':26s} {'bit_exact':>10s} {'ks_p':>7s} {'spearman':>9s} "
           f"{'misrank_close10%':>17s} | {'%seed':>7s} {'%machine':>9s} {'%interact':>10s}")
    print(hdr)
    for model_name, csv_name, yaml_name in CONDITIONS:
        ref = load_reference(csv_name, model_name)
        try:
            ours = load_ours(model_name, tag, yaml_name)
        except FileNotFoundError:
            print(f"{model_name:26s}  (no data yet for {tag})")
            continue
        common = sorted(set(ref) & set(ours))
        if not common:
            print(f"{model_name:26s}  (no shared seeds)")
            continue
        ref_vals = np.array([ref[s] for s in common])
        our_vals = np.array([ours[s] for s in common])

        bit_exact = np.mean(ref_vals == our_vals)
        ks_p = stats.ks_2samp(ref_vals, our_vals).pvalue
        rho, _ = stats.spearmanr(ref_vals, our_vals)
        _, close_mr = pairwise_misranking(ref_vals, our_vals)

        M = np.column_stack([ref_vals, our_vals])  # (n_seeds, 2 machines)
        decomp = twoWayDecomposition(M)

        print(f"{model_name:26s} {bit_exact:10.2%} {ks_p:7.4f} {rho:9.4f} "
              f"{close_mr:17.4%} | {decomp['pct_seed']:7.2f} "
              f"{decomp['pct_machine']:9.2f} {decomp['pct_resid']:10.2f}")
    print("\n%seed / %machine / %interact add up to ~100% of the total variance "
          "(SS shares, descriptive). %interact = hardware noise that "
          "depends on the seed (this is what causes misranking); "
          "%machine = only the mean systematic shift between "
          "machines.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: compare_and_decompose.py <TAG>")
    main(sys.argv[1])
