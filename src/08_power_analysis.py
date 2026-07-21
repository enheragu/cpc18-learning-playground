#!/usr/bin/env python3
# encoding: utf-8

"""
    Sample-size / power analysis for the headline statistics of the
    manuscript.

    Three families of estimates:
      1. Pairwise misranking P(swap): the bootstrap 95% CI half-width on
         the estimate, given the per-cell N. With N=800 trials per model
         this is on the order of +/- 1-2 percentage points.
      2. Two-way ANOVA (B1, B3): minimum detectable partial eta^2 at
         alpha = 0.05 and power = 0.80, given the degrees of freedom of
         each effect and the observed N per cell.
      3. ICC(3,1): standard error of the estimate. With k = 6 or 9 cells
         and n = 800 paired seeds, even ICC = 0.05 is many SE above 0.

    Outputs go to analysis_results/tables/.

    Usage:
        python 08_power_analysis.py
"""

import os
import re

import path_setup  # noqa: F401

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist, ncf
from scipy.optimize import brentq

from utils.log_utils import log, logTable, bcolors

from cpc18_utils import output_path
from cpc18_utils.safe_yaml import fastLoadYaml
from cpc18_models.ablation import B1_CLASSES, B3_CLASSES


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_")
analysis_path = os.path.join(os.path.dirname(output_path), "analysis_results")
tables_path = os.path.join(analysis_path, "tables")
csv_path = os.path.join(analysis_path, "csv")

# Cross-model regime-2 leaderboard order (matches 10_paper_tables.py).
R2_ORDER = ["LinearLogistic", "BourginMLP", "ContextDepNet",
            "SparseMLP", "PsychForest", "BEAST_GB"]


# ---------------------------------------------------------------------
# Toggles
# ---------------------------------------------------------------------
enable_misranking_precision = True
enable_anova_power          = True
enable_icc_precision        = True


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _loadTrials(path):
    return [v for k, v in fastLoadYaml(path).items()
            if _DATE_RE.match(k) and isinstance(v, dict)]


def _mseX100(trials):
    return np.array([float(t["mse"]) * 100.0 for t in trials])


def iccThreeOne(M):
    """
        ICC(3,1) on a (n_seeds, k_conditions) matrix. Same formula as
        02_analyze.py / 07_analyze_ablation.py / 10_paper_tables.py:
            ICC(3,1) = (MSr - MSe) / (MSr + (k-1) * MSe)
    """
    n, k = M.shape
    gm = M.mean()
    msr = k * ((M.mean(1) - gm) ** 2).sum() / (n - 1)
    mse = ((M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + gm) ** 2).sum() \
        / ((n - 1) * (k - 1))
    if msr + (k - 1) * mse == 0:
        return float("nan")
    return (msr - mse) / (msr + (k - 1) * mse)


def _pairedMatrix(csv_name, conditions):
    """
        Build a (n_paired_seeds, k_conditions) matrix of mse_x100 from a
        committed CSV, keeping only seeds present for every condition.
        Mirrors the paired-seed matrix construction in 10_paper_tables.py.
    """
    df = pd.read_csv(os.path.join(csv_path, csv_name))
    rows = {}
    for _, r in df.iterrows():
        rows.setdefault(int(r["seed"]), {})[r["model"]] = float(r["mse_x100"])
    seeds = [s for s in rows if all(m in rows[s] for m in conditions)]
    M = np.array([[rows[s][m] for m in conditions] for s in seeds])
    return M


def _iccFromCsv(csv_name, conditions=None):
    """Compute ICC(3,1) and (n, k) from a committed CSV."""
    if conditions is None:
        df = pd.read_csv(os.path.join(csv_path, csv_name))
        conditions = sorted(df["model"].unique())
    M = _pairedMatrix(csv_name, conditions)
    return iccThreeOne(M), M.shape[0], M.shape[1]


# ---------------------------------------------------------------------
# 1. Pairwise misranking bootstrap CI
# ---------------------------------------------------------------------

def _pairwiseSwapProbBootstrap(a, b, n_per_model=1, n_boot=10000, rng=None):
    """
        Bootstrap distribution of P(swap rank) when each of `a` and `b`
        is averaged over `n_per_model` samples. Returns the mean estimate
        plus the 95% CI half-width.
    """
    rng = rng or np.random.default_rng(0)
    a = np.asarray(a)
    b = np.asarray(b)
    # Reference ordering uses BEST-RUN order, matching the headline
    # misranking statistic (computeSwitchedProbability orders groups by
    # np.max on negated MSE, i.e. the best/lowest raw MSE run wins).
    # Here `a`/`b` are raw mse*100 (lower = better), so the reference
    # winner is whichever model has the smaller minimum (best run).
    best_better = "a" if a.min() < b.min() else "b"
    swap_estimates = []
    for _ in range(n_boot):
        # Resample `n_per_model` runs per model and keep the best (lowest
        # raw MSE) of each draw, mirroring the best-run leaderboard logic.
        ra = rng.choice(a, size=(n_per_model,), replace=True).min()
        rb = rng.choice(b, size=(n_per_model,), replace=True).min()
        swapped = (best_better == "a" and ra > rb) or \
                  (best_better == "b" and rb > ra)
        swap_estimates.append(int(swapped))
    swap_estimates = np.array(swap_estimates)
    p_hat = swap_estimates.mean()
    se = np.sqrt(p_hat * (1 - p_hat) / n_boot)
    return p_hat, 1.96 * se


def misrankingPrecisionTable():
    """
        Bootstrap CI half-width on P(swap | N=1) for representative
        model pairs. The aim is to show that the precision of the
        estimate at N=800 is on the order of half a percentage point,
        well below the effect sizes we report.
    """
    log("\n=== 1. Pairwise misranking: bootstrap CI precision ===",
        color=bcolors.HEADER)
    pairs = [
        ("BourginMLP", "PsychForest"),
        ("BourginMLP", "BEAST_GB"),
        ("LinearLogistic", "BourginMLP"),
    ]
    rows = [["Pair", "N each", "$P(\\mathrm{swap}|N{=}1)$",
             "95% CI half-width"]]
    for a_name, b_name in pairs:
        a = _mseX100(_loadTrials(os.path.join(
            output_path, a_name,
            "sharedpool_r2_training_metrics.yaml")))
        b = _mseX100(_loadTrials(os.path.join(
            output_path, b_name,
            "sharedpool_r2_training_metrics.yaml")))
        if len(a) == 0 or len(b) == 0:
            log(f"[skip] {a_name} or {b_name} has no trials",
                color=bcolors.WARNING)
            continue
        p, hw = _pairwiseSwapProbBootstrap(
            a, b, n_per_model=1, n_boot=40000,
            rng=np.random.default_rng(0))
        rows.append([f"{a_name} vs {b_name}",
                     f"{min(len(a), len(b))}",
                     f"{p:.4f}", f"{hw:.4f}"])
    logTable(rows, tables_path,
             "Power analysis 1 - misranking estimate precision "
             "(95% CI half-width)")


# ---------------------------------------------------------------------
# 2. ANOVA: min detectable partial eta^2
# ---------------------------------------------------------------------

def minDetectablePartialEta2(df_effect, df_resid, alpha=0.05, power=0.80):
    """
        Minimum detectable partial eta^2 in a fixed-effects ANOVA, given
        the degrees of freedom of the effect and the residual.

        Uses the non-central F-distribution. lambda (non-centrality) is
        related to the population partial eta^2 by:
            f^2 = eta^2 / (1 - eta^2)
            lambda = f^2 * (df_effect + df_resid + 1)
    """
    F_crit = f_dist.ppf(1 - alpha, df_effect, df_resid)

    def gap(lam):
        return ncf.sf(F_crit, df_effect, df_resid, lam) - power

    try:
        lam = brentq(gap, 1e-3, 1e6)
    except ValueError:
        return float("nan")
    n = df_effect + df_resid + 1
    f2 = lam / n
    return f2 / (1 + f2)


def anovaPowerTable():
    log("\n=== 2. ANOVA min detectable partial eta^2 ===",
        color=bcolors.HEADER)
    rows = [["Study", "Effect", "df effect", "df resid",
             r"min detectable $\eta_p^2$"]]
    for label, class_list in (("B1", B1_CLASSES), ("B3", B3_CLASSES)):
        n_per_cell = []
        for cls in class_list:
            path = os.path.join(output_path, cls.__name__,
                                "sharedpool_r2_ablation_training_metrics.yaml")
            n_per_cell.append(len(_loadTrials(path)))
        n_total = sum(n_per_cell)
        df_f1, df_f2, df_inter = 2, 2, 4
        df_resid = n_total - (1 + df_f1 + df_f2 + df_inter)
        for name, df_eff in (("factor 1", df_f1), ("factor 2", df_f2),
                             ("factor1 x factor2", df_inter)):
            eta2 = minDetectablePartialEta2(df_eff, df_resid)
            rows.append([label, name, f"{df_eff}", f"{df_resid}",
                         f"{eta2*100:.4f} %"])
    logTable(rows, tables_path,
             "Power analysis 2 - ANOVA min detectable partial eta^2 "
             "(alpha=0.05, power=0.80)")


# ---------------------------------------------------------------------
# 3. ICC(3,1) standard error
# ---------------------------------------------------------------------

def iccStandardError(icc, n_seeds, k_cells):
    """
        Approximate SE of ICC(3,1). See McGraw & Wong (1996). For small
        ICC values (which is what we observe) the SE under the null is
        a good guide for whether the estimate is distinguishable from 0.

        SE(ICC) ~ sqrt( (1 - ICC^2)^2 * (1 + (k-1) * ICC)^2 /
                         (n_seeds * (k-1)) )
    """
    n = n_seeds
    k = k_cells
    return float(np.sqrt((1 - icc**2) ** 2 * (1 + (k - 1) * icc) ** 2 /
                         (n * (k - 1))))


def iccPrecisionTable():
    log("\n=== 3. ICC precision ===", color=bcolors.HEADER)
    rows = [["Test", "$n$ seeds", "$k$ conditions",
             "observed ICC", "approx SE", "approx z"]]
    # ICC values, n (paired seeds) and k (conditions) are computed at
    # runtime from the committed analysis_results/csv files via the same
    # iccThreeOne helper used in 02_analyze.py / 07_analyze_ablation.py /
    # 10_paper_tables.py, so this table can never drift from the data.
    icc_r2, n_r2, k_r2 = _iccFromCsv("results_sharedpool_r2.csv", R2_ORDER)
    icc_b1, n_b1, k_b1 = _iccFromCsv("results_ablation_b1.csv")
    icc_b3, n_b3, k_b3 = _iccFromCsv("results_ablation_b3.csv")
    icc_r1, n_r1, k_r1 = _iccFromCsv("results_sharedpool_r1.csv")
    observed = [
        ("Cross-model (Regime 2)",      n_r2, k_r2, icc_r2),
        ("B1 ablation",                 n_b1, k_b1, icc_b1),
        ("B3 ablation",                 n_b3, k_b3, icc_b3),
        ("Cross-model (Regime 1, raw)", n_r1, k_r1, icc_r1),
    ]
    for name, n, k, icc in observed:
        se = iccStandardError(icc, n, k)
        z = icc / se if se else float("nan")
        rows.append([name, str(n), str(k),
                     f"{icc:+.4f}", f"{se:.4f}", f"{z:+.2f}"])
    logTable(rows, tables_path, "Power analysis 3 - ICC precision")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def run():
    os.makedirs(tables_path, exist_ok=True)
    if enable_misranking_precision:
        misrankingPrecisionTable()
    if enable_anova_power:
        anovaPowerTable()
    if enable_icc_precision:
        iccPrecisionTable()
    log("\nDone. Tables in: " + tables_path, color=bcolors.OKGREEN)


if __name__ == "__main__":
    run()
