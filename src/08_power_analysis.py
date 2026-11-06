#!/usr/bin/env python3
# encoding: utf-8

"""
    Sample-size / power analysis for the headline statistics of the
    manuscript.

    Three families of estimates:
      1. Pairwise misranking P(swap), under Monte Carlo and Bootstrap: the
         95% half-width due to the finite trial sample (trials resampled
         with replacement), plus the simulation half-width for reference.
      2. Two-way ANOVA (B1, B3): minimum detectable partial eta^2 at
         alpha = 0.05 and power = 0.80, given the degrees of freedom of
         each effect and the observed N per cell.
      3. ICC(3,1): standard error of the estimate. With k = 6 or 9 cells
         and n = 800 paired seeds, even ICC = 0.05 is many SE above 0.

    Outputs go to analysis_results/tables/.

    Usage:
        python 08_power_analysis.py
"""

import importlib.util
import os
import tempfile

import path_setup  # noqa: F401

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist, ncf
from scipy.optimize import brentq

from utils.log_utils import log, logTable, bcolors
from utils.compute_switched_probability import (computeSwitchedProbabilityT,
                                                montecarlo_samples, bootstrap_samples)

from cpc18_utils import output_path
from cpc18_utils.training_logs import loadTrainingLog
from cpc18_models.ablation import B1_CLASSES, B3_CLASSES


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
        flat CSV of analysis_results/csv/, keeping only seeds present for
        every condition.
        Mirrors the paired-seed matrix construction in 10_paper_tables.py.
    """
    df = pd.read_csv(os.path.join(csv_path, csv_name), float_precision="round_trip")
    rows = {}
    for _, r in df.iterrows():
        rows.setdefault(int(r["seed"]), {})[r["model"]] = float(r["mse_x100"])
    seeds = [s for s in rows if all(m in rows[s] for m in conditions)]
    M = np.array([[rows[s][m] for m in conditions] for s in seeds])
    return M


def _iccMatrix(csv_name, conditions=None, expected_k=None):
    """Paired-seed matrix of the ICC(3,1) from a flat CSV of analysis_results/csv/."""
    if conditions is None:
        df = pd.read_csv(os.path.join(csv_path, csv_name))
        conditions = sorted(df["model"].unique())
    M = _pairedMatrix(csv_name, conditions)
    if expected_k is not None and M.shape[1] != expected_k:
        raise SystemExit(f"{csv_name}: {M.shape[1]} conditions instead of {expected_k}")
    return M


# ---------------------------------------------------------------------
# 1. Pairwise misranking precision (Monte Carlo and Bootstrap)
# ---------------------------------------------------------------------

MISRANK_SIMS = 40000
MISRANK_OUTER = 1000


def _swapProbability(a, b, method, n_sims, rng):
    """
        P(swap | N=1) as in computeSwitchedProbability: the reference order
        is the best-run order (lowest raw MSE wins) and one run is drawn per
        model, from Normal(mean, std) (MonteCarlo) or from the runs
        themselves (Bootstrap).
    """
    if method == "MonteCarlo":
        ra = rng.normal(a.mean(), a.std(), n_sims)
        rb = rng.normal(b.mean(), b.std(), n_sims)
    else:
        ra = rng.choice(a, n_sims, replace=True)
        rb = rng.choice(b, n_sims, replace=True)
    if a.min() < b.min():
        return float(np.mean(ra > rb))
    return float(np.mean(rb > ra))


def _swapSamplingHalfWidth(a, b, method, rng):
    """
        Uncertainty due to the finite trial sample: resample the runs of each
        model with replacement, recompute the estimate, and return 1.96 x SD
        over MISRANK_OUTER replicates.
    """
    replicates = [_swapProbability(rng.choice(a, a.size, replace=True),
                                   rng.choice(b, b.size, replace=True),
                                   method, MISRANK_SIMS, rng)
                  for _ in range(MISRANK_OUTER)]
    return 1.96 * float(np.std(replicates, ddof=1))


def _loadNumberedScript(filename, name):
    """Import a numeric-prefixed entry script, which a plain import cannot handle."""
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(os.path.dirname(os.path.abspath(__file__)), filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _headlineSwapProbabilities(negated, a_name, b_name, seed):
    """
        N=1 estimates computed exactly as 02_analyze.py computes them for the
        pair (same helper, sorted names, same seed and draw order), so the
        point estimates equal the pairwise tables reported in the manuscript.
        The helper's own tables go to a throwaway directory.
    """
    names = sorted([a_name, b_name])
    with tempfile.TemporaryDirectory() as scratch:
        os.makedirs(os.path.join(scratch, "tables"))
        np.random.seed(seed)
        mc = computeSwitchedProbabilityT(negated, names, n_simulations=montecarlo_samples,
                                         resampling_method="MonteCarlo", analysis_path=scratch)
        boot = computeSwitchedProbabilityT(negated, names, n_simulations=bootstrap_samples,
                                           resampling_method="Bootstrap", analysis_path=scratch)
    return {"MonteCarlo": float(mc[1][1]), "Bootstrap": float(boot[1][1])}


def misrankingPrecisionTable():
    """
        P(swap | N=1) for representative model pairs under both estimators,
        with the 95% half-width due to the finite trial sample and, for
        reference, the simulation half-width of MISRANK_SIMS draws.
    """
    log("\n=== 1. Pairwise misranking precision (Monte Carlo and Bootstrap) ===",
        color=bcolors.HEADER)
    pairs = [
        ("BourginMLP", "PsychForest"),
        ("BourginMLP", "BEAST_GB"),
        ("LinearLogistic", "BourginMLP"),
    ]
    rows = [["Pair", "N each", "Estimator", "$P(\\mathrm{swap}|N{=}1)$",
             "95% half-width (trials)", "95% half-width (simulation)"]]
    analyze = _loadNumberedScript("02_analyze.py", "analyze_02")
    negated = analyze.negatedMseByModel(analyze.loadMetrics("shared_pool", regime=2))
    for a_name, b_name in pairs:
        if a_name not in negated or b_name not in negated:
            log(f"[skip] {a_name} or {b_name} has no trials",
                color=bcolors.WARNING)
            continue
        a = -100.0 * np.asarray(negated[a_name])
        b = -100.0 * np.asarray(negated[b_name])
        n_each = min(len(a), len(b))
        no_overlap = a.max() < b.min() or b.max() < a.min()
        headline = _headlineSwapProbabilities(negated, a_name, b_name, analyze.MISRANK_SEED)
        for method in ("MonteCarlo", "Bootstrap"):
            p = headline[method]
            if method == "Bootstrap" and no_overlap:
                half_width = f"<{3 / n_each:.3f} (rule of three)"
            else:
                half_width = f"{_swapSamplingHalfWidth(a, b, method, np.random.default_rng(1)):.3f}"
            simulation_half_width = 1.96 * np.sqrt(p * (1 - p) / MISRANK_SIMS)
            rows.append([f"{a_name} vs {b_name}", f"{n_each}", method,
                         f"{p:.3f}", half_width, f"{simulation_half_width:.4f}"])
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
            n_per_cell.append(len(loadTrainingLog(path)))
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
        Large-sample SE of ICC(3,1), from Fisher's variance of the intraclass correlation:
        Var = 2 (1 - ICC)^2 (1 + (k-1) ICC)^2 / (k (k-1) (n-1)).
    """
    n = n_seeds
    k = k_cells
    return float(np.sqrt(2 * (1 - icc) ** 2 * (1 + (k - 1) * icc) ** 2 /
                         (k * (k - 1) * (n - 1))))


def iccExactCi(M, alpha=0.05):
    """ Exact confidence interval of ICC(3,1) from F = MSr / MSe (Shrout and Fleiss, 1979). """
    n, k = M.shape
    gm = M.mean()
    msr = k * ((M.mean(1) - gm) ** 2).sum() / (n - 1)
    mse = ((M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    df_r, df_e = n - 1, (n - 1) * (k - 1)
    f_low = (msr / mse) / f_dist.ppf(1 - alpha / 2, df_r, df_e)
    f_high = (msr / mse) * f_dist.ppf(1 - alpha / 2, df_e, df_r)
    return (f_low - 1) / (f_low + k - 1), (f_high - 1) / (f_high + k - 1)


def iccPrecisionTable():
    log("\n=== 3. ICC precision ===", color=bcolors.HEADER)
    rows = [["Test", "$n$ seeds", "$k$ conditions", "observed ICC", "95% CI (exact)", "approx SE", "approx z"]]
    observed = [
        ("Cross-model (Regime 2)", _iccMatrix("results_sharedpool_r2.csv", R2_ORDER)),
        ("B1 ablation", _iccMatrix("results_ablation_b1.csv", expected_k=len(B1_CLASSES))),
        ("B3 ablation", _iccMatrix("results_ablation_b3.csv", expected_k=len(B3_CLASSES))),
    ]
    # The exploratory first regime is only in the training logs, not in the deposit.
    if os.path.exists(os.path.join(csv_path, "results_sharedpool_r1.csv")):
        observed.append(("Cross-model (Regime 1, raw)", _iccMatrix("results_sharedpool_r1.csv")))
    for name, M in observed:
        n, k = M.shape
        icc = iccThreeOne(M)
        low, high = iccExactCi(M)
        se = iccStandardError(icc, n, k)
        z = icc / se if se else float("nan")
        rows.append([name, str(n), str(k), f"{icc:+.4f}", f"[{low:+.3f}, {high:+.3f}]", f"{se:.4f}", f"{z:+.2f}"])
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
