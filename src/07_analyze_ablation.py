#!/usr/bin/env python3
# encoding: utf-8

"""
    B1 / B3 ablation analyser.

    For each ablation study (B1: BourginMLP batch x learning_rate;
    B3: SparseMLP epsilon x learning_rate), computes:
      1. Per-cell mean and standard deviation of MSE x 100.
      2. Two-way ANOVA (type-III SS) with partial eta^2 per term.
      3. ICC(3,1) across the 9 cells, with paired shared seeds as the
         "subjects".
      4. The closest-pair misranking probability (Monte Carlo + Bootstrap)
         between the two cells with the smallest mean gap.

    Usage:
        python 07_analyze_ablation.py
"""

import os
import re

import path_setup  # noqa: F401

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from utils.log_utils import log, logTable, bcolors
from utils.compute_switched_probability import computeSwitchedProbability

from cpc18_utils import output_path
from cpc18_utils.safe_yaml import fastLoadYaml
from cpc18_models.ablation import B1_CLASSES, B3_CLASSES


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_")

# The submodule misranking helper draws from the global np.random stream
# without seeding it; seed before the call so the closest-pair Monte
# Carlo / Bootstrap table is bit-reproducible across runs.
MISRANK_SEED = 20260522
analysis_path = os.path.join(os.path.dirname(output_path), "analysis_results")
tables_path = os.path.join(analysis_path, "tables")


# ---------------------------------------------------------------------
# Toggles
# ---------------------------------------------------------------------
enable_per_cell_means       = True
enable_two_way_anova        = True
enable_cross_cell_icc       = True
enable_closest_pair_misrank = True


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def loadCell(cls):
    """ Returns the list of (seed, mse * 100) tuples for one ablation cell. """
    path = os.path.join(output_path, cls.__name__,
                        "sharedpool_r2_ablation_training_metrics.yaml")
    data = fastLoadYaml(path)
    out = []
    for k, v in data.items():
        if _DATE_RE.match(k) and isinstance(v, dict) and "mse" in v:
            out.append((int(v["seed"]), float(v["mse"]) * 100.0))
    return out


def buildStudyDataframe(class_list):
    """
        Builds the long-format dataframe (factor1, factor2, seed, mse).
        factor1 is `batch_size` for B1 and `epsilon` for B3.
        factor2 is `learning_rate` in both studies.
    """
    rows = []
    for cls in class_list:
        f1 = getattr(cls, "batch_size", None) if cls in B1_CLASSES else \
             getattr(cls, "epsilon", None)
        f2 = cls.default_lr
        for seed, mse in loadCell(cls):
            rows.append({"f1": f1, "f2": f2, "seed": seed,
                         "mse": mse, "cell": cls.__name__})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Per-cell summary
# ---------------------------------------------------------------------

def perCellMeans(df, factor1_name, factor2_name):
    g = df.groupby(["f1", "f2"]).agg(
        n=("mse", "size"),
        mean=("mse", "mean"),
        std=("mse", "std"),
    ).reset_index()
    return g.rename(columns={"f1": factor1_name, "f2": factor2_name})


# ---------------------------------------------------------------------
# Two-way ANOVA
# ---------------------------------------------------------------------

def twoWayAnova(df):
    """ Type-III SS ANOVA with partial eta^2 per term.

        Uses sum-to-zero contrasts (C(..., Sum)). With the default
        treatment contrasts, type-III sums of squares for the main
        effects are simple effects at the reference level of the other
        factor, not genuine main effects. Sum coding makes the type-III
        main-effect SS the proper marginal effect; on the balanced
        design used here it coincides with the type-I decomposition.
    """
    formula = "mse ~ C(f1, Sum) * C(f2, Sum)"
    model = ols(formula, data=df).fit()
    aov = anova_lm(model, typ=3).copy()
    if "Intercept" in aov.index:
        aov = aov.drop("Intercept")
    ss_res = float(aov.loc["Residual", "sum_sq"])
    aov["part_eta2"] = aov["sum_sq"] / (aov["sum_sq"] + ss_res)
    aov.loc["Residual", "part_eta2"] = np.nan
    aov["pct_ss"] = 100.0 * aov["sum_sq"] / aov["sum_sq"].sum()
    return aov


# ---------------------------------------------------------------------
# Cross-cell ICC(3,1)
# ---------------------------------------------------------------------

def iccThreeOne(matrix):
    """
        ICC(3,1) on a (n_seeds, k_cells) matrix.
        Replicated from 02_analyze.iccThreeOne so this script is
        self-contained.
    """
    n, k = matrix.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = matrix.mean()
    seed_means = matrix.mean(axis=1)
    cond_means = matrix.mean(axis=0)
    SSb = k * np.sum((seed_means - grand) ** 2)
    SSm = n * np.sum((cond_means - grand) ** 2)
    SStot = np.sum((matrix - grand) ** 2)
    SSres = SStot - SSb - SSm
    MSb = SSb / (n - 1)
    MSres = SSres / ((n - 1) * (k - 1))
    if MSb + (k - 1) * MSres == 0:
        return float("nan")
    return (MSb - MSres) / (MSb + (k - 1) * MSres)


def buildSeedXCellMatrix(df):
    """
        Returns (cell_names, seed_list, matrix [n_seeds, k_cells]).
        Keeps only seeds present in every cell (intersection).
    """
    cells = sorted(df["cell"].unique())
    pivot = df.pivot_table(index="seed", columns="cell", values="mse")
    pivot = pivot.dropna(how="any")
    return cells, list(pivot.index), pivot.values


# ---------------------------------------------------------------------
# Closest-pair misranking
# ---------------------------------------------------------------------

def closestPairMisranking(df):
    """
        Finds the cell pair with the smallest |mean gap| and computes
        the Monte Carlo + Bootstrap P(swap) using the helper from the
        parent submodule (sign-flips MSE because that helper assumes
        higher-is-better).
    """
    cell_means = df.groupby("cell")["mse"].mean()
    cells = list(cell_means.index)
    best_gap = float("inf")
    best_pair = None
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            gap = abs(cell_means[cells[i]] - cell_means[cells[j]])
            if gap < best_gap:
                best_gap = gap
                best_pair = (cells[i], cells[j])
    a, b = best_pair
    neg = {a: (-df[df.cell == a]["mse"]).tolist(),
           b: (-df[df.cell == b]["mse"]).tolist()}
    log(f"[closest-pair] {a} (mean={cell_means[a]:.4f}) vs "
        f"{b} (mean={cell_means[b]:.4f}), gap = {best_gap:.5f}",
        color=bcolors.OKCYAN)
    np.random.seed(MISRANK_SEED)  # deterministic Monte Carlo / Bootstrap
    computeSwitchedProbability(neg, [a, b], analysis_path=analysis_path)
    return best_pair, best_gap


# ---------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------

def printPerCell(df_means, label):
    header = ["factor1", "factor2", "$N$", "mean", "std"]
    rows = [header]
    for _, r in df_means.iterrows():
        rows.append([
            str(r.iloc[0]), str(r.iloc[1]),
            f"{int(r['n'])}", f"{r['mean']:.4f}", f"{r['std']:.4f}",
        ])
    logTable(rows, tables_path, f"Ablation {label} - per-cell mean MSEx100")


def printAnova(aov, label):
    header = ["predictor", "df", "F", "p", "SS", "% SS", r"$\eta_p^2$"]
    rows = [header]
    for idx, r in aov.iterrows():
        rows.append([
            str(idx),
            f"{r.get('df', float('nan')):.0f}" if not np.isnan(
                r.get('df', float('nan'))) else "-",
            f"{r['F']:.2f}" if not np.isnan(r['F']) else "-",
            f"{r['PR(>F)']:.2e}" if not np.isnan(r['PR(>F)']) else "-",
            f"{r['sum_sq']:.2f}",
            f"{r['pct_ss']:.2f}",
            f"{r['part_eta2']:.4f}"
            if not np.isnan(r['part_eta2']) else "-",
        ])
    logTable(rows, tables_path,
             f"Ablation {label} - two-way ANOVA (type-III)")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def runStudy(label, class_list, factor1_name):
    log(f"\n========== Ablation {label} ==========", color=bcolors.HEADER)
    df = buildStudyDataframe(class_list)
    if df.empty:
        log(f"[skip] no data for {label}", color=bcolors.WARNING)
        return

    log(f"[summary] {len(df)} trials across {df['cell'].nunique()} cells "
        f"(min N per cell = {df.groupby('cell').size().min()}, "
        f"max = {df.groupby('cell').size().max()})")

    if enable_per_cell_means:
        printPerCell(perCellMeans(df, factor1_name, "learning_rate"), label)

    if enable_two_way_anova:
        printAnova(twoWayAnova(df), label)

    if enable_cross_cell_icc:
        cells, seeds, M = buildSeedXCellMatrix(df)
        icc = iccThreeOne(M)
        log(f"[ICC] ICC(3,1) across {M.shape[1]} cells on {M.shape[0]} "
            f"shared seeds = {icc:.5f}", color=bcolors.OKGREEN)

    if enable_closest_pair_misrank:
        closestPairMisranking(df)


def run():
    os.makedirs(tables_path, exist_ok=True)
    runStudy("B1", B1_CLASSES, factor1_name="batch_size")
    runStudy("B3", B3_CLASSES, factor1_name="epsilon")


if __name__ == "__main__":
    run()
