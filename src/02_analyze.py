#!/usr/bin/env python3
# encoding: utf-8

"""
    CPC18 cross-model analysis orchestrator.

    Reads per-model `<mode>_training_metrics.yaml` files produced by
    `00_train_models.py` (regime 1) and `01_train_regime2.py` (regime 2)
    and computes:
      1. Summary table (n, mean, std, min, max, 99.9% normal amplitude).
      2. Shapiro-Wilk normality test per model.
      3. Pairwise misranking probability tables (MNIST paper Tables 2-3),
         computed with both MonteCarlo and Bootstrap.
      4. Cross-model Spearman correlation matrix on shared seeds (only
         meaningful when --mode shared_pool).
      5. ICC(3,1) across the seed x model matrix (extension of the MNIST
         paper's ablation Spearman/ICC to model-level comparison).

    For MSE-as-loss we flip the sign before passing data to
    computeSwitchedProbability because that helper uses np.max as
    "better than" - i.e. it expects higher-is-better semantics.

    Usage:
        python 02_analyze.py --mode shared_pool --regime 2
        python 02_analyze.py --mode random --regime 1
"""

import argparse
import os

import numpy as np
from scipy.stats import spearmanr

import path_setup  # noqa: F401 - exposes submodule utils.*

from utils.log_utils import log, logTable, bcolors
from utils.distribution_analysis import (
    normalityTest, maxAmplitude, count_trials, normal_amplitude,
)
from utils.compute_switched_probability import computeSwitchedProbability

from cpc18_utils import output_path
from cpc18_utils.training_logs import loadTrainingLog, trainingConditions


# The submodule misranking helper draws from the global np.random stream
# without seeding it. We seed before each call so the Monte Carlo /
# Bootstrap tables are bit-reproducible across runs. The submodule is not
# modified (see safe_yaml for the same wrapping pattern).
MISRANK_SEED = 20260522
_MODE_TO_PREFIX = {
    "random":      "randomseed",
    "same":        "sameseed",
    "shared_pool": "sharedpool",
}


# ---------------------------------------------------------------------
# Toggles. Switch off any block that is not relevant for the run.
# ---------------------------------------------------------------------
enable_count_trials              = True
enable_normality_test            = True
enable_max_amplitude             = True
enable_pairwise_switched_prob    = True
enable_cross_model_spearman_icc  = True


def _prefixFor(mode, regime):
    """
        Return the YAML filename prefix for the (mode, regime) pair.
        Regime 1 keeps the historical bare prefix; regime 2 (BEAST
        features) adds a `_r2` suffix so the YAMLs don't collide.
    """
    base = _MODE_TO_PREFIX[mode]
    return base if regime == 1 else f"{base}_r{regime}"


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def loadMetrics(mode, regime=1):
    """
        Returns `{model_name: {timestamp: entry, ...}}` for all models
        with a YAML file matching the requested seed mode + regime.
    """
    suffix = f"{_prefixFor(mode, regime)}_training_metrics.yaml"
    out = {}
    for model_name in trainingConditions():
        path = os.path.join(output_path, model_name, suffix)
        try:
            trials = loadTrainingLog(path)
        except FileNotFoundError:
            continue
        if trials:
            out[model_name] = trials
    return out


def buildSeedXModelMatrix(metrics_data):
    """
        Wide matrix indexed by seed, with one column per model.
        Only retains seeds that appear in ALL models (intersection).
        Returns (seed_list, model_list, matrix) where matrix[i, j] is
        the MSE of model_list[j] on seed_list[i].
    """
    seeds_per_model = {
        m: {t["seed"]: t["mse"] for t in trials.values()}
        for m, trials in metrics_data.items()
    }
    common_seeds = sorted(set.intersection(
        *(set(d.keys()) for d in seeds_per_model.values())))
    model_list = sorted(seeds_per_model.keys())
    matrix = np.array(
        [[seeds_per_model[m][s] for m in model_list] for s in common_seeds],
        dtype=float,
    )
    return common_seeds, model_list, matrix


def negatedMseByModel(metrics_data):
    """
        For the switched-probability helper: convert MSE to -MSE so the
        helper's "max-is-best" convention picks the lowest-MSE model.
    """
    return {m: [-t["mse"] for t in trials.values()]
            for m, trials in metrics_data.items()}


# ---------------------------------------------------------------------
# Cross-model statistics
# ---------------------------------------------------------------------

def spearmanMatrix(model_list, matrix):
    """ Pairwise Spearman correlation of MSE ranks across seeds. """
    K = len(model_list)
    rho = np.eye(K)
    for i in range(K):
        for j in range(i + 1, K):
            r, _ = spearmanr(matrix[:, i], matrix[:, j])
            rho[i, j] = rho[j, i] = r
    return rho


def iccThreeOne(matrix):
    """
        ICC(3,1) on a (n_seeds, k_models) matrix following Eq. 1 of the
        MNIST variance paper.

        ICC(3,1) = (MSb - MSw) / (MSb + (k-1) * MSw)
        where MSb = between-seed MS and MSw = within-seed (residual) MS.
    """
    n, k = matrix.shape
    grand_mean = matrix.mean()
    seed_means = matrix.mean(axis=1)
    model_means = matrix.mean(axis=0)

    SSb = k * np.sum((seed_means - grand_mean) ** 2)
    SSm = n * np.sum((model_means - grand_mean) ** 2)
    SStot = np.sum((matrix - grand_mean) ** 2)
    SSres = SStot - SSb - SSm

    MSb = SSb / (n - 1)
    MSres = SSres / ((n - 1) * (k - 1))
    if MSb + (k - 1) * MSres == 0:
        return float("nan")
    return (MSb - MSres) / (MSb + (k - 1) * MSres)


def runPairwiseSwitchedProbability(metrics_data, analysis_path):
    """
        Pairwise misranking probability between every model pair, as in
        the MNIST paper's Tables 2-3. Computes Monte Carlo + Bootstrap.
    """
    negated = negatedMseByModel(metrics_data)
    model_names = sorted(negated.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            np.random.seed(MISRANK_SEED)  # deterministic per-pair tables
            computeSwitchedProbability(
                negated, [model_names[i], model_names[j]],
                analysis_path=analysis_path,
            )


def runCrossModelRankAnalysis(metrics_data, analysis_path, mode):
    """
        Cross-model Spearman matrix + ICC(3,1) on the seed x model
        matrix. Only meaningful for the `shared_pool` mode where seeds
        are paired across models.
    """
    seeds, model_list, mat = buildSeedXModelMatrix(metrics_data)
    log(f"\nShared-seed matrix: {len(seeds)} seeds x {len(model_list)} models",
        color=bcolors.OKCYAN)

    rho = spearmanMatrix(model_list, mat)
    header = ["model"] + model_list
    rows = [header]
    for i, m in enumerate(model_list):
        rows.append([m] + [f"{rho[i, j]:.3f}" for j in range(len(model_list))])
    logTable(rows, f"{analysis_path}/tables",
             f"Spearman rank correlation across seeds ({mode})")

    icc = iccThreeOne(mat)
    log(f"\nICC(3,1) across {len(model_list)} models on {len(seeds)} "
        f"shared seeds: {icc:.5f}", color=bcolors.OKGREEN)
    log("(Low ICC -> a seed that is 'good' for one model is not 'good' "
        "for another -> model ranking is seed-dependent.)")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def run(mode, analysis_path, regime=1):
    metrics_data = loadMetrics(mode, regime=regime)
    if not metrics_data:
        log(f"No '{_prefixFor(mode, regime)}' YAMLs found under {output_path}",
            color=bcolors.WARNING)
        return

    log(f"\n========== CPC18 analysis (regime {regime}, mode {mode}) ==========",
        color=bcolors.HEADER)

    if enable_count_trials:
        count_trials(metrics_data, title_tag=mode, analysis_path=analysis_path)

    if enable_normality_test:
        normalityTest(metrics_data, metric='mse', title_tag=mode,
                      analysis_path=analysis_path, unit_multiplier=100)

    if enable_max_amplitude:
        maxAmplitude(metrics_data, metric='mse', unit=' (x100)',
                     unit_multiplier=100, format='.3f',
                     amplitude_function=normal_amplitude,
                     title_tag=mode, analysis_path=analysis_path)

    if enable_pairwise_switched_prob:
        runPairwiseSwitchedProbability(metrics_data, analysis_path)

    if not enable_cross_model_spearman_icc:
        return
    if mode != "shared_pool":
        log("Skipping Spearman/ICC: only meaningful for shared_pool mode "
            "(seeds must be paired across models).", color=bcolors.WARNING)
        return
    runCrossModelRankAnalysis(metrics_data, analysis_path, mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="shared_pool",
                        choices=list(_MODE_TO_PREFIX))
    parser.add_argument("--regime", type=int, default=1, choices=(1, 2),
                        help="1 = raw-features experiment, 2 = BEAST-features experiment.")
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(output_path), "analysis_results"))
    args = parser.parse_args()
    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)
    run(args.mode, args.out, regime=args.regime)
