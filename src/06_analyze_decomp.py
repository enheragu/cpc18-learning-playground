#!/usr/bin/env python3
# encoding: utf-8

"""
    Phase C variance-decomposition analyser.

    For each neural model with a factorial init x order grid produced by
    `03_train_decomp.py`, computes:
      1. The two-way sum-of-squares decomposition (SS_init, SS_order,
         SS_residual) and the percentages of total SS.
      2. Six "slicing modes" that give different empirical standard
         deviations depending on which factor is held fixed (init-only,
         order-only, init-mean, order-mean, diagonal, full factorial).
      3. A consolidated table per model in plain text and LaTeX.

    Usage:
        python 06_analyze_decomp.py

    Outputs go to analysis_results/tables/.
"""

import os
import re

import path_setup  # noqa: F401

import numpy as np

from utils.log_utils import log, logTable, bcolors
from cpc18_utils import output_path
from cpc18_utils.safe_yaml import fastLoadYaml


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_")
_YAML_NAME = "factorial_init_order_r2_training_metrics.yaml"

# Models to analyse. PsychForest and BEAST_GB are excluded by design:
# they do not consume a DataLoader, so they have no batch-order axis.
_DECOMP_MODELS = ("BourginMLP", "LinearLogistic", "ContextDepNet", "SparseMLP")

analysis_path = os.path.join(os.path.dirname(output_path), "analysis_results")
tables_path = os.path.join(analysis_path, "tables")


# ---------------------------------------------------------------------
# Toggles
# ---------------------------------------------------------------------
enable_per_model_decomposition = True
enable_slicing_modes           = True


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def loadTrials(model_name):
    """ Returns the per-trial entry list for one model's factorial YAML. """
    path = os.path.join(output_path, model_name, _YAML_NAME)
    data = fastLoadYaml(path)
    return [v for k, v in data.items()
            if _DATE_RE.match(k) and isinstance(v, dict)]


def buildGrid(trials):
    """
        Returns (init_seeds_sorted, order_seeds_sorted, mse_matrix).
        mse_matrix has shape (n_init, n_order) with MSE x 100 values;
        cells with no trial are np.nan. Trials must carry 'init_seed'
        and 'order_seed' keys.
    """
    by_pair = {(int(t["init_seed"]), int(t["order_seed"])): t["mse"] * 100.0
               for t in trials
               if "init_seed" in t and "order_seed" in t and "mse" in t}
    inits = sorted({p[0] for p in by_pair})
    orders = sorted({p[1] for p in by_pair})
    M = np.full((len(inits), len(orders)), np.nan)
    for i, s_i in enumerate(inits):
        for j, s_o in enumerate(orders):
            v = by_pair.get((s_i, s_o))
            if v is not None:
                M[i, j] = v
    return inits, orders, M


def _dropIncomplete(M):
    """
        Keep only rows AND columns with no NaN. Returns a complete grid
        suitable for the two-way decomposition.
    """
    keep_rows = ~np.isnan(M).any(axis=1)
    keep_cols = ~np.isnan(M).any(axis=0)
    return M[np.ix_(keep_rows, keep_cols)]


# ---------------------------------------------------------------------
# Decomposition + slicing
# ---------------------------------------------------------------------

def _varianceComponentShares(SS_init, SS_order, SS_resid, n_i, n_o):
    """Random-effects variance-component shares for a two-way design with
    one observation per cell, via expected mean squares
    (E[MS_init] = sigma2_inter + n_o * sigma2_init, and symmetrically).

    A descriptive cross-check on the eta-squared (SS-share) reading: for a
    balanced, deterministic grid the two nearly coincide. Returns the three
    shares in percent (init, order, interaction); NaN if the grid is too
    small to estimate them.
    """
    if n_i < 2 or n_o < 2:
        return float("nan"), float("nan"), float("nan")
    MS_init  = SS_init  / (n_i - 1)
    MS_order = SS_order / (n_o - 1)
    MS_inter = SS_resid / ((n_i - 1) * (n_o - 1))
    s2_inter = MS_inter
    s2_init  = max(0.0, MS_init  - MS_inter) / n_o
    s2_order = max(0.0, MS_order - MS_inter) / n_i
    tot = s2_init + s2_order + s2_inter
    if tot <= 0:
        return float("nan"), float("nan"), float("nan")
    return 100 * s2_init / tot, 100 * s2_order / tot, 100 * s2_inter / tot


def twoWayDecomposition(M):
    """
        One-observation-per-cell two-way SS decomposition (descriptive).

        Normally one observation per cell confounds the interaction with
        within-cell noise. Here the pipeline is deterministic given the two
        seeds (a cell reproduces its MSE bit-identically), so there is no
        within-cell noise and SS_resid IS exactly the init x order
        interaction. The shares are eta-squared on the observed grid, read
        descriptively: not an inferential ANOVA, and not a random-effects
        variance-component estimate for the seed population (for this
        balanced, deterministic grid the two coincide closely -- the
        var-component shares are returned alongside for cross-checking).
    """
    grand = M.mean()
    row_means = M.mean(axis=1, keepdims=True)
    col_means = M.mean(axis=0, keepdims=True)
    SS_total = float(np.sum((M - grand) ** 2))
    SS_init = float(M.shape[1] * np.sum((row_means - grand) ** 2))
    SS_order = float(M.shape[0] * np.sum((col_means - grand) ** 2))
    SS_resid = SS_total - SS_init - SS_order
    n_i, n_o = M.shape[0], M.shape[1]
    pct_init  = 100 * SS_init  / SS_total if SS_total > 0 else float("nan")
    pct_order = 100 * SS_order / SS_total if SS_total > 0 else float("nan")
    pct_resid = 100 * SS_resid / SS_total if SS_total > 0 else float("nan")
    vc_init, vc_order, vc_resid = _varianceComponentShares(
        SS_init, SS_order, SS_resid, n_i, n_o)
    shift = (max(abs(pct_init - vc_init), abs(pct_order - vc_order),
                 abs(pct_resid - vc_resid))
             if SS_total > 0 else float("nan"))
    return {
        "SS_total": SS_total,
        "SS_init":  SS_init,
        "SS_order": SS_order,
        "SS_resid": SS_resid,
        "pct_init":  pct_init,
        "pct_order": pct_order,
        "pct_resid": pct_resid,
        "vc_init":  vc_init,
        "vc_order": vc_order,
        "vc_resid": vc_resid,
        "eta2_vc_shift": shift,
        "n_init":  n_i,
        "n_order": n_o,
    }


def slicingModes(M):
    """
        Six slicing modes giving different empirical standard deviations.

        - init-only:  vary init seeds with the order seed held fixed,
                      averaged over the choice of fixed order seed (the
                      std across init is computed for each order column
                      and the column-stds are averaged, so the value does
                      not depend on which single column happens to be
                      picked -- it did in the original M[:, 0] version)
        - order-only: symmetric (std across order per init row, averaged)
        - init-mean:  for each init seed, average MSE over all orders;
                      then take the std across init seeds
        - order-mean: symmetric
        - diagonal:   (init_i, order_i) paired -- the regime a standard
                      pipeline samples
        - full:       all cells of the factorial
    """
    n = min(M.shape)
    return {
        "init_only":  (float(M.std(axis=0, ddof=1).mean()), M.shape[0]),
        "order_only": (float(M.std(axis=1, ddof=1).mean()), M.shape[1]),
        "init_mean":  (float(M.mean(axis=1).std(ddof=1)), M.shape[0]),
        "order_mean": (float(M.mean(axis=0).std(ddof=1)), M.shape[1]),
        "diagonal":   (float(np.array([M[i, i] for i in range(n)])
                             .std(ddof=1)), n),
        "full":       (float(M.flatten().std(ddof=1)),    M.size),
    }


# ---------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------

def perModelTable(rows):
    header = ["Model", "$N$ cells", "init %", "order %", "resid/inter %"]
    out = [header]
    for r in rows:
        out.append([
            r["model"],
            f"{r['n_init']}x{r['n_order']}",
            f"{r['pct_init']:.1f}",
            f"{r['pct_order']:.1f}",
            f"{r['pct_resid']:.1f}",
        ])
    logTable(out, tables_path,
             "Phase C decomposition (regime 2, factorial init x order)")


def slicingTable(model_name, modes):
    header = ["Slicing mode", "$n$", "$\\sigma(\\mathrm{MSE} \\times 100)$"]
    rows = [header]
    label = {
        "init_only":  "init-only (fix one order seed, vary inits)",
        "order_only": "order-only (fix one init seed, vary orders)",
        "init_mean":  "init-mean (vary inits, mean over orders)",
        "order_mean": "order-mean (vary orders, mean over inits)",
        "diagonal":   "diagonal (init$_i$ paired with order$_i$)",
        "full":       "full factorial",
    }
    for key in ("init_only", "order_only", "init_mean", "order_mean",
                "diagonal", "full"):
        sigma, n = modes[key]
        rows.append([label[key], f"{n}", f"{sigma:.3f}"])
    logTable(rows, tables_path, f"Phase C slicing modes - {model_name}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def run():
    os.makedirs(tables_path, exist_ok=True)
    log("\n========== Phase C variance decomposition ==========\n",
        color=bcolors.HEADER)

    decomp_rows = []
    for model_name in _DECOMP_MODELS:
        trials = loadTrials(model_name)
        if not trials:
            log(f"[skip] no trials for {model_name}", color=bcolors.WARNING)
            continue
        _, _, M = buildGrid(trials)
        Mc = _dropIncomplete(M)
        if Mc.size == 0:
            log(f"[skip] incomplete grid for {model_name} "
                f"(no complete row x column block)", color=bcolors.WARNING)
            continue

        if enable_per_model_decomposition:
            decomp = twoWayDecomposition(Mc)
            decomp["model"] = model_name
            decomp_rows.append(decomp)
            log(f"\n[{model_name}] grid {Mc.shape[0]}x{Mc.shape[1]} "
                f"(complete) of {M.shape[0]}x{M.shape[1]} (raw)",
                color=bcolors.OKCYAN)
            log(f"  SS_total = {decomp['SS_total']:.3f}")
            log(f"  init  : {decomp['pct_init']:6.2f} % (eta2)  {decomp['vc_init']:6.2f} % (var-comp)")
            log(f"  order : {decomp['pct_order']:6.2f} % (eta2)  {decomp['vc_order']:6.2f} % (var-comp)")
            log(f"  resid : {decomp['pct_resid']:6.2f} % (eta2)  {decomp['vc_resid']:6.2f} % (var-comp)  "
                f"(= init x order interaction; deterministic cells, no noise)")
            log(f"  eta2 vs variance-component shares agree to "
                f"{decomp['eta2_vc_shift']:.1f} pts")

        if enable_slicing_modes:
            slicingTable(model_name, slicingModes(Mc))

    if enable_per_model_decomposition and decomp_rows:
        perModelTable(decomp_rows)
    log("\nWrote tables to: " + tables_path, color=bcolors.OKGREEN)


if __name__ == "__main__":
    run()
