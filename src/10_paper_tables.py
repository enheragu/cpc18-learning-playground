#!/usr/bin/env python3
# encoding: utf-8
"""
Regenerate the manuscript's numeric tables from the committed flat CSVs in
analysis_results/csv/ -- no YAML, no retraining. This is the lightweight
reproducibility path: clone, run 09_export_csv.py once (or use the committed
CSVs), then run this to reproduce the paper's headline numbers.

Covers: the six-model summary / reporting table (best, mean, std, P10), the
per-model normality stats, the cross-model ICC(3,1) and Spearman matrix, the
exact paired-win matrix, the B1/B3 two-way ANOVA decomposition (sum-to-zero
contrasts), and the Phase C init x order decomposition. Figures and per-epoch
analyses need the full per-trial data (released separately) and are produced
by 05_make_paper_figures.py.

Run:  python 10_paper_tables.py
"""

import os
import itertools

import path_setup  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, shapiro, kurtosis
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from utils.log_utils import log, bcolors
from cpc18_utils import output_path

CSV_DIR = os.path.join(os.path.dirname(output_path), "analysis_results", "csv")
R2_ORDER = ["LinearLogistic", "BourginMLP", "ContextDepNet",
            "SparseMLP", "PsychForest", "BEAST_GB"]
MISRANK_SEED = 20260522


def _load(name):
    return pd.read_csv(os.path.join(CSV_DIR, name))


def iccThreeOne(M):
    n, k = M.shape
    gm = M.mean()
    msr = k * ((M.mean(1) - gm) ** 2).sum() / (n - 1)
    mse = ((M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + gm) ** 2).sum() \
        / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse)


def sixModelTables():
    df = _load("results_sharedpool_r2.csv")
    by = {m: df[df.model == m]["mse_x100"].values for m in R2_ORDER}
    log("\n=== Six-model summary (regime 2, BEAST features) ===", bcolors.HEADER)
    log(f"{'model':16s} {'N':>4} {'best':>7} {'mean':>7} {'std':>7} "
        f"{'P10':>7} {'kurt':>7} {'SW p':>9}")
    for m in sorted(R2_ORDER, key=lambda k: by[k].mean()):
        a = by[m]
        log(f"{m:16s} {len(a):4d} {a.min():7.3f} {a.mean():7.3f} "
            f"{a.std(ddof=1):7.3f} {np.percentile(a,10):7.3f} "
            f"{kurtosis(a):7.2f} {shapiro(a).pvalue:9.2e}")

    # paired matrix on shared seeds
    rows = {}
    for _, r in df.iterrows():
        rows.setdefault(int(r["seed"]), {})[r["model"]] = float(r["mse_x100"])
    seeds = [s for s in rows if all(m in rows[s] for m in R2_ORDER)]
    M = np.array([[rows[s][m] for m in R2_ORDER] for s in seeds])
    log(f"\nShared seeds: {len(seeds)} | ICC(3,1) six-model = {iccThreeOne(M):.4f}",
        bcolors.OKCYAN)
    log("Spearman off-diagonal range: "
        f"[{min(spearmanr(M[:,i],M[:,j]).correlation for i,j in itertools.combinations(range(6),2)):+.3f}, "
        f"{max(spearmanr(M[:,i],M[:,j]).correlation for i,j in itertools.combinations(range(6),2)):+.3f}]")
    log("Exact paired-win matrix P(row<col):")
    abbr = ["LL", "BMLP", "CDN", "SMLP", "PF", "BGB"]
    log("        " + " ".join(f"{a:>6}" for a in abbr))
    for i, a in enumerate(R2_ORDER):
        cells = ["   -- " if i == j else
                 f"{np.mean([rows[s][a] < rows[s][R2_ORDER[j]] for s in seeds]):6.3f}"
                 for j in range(6)]
        log(f"{abbr[i]:7s} " + " ".join(cells))


def twoWayAnova(df):
    aov = anova_lm(ols("mse ~ C(f1, Sum) * C(f2, Sum)", data=df).fit(), typ=3)
    aov = aov.drop("Intercept", errors="ignore")
    ss = aov["sum_sq"]; ssr = ss["Residual"]
    aov["pct_ss"] = 100.0 * ss / ss.sum()
    aov["eta2p"] = [ss[i] / (ss[i] + ssr) if i != "Residual" else np.nan
                    for i in aov.index]
    return aov[["F", "PR(>F)", "pct_ss", "eta2p"]]


def ablationTables():
    for study, (f1c, lab) in {"b1": ("batch_size", "BourginMLP batch x lr"),
                              "b3": ("epsilon", "SparseMLP eps x lr")}.items():
        df = _load(f"results_ablation_{study}.csv")
        f1 = f1c if f1c in df.columns else ("epsilon" if "epsilon" in df.columns else "batch_size")
        d = df.rename(columns={f1: "f1", "learning_rate": "f2"})[["f1", "f2", "mse_x100"]]
        d = d.rename(columns={"mse_x100": "mse"})
        log(f"\n=== Ablation {study.upper()} ({lab}) two-way ANOVA (sum contrasts) ===",
            bcolors.HEADER)
        log(twoWayAnova(d).to_string())


def decompTable():
    df = _load("results_factorial_decomp.csv")
    log("\n=== Phase C init x order decomposition (% of total SS) ===", bcolors.HEADER)
    log(f"{'model':16s} {'init%':>7} {'order%':>8} {'interaction%':>13}")
    for m in ["BourginMLP", "LinearLogistic", "ContextDepNet", "SparseMLP"]:
        c = {(int(r.init_seed), int(r.order_seed)): r.mse_x100
             for r in df[df.model == m].itertuples()}
        inits = sorted({k[0] for k in c}); orders = sorted({k[1] for k in c})
        M = np.array([[c[(i, o)] for o in orders] for i in inits
                      if all((i, o) in c for o in orders)])
        gm = M.mean()
        ss_i = M.shape[1] * ((M.mean(1) - gm) ** 2).sum()
        ss_o = M.shape[0] * ((M.mean(0) - gm) ** 2).sum()
        tot = ((M - gm) ** 2).sum()
        log(f"{m:16s} {100*ss_i/tot:7.1f} {100*ss_o/tot:8.1f} "
            f"{100*(tot-ss_i-ss_o)/tot:13.1f}")


def main():
    if not os.path.isdir(CSV_DIR):
        raise SystemExit(f"No CSVs at {CSV_DIR}. Run 09_export_csv.py first.")
    sixModelTables()
    ablationTables()
    decompTable()
    log("\nAll paper tables regenerated from CSV.", bcolors.OKGREEN)


if __name__ == "__main__":
    main()
