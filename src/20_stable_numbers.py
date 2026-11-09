#!/usr/bin/env python3
# encoding: utf-8

"""
Numbers quoted in the text of the article in the stable configuration
(cpc18_utils/protocol.py), and the checks of the fragile one, from the data deposit only (version 1.1 or later,
--deposit, by default zenodo_export/), read as cpc18_utils/stable_results.py
describes: the neural models at the fixed epoch of each configuration on M00
and M01, the tree-based models in the original protocol on M00.

The output is a sequence of blocks, each with its own random generator, in this
order: the main analysis (summary of the six models, misranking, factorial
decomposition, ablations, test-sampling and split variance, forced arithmetic
paths, check of the fit with few trials), the precision of the comparisons of
Sections 3.2 to 3.4, the fitted family and its check (Section 3.1), the
misranking tables and the inverted order by best, mean and P10 of N trials
(Section 3.2), Monte Carlo against Bootstrap, the
decomposition (Section 3.4.1), the ablations (Section 3.3), the per-problem
concentration of the seed variance (Section 3.4.2), the reporting format
(Section 4), the appendices and the fragile configuration (Section 3.6). Within a block the draws follow a fixed order,
so every number is reproducible. Monte Carlo draws from the lognormal fitted to
the trials (mean and ddof=1 std of their logarithms), the family of Section
3.1, except in the block that compares it with the Gaussian and in the misranking
of the six-model mean, which draws the mean of N trials from its normal
approximation. Resampling draws trials by position in a frame ordered by seed,
so a number can change by a few units in its last digit if the trials are
ordered otherwise.

Usage:
    cd src && python 20_stable_numbers.py [--deposit PATH]
"""

import argparse
import os
import types
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from cpc18_utils.stable_results import StableResults, NEURAL_MODELS

warnings.filterwarnings("ignore")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NN = list(NEURAL_MODELS)
ALL = NN + ["PsychForest", "BEAST_GB"]
Zq = stats.norm.ppf(0.1)
MASTER_SEED_FOR_POOL = 20260522


# ---------------------------------------------------------------------
# Statistics shared by the blocks
# ---------------------------------------------------------------------

def p10(x):
    """ 10th percentile of the lognormal fitted to x (maximum likelihood: mean and ddof=0 std of log x). """
    lx = np.log(x)
    return float(np.exp(lx.mean() + Zq * lx.std()))


def desc(x):
    return (f"mean {x.mean():.3f} median {np.median(x):.3f} sd {x.std(ddof=1):.3f} kurtosis "
            f"{stats.kurtosis(x):+.2f} check {100 * (x <= p10(x)).mean():.1f}% >1.5 {100 * (x > 1.5).mean():.1f}%")


def icc(M):
    """ ICC(3,1) of a seeds x conditions matrix. """
    M = np.asarray(M)
    n, k = M.shape
    g = M.mean()
    msr = k * ((M.mean(1) - g) ** 2).sum() / (n - 1)
    mse = ((M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + g) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse)


def iccCi(M):
    """ ICC(3,1) with its exact F-based 95 % interval. """
    M = np.asarray(M)
    n, k = M.shape
    g = M.mean()
    msr = k * ((M.mean(1) - g) ** 2).sum() / (n - 1)
    mse = ((M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + g) ** 2).sum() / ((n - 1) * (k - 1))
    F = msr / mse
    d1, d2 = n - 1, (n - 1) * (k - 1)
    FL, FU = F / stats.f.ppf(.975, d1, d2), F * stats.f.ppf(.975, d2, d1)
    return (msr - mse) / (msr + (k - 1) * mse), (FL - 1) / (FL + k - 1), (FU - 1) / (FU + k - 1)


def comps(X):
    """ Random-effects variance components (%) of a grid: initialization (rows), order (columns), interaction,
    and the total variance. """
    X = np.asarray(X)
    a, b = X.shape
    g = X.mean()
    ssr = b * ((X.mean(1) - g) ** 2).sum()
    ssc = a * ((X.mean(0) - g) ** 2).sum()
    sse = ((X - g) ** 2).sum() - ssr - ssc
    msr, msc, mse = ssr / (a - 1), ssc / (b - 1), sse / ((a - 1) * (b - 1))
    vi, vo = max((msr - mse) / b, 0), max((msc - mse) / a, 0)
    t = vi + vo + mse
    return 100 * vi / t, 100 * vo / t, 100 * mse / t, t


def eta(X):
    """ Shares (%) of the sum of squares of a grid: rows, columns, residual. """
    X = np.asarray(X)
    a, b = X.shape
    g = X.mean()
    ssr = b * ((X.mean(1) - g) ** 2).sum()
    ssc = a * ((X.mean(0) - g) ** 2).sum()
    sst = ((X - g) ** 2).sum()
    return 100 * ssr / sst, 100 * ssc / sst, 100 * (sst - ssr - ssc) / sst


def anova(df, a, b, digits=2):
    """ Shares (%) of the total sum of squares of a balanced two-way design: a, b, interaction, residual. """
    g = df.y.mean()
    A_ = df.groupby(a).y.transform("mean")
    B_ = df.groupby(b).y.transform("mean")
    C_ = df.groupby([a, b]).y.transform("mean")
    t = ((df.y - g) ** 2).sum()
    shares = (((A_ - g) ** 2).sum() / t, ((B_ - g) ** 2).sum() / t, ((C_ - A_ - B_ + g) ** 2).sum() / t,
              ((df.y - C_) ** 2).sum() / t)
    return [round(100 * v, digits) for v in shares]


def anovaF(long, a):
    """ Degrees of freedom, F statistic and p-value of each term of the two-way fixed-effects ANOVA (type III,
    sum-to-zero contrasts), and the degrees of freedom of the residual. """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    model = smf.ols(f"y ~ C({a}, Sum) * C(lr, Sum)", data=long).fit()
    table = sm.stats.anova_lm(model, typ=3)
    terms = {i: (int(table.loc[i, "df"]), round(float(table.loc[i, "F"]), 1), float(f"{table.loc[i, 'PR(>F)']:.2g}"))
             for i in table.index if i not in ("Intercept", "Residual")}
    return {**terms, "residual df": int(table.loc["Residual", "df"])}


def greedy(M):
    """ End cell of the one-pass coordinate search from each of the six starting levels of a 3 x 3 grid. """
    r = []
    for e0 in range(3):
        lev = int(np.argmin(M[e0]))
        r.append((int(np.argmin(M[:, lev])), lev))
    for l0 in range(3):
        e = int(np.argmin(M[:, l0]))
        r.append((e, int(np.argmin(M[e]))))
    return r


def lognormalDraw(rng, x, shape):
    """ Monte Carlo trials from the lognormal fitted to x (mean and ddof=1 std of log x). """
    lx = np.log(x)
    return np.exp(rng.normal(lx.mean(), lx.std(ddof=1), shape))


def fourLevels(S):
    """ The ordering that pools the two tied pairs: BEAST-GB < PsychForest < {ContextDepNet, SparseMLP} <
    {LinearLogistic, BourginMLP}. """
    return ((S["BEAST_GB"] < S["PsychForest"]) & (S["PsychForest"] < np.minimum(S["ContextDepNet"], S["SparseMLP"]))
            & (np.maximum(S["ContextDepNet"], S["SparseMLP"]) < np.minimum(S["LinearLogistic"], S["BourginMLP"])))


def cellName(c):
    return f"({c[0]:g}, {c[1]:g})"


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def loadData(deposit):
    S = StableResults(deposit)
    D = types.SimpleNamespace(S=S)
    cv = S.cvEpochs()
    D.EMAIN = {m: cv[f"{m}_B32_lr0.001" + ("_E20" if m == "SparseMLP" else "")] for m in NN}
    # Main run: one Series per model, indexed by seed in the order of the deposit
    D.V, D.Vmin = {}, {}
    for m in NN:
        rows = S.mainRows(m).sort_values("trial_id")
        D.V[m] = rows.set_index("seed").mse_x100
        D.Vmin[m] = rows.set_index("seed").mse_x100_testmin
    for m in ("PsychForest", "BEAST_GB"):
        rows = S.reference[(S.reference.experiment == "main") & (S.reference.condition == m)]
        D.V[m] = rows.set_index("seed").mse_x100
    D.P = pd.DataFrame(D.V)[ALL]          # seeds sorted
    # Factorial grids: initialization seeds in rows, order seeds in columns; also read at the test minimum
    D.G = {m: S.factorial(m) for m in NN}
    ref = S.reference
    fac = ref[ref.experiment == "factorial_init_order"]
    D.Gmin = {m: fac[fac.condition == m].pivot(index="init_seed", columns="order_seed", values="mse_x100_testmin")
              for m in NN}
    # Ablation BourginMLP, with the 3.16e-4 column, and Ablation SparseMLP: long frames and seed x cell matrices
    b1 = S.ablation("ablation_bourginmlp", include_316=True)
    b1min = ref[ref.experiment == "ablation_bourginmlp"].set_index("trial_id").mse_x100_testmin
    b1["ymin"] = b1.trial_id.map(b1min)
    D.LB = b1.rename(columns={"f1": "B"})
    D.full = D.LB[D.LB.lr.isin([1e-4, 1e-3, 1e-2])]
    D.conv = D.LB[D.LB.lr.isin([1e-4, 3.16e-4, 1e-3])]
    b3 = S.ablation("ablation_sparsemlp")
    b3min = ref[ref.experiment == "ablation_sparsemlp"].set_index("trial_id").mse_x100_testmin
    b3["ymin"] = b3.trial_id.map(b3min)
    D.LS = b3.rename(columns={"f1": "eps"})
    D.PA = D.full.pivot_table(index="seed", columns=["B", "lr"], values="y")
    D.PS = D.LS.pivot_table(index="seed", columns=["eps", "lr"], values="y").dropna()
    D.BB, D.LR = sorted(D.full.B.unique()), sorted(D.full.lr.unique())
    D.AB3 = np.stack([[D.PA[(b, lr)].to_numpy() for lr in D.LR] for b in D.BB])
    D.MmB = D.AB3.mean(2)
    D.optB = np.unravel_index(np.argmin(D.MmB), D.MmB.shape)
    D.eps, D.lrs = sorted(D.LS.eps.unique()), sorted(D.LS.lr.unique())
    cmS = D.LS.groupby(["eps", "lr"]).y.mean()
    D.Mm = np.array([[cmS[(e, lr)] for lr in D.lrs] for e in D.eps])
    D.opt = np.unravel_index(np.argmin(D.Mm), D.Mm.shape)
    D.A3 = np.stack([[D.PS[(e, lr)].to_numpy() for lr in D.lrs] for e in D.eps])
    # Re-runs and the original protocol
    D.split = S.depositFrame("split_variance_stable.csv")
    D.split_original = S.depositFrame("split_variance.csv")
    D.pp = S.depositFrame("perproblem_errors_stable.csv")
    D.pp_original = S.depositFrame("perproblem_errors.csv")
    tr = S.trials
    D.MAIN_original = tr[(tr.protocol == "original") & (tr.machine_id == "M00") & (tr.experiment == "main")]
    D.features = S.depositFrame("cpc18_features.csv")
    return D


# ---------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------

def blockMain(D):
    """ Main analysis of the stable configuration at the fixed epoch. """
    rng = np.random.default_rng(11)
    V, P = D.V, D.P
    print("== Main analysis at the fixed epoch ==")
    print("fixed epochs (main run):", D.EMAIN)

    print("\n-- Main run, 800 seeds")
    for m in NN:
        print(f"{m:15s} {desc(V[m].to_numpy())} | short 0 | (test minimum: {D.Vmin[m].mean():.3f})")
    for m in ("PsychForest", "BEAST_GB"):
        print(f"{m:15s} {desc(V[m].to_numpy())}")
    order = sorted(ALL, key=lambda m: V[m].mean())
    print("order:", " < ".join(f"{m} {V[m].mean():.3f}" for m in order))

    def anyswap(N, R=20000):
        ref = np.argsort([V[m].mean() for m in ALL])
        S_ = np.stack([V[m].to_numpy()[rng.integers(0, 800, (R, N))].mean(1) for m in ALL], 1)
        return 1 - float((np.argsort(S_, 1) == ref).all(1).mean())
    print("P(any swap) Bootstrap N=1/5/50/800:", [round(anyswap(n), 3) for n in (1, 5, 50, 800)])
    for a, b in [(order[i], order[i + 1]) for i in range(5)]:
        d = P[a] - P[b]
        w = 100 * (P[a] < P[b]).mean()
        half = 1.645 * d.std(ddof=1) / np.sqrt(800)
        print(f"  neighbour pair {a}-{b}: mean difference {d.mean():+.3f} CI90 [{d.mean() - half:+.3f},"
              f"{d.mean() + half:+.3f}] wins {w:.1f}%")

    def pswap(a, b, N=1, R=40000):
        A, B = P[a].to_numpy(), P[b].to_numpy()
        ref = A.min() < B.min()
        bo = float(((A[rng.integers(0, 800, (R, N))].min(1) < B[rng.integers(0, 800, (R, N))].min(1)) != ref).mean())
        mc = float(((lognormalDraw(rng, A, (R, N)).min(1) < lognormalDraw(rng, B, (R, N)).min(1)) != ref).mean())
        return bo, mc
    for a, b in (("BourginMLP", "PsychForest"), ("SparseMLP", "PsychForest"), ("ContextDepNet", "SparseMLP"),
                 ("LinearLogistic", "BourginMLP")):
        bo, mc = pswap(a, b)
        print(f"  {a}-{b}: best of 800 {P[a].min():.3f} vs {P[b].min():.3f}; P(swap|N=1) Bootstrap {bo:.3f} "
              f"Monte Carlo (lognormal) {mc:.3f}")
    sp = P.corr(method="spearman").to_numpy()[np.triu_indices(6, 1)]
    print(f"Spearman between models {sp.min():+.3f} to {sp.max():+.3f}; ICC(3,1) {icc(P):.3f}")

    print("\n-- Factorial grids at the fixed epoch: initialization / order / interaction (%), total variance")
    for m in NN:
        ci, co, cx, t = comps(D.G[m])
        mn = comps(D.Gmin[m])
        print(f"  {m:15s} {ci:5.1f} / {co:4.1f} / {cx:5.1f}  variance {t:.4f} | short 0 | (at the test minimum: "
              f"{mn[0]:.1f}/{mn[1]:.1f}/{mn[2]:.1f})")
    tr = D.S.trials
    avx2 = tr[(tr.protocol == "stable") & (tr.machine_id == "M01-forced-AVX2")
              & (tr.experiment == "factorial_init_order") & (tr.condition == "BourginMLP")]
    X1 = avx2.pivot(index="init_seed", columns="order_seed", values="mse_x100").reindex(
        index=D.G["BourginMLP"].index, columns=D.G["BourginMLP"].columns)
    print(f"  BourginMLP on the forced AVX2 path against the reference: max |difference| "
          f"{np.abs(X1.to_numpy() - D.G['BourginMLP'].to_numpy()).max():.2e}")
    main = tr[(tr.protocol == "stable") & (tr.experiment == "main")]
    reference = main[main.machine_id == "M00"].set_index(["condition", "seed"])
    for path in ("M01-forced-AVX2", "M01-forced-DEFAULT"):
        forced = main[main.machine_id == path].set_index(["condition", "seed"]).reindex(reference.index)
        print(f"  {path}, main run of the networks, max |difference|: fixed epoch "
              f"{(forced.mse_x100 - reference.mse_x100).abs().max():.1e}, minimum of the test curve "
              f"{(forced.mse_x100_testmin - reference.mse_x100_testmin).abs().max():.1e}")

    print("\n-- Ablation BourginMLP at the fixed epoch of each cell")
    print("  cell E n mean median sd check short test-minimum")
    for (b, lr), g in D.LB.groupby(["B", "lr"]):
        x = g.y.to_numpy()
        print(f"  {cellName((b, lr))} {g.readout_epoch.iloc[0] + 1} {len(x)} {x.mean():.3f} {np.median(x):.3f} "
              f"{x.std(ddof=1):.3f} {100 * (x <= p10(x)).mean():.3f} 0 {g.ymin.mean():.3f}")
    print("%SS batch/rate/interaction/residual: original grid", anova(D.full, "B", "lr"), "; with 3.16e-4 instead of "
          "1e-2", anova(D.conv, "B", "lr"))
    cm = D.full.groupby(["B", "lr"]).y.mean().sort_values()
    print("cells sorted:", [(cellName(k), round(v, 3)) for k, v in cm.items()][:4])
    a, b = cm.index[0], cm.index[1]
    xa, xb = D.PA[a].to_numpy(), D.PA[b].to_numpy()
    ref = xa.mean() < xb.mean()
    bo = float(((xa[rng.integers(0, 800, 40000)] < xb[rng.integers(0, 800, 40000)]) != ref).mean())
    print(f"closest pair {cellName(a)} vs {cellName(b)}: difference {xa.mean() - xb.mean():+.4f}, P(swap|N=1) "
          f"Bootstrap {bo:.3f}, paired wins {100 * (xa < xb).mean():.1f}%, paired p {stats.ttest_rel(xa, xb).pvalue:.3g}")
    print(f"ICC(3,1) 9 cells {icc(D.PA.dropna()):.3f}; winner per seed: "
          f"{100 * (D.PA.idxmin(axis=1) == cm.index[0]).mean():.1f}% for {cellName(cm.index[0])}")

    print("\n-- Ablation SparseMLP at the fixed epoch of each cell")
    print("  cell E n mean median sd check short test-minimum")
    for (e, lr), g in D.LS.groupby(["eps", "lr"]):
        x = g.y.to_numpy()
        print(f"  {cellName((e, lr))} {g.readout_epoch.iloc[0] + 1} {len(x)} {x.mean():.3f} {np.median(x):.3f} "
              f"{x.std(ddof=1):.3f} {100 * (x <= p10(x)).mean():.3f} 0 {g.ymin.mean():.3f}")
    print("%SS eps/rate/interaction/residual", anova(D.LS, "eps", "lr"))
    print("means", np.round(D.Mm, 3).tolist(), "optimum", (D.eps[D.opt[0]], D.lrs[D.opt[1]]),
          "; greedy on the means:", sum(r == tuple(D.opt) for r in greedy(D.Mm)), "of 6")
    g1 = [r for s in range(D.A3.shape[2]) for r in greedy(D.A3[:, :, s])]
    print(f"greedy with 1 trial ({D.A3.shape[2]} complete seeds): optimum {100 * np.mean([r == tuple(D.opt) for r in g1]):.1f}%, "
          f"within 0.01 {100 * np.mean([D.Mm[r] - D.Mm[tuple(D.opt)] < .01 for r in g1]):.1f}%")

    print("\n-- Training seed against test sampling (sigma_test: 2 x 10^4 bootstrap means of the 60-problem profile)")
    for m in NN:
        prof = D.pp[D.pp.model == m].groupby("problem_id").mean_sq_error_x100.mean().to_numpy()
        bs = prof[rng.integers(0, 60, (20000, 60))].mean(1)
        print(f"  {m:15s} sigma_seed {V[m].std(ddof=1):.3f} sigma_test {bs.std():.3f} ratio "
              f"{V[m].std(ddof=1) / bs.std():.2f} (closed form {np.sqrt(prof.var() / 60):.4f})")
    for m in ("BEAST_GB", "PsychForest"):
        prof = D.pp_original[D.pp_original.model == m].groupby("problem_id").mean_sq_error_x100.mean().to_numpy()
        bs = prof[rng.integers(0, 60, (20000, 60))].mean(1)
        print(f"  {m:15s} sigma_seed {V[m].std(ddof=1):.3f} sigma_test {bs.std():.3f} (closed form "
              f"{np.sqrt(prof.var() / 60):.4f})")

    print("\n-- Random re-splits (20) at the fixed epoch")
    for m in NN:
        x = D.split[D.split.model == m].mse_x100.to_numpy()
        print(f"  {m:15s} sigma_split {x.std(ddof=1):.3f} mean {x.mean():.3f} (canonical {V[m].mean():.3f}) short 0")
    for m in ("BEAST_GB", "PsychForest"):
        x = D.split_original[D.split_original.model == m].mse_x100.to_numpy()
        print(f"  {m:15s} sigma_split {x.std(ddof=1):.3f} mean {x.mean():.3f} (canonical {V[m].mean():.3f}), "
              f"original protocol")

    print("\n-- Forced arithmetic paths (main run, fixed epoch)")
    for m in NN:
        for path in ("AVX2", "DEFAULT"):
            d = tr[(tr.protocol == "stable") & (tr.machine_id == f"M01-forced-{path}") & (tr.experiment == "main")
                   & (tr.condition == m)].set_index("seed").mse_x100.reindex(V[m].index)
            v = V[m]
            print(f"  {m:15s} {path:7s} max |difference| {np.abs(d - v).max():.2e} Spearman "
                  f"{stats.spearmanr(d, v).statistic:.3f} dP10 {100 * (p10(d.to_numpy()) / p10(v.to_numpy()) - 1):+.2f}% "
                  f"dmean {100 * (d.mean() / v.mean() - 1):+.2f}% KS p {stats.ks_2samp(d, v).pvalue:.2f}")

    print("\n-- Ablation BourginMLP: greedy search and winner per seed")
    gB = [r for s in range(D.AB3.shape[2]) for r in greedy(D.AB3[:, :, s])]
    print("means", np.round(D.MmB, 3).tolist(), "optimum", (D.BB[D.optB[0]], D.LR[D.optB[1]]), "; greedy on the means",
          sum(r == tuple(D.optB) for r in greedy(D.MmB)), "of 6; with 1 trial optimum %.1f%%, within 0.01 %.1f%%" % (
              100 * np.mean([r == tuple(D.optB) for r in gB]),
              100 * np.mean([D.MmB[r] - D.MmB[tuple(D.optB)] < .01 for r in gB])))

    print("\n-- Ablation SparseMLP: closest pair and winner per seed")
    cmS2 = D.LS.groupby(["eps", "lr"]).y.mean().sort_values()
    a, b = cmS2.index[0], cmS2.index[1]
    xa, xb = D.PS[a].to_numpy(), D.PS[b].to_numpy()
    ref = xa.mean() < xb.mean()
    bo = float(((xa[rng.integers(0, len(xa), 40000)] < xb[rng.integers(0, len(xb), 40000)]) != ref).mean())
    print(f"best cells {cellName(a)} {xa.mean():.3f} and {cellName(b)} {xb.mean():.3f}; P(swap|N=1) Bootstrap {bo:.3f}; "
          f"paired wins {100 * (xa < xb).mean():.1f}%; paired p {stats.ttest_rel(xa, xb).pvalue:.2g}; winner per seed "
          f"{100 * (D.PS.idxmin(axis=1) == a).mean():.1f}%; ICC 9 cells {icc(D.PS):.3f}")

    print("\n-- Check of the fit with few trials: share of subsamples with no trial at or below the fitted target")
    for m in ("BourginMLP", "ContextDepNet", "SparseMLP", "LinearLogistic"):
        v = V[m].to_numpy()
        out = []
        for N in (10, 20):
            sub = v[np.argsort(rng.random((20000, len(v))), 1)[:, :N]]
            lsub = np.log(sub)
            q = np.exp(lsub.mean(1) + Zq * lsub.std(1))
            out.append(100 * float(((sub <= q[:, None]).mean(1) == 0).mean()))
        print(f"  {m:15s} no trial below the target: N=10 {out[0]:.1f}%  N=20 {out[1]:.1f}%")


def blockComparisons(D):
    """ Precision of the comparisons of Sections 3.2 to 3.4. """
    rng = np.random.default_rng(23)
    P = D.P
    print("\n== Comparisons of Sections 3.2 to 3.4 ==")
    print("-- BourginMLP against PsychForest: P(swap) per N (Monte Carlo lognormal / Bootstrap)")
    A, B = P["BourginMLP"].to_numpy(), P["PsychForest"].to_numpy()
    ref = A.min() < B.min()
    for N in range(1, 6):
        bo = float(((A[rng.integers(0, 800, (40000, N))].min(1) < B[rng.integers(0, 800, (40000, N))].min(1)) != ref).mean())
        mc = float(((lognormalDraw(rng, A, (40000, N)).min(1) < lognormalDraw(rng, B, (40000, N)).min(1)) != ref).mean())
        print(f"  N={N} MC {mc:.3f} Boot {bo:.3f}")

    def ebest(x, N, R=4000):
        return x[rng.integers(0, len(x), (R, N))].min(1).mean()
    crossing = next((N for N in range(1, 81) if ebest(A, N) < ebest(B, N)), None)
    print("  best-of-N BourginMLP < PsychForest:", f"N = {crossing}" if crossing else "not before 80")

    print("-- Six models: P(any swap) per N (Monte Carlo normal approximation of the mean of N / Bootstrap)")
    mus = np.array([P[m].mean() for m in ALL])
    sds = np.array([P[m].std(ddof=1) for m in ALL])
    refo = np.argsort(mus)
    for N in range(1, 6):
        bo = 1 - float((np.argsort(np.stack([P[m].to_numpy()[rng.integers(0, 800, (40000, N))].mean(1) for m in ALL],
                                             1), 1) == refo).all(1).mean())
        mc = 1 - float((np.argsort(rng.normal(mus, sds / np.sqrt(N), (40000, 6)), 1) == refo).all(1).mean())
        print(f"  N={N} MC {mc:.3f} Boot {bo:.3f}")
    print("-- Ordering in four levels: BEAST-GB < PsychForest < {ContextDepNet, SparseMLP} < "
          "{LinearLogistic, BourginMLP} (Bootstrap)")
    for N in (1, 5):
        S_ = {m: P[m].to_numpy()[rng.integers(0, 800, (40000, N))].mean(1) for m in ALL}
        print(f"  N={N} {100 * fourLevels(S_).mean():.1f}%")
    print("-- TOST of the close pairs (paired 90 % CI; margins 0.05 and 0.01)")
    for a, b in (("ContextDepNet", "SparseMLP"), ("LinearLogistic", "BourginMLP"), ("SparseMLP", "LinearLogistic")):
        d = P[a] - P[b]
        se = d.std(ddof=1) / np.sqrt(len(d))
        lo, hi = d.mean() - 1.645 * se, d.mean() + 1.645 * se
        print(f"  {a}-{b}: difference {d.mean():+.4f} CI90 [{lo:+.3f},{hi:+.3f}] equivalent at 0.05: "
              f"{max(abs(lo), abs(hi)) < 0.05}; at 0.01: {max(abs(lo), abs(hi)) < 0.01}; wins {100 * (P[a] < P[b]).mean():.1f}% "
              f"sign test p {stats.binomtest(int((P[a] < P[b]).sum()), 800).pvalue:.2g}")
    print("-- Paired wins (share of seeds on which the row model wins)")
    W = pd.DataFrame({a: {b: (P[a] < P[b]).mean() for b in ALL} for a in ALL}).T
    print(W.round(3).to_string())
    print("-- Spearman with Holm correction")
    from statsmodels.stats.multitest import multipletests
    pr = [(a, b, *stats.spearmanr(P[a], P[b])) for i, a in enumerate(ALL) for b in ALL[i + 1:]]
    rej = multipletests([p for *_, p in pr], method="holm")[0]
    top = max(pr, key=lambda t: abs(t[2]))
    print(f"  largest {top[0]}-{top[1]} {top[2]:+.3f}; significant after Holm:",
          [(a, b, round(float(r), 3)) for (a, b, r, p), k in zip(pr, rej) if k])
    v, lo, hi = iccCi(P.to_numpy())
    print(f"  ICC(3,1) {v:.3f} CI [{lo:.3f}, {hi:.3f}]")
    print("-- Factorial: bootstrap CIs of the components (rows and columns resampled, 2000)")
    for m in NN:
        X = D.G[m].to_numpy()
        bs = [comps(X[np.ix_(rng.integers(0, 35, 35), rng.integers(0, 35, 35))])[:3] for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        c = comps(X)
        print(f"  {m:15s} init {c[0]:.1f} [{lo[0]:.0f}-{hi[0]:.0f}] order {c[1]:.1f} [{lo[1]:.0f}-{hi[1]:.0f}] "
              f"interaction {c[2]:.1f} [{lo[2]:.0f}-{hi[2]:.0f}]")
    print("-- Ablations: df, F and p, and the closest pair per N (Monte Carlo lognormal / Bootstrap)")
    print("  BourginMLP", anovaF(D.full, "B"))
    print("  SparseMLP", anovaF(D.LS, "eps"))
    for lab, PX, cm_ in (("BourginMLP", D.PA, D.full.groupby(["B", "lr"]).y.mean()),
                         ("SparseMLP", D.PS, D.LS.groupby(["eps", "lr"]).y.mean())):
        c2 = cm_.sort_values()
        a, b = c2.index[0], c2.index[1]
        xa, xb = PX[a].to_numpy(), PX[b].to_numpy()
        ref = xa.min() < xb.min()
        out = []
        for N in (1, 5):
            bo = float(((xa[rng.integers(0, len(xa), (40000, N))].min(1) < xb[rng.integers(0, len(xb), (40000, N))].min(1))
                        != ref).mean())
            mc = float(((lognormalDraw(rng, xa, (40000, N)).min(1) < lognormalDraw(rng, xb, (40000, N)).min(1))
                        != ref).mean())
            out.append((N, round(mc, 3), round(bo, 3)))
        print(f"  {lab} pair {cellName(a)} {xa.mean():.3f} / {cellName(b)} {xb.mean():.3f}: P(swap) (N, MC, Boot) {out}")
    print("-- ICC with its CI per ablation")
    for lab, PX in (("BourginMLP", D.PA), ("SparseMLP", D.PS)):
        v, lo, hi = iccCi(PX.dropna().to_numpy())
        print(f"  {lab} ICC {v:.3f} [{lo:.3f}, {hi:.3f}]")


def blockFamilies(D):
    """ Section 3.1: fitted family, target and its check. """
    rng = np.random.default_rng(41)
    P = D.P
    print("\n== Section 3.1: fitted family and target ==")
    print("-- Share (%) of trials at or below the fitted P10 (Gaussian / Gamma / lognormal); Bootstrap = trials to "
          "the target (1 / share)")
    for m in ALL:
        x = P[m].to_numpy()
        g = x.mean() + Zq * x.std(ddof=1)
        a, _, sc = stats.gamma.fit(x, floc=0)
        ga = stats.gamma.ppf(.1, a, 0, sc)
        ln = p10(x)
        sh = (x <= ln).mean()
        print(f"  {m:15s} {100 * (x <= g).mean():5.1f} {100 * (x <= ga).mean():5.1f} {100 * sh:5.1f} | Bootstrap "
              f"{1 / sh:.1f} trials | P10 {ln:.3f}")

    def estimators(sub):
        lsub = np.log(sub)
        return (np.percentile(sub, 10, axis=1), sub.mean(1) + Zq * sub.std(1, ddof=1),
                np.exp(lsub.mean(1) + Zq * lsub.std(1)))
    print("-- ContextDepNet: RMSE (%) against the empirical P10 of the 800 trials, and bias (%) "
          "(empirical / Gaussian / lognormal), 5000 subsamples without replacement")
    x = P["ContextDepNet"].to_numpy()
    emp = np.percentile(x, 10)
    for N in (5, 10, 20, 50, 100):
        e = estimators(x[np.argsort(rng.random((5000, 800)), 1)[:, :N]])
        print(f"  N={N:3d} RMSE " + " / ".join(f"{100 * np.sqrt(np.mean((v / emp - 1) ** 2)):.2f}" for v in e)
              + "   bias " + " / ".join(f"{100 * (v.mean() / emp - 1):+.2f}" for v in e))
    print("-- Other models: RMSE at N=5 and N=50 (empirical / Gaussian / lognormal)")
    for m in ("LinearLogistic", "BourginMLP", "SparseMLP", "PsychForest", "BEAST_GB"):
        x = P[m].to_numpy()
        emp = np.percentile(x, 10)
        out = []
        for N in (5, 50):
            e = estimators(x[np.argsort(rng.random((5000, 800)), 1)[:, :N]])
            out.append(tuple(round(100 * np.sqrt(np.mean((v / emp - 1) ** 2)), 2) for v in e))
        print(f"  {m:15s} N=5 {out[0]}  N=50 {out[1]}")


def blockFitCheck(D):
    """ Section 3.1: how often a subsample shows no trial at or below its own fitted target. """
    rng = np.random.default_rng(43)
    P = D.P
    print("\n== Section 3.1: power of the check ==")
    cell = D.PA[(16, 0.01)].to_numpy()

    def rates(v, N, R=40000):
        sub = v[np.argsort(rng.random((R, len(v))), 1)[:, :N]]
        lsub = np.log(sub)
        q = np.exp(lsub.mean(1) + Zq * lsub.std(1))
        share = (sub <= q[:, None]).mean(1)
        return 100 * float((share == 0).mean()), 100 * float((share <= 0.05).mean())
    for N in (10, 20, 50):
        others = [rates(P[m].to_numpy(), N) for m in ALL]
        c = rates(cell, N)
        print(f"N={N}: cell B16/1e-2 none {c[0]:.1f}% (<=5%: {c[1]:.1f}%) | models whose fit validates none "
              f"{min(o[0] for o in others):.1f}-{max(o[0] for o in others):.1f}% (<=5%: "
              f"{min(o[1] for o in others):.1f}-{max(o[1] for o in others):.1f}%)")


def blockFamilyShape(D):
    """ Section 3.1: shape of the four networks and the error of the target at small N. """
    rng = np.random.default_rng(5)
    print("\n== Section 3.1: shape of the networks and error of the target ==")
    for m in ("LinearLogistic", "BourginMLP", "ContextDepNet", "SparseMLP"):
        x = D.V[m].to_numpy()
        g = x.mean() + Zq * x.std(ddof=1)
        a, _, sc = stats.gamma.fit(x, floc=0)
        ga = stats.gamma.ppf(.1, a, 0, sc)
        ln = p10(x)
        emp = np.percentile(x, 10)
        out = []
        for N in (5, 10, 50):
            sub = x[np.argsort(rng.random((5000, 800)), 1)[:, :N]]
            lsub = np.log(sub)
            e = (np.percentile(sub, 10, axis=1), sub.mean(1) + Zq * sub.std(1, ddof=1),
                 np.exp(lsub.mean(1) + Zq * lsub.std(1)))
            out.append(tuple(round(100 * np.sqrt(np.mean(((v / emp) - 1) ** 2)), 2) for v in e))
        print(f"{m:15s} skewness {stats.skew(x):+.2f} kurtosis {stats.kurtosis(x):+.2f} S-W p "
              f"{stats.shapiro(x).pvalue:.2g} | under P10: Gaussian {100 * (x <= g).mean():.1f} Gamma "
              f"{100 * (x <= ga).mean():.1f} lognormal {100 * (x <= ln).mean():.1f} | RMSE % (empirical, Gaussian, "
              f"lognormal) N=5 {out[0]} N=10 {out[1]} N=50 {out[2]}")


def familyFits(x):
    """ Gaussian, Gamma and lognormal fitted to a sample of trials. """
    a, _, sc = stats.gamma.fit(x, floc=0)
    lx = np.log(x)
    return {"Gaussian": stats.norm(x.mean(), x.std(ddof=1)), "Gamma": stats.gamma(a, 0, sc),
            "Lognormal": stats.lognorm(lx.std(), 0, np.exp(lx.mean()))}


def blockFamilyTables(D, deposit):
    """ Appendix A: fit check and log-likelihood per model, and calibration of each family. """
    print("\n== Appendix A: distribution families ==")
    grid = np.linspace(0.02, 14, 1400001)
    integrate = getattr(np, "trapezoid", np.trapz)
    order = ["BEAST_GB", "PsychForest", "LinearLogistic", "ContextDepNet", "BourginMLP", "SparseMLP"]

    def logLikelihoods(x):
        fits = familyFits(x)
        return {"Gaussian": stats.norm(x.mean(), x.std()).logpdf(x).sum(), "Gamma": fits["Gamma"].logpdf(x).sum(),
                "Lognormal": fits["Lognormal"].logpdf(x).sum()}

    print("-- Stable: trials at or below the fitted P10 (%), log-likelihood minus the lognormal")
    stable = {m: D.P[m].to_numpy() for m in order}
    for m, x in stable.items():
        fits, ll = familyFits(x), logLikelihoods(x)
        print(f"  {m:15s}", {k: round(100 * float((x <= d.ppf(.1)).mean()), 1) for k, d in fits.items()},
              {k: round(float(ll[k] - ll["Lognormal"]), 1) for k in ("Gaussian", "Gamma")})
    print("-- Stable: mass at or below the empirical 5th, 10th and 25th percentiles (range over the models), largest "
          "error of the single-run win probabilities over the 15 pairs")
    for fam in ("Gaussian", "Gamma", "Lognormal"):
        mass = np.array([familyFits(x)[fam].cdf(np.quantile(x, [.05, .1, .25])) for x in stable.values()])
        errors = {}
        for i, a in enumerate(order):
            for b in order[i + 1:]:
                A, B = stable[a], stable[b]
                empirical = 1 - np.searchsorted(np.sort(B), A, side="right").mean() / len(B)
                fitted = integrate(familyFits(A)[fam].pdf(grid) * familyFits(B)[fam].sf(grid), grid)
                errors[a, b] = abs(fitted - empirical)
        print(f"  {fam:10s}", [f"{mass[:, j].min():.3f}-{mass[:, j].max():.3f}" for j in range(3)],
              round(max(errors.values()), 3), max(errors, key=errors.get))
    trials = pd.read_csv(os.path.join(deposit, "trials.csv"), low_memory=False)
    main = trials[(trials.protocol == "original") & (trials.machine_id == "M00") & (trials.experiment == "main")]
    fragile = {m: main[main.condition == m].mse_x100.to_numpy() for m in order}
    print("-- Fragile: family with the highest log-likelihood per model",
          {m: max(logLikelihoods(x), key=logLikelihoods(x).get) for m, x in fragile.items()})
    bourgin = fragile["BourginMLP"]
    low, high = bourgin[bourgin < 1.5], bourgin[bourgin >= 1.5]
    weight = len(low) / len(bourgin)
    lognormalOf = lambda y: stats.lognorm(np.log(y).std(), 0, np.exp(np.log(y).mean()))
    fits = familyFits(bourgin)
    q = np.percentile(bourgin, [5, 10, 25])
    rows = {k: (d.cdf(q), d.pdf(grid)) for k, d in fits.items()}
    rows["Two-component mixture"] = (weight * lognormalOf(low).cdf(q) + (1 - weight) * lognormalOf(high).cdf(q),
                                     weight * lognormalOf(low).pdf(grid) + (1 - weight) * lognormalOf(high).pdf(grid))
    others = [m for m in order if m != "BourginMLP"]
    print("-- Fragile, BourginMLP: mass at or below the same percentiles, largest error over its five pairs (the other "
          "model of each pair is a Gaussian)")
    for name, (mass, pdf) in rows.items():
        if name == "Gaussian":
            # Closed form: the fitted Gaussian extends below the integration grid
            fitted = [stats.norm.cdf((fragile[p].mean() - bourgin.mean())
                                     / np.hypot(bourgin.std(ddof=1), fragile[p].std(ddof=1))) for p in others]
        else:
            fitted = [integrate(pdf * stats.norm(fragile[p].mean(), fragile[p].std(ddof=1)).sf(grid), grid)
                      for p in others]
        largest = max(abs(w - np.mean(bourgin[:, None] < fragile[p][None, :])) for w, p in zip(fitted, others))
        print(f"  {name:22s}", [round(float(v), 3) for v in mass], round(float(largest), 3))


def blockMisranking(D):
    """ Section 3.2: misranking tables (Monte Carlo lognormal / Bootstrap). """
    rng = np.random.default_rng(53)
    P = D.P
    R = 40000

    def draw(x, shape, fam):
        return x[rng.integers(0, len(x), shape)] if fam == "boot" else lognormalDraw(rng, x, shape)
    print("\n== Section 3.2: misranking ==")
    A, B = P["BourginMLP"].to_numpy(), P["PsychForest"].to_numpy()
    for N in (1, 5, 800):
        for f in ("logn", "boot"):
            S_ = {m: draw(P[m].to_numpy(), (20000 if N == 800 else R, N), f).mean(1) for m in ALL}
            print(f"  ordering in four levels N={N} {f}: {100 * fourLevels(S_).mean():.1f}%")
    r, p = stats.spearmanr(P["BourginMLP"], P["SparseMLP"])
    print(f"Spearman BourginMLP-SparseMLP {r:.3f} p {p:.2g}")
    print("median BourginMLP %.3f; worst PsychForest %.3f; best PsychForest %.3f; BourginMLP beats PsychForest on "
          "%.1f%% of seeds; independent pairs %.3f" % (np.median(A), B.max(), B.min(),
                                                      100 * (P["BourginMLP"] < P["PsychForest"]).mean(),
                                                      np.mean(A[:, None] < B[None, :])))
    print("SE of the mean of ContextDepNet %.4f, BourginMLP %.4f" % (P["ContextDepNet"].std() / np.sqrt(800),
                                                                       P["BourginMLP"].std() / np.sqrt(800)))


def blockInvertedOrder(D):
    """ Section 3.2: inverted order by the best, the mean and the P10 of N trials per model, and the order of the six
    models by the mean of N trials, in full and in four levels (Monte Carlo from the lognormal with the ddof=0 std of
    the logarithms, and Bootstrap). """
    rng = np.random.default_rng(32)
    R = 40000
    NS = (1, 2, 3, 5, 10, 20)
    M = ["BEAST_GB", "PsychForest", "ContextDepNet", "SparseMLP", "LinearLogistic", "BourginMLP"]
    V = {m: D.P[m].to_numpy() for m in M}

    def draw(x, N, fam):
        if fam == "boot":
            return x[rng.integers(0, len(x), (R, N))]
        lx = np.log(x)
        return np.exp(rng.normal(lx.mean(), lx.std(), (R, N)))

    def p10of(a):
        la = np.log(a)
        return np.exp(la.mean(-1) + Zq * la.std(-1))
    STAT = {"best": lambda a: a.min(-1), "mean": lambda a: a.mean(-1), "p10": p10of}

    def swap(a, b, N, st, fam):
        if st == "p10" and N < 3:   # the fit needs three trials
            return np.nan
        f = STAT[st]
        return float(((f(draw(a, N, fam)) < f(draw(b, N, fam))) != (f(a) < f(b))).mean())
    print("\n== Section 3.2: inverted order by the best, the mean and the P10 of N trials ==")
    print("-- Pairs: rows N; columns best MC/Boot, mean MC/Boot, P10 MC/Boot")
    for a, b in (("ContextDepNet", "BourginMLP"), ("ContextDepNet", "LinearLogistic"), ("LinearLogistic", "BourginMLP"),
                 ("BourginMLP", "PsychForest")):
        print(a, b, "references best", round(V[a].min(), 3), round(V[b].min(), 3), "mean", round(V[a].mean(), 3),
              round(V[b].mean(), 3), "P10", round(float(p10of(V[a])), 3), round(float(p10of(V[b])), 3))
        for N in NS:
            print(f"  N={N:2d} " + " | ".join(f'{swap(V[a], V[b], N, s_, "logn"):.3f} {swap(V[a], V[b], N, s_, "boot"):.3f}'
                                              for s_ in STAT))
    print("-- Six models, mean of N trials: order in full and in four levels (% correct), MC and Boot")

    def six(N, fam):
        Q = {m: draw(V[m], N, fam).mean(1) for m in M}
        A = np.stack([Q[m] for m in M], 1)
        ref = np.argsort([V[m].mean() for m in M])
        return 100 * float((np.argsort(A, 1) == ref).all(1).mean()), 100 * float(fourLevels(Q).mean())
    for N in NS + (50,):
        a_, b_ = six(N, "logn"), six(N, "boot")
        print(f"  N={N:2d} full MC {a_[0]:.1f} Boot {b_[0]:.1f} | four levels MC {a_[1]:.1f} Boot {b_[1]:.1f}")


def blockMonteCarlo(D):
    """ Monte Carlo from a Gaussian and from a lognormal, against the Bootstrap. """
    rng = np.random.default_rng(31)
    P = D.P
    R = 40000

    def draw(x, shape, fam):
        if fam == "boot":
            return x[rng.integers(0, len(x), shape)]
        if fam == "gauss":
            return rng.normal(x.mean(), x.std(ddof=1), shape)
        return lognormalDraw(rng, x, shape)

    def pswapBest(a, b, N, fam):
        ref = a.min() < b.min()
        return float(((draw(a, (R, N), fam).min(1) < draw(b, (R, N), fam).min(1)) != ref).mean())
    print("\n== Monte Carlo (Gaussian / lognormal) against the Bootstrap ==")
    print("P(swap) of the best of N   [Gaussian / lognormal / Bootstrap]")
    err = {"gauss": [], "logn": []}
    for a, b in (("BourginMLP", "PsychForest"), ("ContextDepNet", "SparseMLP"), ("LinearLogistic", "BourginMLP"),
                 ("SparseMLP", "LinearLogistic"), ("BEAST_GB", "PsychForest")):
        A, B = P[a].to_numpy(), P[b].to_numpy()
        row = []
        for N in (1, 5):
            g, lg, bo = (pswapBest(A, B, N, f) for f in ("gauss", "logn", "boot"))
            row.append(f"N={N}: {g:.3f} / {lg:.3f} / {bo:.3f}")
            err["gauss"].append(abs(g - bo))
            err["logn"].append(abs(lg - bo))
        print(f"  {a}-{b}: " + " | ".join(row))
    for lab, PX in (("ablation BourginMLP", D.PA), ("ablation SparseMLP", D.PS)):
        cm_ = PX.mean().sort_values()
        a, b = cm_.index[0], cm_.index[1]
        A, B = PX[a].dropna().to_numpy(), PX[b].dropna().to_numpy()
        row = []
        for N in (1, 5):
            g, lg, bo = (pswapBest(A, B, N, f) for f in ("gauss", "logn", "boot"))
            row.append(f"N={N}: {g:.3f} / {lg:.3f} / {bo:.3f}")
            err["gauss"].append(abs(g - bo))
            err["logn"].append(abs(lg - bo))
        print(f"  {lab} {cellName(a)} vs {cellName(b)}: " + " | ".join(row))
    print("P(any swap) of the six models by the mean of N   [Gaussian / lognormal / Bootstrap]")
    refo = np.argsort([P[m].mean() for m in ALL])
    for N in (1, 5):
        out = []
        for f in ("gauss", "logn", "boot"):
            S_ = np.stack([draw(P[m].to_numpy(), (R, N), f).mean(1) for m in ALL], 1)
            out.append(1 - float((np.argsort(S_, 1) == refo).all(1).mean()))
        print(f"  N={N}: {out[0]:.3f} / {out[1]:.3f} / {out[2]:.3f}")
        err["gauss"].append(abs(out[0] - out[2]))
        err["logn"].append(abs(out[1] - out[2]))
    print(f"Absolute difference with the Bootstrap: Gaussian mean {np.mean(err['gauss']):.4f} max "
          f"{np.max(err['gauss']):.4f} | lognormal mean {np.mean(err['logn']):.4f} max {np.max(err['logn']):.4f}")
    print("The same comparison in the original (fragile) configuration: BourginMLP-PsychForest N=1")
    O = D.MAIN_original.pivot(index="seed", columns="condition", values="mse_x100")
    A, B = O["BourginMLP"].to_numpy(), O["PsychForest"].to_numpy()
    print("  ", [round(pswapBest(A, B, 1, f), 3) for f in ("gauss", "logn", "boot")])


def blockDecomposition(D):
    """ Section 3.4.1: components against eta^2, slicing and subgrids. """
    rng = np.random.default_rng(61)
    print("\n== Section 3.4.1: factorial decomposition ==")
    for m in NN:
        X = D.G[m].to_numpy()
        c = comps(X)[:3]
        e = eta(X)
        print(f"{m:15s} components {np.round(c, 1)} eta2 {np.round(e, 1)} max difference "
              f"{max(abs(np.array(c) - np.array(e))):.1f}; total sd {X.std(ddof=1):.3f}")
    X = D.G["BourginMLP"].to_numpy()
    io_, oo = X.std(0, ddof=1), X.std(1, ddof=1)
    print("BourginMLP slicing: init-only %.3f (range %.3f-%.3f) order-only %.3f (range %.3f-%.3f) init-mean %.3f "
          "order-mean %.3f diagonal %.3f full %.3f" % (io_.mean(), io_.min(), io_.max(), oo.mean(), oo.min(), oo.max(),
                                                       X.mean(1).std(ddof=1), X.mean(0).std(ddof=1),
                                                       np.diag(X).std(ddof=1), X.std(ddof=1)))
    for m in ("ContextDepNet", "LinearLogistic", "SparseMLP"):
        Y = D.G[m].to_numpy()
        print(f"  {m}: init-only {Y.std(0, ddof=1).mean():.3f} order-only {Y.std(1, ddof=1).mean():.3f} "
              f"full {Y.std(ddof=1):.3f}")
    for m in ("BourginMLP", "SparseMLP", "ContextDepNet"):
        Y = D.G[m].to_numpy()
        out = []
        for k in (10, 20):
            r = np.array([comps(Y[np.ix_(rng.permutation(35)[:k], rng.permutation(35)[:k])])[:3] for _ in range(4000)])
            out.append((k, np.round(np.percentile(r[:, 1], [2.5, 97.5]), 0).tolist()))
        print(f"  {m} order share 35x35 {comps(Y)[1]:.1f}; 95% range over subgrids {out}")


def blockAblations(D):
    """ Section 3.3: ablations. """
    rng = np.random.default_rng(34)
    print("\n== Section 3.3: ablations ==")
    print("-- %SS, and df, F and p of each term")
    print("  BourginMLP grid", anova(D.full, "B", "lr"), anovaF(D.full, "B"))
    print("  BourginMLP with 3.16e-4", anova(D.conv, "B", "lr"), anovaF(D.conv, "B"))
    print("  SparseMLP", anova(D.LS, "eps", "lr"), anovaF(D.LS, "eps"))
    print("  log scale: BourginMLP", anova(D.full.assign(y=np.log(D.full.y)), "B", "lr"), " SparseMLP",
          anova(D.LS.assign(y=np.log(D.LS.y)), "eps", "lr"))

    def greedyDetail(M, rows, cols, name):
        out = []
        for i0 in range(3):
            lev = int(np.argmin(M[i0]))
            e = int(np.argmin(M[:, lev]))
            out.append((f"{name}={rows[i0]:g} first", (rows[e], cols[lev])))
        for l0 in range(3):
            e = int(np.argmin(M[:, l0]))
            lev = int(np.argmin(M[e]))
            out.append((f"lr={cols[l0]:g} first", (rows[e], cols[lev])))
        return out
    for lab, A3_, Mm_, rows, cols, nm, opt_ in (("BourginMLP", D.AB3, D.MmB, D.BB, D.LR, "B", D.optB),
                                                ("SparseMLP", D.A3, D.Mm, D.eps, D.lrs, "eps", D.opt)):
        print(f"-- greedy {lab}: optimum {cellName((rows[opt_[0]], cols[opt_[1]]))}")
        for s, r in greedyDetail(Mm_, rows, cols, nm):
            print("   on the means, start", s, "->", cellName(r))
        for N in (1, 5, 20, 100):
            ok = []
            for _ in range(2000 if N > 1 else 1):
                if N == 1:
                    ok += [r == tuple(opt_) for s_ in range(A3_.shape[2]) for r in greedy(A3_[:, :, s_])]
                else:
                    idx = rng.integers(0, A3_.shape[2], N)
                    ok += [r == tuple(opt_) for r in greedy(A3_[:, :, idx].mean(2))]
            print(f"   N={N}: optimum {100 * np.mean(ok):.1f}%")
        win = pd.Series([np.unravel_index(np.argmin(A3_[:, :, s_]), (3, 3)) for s_ in range(A3_.shape[2])]
                        ).value_counts(normalize=True)
        print("   winner per seed (full grid, 1 trial):",
              [(cellName((rows[i], cols[j])), round(100 * v, 1)) for (i, j), v in win.items()])
    print("-- Closest pair: paired Hodges-Lehmann and bootstrap CI")

    def hl(v):
        w = (v[:, None] + v[None, :]) / 2
        return np.median(w[np.triu_indices(len(v))])
    for lab, PX, cm_ in (("BourginMLP", D.PA, D.full.groupby(["B", "lr"]).y.mean()),
                         ("SparseMLP", D.PS, D.LS.groupby(["eps", "lr"]).y.mean())):
        c2 = cm_.sort_values()
        a, b = c2.index[0], c2.index[1]
        d = (PX[a] - PX[b]).dropna().to_numpy()
        bs = [hl(d[rng.integers(0, len(d), len(d))]) for _ in range(300)]
        print(f"  {lab} {cellName(a)} vs {cellName(b)}: mean difference {d.mean():+.4f} HL {hl(d):+.4f} CI95 "
              f"[{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}] wins {100 * (d < 0).mean():.1f}% "
              f"medians {PX[a].median():.3f} / {PX[b].median():.3f}")
    print("-- ICC without the 1e-2 column and largest correlation between cells")
    for lab, PX in (("BourginMLP", D.PA), ("SparseMLP", D.PS)):
        sub = PX[[c for c in PX.columns if c[1] != 0.01]].dropna()
        C = PX.dropna().corr().to_numpy()
        iu = np.triu_indices(9, 1)
        k = np.argmax(C[iu])
        cols_ = list(PX.columns)
        print(f"  {lab}: ICC without 1e-2 {icc(sub):.3f}; largest r {C[iu][k]:.2f} between {cellName(cols_[iu[0][k]])} "
              f"and {cellName(cols_[iu[1][k]])}; median r {np.median(C[iu]):.2f}")
    print("-- Cell with a tail (16, 1e-2)")
    x = D.PA[(16, 0.01)].to_numpy()
    medians = D.full.groupby(["B", "lr"]).y.median().sort_values().round(3).head(4)
    print(f"  mean {x.mean():.3f} median {np.median(x):.3f} sd {x.std(ddof=1):.3f} >1.5 {100 * (x > 1.5).mean():.1f}% "
          f"kurtosis {stats.kurtosis(x):.1f} max {x.max():.2f}; medians of the cells:",
          {cellName(k): v for k, v in medians.items()})
    print("-- %SS with 3 decimals: BourginMLP", [f"{v:.3f}" for v in anova(D.full, "B", "lr", digits=9)],
          " 3.16e-4", [f"{v:.3f}" for v in anova(D.conv, "B", "lr", digits=9)],
          " SparseMLP", [f"{v:.3f}" for v in anova(D.LS, "eps", "lr", digits=9)])
    print("-- Lognormal P10, empirical 10th percentile and check per cell (BourginMLP)")
    for c in D.PA.columns:
        x = D.PA[c].to_numpy()
        q = p10(x)
        print(f"  {cellName(c)}: mean {x.mean():.3f} P10 {q:.3f} empirical q10 {np.percentile(x, 10):.3f} check "
              f"{100 * (x <= q).mean():.1f}%")
    for lab, PX, frame, f1 in (("BourginMLP", D.PA, D.full, "B"), ("SparseMLP", D.PS, D.LS, "eps")):
        C = PX.corr()
        big = [c for c in PX.columns if c[1] == 0.01]
        rest = [c for c in PX.columns if c[1] != 0.01]
        across = np.mean([C.loc[a, b] for a in big for b in rest])
        among = np.mean([C.loc[a, b] for i, a in enumerate(rest) for b in rest[i + 1:]])
        within = np.mean([C.loc[a, b] for i, a in enumerate(big) for b in big[i + 1:]])
        print(f"-- {lab}: mean r of 1e-2 with the rest {across:.2f}; between cells without 1e-2 {among:.2f}; "
              f"within 1e-2 {within:.2f}")
        above = {cellName(c): round(float(100 * (PX[c] > 1.5).mean()), 1) for c in PX.columns if (PX[c] > 1.5).any()}
        print("   >1.5 per cell:", above)
        print("   marginal means:", frame.groupby(f1).y.mean().round(3).to_dict(),
              frame.groupby("lr").y.mean().round(3).to_dict())


def blockConcentration(D):
    """ Section 3.4.2: concentration of the seed variance on a few test problems (first 50 seeds, fixed epoch). """
    print("\n== Section 3.4.2: per-problem concentration ==")
    test = D.features[(D.features.Train_Test == "Test") & (D.features.block == 1)].set_index("problem_id")
    for m in ("BourginMLP", "LinearLogistic", "ContextDepNet", "SparseMLP"):
        W = D.pp[D.pp.model == m].pivot(index="seed", columns="problem_id", values="mean_sq_error_x100")
        var = W.var(ddof=1).sort_values(ascending=False)
        agg = W.mean(1)
        k = W.shape[1]
        att = W.apply(lambda c: np.cov(c, agg, ddof=1)[0, 1]) / k / agg.var(ddof=1)
        s = [f"{m}: seeds {len(W)}"]
        for n in (5, 10):
            s.append(f"top {n} {100 * var.iloc[:n].sum() / var.sum():.1f}% of the sum, {100 * att[var.index[:n]].sum():.1f}% "
                     f"of the aggregate")
        top, bot = var.index[:10], var.index[-10:]
        s.append(f"median |diffEV| top10 {test.loc[top, 'diffEV'].abs().median():.1f} vs stable "
                 f"{test.loc[bot, 'diffEV'].abs().median():.1f}; median diffEV {test.loc[top, 'diffEV'].median():+.1f} vs "
                 f"{test.loc[bot, 'diffEV'].median():+.1f}")
        s.append(f"Spearman variance-mean error {stats.spearmanr(W.var(ddof=1), W.mean()).statistic:.2f}")
        amb = test.loc[var.index, "Amb"] == 1
        s.append(f"ambiguous/not {var[amb].mean() / var[~amb].mean():.1f}x ({int(amb.sum())} ambiguous)")
        print(" | ".join(s))


def blockReporting(D):
    """ Section 4: reporting format and the stability of the order by P10 against the mean. """
    rng = np.random.default_rng(4)
    print("\n== Section 4: reporting format ==")
    for m in ALL:
        x = D.V[m].to_numpy()
        q = p10(x)
        c = (x <= q).sum()
        print(f"  {m:15s} best {x.min():.3f} mean {x.mean():.3f} sd {x.std(ddof=1):.3f} P10 {q:.3f} check {c}/800 = "
              f"{100 * c / 800:.3f}%")

    def p10v(S_):
        lS = np.log(S_)
        return np.exp(lS.mean(-1) + Zq * lS.std(-1))
    Vn = {m: D.V[m].to_numpy() for m in ALL}
    for lab, MOD in (("six models", ALL), ("five, without BourginMLP of the tied pair", [m for m in ALL if m != "BourginMLP"])):
        rp = np.argsort([p10(Vn[m]) for m in MOD])
        rm = np.argsort([Vn[m].mean() for m in MOD])
        print(f"-- Full order reproduced, {lab}: order by P10 {[MOD[i] for i in rp]} | order by mean {[MOD[i] for i in rm]}")
        for N in (5, 10, 20, 50, 100):
            S_ = np.stack([Vn[m][np.argsort(rng.random((20000, 800)), 1)[:, :N]] for m in MOD], 1)
            print(f"  N={N:3d} P10 {100 * (np.argsort(p10v(S_), 1) == rp).all(1).mean():.0f}%  mean "
                  f"{100 * (np.argsort(S_.mean(2), 1) == rm).all(1).mean():.0f}%")
    print("-- Ordering in four levels by P10 and by mean")
    for N in (5, 10, 20, 50):
        S_ = {m: Vn[m][np.argsort(rng.random((20000, 800)), 1)[:, :N]] for m in ALL}
        line = []
        for lab, f in (("P10", p10v), ("mean", lambda a: a.mean(-1))):
            T = {m: f(S_[m]) for m in ALL}
            line.append(f"N={N} {lab} {100 * fourLevels(T).mean():.1f}%")
        print("  " + ";  ".join(line) + ";")


def blockAppendices(D):
    """ Appendices on the distribution families (A) and on the sample size (C). """
    rng = np.random.default_rng(12)
    Vn = {m: D.V[m].to_numpy() for m in ALL}
    print("\n== Appendices ==")
    print("-- A. Shape of the distribution (main run, 800)")
    for m in ALL:
        x = Vn[m]
        print(f"  {m:15s} median {np.median(x):.3f} mean {x.mean():.3f} kurtosis {stats.kurtosis(x):+.2f} "
              f"S-W p {stats.shapiro(x).pvalue:.2g}")
    print("-- A. Calibration of the lower tail: fitted mass at the empirical 5th, 10th and 25th percentiles")
    for m in ALL:
        x = Vn[m]
        q = np.quantile(x, [.05, .1, .25])
        lx = np.log(x)
        g = stats.norm.cdf(q, x.mean(), x.std(ddof=1))
        ln = stats.norm.cdf(np.log(q), lx.mean(), lx.std())
        print(f"  {m:15s} Gaussian {np.round(g, 3).tolist()} lognormal {np.round(ln, 3).tolist()}")
    print("-- A. Probability that one trial wins, independent pairs: empirical, Gaussian, lognormal (15 pairs)")
    dg, dl = [], []
    for i, a in enumerate(ALL):
        for b in ALL[i + 1:]:
            Bs = np.sort(Vn[b])
            emp = 1 - np.searchsorted(Bs, np.sort(Vn[a]), side="right").astype(float).mean() / len(Bs)
            g = stats.norm.cdf((Vn[b].mean() - Vn[a].mean()) / np.hypot(Vn[a].std(ddof=1), Vn[b].std(ddof=1)))
            la, lb = np.log(Vn[a]), np.log(Vn[b])
            ln = stats.norm.cdf((lb.mean() - la.mean()) / np.hypot(la.std(), lb.std()))
            dg.append(abs(g - emp))
            dl.append(abs(ln - emp))
            if max(abs(g - emp), abs(ln - emp)) > .02:
                print(f"   {a}-{b}: empirical {emp:.3f} Gaussian {g:.3f} lognormal {ln:.3f}")
    print(f"  max |difference| Gaussian {max(dg):.3f} lognormal {max(dl):.3f}; mean Gaussian {np.mean(dg):.3f} "
          f"lognormal {np.mean(dl):.3f}")

    print("-- C. Simulation noise with 4 x 10^4 resamples (30 repetitions)")

    def psBoot(A, B, R, r):
        ref = A.min() < B.min()
        return float(((A[r.integers(0, len(A), R)] < B[r.integers(0, len(B), R)]) != ref).mean())

    def psMc(A, B, R, r):
        ref = A.min() < B.min()
        la, lb = np.log(A), np.log(B)
        return float(((np.exp(r.normal(la.mean(), la.std(), R)) < np.exp(r.normal(lb.mean(), lb.std(), R)))
                      != ref).mean())
    for a, b, A, B in (("BourginMLP", "PsychForest", Vn["BourginMLP"], Vn["PsychForest"]),
                       ("ablation (32,1e-2)-(16,1e-4)", "", D.PA[(32, 0.01)].to_numpy(), D.PA[(16, 0.0001)].to_numpy())):
        sb = [psBoot(A, B, 40000, np.random.default_rng(k)) for k in range(30)]
        sm = [psMc(A, B, 40000, np.random.default_rng(100 + k)) for k in range(30)]
        print(f"  {a} {b}: Boot mean {np.mean(sb):.3f} SD {np.std(sb):.4f} | MC mean {np.mean(sm):.3f} SD {np.std(sm):.4f}")

    print("-- C. Precision of P(swap|N=1): double bootstrap (300), exact Bootstrap and closed-form lognormal "
          "Monte Carlo")

    def bootExact(A, B, ref=None):
        ref = A.min() < B.min() if ref is None else ref
        p_alt = 1 - np.searchsorted(np.sort(B), A, side="right").mean() / len(B)
        return 1 - p_alt if ref else p_alt

    def mcExact(A, B, ref=None):
        ref = A.min() < B.min() if ref is None else ref
        la, lb = np.log(A), np.log(B)
        p = stats.norm.cdf((lb.mean() - la.mean()) / np.hypot(la.std(), lb.std()))
        return 1 - p if ref else p
    for a, b in (("BourginMLP", "PsychForest"), ("BourginMLP", "BEAST_GB"), ("LinearLogistic", "BourginMLP")):
        A, B = Vn[a], Vn[b]
        ob, om = [], []
        for _ in range(300):
            A2, B2 = A[rng.integers(0, 800, 800)], B[rng.integers(0, 800, 800)]
            R0 = A.min() < B.min()
            ob.append(bootExact(A2, B2, R0))
            om.append(mcExact(A2, B2, R0))
        print(f"  {a}-{b}: MC {mcExact(A, B):.3f} HW {1.96 * np.std(om):.3f} | Boot {bootExact(A, B):.3f} HW "
              f"{1.96 * np.std(ob):.3f} | observed wins of {a}: {int((A[:, None] < B[None, :]).sum())} of 640000; "
              f"min {a} {A.min():.3f} max {b} {B.max():.3f}")
    print("-- C. Partial eta^2 of the effects of the ablations")
    for lab, L_, f in (("BourginMLP", D.full, "B"), ("SparseMLP", D.LS, "eps")):
        g = L_.y.mean()
        A_ = L_.groupby(f).y.transform("mean")
        B_ = L_.groupby("lr").y.transform("mean")
        C_ = L_.groupby([f, "lr"]).y.transform("mean")
        ssa, ssb = ((A_ - g) ** 2).sum(), ((B_ - g) ** 2).sum()
        ssi, sse = ((C_ - A_ - B_ + g) ** 2).sum(), ((L_.y - C_) ** 2).sum()
        print(f"  {lab}: {f} {ssa / (ssa + sse):.4f} lr {ssb / (ssb + sse):.6f} interaction {ssi / (ssi + sse):.4f}")

    A, B = Vn["BourginMLP"], Vn["BEAST_GB"]
    print(f"closed-form lognormal Monte Carlo, P(BourginMLP < BEAST-GB) for one trial each: {mcExact(A, B, ref=False):.2e}")


def gridShares(C):
    """ Shares (%) of the interaction and of the first factor in an array of seeds x levels x levels. """
    n, _, lb = C.shape
    g = C.mean()
    a, b, cell = C.mean((0, 2)), C.mean((0, 1)), C.mean(0)
    total = ((C - g) ** 2).sum()
    return 100 * n * ((cell - a[:, None] - b[None, :] + g) ** 2).sum() / total, 100 * n * lb * ((a - g) ** 2).sum() / total


def blockSmallerSamples(D):
    """ Sample-size appendix: headline statistics on the first seeds of the pool and on random subsets of the trials. """
    print("\n== Sample-size appendix: headline statistics at smaller sample sizes ==")
    sizes = (100, 200, 400, 800)
    pool = np.random.default_rng(MASTER_SEED_FOR_POOL).integers(0, 2 ** 32, size=800, dtype=np.uint64).astype(np.int64)
    first = {n: set(int(s) for s in pool[:n]) for n in sizes}
    P = {n: D.P[[int(s) in first[n] for s in D.P.index]] for n in sizes}
    print("-- First seeds of the pool: mean (std), P10 and best trial at N =", sizes)
    for m in ALL:
        print(f"  {m:15s}", " | ".join(f"{P[n][m].mean():.3f} ({P[n][m].std(ddof=1):.3f})" for n in sizes), "; P10",
              " ".join(f"{p10(P[n][m].to_numpy()):.3f}" for n in sizes), "; best",
              " ".join(f"{P[n][m].min():.3f}" for n in sizes))
    print("  P(swap|N=1) BourginMLP-PsychForest, exact",
          [round(float(np.searchsorted(np.sort(P[n].PsychForest.to_numpy()), P[n].BourginMLP.to_numpy(), side="right").mean()
                       / n), 3) for n in sizes])
    print("  ICC six models", [round(float(icc(P[n])), 3) for n in sizes])
    full = {n: D.full[D.full.seed.isin(first[n])] for n in sizes}
    print("  Ablation BourginMLP interaction %SS", [anova(full[n], "B", "lr")[2] for n in sizes])
    print("  Ablation SparseMLP eps %SS", [anova(D.LS[D.LS.seed.isin(first[n])], "eps", "lr")[0] for n in sizes])
    print("  trials above 1.5 in the cell B=16, lr=1e-2",
          [int((full[n][(full[n].B == 16) & (full[n].lr == 1e-2)].y > 1.5).sum()) for n in sizes])
    print("-- 2000 random subsets of the 800 trials: same order of the six models as the full sample (mean, P10), best cell "
          "of each ablation by the mean recovered, central 95% of the largest share")
    rng = np.random.default_rng(8)
    X = D.P[ALL].to_numpy()
    cubeB, cubeS = D.PA.to_numpy().reshape(800, 3, 3), D.PS.to_numpy().reshape(800, 3, 3)
    lx = np.log(X)
    orderMean, orderP10 = tuple(np.argsort(X.mean(0))), tuple(np.argsort(lx.mean(0) + Zq * lx.std(0)))
    bestB, bestS = cubeB.mean(0).argmin(), cubeS.mean(0).argmin()
    for n in sizes[:-1]:
        same, shares, tail = np.zeros(4), [], []
        for _ in range(2000):
            i = rng.choice(800, n, replace=False)
            ls = np.log(X[i])
            same += (tuple(np.argsort(X[i].mean(0))) == orderMean, tuple(np.argsort(ls.mean(0) + Zq * ls.std(0))) == orderP10,
                     cubeB[i].mean(0).argmin() == bestB, cubeS[i].mean(0).argmin() == bestS)
            shares.append((gridShares(cubeB[i])[0], gridShares(cubeS[i])[1]))
            tail.append(int((cubeB[i][:, 0, 2] > 1.5).sum()))
        low, high = np.percentile(shares, [2.5, 97.5], axis=0)
        print(f"  N={n}: order mean {same[0] / 2000:.3f}, P10 {same[1] / 2000:.3f} | best cell BourginMLP {same[2] / 2000:.3f}, "
              f"SparseMLP {same[3] / 2000:.3f} | BourginMLP interaction {low[0]:.2f}-{high[0]:.2f}%, SparseMLP eps "
              f"{low[1]:.1f}-{high[1]:.1f}% | Spearman of the interaction share with the trials above 1.5 "
              f"{stats.spearmanr(np.array(shares)[:, 0], tail)[0]:.2f}")


def blockIntroduction(D):
    """ Section 1 and the literature comparison of Section 3.2. """
    print("\n== Section 1 and the literature table ==")
    x = D.P["BourginMLP"].to_numpy()
    q25, q75 = np.percentile(x, [25, 75])
    print(f"BourginMLP: range {x.min():.3f} to {x.max():.3f} (amplitude {x.max() - x.min():.2f}), interquartile range "
          f"{q75 - q25:.2f}")
    print("means:", {m: round(float(D.P[m].mean()), 2) for m in ALL})
    test = D.features[D.features.Train_Test == "Test"]
    print(f"BEAST theory only, canonical test rows: {100 * ((test.BEASTpred - test.B_rate) ** 2).mean():.3f}")
    beast = D.P["BEAST_GB"].to_numpy()
    print(f"share of the BEAST-GB trials at or below 0.556: {100 * (beast <= 0.556).mean():.1f}%")


def blockPairSeparation(D):
    """ Section 3.2: reversal of each pair of networks with one trial per model against how far apart the pair is, and
    the overlap of the fitted distributions of the pair of the table (Monte Carlo from the lognormal and Bootstrap). """
    rng = np.random.default_rng(20261109)
    R = 40000
    V = {m: D.P[m].to_numpy() for m in NN}
    print("\n== Section 3.2: reversal with one trial per model ==")
    for i, a in enumerate(NN):
        for b in NN[i + 1:]:
            good, bad = sorted((V[a], V[b]), key=np.mean)
            mc = float((lognormalDraw(rng, good, R) >= lognormalDraw(rng, bad, R)).mean())
            bo = float((good[rng.integers(0, len(good), R)] >= bad[rng.integers(0, len(bad), R)]).mean())
            d = abs(good.mean() - bad.mean()) / np.sqrt((good.var() + bad.var()) / 2)
            print(f"  {a}-{b}: means {d:.2f} standard deviations apart, P(swap | N=1) MC {100 * mc:.1f}%, Boot {100 * bo:.1f}%")
    grid = np.linspace(0.3, 2, 20001)

    def fitted(v):
        lv = np.log(v)
        return stats.lognorm.pdf(grid, lv.std(), scale=np.exp(lv.mean()))
    shared = np.minimum(fitted(V["ContextDepNet"]), fitted(V["LinearLogistic"])).sum() * (grid[1] - grid[0])
    print(f"  area shared by the fitted lognormals of ContextDepNet and LinearLogistic: {100 * shared:.1f}%")


def blockP10AgainstMean(D):
    """ Section 4.2: how the P10 and the mean relate to the best value of a model, and how well each one orders the
    models from few trials (Monte Carlo from the lognormal and Bootstrap). """
    rng = np.random.default_rng(1077)
    R = 40000
    V = {m: D.P[m].to_numpy() for m in ALL}

    def p10of(a):
        la = np.log(a)
        return np.exp(la.mean(-1) + Zq * la.std(-1))

    def draw(x, shape, fam):
        return x[rng.integers(0, len(x), shape)] if fam == "boot" else lognormalDraw(rng, x, shape)

    def rhoBest(X):
        best = [x.min() for x in X]
        return (stats.spearmanr([p10of(x) for x in X], best).statistic,
                stats.spearmanr([x.mean() for x in X], best).statistic)
    print("\n== Section 4.2: the P10 against the mean ==")
    print("Rank correlation with the best of the 800 trials (P10, mean)")
    print("  six models: %.2f %.2f" % rhoBest(list(V.values())))
    print("  Ablation BourginMLP, without the cell whose fit does not validate: %.2f %.2f"
          % rhoBest([D.PA[c].to_numpy() for c in D.PA.columns if c != (16, 0.01)]))
    print("  Ablation SparseMLP: %.2f %.2f" % rhoBest([D.PS[c].to_numpy() for c in D.PS.columns]))
    for fam in ("logn", "boot"):
        by_mean, by_p10 = [], []
        for i, a in enumerate(ALL):
            for b in ALL[i + 1:]:
                ref = V[a].min() < V[b].min()
                A, B = draw(V[a], (R, 10), fam), draw(V[b], (R, 10), fam)
                by_mean.append(((A.mean(1) < B.mean(1)) == ref).mean())
                by_p10.append(((p10of(A) < p10of(B)) == ref).mean())
        print(f"Order of the best runs from ten trials per model, average over the pairs ({fam}): "
              f"P10 {100 * np.mean(by_p10):.1f}%, mean {100 * np.mean(by_mean):.1f}%")
    for N in (10, 50):
        row = []
        for name, stat in (("P10", p10of), ("mean", lambda a: a.mean(-1))):
            ref = np.argsort([stat(V[m]) for m in ALL])
            for fam in ("logn", "boot"):
                S_ = np.stack([stat(draw(V[m], (R, N), fam)) for m in ALL], 1)
                row.append(f"{name} {fam} {100 * float((np.argsort(S_, 1) == ref).all(1).mean()):.1f}%")
        print(f"Order of the six models from {N} trials per model: " + ", ".join(row))


def blockTrialBudget(D):
    """ Section 5.1: the trials needed to defend a difference, checked by running a two-sample test on resampled real
    trials: power against the difference, and false positives without it. """
    V = {m: D.P[m].to_numpy() for m in ALL}

    def rejected(a, b):
        return 100 * float((stats.ttest_ind(a, b, axis=1, equal_var=False).pvalue < 0.05).mean())
    print("\n== Section 5.1: budgets of the table of trials needed, on resampled trials ==")
    for m, delta, N in (("BEAST_GB", .01, 8), ("PsychForest", .01, 27), ("LinearLogistic", .05, 31), ("ContextDepNet", .05, 31),
                        ("SparseMLP", .05, 31), ("LinearLogistic", .1, 8), ("ContextDepNet", .1, 8), ("SparseMLP", .1, 8),
                        ("BourginMLP", .05, 91), ("BourginMLP", .1, 23)):
        rng = np.random.default_rng(77 + N)
        x, R = V[m], 20000
        pick = lambda: x[rng.integers(0, len(x), (R, N))]
        print(f"  {m}, difference {delta}, {N} trials: power {rejected(pick(), pick() + delta):.1f}%, "
              f"false positives {rejected(pick(), pick()):.1f}%")


def blockFragile(D):
    """ Section 3.6: the fragile configuration re-trained with a patience of 300 epochs, and its interaction on the
    reference and the AVX2 arithmetic paths. """
    print("\n== Section 3.6: patience and arithmetic path of the fragile configuration ==")
    tr = D.S.trials
    main = tr[tr.experiment == "main"]
    base = main[(main.protocol == "original") & (main.machine_id == "M00")
                & (main.condition == "BourginMLP")].set_index("seed").mse_x100
    longer = main[main.protocol == "original_patience300"].set_index("seed").mse_x100
    tail = base[base > 1.5].index
    print(f"  BourginMLP with patience 300 ({len(longer)} seeds): of the {len(tail)} trials above 1.5 with patience 50, "
          f"{int((longer.loc[tail] < 1.5).sum())} end below it, median {longer.loc[tail].median():.3f}; "
          f"std of all the trials {longer.std(ddof=1):.3f}")

    def grid(machine, model):
        rows = tr[(tr.experiment == "factorial_init_order") & (tr.protocol == "original")
                  & (tr.machine_id == machine) & (tr.condition == model)]
        return rows.pivot(index="init_seed", columns="order_seed", values="mse_x100").to_numpy()

    def residuals(X):
        return X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + X.mean()

    ref, avx2 = grid("M00", "BourginMLP"), grid("M00-forced-AVX2", "BourginMLP")
    _, _, share, total = comps(ref)
    rx, ry = residuals(ref).ravel(), residuals(avx2).ravel()
    pearson = stats.pearsonr(rx, ry)
    ci = pearson.confidence_interval()
    common = np.mean(rx * ry) * ref.size / ((ref.shape[0] - 1) * (ref.shape[1] - 1))
    print(f"  BourginMLP interaction: {share:.1f}% on the reference path, {comps(avx2)[2]:.1f}% on AVX2; correlation "
          f"between the paths of the row means {np.corrcoef(ref.mean(1), avx2.mean(1))[0, 1]:.2f} and of the interaction "
          f"residuals {pearson.statistic:.2f} [{ci.low:.2f}, {ci.high:.2f}]; part of the interaction common to both, "
          f"{100 * max(min(common, share / 100 * total), 0) / total:.1f}% of the variance")
    for m in ("LinearLogistic", "ContextDepNet", "SparseMLP"):
        print(f"  {m}: largest difference between the grids of the two paths "
              f"{np.abs(grid('M00', m) - grid('M00-forced-AVX2', m)).max():.1e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deposit", default=os.path.join(_REPO_ROOT, "zenodo_export"),
                        help="folder of the data deposit, version 1.1 or later (default: <repo>/zenodo_export)")
    deposit = parser.parse_args().deposit
    D = loadData(deposit)
    blockMain(D)
    blockComparisons(D)
    blockFamilies(D)
    blockFitCheck(D)
    blockFamilyShape(D)
    blockFamilyTables(D, deposit)
    blockMisranking(D)
    blockInvertedOrder(D)
    blockPairSeparation(D)
    blockP10AgainstMean(D)
    blockMonteCarlo(D)
    blockDecomposition(D)
    blockAblations(D)
    blockConcentration(D)
    blockReporting(D)
    blockTrialBudget(D)
    blockAppendices(D)
    blockSmallerSamples(D)
    blockIntroduction(D)
    blockFragile(D)


if __name__ == "__main__":
    main()
