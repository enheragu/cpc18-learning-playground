#!/usr/bin/env python3
# encoding: utf-8

"""
Numbers quoted in the text of the CPC18 variance manuscript that no other script
of the repository prints, computed from the public data deposit only (the output
of 16_export_zenodo.py, the files archived in Zenodo). Prints one block per
section of the article.

Conventions:

  - Errors are MSE x 100 unless a line says otherwise, and the reference machine
    is M00. Paired statistics use the seeds shared by the two columns.
  - P10 is the 10th percentile of a lognormal fitted to the trials (mean and
    ddof=0 std of the log-values, the closed form of lognorm.fit(x, floc=0)).
  - The half-width of a bootstrap interval is (hi - lo) / 2. Where the longer side
    max(hi - p, p - lo) is also printed, the line says so.
  - Misranking probabilities are given under both estimators, as the exact value
    that the seeded simulations of 02_analyze and 07_analyze_ablation converge to:
    Monte Carlo draws from a Gaussian fitted to the trials, Bootstrap resamples
    them. The reference order is the best run of each column. A tie is not a
    reversal: every comparison is strict.
  - Every simulation has a fixed seed. Values are ordered by seed before resampling,
    except in the error curve of the fitted target, which follows the figure of
    Section 3.1 and so keeps the row order of the deposit, the order of the logs.

Usage:
    cd src && python 18_text_numbers.py [--deposit PATH]
"""

import argparse
import itertools
import os
import types
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import trapezoid
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MASTER_SEED_FOR_POOL = 20260522
MASTER_SEED_ORDER = 99999777
BOOT_SEED = 20260522
FIGURE_RESAMPLES = 2000  # error curve of the fitted target, as in its figure
FAMILY_RESAMPLES = 20000  # same statistic on the other models
CROSSING_RESAMPLES = 2000
HL_RESAMPLES = 2000
P10_RESAMPLES = 5000
SMALL_SAMPLES, SMALL_RESAMPLES = 10000, 2000
Z10 = norm.ppf(0.10)
LR = {"0001": 1e-4, "000316": 3.16e-4, "001": 1e-3, "01": 1e-2}
FAIL_THRESHOLD = 1.5
TOST_MARGINS = (0.05, 0.10)

MODELS = ["LinearLogistic", "BourginMLP", "ContextDepNet", "SparseMLP", "PsychForest", "BEAST_GB"]
NEURAL = ["LinearLogistic", "BourginMLP", "ContextDepNet", "SparseMLP"]
LABELS = {"BEAST_GB": "BEAST-GB"}

# Arithmetic paths of the cross-machine study (machine identifiers of machines.csv)
PATHS = {"AVX512": ["M00", "M01", "M02"], "AVX2": ["M03", "M04", "M05", "M01-forced-AVX2"], "AMD": ["C02"],
         "DEFAULT": ["M01-forced-DEFAULT", "M03-forced-DEFAULT", "M04-forced-DEFAULT"], "ARM": ["M06"]}
COMPARISON = ["AVX2", "AMD", "DEFAULT", "ARM"]
CONDITIONS = {  # key: (experiment, condition, label)
    "B32": ("ablation_bourginmlp", "BourginMLP_B32_L0_001", "BourginMLP, B = 32, lr = 1e-3"),
    "B16mild": ("ablation_bourginmlp", "BourginMLP_B16_L0_0001", "BourginMLP, B = 16, lr = 1e-4"),
    "E20": ("ablation_sparsemlp", "SparseMLP_E20_L0_001", "SparseMLP, eps = 20, lr = 1e-3"),
    "E40": ("ablation_sparsemlp", "SparseMLP_E40_L0_0001", "SparseMLP, eps = 40, lr = 1e-4"),
    "ContextDepNet": ("main", "ContextDepNet", "ContextDepNet"),
    "BEAST_GB": ("main", "BEAST_GB", "BEAST-GB"),
}


def label(model):
    return LABELS.get(model, model)


def percent(x, d=1):
    """Percentage rounded half-up, as printed in the article (74/800 = 9.25 % -> 9.3)."""
    return str(Decimal(repr(round(100 * float(x), 9))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def seedPool(master, size):
    return np.random.default_rng(master).integers(0, 2 ** 32, size=size, dtype=np.uint64).astype(np.int64)


def p10Lognormal(x):
    lx = np.log(x)
    return float(np.exp(lx.mean() + Z10 * lx.std()))


def p10Gaussian(x):
    return float(x.mean() + Z10 * x.std(ddof=1))


def p10Draws(values, idx):
    logs = np.log(values[idx])
    return np.exp(logs.mean(axis=1) + Z10 * logs.std(axis=1))


def bootstrapSides(values, n, seed):
    """Half-width and longer side of the bootstrap 95 % CI of P10, in % of the estimate on these values."""
    idx = np.random.default_rng(seed).integers(0, len(values), size=(n, len(values)))
    return intervalWidths(p10Lognormal(values), p10Draws(values, idx))


def intervalWidths(estimate, draws):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return 100 * (hi - lo) / 2 / estimate, 100 * max(hi - estimate, estimate - lo) / estimate


def iccExact(M):
    """ICC(3,1) with its exact 95 % CI from F = MSr / MSe (Shrout and Fleiss, 1979)."""
    n, k = M.shape
    gm = M.mean()
    msr = k * ((M.mean(1) - gm) ** 2).sum() / (n - 1)
    mse = ((M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    df_r, df_e = n - 1, (n - 1) * (k - 1)
    f_low = (msr / mse) / stats.f.ppf(0.975, df_r, df_e)
    f_high = (msr / mse) * stats.f.ppf(0.975, df_e, df_r)
    return (msr - mse) / (msr + (k - 1) * mse), (f_low - 1) / (f_low + k - 1), (f_high - 1) / (f_high + k - 1)


def twoWayAnova(df, a, b, y="mse_x100"):
    """Balanced two-way ANOVA with sum-to-zero contrasts: per term SS, % of the total SS, F, p and partial eta^2."""
    cells = df.groupby([a, b])[y]
    counts = cells.size()
    assert counts.nunique() == 1, "unbalanced grid"
    n = counts.iloc[0]
    grand = df[y].mean()
    ma, mb, mab = df.groupby(a)[y].mean(), df.groupby(b)[y].mean(), cells.mean()
    inter = np.array([mab[(i, j)] - ma[i] - mb[j] + grand for i, j in mab.index])
    ss = {"A": n * len(mb) * ((ma - grand) ** 2).sum(), "B": n * len(ma) * ((mb - grand) ** 2).sum(),
          "AB": n * (inter ** 2).sum(), "R": ((df[y] - cells.transform("mean")) ** 2).sum()}
    dof = {"A": len(ma) - 1, "B": len(mb) - 1, "AB": (len(ma) - 1) * (len(mb) - 1), "R": len(ma) * len(mb) * (n - 1)}
    total = sum(ss.values())
    out = {"R": dict(ss=ss["R"], pct=100 * ss["R"] / total)}
    for term in ("A", "B", "AB"):
        F = (ss[term] / dof[term]) / (ss["R"] / dof["R"])
        out[term] = dict(ss=ss[term], pct=100 * ss[term] / total, F=F, p=stats.f.sf(F, dof[term], dof["R"]),
                         eta2p=ss[term] / (ss[term] + ss["R"]))
    return out


def ablationFrame(df, prefix):
    d = df.copy()
    d["f1"] = d.condition.str.extract(prefix + r"(\d+)_")[0].astype(int)
    d["lr"] = d.condition.str.extract(r"_L0_(\d+)$")[0].map(LR)
    return d


def seedGrid(frame):
    return frame.pivot_table(index="seed", columns=["f1", "lr"], values="mse_x100")


def cellValues(frame, f1, lr):
    return frame[(frame.f1 == f1) & (frame.lr == lr)].sort_values("seed").mse_x100.to_numpy()


def cellName(cell):
    return f"({cell[0]:g}, {cell[1]:g})"


def closestPair(frame):
    means = frame.groupby(["f1", "lr"]).mse_x100.mean()
    return min(itertools.combinations(means.index, 2), key=lambda p: abs(means[p[0]] - means[p[1]]))


def bestRunOrder(a, b):
    assert a.min() != b.min(), "tied best runs define no reference order"
    return (a, b) if a.min() < b.min() else (b, a)


def swapBootstrap(a, b, n):
    """Exact Bootstrap P(best of n of the best-run leader is strictly worse than best of n of the other)."""
    leader, other = bestRunOrder(a, b)
    other = np.sort(other)
    m = len(other)
    i = np.arange(m)
    pmf_of_min = ((m - i) / m) ** n - ((m - i - 1) / m) ** n
    above = 1 - np.searchsorted(np.sort(leader), other, side="right") / len(leader)
    return float((pmf_of_min * above ** n).sum())


def swapMonteCarlo(a, b, n):
    """The same probability under Gaussians fitted to the trials (mean, ddof=0 std), by numerical integration."""
    leader, other = bestRunOrder(a, b)
    ml, sl, mo, so = leader.mean(), leader.std(), other.mean(), other.std()
    x = np.linspace(min(ml - 12 * sl, mo - 12 * so), max(ml + 12 * sl, mo + 12 * so), 400001)
    density_of_min = n * norm.pdf(x, mo, so) * norm.sf(x, mo, so) ** (n - 1)
    return float(trapezoid(density_of_min * norm.sf(x, ml, sl) ** n, x))


def pairedCi(d, level):
    half = stats.t.ppf(0.5 + level / 2, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean() - half, d.mean() + half


def hodgesLehmann(d, walsh):
    return float(np.median((d[walsh[0]] + d[walsh[1]]) / 2))


def hodgesLehmannCi(d, walsh, seed):
    rng = np.random.default_rng(seed)
    draws = [hodgesLehmann(d[rng.integers(0, len(d), len(d))], walsh) for _ in range(HL_RESAMPLES)]
    return np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def seedPairInversions(x, y):
    """Share of seed pairs in strictly opposite order, over all pairs and over the closest 10 % of pairs of x."""
    i, j = np.triu_indices(len(x), 1)
    dx, dy = x[i] - x[j], y[i] - y[j]
    inverted = dx * dy < 0
    close = np.abs(dx) <= np.percentile(np.abs(dx), 10)
    return float(inverted.mean()), float(inverted[close].mean())


def seedMachineInteraction(x, y):
    """Share (%) of the total SS of the seed x machine table that is not seed nor machine main effect."""
    M = np.column_stack([x, y])
    gm = M.mean()
    total = ((M - gm) ** 2).sum()
    return 100 * (total - 2 * ((M.mean(1) - gm) ** 2).sum() - len(x) * ((M.mean(0) - gm) ** 2).sum()) / total


def componentsTotal(grid):
    """Random-effects variance components of a grid (expected mean squares, as 13_variance_components)."""
    a, b = grid.shape
    gm = grid.mean()
    ssa = b * ((grid.mean(axis=1) - gm) ** 2).sum()
    ssb = a * ((grid.mean(axis=0) - gm) ** 2).sum()
    ssab = ((grid - gm) ** 2).sum() - ssa - ssb
    msab = ssab / ((a - 1) * (b - 1))
    return max(0.0, (ssa / (a - 1) - msab) / b), max(0.0, (ssb / (b - 1) - msab) / a), msab


def featuresPath(deposit):
    for path in (os.path.join(deposit, "cpc18_features.csv"), os.path.join(deposit, "features", "cpc18_features.csv")):
        if os.path.exists(path):
            return path
    raise SystemExit("cpc18_features.csv not found in the deposit")


def loadDeposit(deposit):
    def read(path, **options):
        return pd.read_csv(path, float_precision="round_trip", low_memory=False, **options)

    d = types.SimpleNamespace()
    d.trials = read(os.path.join(deposit, "trials.csv"))
    if "protocol" in d.trials:  # deposit 1.1 and later: these numbers are those of the original protocol
        d.trials = d.trials[d.trials.protocol == "original"]
    d.perproblem = read(os.path.join(deposit, "perproblem_errors.csv"))
    d.split = read(os.path.join(deposit, "split_variance.csv"))
    d.features = read(featuresPath(deposit))
    d.probes = read(os.path.join(deposit, "probes.csv"), dtype=str)
    d.machines = read(os.path.join(deposit, "machines.csv"), dtype=str)
    for frame in (d.perproblem, d.split):
        frame["model"] = frame.model.str.replace("BEAST-GB", "BEAST_GB")

    d.ref = d.trials[d.trials.machine_id == "M00"]
    d.main = d.ref[d.ref.experiment == "main"]
    d.P = d.main.pivot(index="seed", columns="condition", values="mse_x100")[MODELS]
    d.V = {m: d.P[m].to_numpy() for m in MODELS}
    b1 = d.ref[d.ref.experiment == "ablation_bourginmlp"]
    d.A1all = ablationFrame(b1, "BourginMLP_B")
    d.A1 = d.A1all[d.A1all.lr != 3.16e-4]
    d.A3 = ablationFrame(d.ref[d.ref.experiment == "ablation_sparsemlp"], "SparseMLP_E")
    d.fac = d.ref[d.ref.experiment == "factorial_init_order"]
    d.A1amd = ablationFrame(d.trials[(d.trials.machine_id == "C02") & (d.trials.experiment == "ablation_bourginmlp")],
                            "BourginMLP_B")
    d.A3amd = ablationFrame(d.trials[(d.trials.machine_id == "C02") & (d.trials.experiment == "ablation_sparsemlp")],
                            "SparseMLP_E")
    return d


def sectionIntroduction(d):
    print("\n== Section 1: the variance phenomenon, BourginMLP main run ==")
    x = d.V["BourginMLP"]
    epochs = d.main[d.main.condition == "BourginMLP"].best_epoch
    print(f"  epoch of the best test MSE, 0-based: from {epochs.min():.0f} to {epochs.max():.0f}, "
          f"{epochs.max() / max(epochs.min(), 1):.0f} times (more than an order of magnitude)")
    print(f"  runs above MSE x 100 = 1.8, counted but not drawn in the six-model figure: {int((x > 1.8).sum())} of "
          f"{len(x)}, up to {x.max():.1f}")
    pf = d.V["PsychForest"]
    print(f"  overlap figure: PsychForest runs span {pf.min():.3f} to {pf.max():.3f}, inside the BourginMLP runs below their "
          f"median: best run {x.min():.3f}, median {np.median(x):.3f}")


def sectionMethodology(d):
    print("\n== Section 2: data, protocol and cost ==")
    f = d.features
    train, test = f[f.Train_Test == "Train"], f[f.Train_Test == "Test"]
    inputs = f.shape[1] - 4 - 2 + f.LotShapeA.nunique() + f.LotShapeB.nunique()
    print(f"  training rows: {len(train)} ({train.problem_id.nunique()} problems x 5 blocks); "
          f"test rows: {len(test)} ({test.problem_id.nunique()} problems x 5 blocks)")
    print(f"  problems excluded from the 210 of the canonical training pool: "
          f"{sorted(set(range(1, 211)) - set(train.problem_id))}")
    print(f"  feature file: {f.shape[0]} rows x {f.shape[1]} columns; model inputs: {inputs} "
          f"(without 4 identifier and target columns, 2 shape columns one-hot encoded)")
    p1 = f[(f.problem_id == 1) & (f.Train_Test == "Train")].set_index("block")
    print(f"  worked example, problem 1: A pays {p1.Ha[1]:g} with probability {p1.pHa[1]:g}; B pays {p1.Hb[1]:g} with "
          f"probability {p1.pHb[1]:g} and {p1.Lb[1]:g} otherwise; B rate {percent(p1.B_rate[1], 0)} % in block 1 and "
          f"{percent(p1.B_rate[5], 0)} % in block 5")

    nn = d.ref[d.ref.model.isin(NEURAL) & d.ref.experiment.isin(
        ["main", "ablation_bourginmlp", "ablation_sparsemlp", "factorial_init_order"])]
    capped = nn[nn.n_epochs != nn.best_epoch + nn.patience + 1]
    longest = nn.groupby(nn.experiment.str.startswith("ablation")).n_epochs.max()
    print(f"  epoch cap: reached in {len(capped)} of {len(nn)} neural trials ({sorted(capped.model.unique())}); "
          f"still improving within the last 50 epochs: {int((capped.best_epoch >= capped.n_epochs - 50).sum())}; "
          f"longest run {longest[False]:.0f} epochs (main, factorial), {longest[True]:.0f} (ablations)")

    sizes = d.ref.groupby("experiment").size()
    grid = d.A1all[d.A1all.lr == 3.16e-4]
    print(f"  trials on the reference machine: main {sizes['main']} (6 x {len(d.V['BourginMLP'])}); Ablation BourginMLP "
          f"{len(d.A1)} ({d.A1.condition.nunique()} x {d.A1.groupby('condition').size().iloc[0]}); grid-sensitivity "
          f"column {len(grid)} ({grid.condition.nunique()} x {grid.groupby('condition').size().iloc[0]}); Ablation "
          f"SparseMLP {len(d.A3)} ({d.A3.condition.nunique()} x {d.A3.groupby('condition').size().iloc[0]}); factorial "
          f"{len(d.fac)} (4 x {d.fac.groupby('model').size().iloc[0]})")

    pool = seedPool(MASTER_SEED_FOR_POOL, 800)
    orders = seedPool(MASTER_SEED_ORDER, 35)
    cells = pd.concat([d.A1all, d.A3]).groupby("condition").seed.apply(set)
    print(f"  shared pool regenerated from {MASTER_SEED_FOR_POOL}: equal to the seeds of the main run "
          f"{set(d.P.index) == set(pool.tolist())} and of every ablation cell "
          f"{all(s == set(pool.tolist()) for s in cells)}; factorial init seeds equal to its first 35 "
          f"{all(set(g.init_seed.astype(np.int64)) == set(pool[:35].tolist()) for _, g in d.fac.groupby('model'))}; "
          f"factorial order seeds equal to the 35 of {MASTER_SEED_ORDER} "
          f"{all(set(g.order_seed.astype(np.int64)) == set(orders.tolist()) for _, g in d.fac.groupby('model'))}")

    hours = d.ref[d.ref.experiment != "validation_split"].train_duration_s.sum() / 3600
    with_split = d.ref.train_duration_s.sum() / 3600
    other = d.trials[d.trials.machine_id != "M00"].train_duration_s.sum() / 3600
    print(f"  single-core compute: {hours:.1f} h on the reference machine (main, ablations, factorial), {hours / 6:.1f} h "
          f"of wall-clock on six workers; {with_split:.1f} h and {with_split / 6:.1f} h with the validation split; "
          f"{other:.1f} h on the other machines")


def sectionInformedTraining(d):
    print("\n== Section 3.1: the fitted target, main run ==")
    print("  share of the trials at or below the fitted P10 (nominal 10 %) and expected trials to reach it, 1 / share:")
    shares = {}
    for m in MODELS:
        x = d.V[m]
        shares[m] = np.mean(x <= p10Lognormal(x))
        print(f"    {label(m):15s} Gaussian {percent(np.mean(x <= p10Gaussian(x))):>5} %   lognormal "
              f"{percent(shares[m]):>5} %   Bootstrap trials {1 / shares[m]:6.1f}")
    five = [m for m in MODELS if m != "BourginMLP"]
    x = d.V["BourginMLP"]
    print(f"  five models that validate: Bootstrap {min(1 / shares[m] for m in five):.1f} to "
          f"{max(1 / shares[m] for m in five):.1f} trials, lognormal share {percent(min(shares[m] for m in five))} to "
          f"{percent(max(shares[m] for m in five))} %")
    print(f"  BourginMLP (also Appendix A): {int((x <= p10Lognormal(x)).sum())} of {len(x)} trials at or below "
          f"{p10Lognormal(x):.3f}, {100 * shares['BourginMLP']:.2f} %; Bootstrap {1 / shares['BourginMLP']:.0f} trials "
          f"against 10 from Monte Carlo, a factor of {0.1 / shares['BourginMLP']:.0f}")

    fits = {"empirical": lambda X: np.percentile(X, 10, axis=1),
            "Gaussian": lambda X: X.mean(axis=1) + Z10 * X.std(axis=1, ddof=1),
            "lognormal": lambda X: np.exp(np.log(X).mean(axis=1) + Z10 * np.log(X).std(axis=1))}
    print(f"  ContextDepNet, error (%) of the estimated target against the empirical P10 of the {len(x)} trials, "
          f"{FIGURE_RESAMPLES} resamples per point, row order of the deposit, as the figure: RMSE (mean signed error)")
    sizes = (5, 10, 20, 50, 100)
    print("    " + " " * 10 + "".join(f"{'N = ' + str(n):>16s}" for n in sizes))
    values = d.main[d.main.condition == "ContextDepNet"].mse_x100.to_numpy()
    truth = np.percentile(values, 10)
    rng = np.random.default_rng(BOOT_SEED)
    for name, fit in fits.items():
        cells = []
        for n in sizes:
            error = 100 * (fit(values[rng.integers(0, len(values), size=(FIGURE_RESAMPLES, n))]) / truth - 1)
            cells.append(f"{np.sqrt((error ** 2).mean()):7.2f} ({error.mean():+5.2f})")
        print(f"    {name:10s}" + "".join(f"{c:>16s}" for c in cells))

    print(f"  other models whose fit validates, RMSE (%) of the target, {FAMILY_RESAMPLES} resamples "
          "(empirical, Gaussian, lognormal):")
    rng = np.random.default_rng(BOOT_SEED)
    gap, fits_ahead = -np.inf, True
    for m in ("LinearLogistic", "SparseMLP", "PsychForest", "BEAST_GB"):
        cells = []
        for n in (5, 50):
            X = d.V[m][rng.integers(0, 800, size=(FAMILY_RESAMPLES, n))]
            rmse = {k: np.sqrt(((100 * (fit(X) / np.percentile(d.V[m], 10) - 1)) ** 2).mean()) for k, fit in fits.items()}
            gap = max(gap, rmse["lognormal"] - rmse["Gaussian"])
            fits_ahead &= rmse["Gaussian"] < rmse["empirical"] and rmse["lognormal"] < rmse["empirical"]
            cells.append(f"N = {n}: {rmse['empirical']:.2f}, {rmse['Gaussian']:.2f}, {rmse['lognormal']:.2f}")
        print(f"    {label(m):15s} " + "   ".join(cells))
    print(f"    both fits ahead of the empirical percentile: {fits_ahead}; largest lognormal - Gaussian: {gap:+.3f} points")


def bestOfNWeights(m, sizes):
    i = np.arange(m)
    return ((m - i) / m) ** sizes[:, None] - ((m - i - 1) / m) ** sizes[:, None]


def crossingOf(a, b, weights, sizes):
    """Smallest N at which the expected best of N of a is strictly below that of b, None if there is none."""
    below = np.nonzero(weights @ np.sort(a) < weights @ np.sort(b))[0]
    return int(sizes[below[0]]) if below.size else None


def sectionComparison(d):
    print("\n== Section 3.2: replication comparison, main run ==")
    V = d.V
    sizes = np.arange(1, 201)
    weights = bestOfNWeights(800, sizes)
    rng = np.random.default_rng(BOOT_SEED)
    crossing = crossingOf(V["BourginMLP"], V["PsychForest"], weights, sizes)
    draws = [crossingOf(rng.choice(V["BourginMLP"], 800), rng.choice(V["PsychForest"], 800), weights, sizes)
             for _ in range(CROSSING_RESAMPLES)]
    found = [c for c in draws if c is not None]
    lo, hi = np.percentile(found, [2.5, 97.5])
    print(f"  best-of-N crossing, BourginMLP overtakes PsychForest in the exact E[min of N trials] at N = {crossing}, "
          f"bootstrap 95 % CI [{lo:.0f}, {hi:.0f}] ({CROSSING_RESAMPLES} resamples, "
          f"{len(draws) - len(found)} without a crossing up to N = {sizes[-1]})")
    worst_gb, best_rest = V["BEAST_GB"].max(), min(V[m].min() for m in MODELS if m != "BEAST_GB")
    print(f"  BEAST-GB never overtaken: its worst run {worst_gb:.3f} is below the best run of any other model "
          f"{best_rest:.3f}")

    mu = {m: V[m].mean() for m in MODELS}
    sd = {m: V[m].std(ddof=1) for m in MODELS}
    cluster = ("ContextDepNet", "BourginMLP", "SparseMLP")
    print(f"  neural cluster: mean gap ContextDepNet - BourginMLP {abs(mu['ContextDepNet'] - mu['BourginMLP']):.3f}, "
          f"standard error of the BourginMLP mean at N = 800 {sd['BourginMLP'] / np.sqrt(800):.3f}; width of the three "
          f"means {max(mu[m] for m in cluster) - min(mu[m] for m in cluster):.3f}, smallest trial std of the three "
          f"{min(sd[m] for m in cluster):.3f}")

    print("  TOST on the paired differences, first - second, 800 seeds:")
    for a, b in (("BourginMLP", "ContextDepNet"), ("ContextDepNet", "SparseMLP"), ("BourginMLP", "SparseMLP")):
        diff = V[a] - V[b]
        ci90, ci95 = pairedCi(diff, 0.90), pairedCi(diff, 0.95)
        bound = max(abs(ci90[0]), abs(ci90[1]))
        print(f"    {a}-{b}: mean {diff.mean():+.3f}; 90 % CI [{ci90[0]:+.3f}, {ci90[1]:+.3f}]; 95 % CI "
              f"[{ci95[0]:+.3f}, {ci95[1]:+.3f}]; equivalent at delta = 0.05 {bound < 0.05} (bound - margin "
              f"{bound - 0.05:+.3f}), at delta = 0.01 {bound < 0.01}")

    pairs = list(itertools.combinations(MODELS, 2))
    tests = [stats.spearmanr(V[a], V[b]) for a, b in pairs]
    p = np.array([t.pvalue for t in tests])
    rejected, corrected = multipletests(p, alpha=0.05, method="holm")[:2]
    top = int(np.argmax([t.correlation for t in tests]))
    print(f"  Spearman over the 15 pairs: {min(t.correlation for t in tests):+.3f} to "
          f"{max(t.correlation for t in tests):+.3f}; largest {pairs[top][0]}-{pairs[top][1]} "
          f"{tests[top].correlation:+.3f}, explains {100 * tests[top].correlation ** 2:.2f} % of rank variance, "
          f"uncorrected p = {p[top]:.3f}; smallest Holm-corrected p = {corrected.min():.3f}, "
          f"significant at 0.05 after Holm: {int(rejected.sum())}")


def factorialGrid(fac, model):
    return fac[fac.model == model].pivot_table(index="init_seed", columns="order_seed", values="mse_x100")


def sectionDecomposition(d):
    print("\n== Section 3.4.1: init x order factorial ==")
    grids = {m: factorialGrid(d.fac, m) for m in NEURAL}
    bm = grids["BourginMLP"].to_numpy()
    spread = bm.std(axis=0, ddof=1)
    print(f"  BourginMLP init-only std over the 35 fixed order seeds: from {spread.min():.2f} to {spread.max():.2f}")
    print("  total variance of the components, squared MSE x 100 (init + order + interaction):")
    totals = {m: sum(componentsTotal(g.to_numpy())) for m, g in grids.items()}
    for m in NEURAL:
        print(f"    {m:15s} {totals[m]:.3f}")
    print(f"    BourginMLP interaction component alone {componentsTotal(bm)[2]:.3f}")


def greedyEnds(grid, levels1, levels2):
    """One-pass coordinate descent from each of the six starting levels, per seed: (axis, start, end1, end2)."""
    cube = np.stack([[grid[(a, b)].to_numpy() for b in levels2] for a in levels1])
    ends = []
    for s in range(cube.shape[2]):
        g = cube[:, :, s]
        for start in range(len(levels1)):
            j = int(np.argmin(g[start, :]))
            ends.append(("f1", start, int(np.argmin(g[:, j])), j))
        for start in range(len(levels2)):
            i = int(np.argmin(g[:, start]))
            ends.append(("f2", start, i, int(np.argmin(g[i, :]))))
    return ends


def tiedMinima(grid):
    """Seeds with a tie at the minimum of the grid, or of a row or a column of it, where argmin takes the first."""
    cube = np.stack([grid[c].to_numpy() for c in grid.columns]).reshape(3, 3, -1)
    def tied(values, axis):
        return (values == values.min(axis=axis, keepdims=True)).sum(axis=axis) > 1
    return int((tied(cube, 0).any(axis=0) | tied(cube, 1).any(axis=0)).sum())


def sectionAblation(d):
    print("\n== Section 3.3: ablation studies ==")
    print("  -- Sensitivity to the grid, % of the total SS --")
    converged = d.A1all[d.A1all.lr < 5e-3]
    means = converged.groupby(["f1", "lr"]).mse_x100.mean()
    shares = twoWayAnova(converged, "f1", "lr")
    full = twoWayAnova(d.A1, "f1", "lr")
    dropped = twoWayAnova(d.A1[d.A1.lr != 1e-2], "f1", "lr")
    sparse = twoWayAnova(d.A3[d.A3.lr != 1e-2], "f1", "lr")
    print(f"  all-converged Ablation BourginMLP (lr 1e-4, 3.16e-4, 1e-3): cell means {means.min():.2f} to "
          f"{means.max():.2f}; learning rate {shares['B']['pct']:.1f}, batch size {shares['A']['pct']:.1f}, "
          f"interaction {shares['AB']['pct']:.1f}, residual {shares['R']['pct']:.1f} "
          f"(full grid: batch size {full['A']['pct']:.2f}, residual {full['R']['pct']:.1f})")
    print(f"  dropping the lr = 1e-2 column: batch size {dropped['A']['pct']:.1f}, interaction {dropped['AB']['pct']:.1f}, "
          f"residual {dropped['R']['pct']:.1f}")
    print("  cell mean at lr = 3.16e-4, by batch size: "
          + ", ".join(f"B = {b}: {means[b, 3.16e-4]:.2f}" for b in (16, 32, 64)))
    print(f"  Ablation SparseMLP without lr = 1e-2 (3 x 2): interaction {sparse['AB']['pct']:.1f}, epsilon "
          f"{sparse['A']['pct']:.1f}, learning rate {sparse['B']['pct']:.1f}")
    cells = d.A3.groupby(["f1", "lr"]).mse_x100.mean().sort_values()
    print(f"  catastrophic cell (eps = 40, lr = 1e-2): mean {cells.iloc[-1]:.1f}, {cells.iloc[-1] / cells.iloc[-2]:.1f} "
          f"times the next worst {cells.iloc[-2]:.3f}, {cells.iloc[-1] / cells.iloc[-3]:.1f} times the third worst")
    rest = d.A3[(d.A3.f1 != 40) | (d.A3.lr != 1e-2)].mse_x100
    print(f"  trials of the other eight cells: 1st percentile {np.percentile(rest, 1):.2f}, 99th {np.percentile(rest, 99):.2f}")

    print("  -- Misranking between the closest cells --")
    for name, frame in (("Ablation BourginMLP", d.A1), ("Ablation SparseMLP", d.A3)):
        p, q = closestPair(frame)
        a, b = cellValues(frame, *p), cellValues(frame, *q)
        print(f"  {name}, closest pair {cellName(p)} and {cellName(q)}: mean gap {abs(a.mean() - b.mean()):.3f}, std "
              f"{a.std(ddof=1):.3f} and {b.std(ddof=1):.3f}; exact P(swap) for N = 1 to 5:")
        print("    Monte Carlo " + " ".join(f"{swapMonteCarlo(a, b, n):.3f}" for n in range(1, 6)))
        print("    Bootstrap   " + " ".join(f"{swapBootstrap(a, b, n):.3f}" for n in range(1, 6)))
    c16, c32 = cellValues(d.A1, 16, 1e-3), cellValues(d.A1, 32, 1e-3)
    print(f"  B = 16 strictly below B = 32 at lr = 1e-3: {percent(np.mean(c16 < c32))} % of the shared seeds "
          "(Hodges-Lehmann estimate: last lines of Appendix D)")

    print("  -- Per-seed best cell and greedy search --")
    for name, frame in (("Ablation BourginMLP", d.A1), ("Ablation SparseMLP", d.A3)):
        grid = seedGrid(frame)
        winners = grid.idxmin(axis=1).value_counts(normalize=True)
        top = ", ".join(f"{cellName(c)} {percent(v)} %" for c, v in winners.iloc[:3].items())
        print(f"  {name}: share of the seeds won by each cell, top three: {top}; aggregate-mean winner "
              f"{cellName(grid.mean().idxmin())}")
        print(f"    seeds with a tie at a minimum of the grid, a row or a column: {tiedMinima(grid)}")
    levels1, levels2 = (10, 20, 40), (1e-4, 1e-3, 1e-2)
    ends = greedyEnds(seedGrid(d.A3), levels1, levels2)
    optimum = (1, 2)
    hit = np.mean([e[2:] == optimum for e in ends])
    print(f"  Ablation SparseMLP, greedy search: ends at the grand-mean optimum (eps = 20, lr = 1e-2) in "
          f"{percent(hit)} % of {len(ends)} searches (800 seeds x 6 starts), suboptimal in {percent(1 - hit)} %")
    for axis, levels in (("f1", levels1), ("f2", levels2)):
        for k, level in enumerate(levels):
            chosen = [e for e in ends if e[0] == axis and e[1] == k]
            finish = pd.Series([e[3] for e in chosen]).map(lambda j: levels2[j]).value_counts(normalize=True)
            print(f"    start {'eps' if axis == 'f1' else 'lr'} = {level:g}: reaches the optimum in "
                  f"{percent(np.mean([e[2:] == optimum for e in chosen]))} %; most often ends at lr = "
                  f"{finish.index[0]:g} ({percent(finish.iloc[0])} %)")
    means = d.A1.groupby(["f1", "lr"]).mse_x100.mean()
    batch = min((16, 32, 64), key=lambda b: means[b, 1e-2])
    rate = min((1e-4, 1e-3, 1e-2), key=lambda a: means[batch, a])
    print(f"  Ablation BourginMLP on the cell means, start inside the lr = 1e-2 column, batch size first: ends at "
          f"(B = {batch}, lr = {rate:g}); optimum {cellName(means.idxmin())}")

    print("  -- Cross-condition reliability, ICC(3,1) [exact 95 % CI] --")
    for name, frame in (("Ablation BourginMLP", d.A1), ("Ablation SparseMLP", d.A3)):
        grid = seedGrid(frame)
        icc, lo, hi = iccExact(grid[[c for c in grid.columns if c[1] != 1e-2]].to_numpy())
        print(f"  {name} without the lr = 1e-2 column: {icc:.3f} [{lo:.3f}, {hi:.3f}]")
    grid = seedGrid(d.A3)
    print(f"  Pearson r between lr = 1e-4 and lr = 1e-3 at eps = 10, Ablation SparseMLP: "
          f"{stats.pearsonr(grid[(10, 1e-4)], grid[(10, 1e-3)])[0]:.2f}")


def sectionVarianceSources(d):
    print("\n== Section 3.4.2: variance sources ==")
    print("  BourginMLP, per-problem squared error x 100 over the first 50 shared seeds:")
    W = d.perproblem[d.perproblem.model == "BourginMLP"].pivot(index="seed", columns="problem_id",
                                                               values="mean_sq_error_x100")
    variance = W.var(ddof=1).sort_values(ascending=False)
    aggregate = W.mean(axis=1)
    k = W.shape[1]
    attribution = W.apply(lambda c: np.cov(c, aggregate, ddof=1)[0, 1]) / k / aggregate.var(ddof=1)
    print("    share of the aggregate = covariance attribution, Cov(problem, aggregate) / (60 Var(aggregate)), which sums "
          "to 100 % over the 60 problems")
    for n in (5, 10):
        top = variance.index[:n]
        print(f"    top {n:2d} problems by across-seed variance: {100 * variance.iloc[:n].sum() / variance.sum():.1f} % of "
              f"the summed per-problem variance, {100 * attribution[top].sum():.1f} % of the variance of the aggregate MSE")
    print(f"    summed per-problem variance / variance of the sum over the 60 problems (3600 x variance of the aggregate): "
          f"{100 * variance.sum() / (k ** 2 * aggregate.var(ddof=1)):.1f} %; the rest is covariance between problems")

    test = d.features[(d.features.Train_Test == "Test") & (d.features.block == 1)].set_index("problem_id")
    top, bottom = variance.index[:10], variance.index[-10:]
    ambiguous = test.loc[variance.index, "Amb"] == 1
    print(f"    ten most variable against ten most stable problems: median diffEV {test.loc[top, 'diffEV'].median():+.1f} "
          f"against {test.loc[bottom, 'diffEV'].median():+.1f}; mean pHa {test.loc[top, 'pHa'].mean():.1f} against "
          f"{test.loc[bottom, 'pHa'].mean():.1f}")
    print(f"    mean across-seed variance of the {int(ambiguous.sum())} ambiguous problems / of the "
          f"{int((~ambiguous).sum())} unambiguous ones: {variance[ambiguous].mean() / variance[~ambiguous].mean():.1f} "
          "times")

    sigma = d.split.groupby("model").mse_x100.std(ddof=1)
    runs = d.split.groupby("model").size()
    print(f"  split variance: {d.features.problem_id.nunique()} problems in the feature set, {runs.iloc[0]} random splits "
          f"per model; sigma_split, std of MSE x 100 (ddof = 1): BEAST-GB {sigma['BEAST_GB']:.2f}, PsychForest "
          f"{sigma['PsychForest']:.2f}, LinearLogistic {sigma['LinearLogistic']:.2f}")


def sectionSuggestions():
    print("\n== Section 4: arithmetic of the recommendations ==")
    print(f"  chance that at least one of k trials reaches the model's own P10, 1 - 0.9^k: k = 6: "
          f"{percent(1 - 0.9 ** 6, 0)} %, k = 7: {percent(1 - 0.9 ** 7, 0)} %")
    print(f"  fall of the standard error of the mean, 1 - 1 / sqrt(N): N = 3: {percent(1 - 1 / np.sqrt(3), 0)} %, "
          f"N = 5: {percent(1 - 1 / np.sqrt(5), 0)} %; sqrt(5) = {np.sqrt(5):.2f}")


def sectionDiscussion(d):
    print("\n== Section 5: discussion ==")
    z = norm.ppf(0.975) + norm.ppf(0.80)
    print(f"  -- Section 5.1, trials per model to defend an improvement, 2 sigma^2 z^2 / delta^2 with z = {z:.2f} --")
    for name, sigma in (("BEAST-GB", 0.007), ("PsychForest", 0.013), ("ContextDepNet / SparseMLP", 0.15),
                        ("BourginMLP", 0.48)):
        cells = ", ".join(f"delta = {delta:g}: {int(np.ceil(2 * sigma ** 2 * z ** 2 / delta ** 2))}"
                          for delta in (0.01, 0.05, 0.10))
        print(f"    {name} (sigma = {sigma:g}): {cells}")
    print("    trial std of the main run behind the representative sigma: "
          + ", ".join(f"{label(m)} {d.V[m].std(ddof=1):.3f}"
                      for m in ("BEAST_GB", "PsychForest", "ContextDepNet", "SparseMLP", "BourginMLP")))

    print("  -- Section 3.6 and Appendix A, failure rate of BourginMLP, main run --")
    x = d.V["BourginMLP"]
    print(f"    threshold {FAIL_THRESHOLD}: converged {percent(np.mean(x <= FAIL_THRESHOLD))} % of the trials, mean "
          f"{x[x <= FAIL_THRESHOLD].mean():.2f}; failed {percent(np.mean(x > FAIL_THRESHOLD))} %, mean "
          f"{x[x > FAIL_THRESHOLD].mean():.1f}")
    print(f"    failure rate at threshold 1.2: {percent(np.mean(x > 1.2))} %, at threshold 2.0: {percent(np.mean(x > 2.0))} %")
    trial = d.main[d.main.condition == "BourginMLP"]
    failed = trial.mse_x100 > FAIL_THRESHOLD
    print(f"    median best epoch: failed {trial[failed].best_epoch.median():.0f}, converged "
          f"{trial[~failed].best_epoch.median():.0f}; Pearson r between MSE and best epoch "
          f"{stats.pearsonr(trial.mse_x100, trial.best_epoch)[0]:.2f}")
    print(f"    chance of no failed trial at a 7 % failure rate: 0.93^10 = {0.93 ** 10:.2f}, 0.93^5 = {0.93 ** 5:.2f}")

    print("  -- Validation split, not reported in the article (early stopping on validation MSE, same trajectories) --")
    runs = d.ref[d.ref.experiment == "validation_split"]
    for m in ("BourginMLP", "ContextDepNet"):
        g = runs[runs.model == m]
        print(f"    {m}: {len(g)} seeds, {g.n_train_problems.iloc[0]:.0f}/{g.n_val_problems.iloc[0]:.0f} problems; std under "
              f"validation selection {g.mse_x100.std(ddof=1):.2f}, under test selection "
              f"{g.mse_x100_testsel.std(ddof=1):.2f}; test selection optimistic by "
              f"{(g.mse_x100 - g.mse_x100_testsel).mean():.2f}")


def appendixPool():
    print("\n== Pool alignment, not reported in the article ==")
    print(f"  first five seeds of the pool of master {MASTER_SEED_FOR_POOL}: {seedPool(MASTER_SEED_FOR_POOL, 5).tolist()}")


def pathValues(trials):
    """Per condition and arithmetic path: seed-indexed MSE x 100 and best epoch, and the seeds that differ within a path."""
    values, epochs, within = {}, {}, {}
    for key, (experiment, condition, _) in CONDITIONS.items():
        d = trials[(trials.experiment == experiment) & (trials.condition == condition)]
        values[key], epochs[key] = {}, {}
        for path, machines in PATHS.items():
            on_path = d[d.machine_id.isin(machines)]
            wide = on_path.pivot_table(index="seed", columns="machine_id", values="mse", aggfunc="first")
            for a, b in itertools.combinations(wide.columns, 2):
                both = wide[[a, b]].dropna()
                within[key, path] = within.get((key, path), 0) + int((both[a] != both[b]).sum())
            values[key][path] = wide.bfill(axis=1).iloc[:, 0] * 100
            if condition != "BEAST_GB":
                best = on_path.pivot_table(index="seed", columns="machine_id", values="best_epoch", aggfunc="first")
                epochs[key][path] = best.bfill(axis=1).iloc[:, 0]
    return values, epochs, within


def appendixCrossMachine(d):
    print("\n== Appendix D and Section 3.5: replication across machines ==")
    trials = d.trials
    print("  -- Design --")
    laboratory = [m for ms in PATHS.values() for m in ms if m not in ("M00", "C02")]
    layer = trials[trials.machine_id.isin(laboratory)]
    counts = layer.groupby(["machine_id", "condition"]).size()
    amd = trials[trials.machine_id == "C02"]
    grids = amd[amd.experiment.str.startswith("ablation")]
    cloud = d.probes[d.probes.machine_id.str.startswith("C")]
    machines = layer.machine_id.str.replace("-forced-.*", "", regex=True).nunique()
    print(f"  re-trained: {len(CONDITIONS)} conditions on {machines} machines and {sum('forced' in m for m in laboratory)} "
          f"forced configurations: {len(layer)} trials, {len(counts) * 800 - len(layer)} short of {len(counts) * 800}, at "
          f"most {800 - counts.min()} missing per condition")
    print(f"  probes: {len(d.probes)} collected, {len(cloud)} on cloud sessions; each runs "
          f"{', '.join(sorted(d.probes.steps.unique()))} optimizer steps; torch 2.10 on every probe "
          f"{d.probes.torch.str.startswith('2.10').all()} and machine {d.machines.torch.str.startswith('2.10').all()}")
    print(f"  AMD sweep: {len(amd)} trials over {amd.condition.nunique()} conditions, {len(grids)} of them over the "
          f"{grids.condition.nunique()} ablation cells, with {grids.groupby('condition').seed.nunique().min()} to "
          f"{grids.groupby('condition').seed.nunique().max()} unique seeds per cell")
    groups = d.probes.assign(init=d.probes.init_hash.str[:8], trained=d.probes.trained_hash.str[:8],
                             cloud=d.probes.machine_id.str.startswith("C"))
    print("  fingerprint groups (machine group, vendor, tier, initial hash, trained hash: probes):")
    for (machine, vendor, tier, init, trained), g in groups.groupby(
            [groups.machine_id.where(~groups.cloud, "cloud"), "vendor", "torch_tier", "init", "trained"]):
        print(f"    {machine}, {vendor}, {tier}, {init}, {trained}: {len(g)}")

    values, epochs, within = pathValues(trials)
    common = {k: sorted(set.intersection(*[set(s.index) for s in values[k].values()])) for k in values}

    def onPath(key, path):
        return values[key][path].loc[common[key]].to_numpy()

    print("  -- Bit identity and order across paths --")
    print(f"  seeds that differ between two machines of one path, over all conditions and shared seeds: "
          f"{sum(within.values())}; seeds on all five paths per condition: {sorted({len(c) for c in common.values()})}")
    print("  over the 10 path pairs: bit-identical seeds, % (lowest, highest); largest relative difference; smallest Spearman; "
          "largest share of seed pairs strictly inverted, %")
    for key in CONDITIONS:
        identical, relative, rho, inverted = [], [], [], []
        for a, b in itertools.combinations(PATHS, 2):
            x, y = onPath(key, a), onPath(key, b)
            identical.append(np.mean(x == y))
            relative.append(np.max(np.abs(x - y) / np.abs(x)))
            rho.append(stats.spearmanr(x, y)[0])
            inverted.append(seedPairInversions(x, y)[0])
        print(f"    {CONDITIONS[key][2]:30s} {100 * min(identical):6.2f} {100 * max(identical):6.2f}  {max(relative):.1e}  "
              f"{min(rho):.4f}  {100 * max(inverted):.4f}")

    print("  -- Selected epoch --")
    keys = ("B32", "B16mild", "E20", "E40", "ContextDepNet")
    print("  median best epoch on the reference path: "
          + ", ".join(f"{k} {epochs[k]['AVX512'].median():.0f}" for k in keys))
    same = {k: all((epochs[k][a].loc[common[k]] == epochs[k][b].loc[common[k]]).all()
                   for a, b in itertools.combinations(PATHS, 2)) for k in keys}
    print("  selected epoch identical on all seeds and path pairs: " + ", ".join(f"{k} {same[k]}" for k in keys))
    for k in ("B32", "B16mild"):
        moved, carried = [], []
        for path in COMPARISON:
            change = (epochs[k]["AVX512"].loc[common[k]] != epochs[k][path].loc[common[k]]).to_numpy()
            squared = (onPath(k, "AVX512") - onPath(k, path)) ** 2
            moved.append(np.mean(change))
            carried.append(squared[change].sum() / squared.sum())
        print(f"  {CONDITIONS[k][2]}, against the reference path: epoch changes on {percent(np.mean(moved), 0)} % of the "
              f"seeds on average; those seeds carry {percent(min(carried), 0)} to {percent(max(carried), 0)} % of the "
              "total squared difference")

    print("  -- BourginMLP, main condition: change of the 90th and 95th percentiles of the MSE against the reference path, % --")
    reference = onPath("B32", "AVX512")
    for path in COMPARISON:
        x = onPath("B32", path)
        print(f"  {path}: " + ", ".join(f"{100 * (np.percentile(x, q) / np.percentile(reference, q) - 1):+.1f}" for q in (90, 95)))

    print("  -- BourginMLP per seed, against the reference path (differences = path - reference) --")
    print("  columns: std of the differences; Spearman; seed x machine interaction, % of the SS; closest 10 % of seed pairs")
    print("  strictly inverted, %; mean shift, also as % of the reference mean and std; Wilcoxon p; paired t p; 90 % CI of the")
    print("  mean shift; TOST at delta = 0.05 and at delta = 0.10; largest |limit| of that CI minus 0.05")
    spread, close = {}, []
    for key in ("B32", "B16mild"):
        r = onPath(key, "AVX512")
        spread[key] = r.std(ddof=1)
        print(f"  {CONDITIONS[key][2]}: reference mean {r.mean():.3f}, std {r.std(ddof=1):.3f}")
        shifts = []
        for path in COMPARISON:
            x = onPath(key, path)
            diff = x - r
            ci = pairedCi(diff, 0.90)
            bound = max(abs(ci[0]), abs(ci[1]))
            closest = seedPairInversions(r, x)[1]
            close.append(closest)
            shifts.append((abs(diff.mean()), 100 * abs(diff.mean()) / r.mean(), bound))
            print(f"    {path:8s} {diff.std(ddof=1):.3f} {stats.spearmanr(r, x)[0]:.3f} {seedMachineInteraction(r, x):4.1f} "
                  f"{100 * closest:4.1f} {diff.mean():+.3f} {100 * diff.mean() / r.mean():+.1f} "
                  f"{100 * diff.mean() / r.std(ddof=1):+.1f} {stats.wilcoxon(x, r).pvalue:.3f} "
                  f"{stats.ttest_rel(x, r).pvalue:.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] "
                  + " ".join(f"{str(bound < m):5s}" for m in TOST_MARGINS) + f" {bound - 0.05:+.4f}")
        print(f"    largest |mean shift| {max(v[0] for v in shifts):.3f} ({max(v[1] for v in shifts):.1f} % of the mean); "
              f"largest |limit| of the 90 % CI {max(v[2] for v in shifts):.3f}")
    print(f"  closest 10 % of seed pairs inverted, mean over the four paths, B = 32: {100 * np.mean(close[:4]):.1f} %; "
          f"reference std of the B = 16 cell / of the B = 32 cell: {spread['B16mild'] / spread['B32']:.1f} times")

    print("  -- Distributional summary, relative to the reference path (%) --")
    r = onPath("B32", "AVX512")
    print("  BourginMLP, B = 32: |mean change| "
          + ", ".join(f"{p} {100 * abs(onPath('B32', p).mean() / r.mean() - 1):.1f}" for p in COMPARISON)
          + "; |std change| "
          + ", ".join(f"{p} {100 * abs(onPath('B32', p).std(ddof=1) / r.std(ddof=1) - 1):.1f}" for p in COMPARISON))
    for key in ("B32", "B16mild"):
        r = onPath(key, "AVX512")
        seeds = {p: int(values[key][p].loc[common[key]].idxmin()) for p in PATHS}
        repeated = {p: sum(s == seeds[p] for s in seeds.values()) for p in ("DEFAULT", "ARM")}
        print(f"  {CONDITIONS[key][2]}: best-run change "
              + ", ".join(f"{p} {100 * (onPath(key, p).min() / r.min() - 1):+.1f}" for p in COMPARISON))
        print(f"    {len(set(seeds.values()))} distinct best seeds on the five paths; paths with the best seed of DEFAULT: "
              f"{repeated['DEFAULT']}, of ARM: {repeated['ARM']} (1 = only itself)")
    r = onPath("B32", "AVX512")
    band = bootstrapSides(r, P10_RESAMPLES, BOOT_SEED)[0]
    outside = [p for p in COMPARISON if abs(100 * (onPath("B32", p).min() / r.min() - 1)) > band]
    print(f"  BourginMLP, B = 32: paths whose best run leaves the half-width of P10 on the reference ({band:.1f}): {outside}")

    print("  -- Resolution of P10 at small N (also Section 4), bootstrap 95 % CI on the reference path, % of the estimate --")
    print(f"  median over {SMALL_SAMPLES} subsamples, {SMALL_RESAMPLES} resamples each. Half-width = (hi - lo) / 2; longer side "
          "= max(hi - p, p - lo); excess = estimate above the P10 of the 800 trials")
    rng = np.random.default_rng(BOOT_SEED)
    for key in ("B32", "B16mild"):
        x = onPath(key, "AVX512")
        large, _ = bootstrapSides(x, P10_RESAMPLES, BOOT_SEED)
        print(f"  {CONDITIONS[key][2]}: half-width at N = {len(x)}: {large:.1f}")
        for n in (5, 10):
            half, longer, excess = [], [], []
            for _ in range(SMALL_SAMPLES):
                sample = rng.choice(x, n, replace=False)
                estimate = p10Lognormal(sample)
                h, side = intervalWidths(estimate, p10Draws(sample, rng.integers(0, n, size=(SMALL_RESAMPLES, n))))
                half.append(h)
                longer.append(side)
                excess.append(100 * (estimate / p10Lognormal(x) - 1))
            print(f"    N = {n:2d}: half-width {np.median(half):.1f}, {np.median(half) / large:.1f} times its value at "
                  f"N = {len(x)}; longer side {np.median(longer):.1f}; excess {np.mean(excess):.1f} in the mean, "
                  f"{np.median(excess):.1f} in the median")
    print("  paired bootstrap of each P10 shift, same resamples on both paths, half-width [95 % CI]:")
    widths = []
    for key in ("B32", "B16mild"):
        a = onPath(key, "AVX512")
        cells = []
        for path in COMPARISON:
            b = onPath(key, path)
            idx = np.random.default_rng(BOOT_SEED).integers(0, len(a), size=(P10_RESAMPLES, len(a)))
            lo, hi = np.percentile(100 * (p10Draws(b, idx) / p10Draws(a, idx) - 1), [2.5, 97.5])
            widths.append((hi - lo) / 2)
            cells.append(f"{path} {(hi - lo) / 2:.2f} [{lo:+.1f}, {hi:+.1f}]")
        print(f"    {CONDITIONS[key][2]}: " + ", ".join(cells))
    print(f"    half-widths from {min(widths):.2f} to {max(widths):.2f} %")

    print("  -- Threshold reading: share of the runs of a path at or below the P10 of the reference path (%) --")
    for key in CONDITIONS:
        target = p10Lognormal(onPath(key, "AVX512"))
        shares = {p: np.mean(onPath(key, p) <= target) for p in PATHS}
        print(f"  {CONDITIONS[key][2]}: " + ", ".join(f"{p} {percent(v, 2)}" for p, v in shares.items())
              + f"; mean {100 * np.mean(list(shares.values())):.2f}; spread "
                f"{100 * (max(shares.values()) - min(shares.values())):.2f} points")

    print("  -- Ablation findings on the AMD machine against the reference --")
    for name, ref_frame, amd_frame in (("Ablation BourginMLP", d.A1, d.A1amd), ("Ablation SparseMLP", d.A3, d.A3amd)):
        ref_aov, amd_aov = twoWayAnova(ref_frame, "f1", "lr"), twoWayAnova(amd_frame, "f1", "lr")
        p_ref, q_ref = closestPair(ref_frame)
        p_amd, q_amd = closestPair(amd_frame)
        a, b = cellValues(ref_frame, *p_ref), cellValues(ref_frame, *q_ref)
        a2, b2 = cellValues(amd_frame, *p_ref), cellValues(amd_frame, *q_ref)
        change = max(abs(ref_aov[t]["pct"] - amd_aov[t]["pct"]) for t in ref_aov)
        print(f"  {name}: largest |change| of a variance share {change:.2f} points; same closest pair "
              f"{set((p_ref, q_ref)) == set((p_amd, q_amd))}")
        print(f"    exact P(swap | N = 1), reference, AMD: Monte Carlo {swapMonteCarlo(a, b, 1):.3f}, "
              f"{swapMonteCarlo(a2, b2, 1):.3f}; Bootstrap {swapBootstrap(a, b, 1):.3f}, {swapBootstrap(a2, b2, 1):.3f}")
    print("  batch-size main effect of Ablation BourginMLP (SS, partial eta^2, F, p), reference, AMD:")
    for name, frame in (("reference", d.A1), ("AMD", d.A1amd)):
        term = twoWayAnova(frame, "f1", "lr")["A"]
        print(f"    {name}: SS {term['ss']:.1f}, {term['eta2p']:.4f}, F {term['F']:.2f}, p {term['p']:.1e}")
    print("  closest pair of Ablation BourginMLP, B = 16 - B = 32 at lr = 1e-3, on 800 shared seeds:")
    walsh = np.triu_indices(800)
    for name, frame in (("reference", d.A1), ("AMD", d.A1amd)):
        a, b = cellValues(frame, 16, 1e-3), cellValues(frame, 32, 1e-3)
        diff = a - b
        lo, hi = hodgesLehmannCi(diff, walsh, BOOT_SEED)
        largest = np.sort(diff)[-10:].sum() / len(diff)
        print(f"    {name}: means {a.mean():.3f}, {b.mean():.3f}; paired t p {stats.ttest_rel(a, b).pvalue:.2f}; share of "
              f"seeds with B = 16 strictly below {np.mean(a < b):.3f}; Hodges-Lehmann {hodgesLehmann(diff, walsh):.3f}, "
              f"bootstrap 95 % CI [{lo:.3f}, {hi:.3f}], {HL_RESAMPLES} resamples")
        print(f"    {name}: Wilcoxon p {stats.wilcoxon(a, b).pvalue:.1e}, median difference {np.median(diff):+.3f}; ten "
              f"largest differences / 800 = {largest:.3f}, mean gap {diff.mean():+.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deposit", default=os.path.join(_REPO_ROOT, "zenodo_export"),
                        help="folder of the data deposit (default: <repo>/zenodo_export)")
    d = loadDeposit(parser.parse_args().deposit)

    print("Errors are MSE x 100 unless a line says otherwise. Reference machine: M00. Other scripts print the rest of the")
    print("article: see the table of analysis scripts in src/README.md.")
    sectionIntroduction(d)
    sectionMethodology(d)
    sectionInformedTraining(d)
    sectionComparison(d)
    sectionDecomposition(d)
    sectionAblation(d)
    sectionVarianceSources(d)
    sectionSuggestions()
    sectionDiscussion(d)
    appendixPool()
    appendixCrossMachine(d)


if __name__ == "__main__":
    main()
