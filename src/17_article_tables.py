#!/usr/bin/env python3
# encoding: utf-8

"""
Tables of the CPC18 variance manuscript that no other script produces, computed
from the public data deposit only (the output of 16_export_zenodo.py, the files
archived in Zenodo). Prints each table as the LaTeX rows of the article:

  Section 3.4.2 training-seed against test-sampling variance
  Section 4     variance-aware reporting format (lognormal P10 and its check)
  Appendix C    headline statistics at smaller sample sizes
  Appendix D    per-trial training time, the cross-machine summary of
                BourginMLP per arithmetic path and the resolution floor of P10

P10 is always the 10th percentile of a lognormal fitted to the trials
(lognorm.fit(x, floc=0) is the MLE on the log-values: mean and ddof=0 std, so
it is computed in closed form), and "check" is the share of the trials at or
below it. Bootstrap CIs of P10 follow 05_make_paper_figures (5,000 resamples,
seed 20260522, values ordered by seed). sigma_test is the standard deviation
of the bootstrap distribution of the mean of the 60-problem profile in closed
form, the value the resampling of Section 3.4.2 converges to; the estimate with
2 x 10^4 resamples is printed next to it, and its Monte Carlo noise (about
1 %) can move the first decimal of a ratio.

With --protocol stable it prints instead the tables of the article in the
stable configuration (cpc18_utils/protocol.py), from a deposit of version 1.1
or later: Section 3.4.2, Section 4, Appendix C and the per-trial training time
of Appendix D, read as cpc18_utils/stable_results.py describes; and the
cross-machine tables of Appendix D, which stay in the fragile (original)
configuration. Each printed row is a row of the article.

Usage:
    cd src && python 17_article_tables.py [--deposit PATH] [--protocol original|stable]
"""

import argparse
import itertools
import os
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
from scipy.stats import norm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MASTER_SEED_FOR_POOL = 20260522
Z10 = norm.ppf(0.10)
P10_BOOT, P10_SEED = 5000, 20260522
TEST_BOOT, TEST_SEED = 20000, 20260522
LR = {"0001": 1e-4, "000316": 3.16e-4, "001": 1e-3, "01": 1e-2}

# Arithmetic paths of the cross-machine study (machine identifiers of machines.csv)
PATHS = {"AVX512": ["M00", "M01", "M02"], "AVX2": ["M03", "M04", "M05", "M01-forced-AVX2"], "AMD": ["C02"],
         "DEFAULT": ["M01-forced-DEFAULT", "M03-forced-DEFAULT", "M04-forced-DEFAULT"], "ARM": ["M06"]}
PATH_LABELS = {"AVX512": r"\texttt{AVX512}, Intel (reference + 2 machines)",
               "AVX2": r"\texttt{AVX2}, Intel (3 machines + 1 forced)",
               "AMD": r"\texttt{AVX2} label, AMD Zen 2 (cloud)",
               "DEFAULT": r"\texttt{DEFAULT}, x86 (3 forced)",
               "ARM": r"\texttt{DEFAULT}, ARM (Raspberry Pi 5)"}
XMACHINE_CONDITIONS = {  # key: (experiment, condition, label in the floor table)
    "B32": ("ablation_bourginmlp", "BourginMLP_B32_L0_001", r"BourginMLP, $B = 32$, $\alpha = 10^{-3}$"),
    "B16mild": ("ablation_bourginmlp", "BourginMLP_B16_L0_0001", r"BourginMLP, $B = 16$, $\alpha = 10^{-4}$"),
    "E20": ("ablation_sparsemlp", "SparseMLP_E20_L0_001", r"SparseMLP, $\varepsilon = 20$, $\alpha = 10^{-3}$"),
    "E40": ("ablation_sparsemlp", "SparseMLP_E40_L0_0001", r"SparseMLP, $\varepsilon = 40$, $\alpha = 10^{-4}$"),
    "ContextDepNet": ("main", "ContextDepNet", "ContextDepNet"),
    "BEAST_GB": ("main", "BEAST_GB", "BEAST-GB"),
}
MODEL_LABELS = {"BEAST_GB": "BEAST-GB", "PsychForest": "PsychForest", "LinearLogistic": "LinearLogistic",
                "ContextDepNet": "ContextDepNet", "BourginMLP": "BourginMLP", "SparseMLP": "SparseMLP"}
SOURCES = {"BEAST_GB": r"\citealp{plonsky2024beastgb}", "PsychForest": r"\citealp{plonsky2017psychforest}",
           "LinearLogistic": "this work", "ContextDepNet": r"\citealp{peterson2021science,thomas2024datasetbias}",
           "BourginMLP": r"\citealp{bourgin2019cognitive}", "SparseMLP": r"\citealp{mocanu2018set,bourgin2019cognitive}"}


def p10Lognormal(x):
    lx = np.log(x)
    return float(np.exp(lx.mean() + Z10 * lx.std()))


def p10BootCi(values, n=P10_BOOT, seed=P10_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    logs = np.log(values[idx])
    draws = np.exp(logs.mean(axis=1) + Z10 * logs.std(axis=1))
    return np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def percent(x, d=1):
    """Percentage rounded half-up, as printed in the article (74/800 = 9.25 % -> 9.3)."""
    return str(Decimal(repr(round(100 * float(x), 9))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def iccThreeOne(M):
    n, k = M.shape
    gm = M.mean()
    msr = k * ((M.mean(1) - gm) ** 2).sum() / (n - 1)
    mse = ((M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse)


def twoWayShares(df, a, b, y="mse_x100"):
    """Percentage of the total sum of squares of each term of a balanced two-way ANOVA (sum-to-zero contrasts)."""
    n = df.groupby([a, b]).size()
    assert n.nunique() == 1, "unbalanced grid"
    n = n.iloc[0]
    g = df[y].mean()
    ma, mb, mab = df.groupby(a)[y].mean(), df.groupby(b)[y].mean(), df.groupby([a, b])[y].mean()
    ssa = n * len(mb) * ((ma - g) ** 2).sum()
    ssb = n * len(ma) * ((mb - g) ** 2).sum()
    inter = np.array([mab[(i, j)] - ma[i] - mb[j] + g for i, j in mab.index])
    ssab = n * (inter ** 2).sum()
    ssr = ((df[y] - df.set_index([a, b]).index.map(mab).to_numpy()) ** 2).sum()
    total = ssa + ssb + ssab + ssr
    return {"A": 100 * ssa / total, "B": 100 * ssb / total, "AB": 100 * ssab / total}


def ablationFrame(df, prefix):
    d = df.copy()
    d["f1"] = d.condition.str.extract(prefix + r"(\d+)_")[0].astype(int)
    d["lr"] = d.condition.str.extract(r"_L0_(\d+)$")[0].map(LR)
    return d


def featuresPath(deposit):
    for path in (os.path.join(deposit, "cpc18_features.csv"), os.path.join(deposit, "features", "cpc18_features.csv")):
        if os.path.exists(path):
            return path
    raise SystemExit("cpc18_features.csv not found in the deposit")


def printCrossMachine(trials):
    """ Appendix D: BourginMLP per arithmetic path and the resolution floor of P10, from the trials of the
    original protocol (the cross-machine study runs in the fragile configuration). """
    # Appendix D: per-path values (machines of a path agree bit for bit; the first machine with the seed is used)
    values, common = {}, {}
    for key, (experiment, condition, _) in XMACHINE_CONDITIONS.items():
        values[key] = {}
        for path, machines in PATHS.items():
            d = trials[(trials.experiment == experiment) & (trials.condition == condition)
                       & trials.machine_id.isin(machines)]
            wide = d.pivot_table(index="seed", columns="machine_id", values="mse", aggfunc="first")
            values[key][path] = wide.bfill(axis=1).iloc[:, 0] * 100
        common[key] = sorted(set.intersection(*[set(s.index) for s in values[key].values()]))

    def onPath(key, path):
        return values[key][path].loc[common[key]].to_numpy()

    print("\n% Appendix D: BourginMLP (B = 32, lr = 1e-3) per arithmetic path: mean, std, best, P10 [95% CI], best seed")
    for path, label in PATH_LABELS.items():
        s = values["B32"][path].loc[common["B32"]]
        x = s.to_numpy()
        lo, hi = p10BootCi(x)
        print(f"{label} & {x.mean():.3f} & {x.std(ddof=1):.3f} & {x.min():.3f} & {p10Lognormal(x):.3f} "
              f"[{lo:.3f}--{hi:.3f}] & {int(s.idxmin())} \\\\")

    print("\n% Appendix D: largest |P10 shift| across the four comparison paths and half-width of the 95% CI of P10 "
          "on the reference, relative to the reference P10")
    for key, (_, _, label) in XMACHINE_CONDITIONS.items():
        x = onPath(key, "AVX512")
        p0 = p10Lognormal(x)
        lo, hi = p10BootCi(x)
        shift = max(abs(100 * (p10Lognormal(onPath(key, p)) / p0 - 1)) for p in PATHS if p != "AVX512")
        print(f"{label} & {shift:.1f}\\% & {100 * (hi - lo) / 2 / p0:.1f}\\% \\\\")

    identical = all((onPath("BEAST_GB", a) == onPath("BEAST_GB", b)).all() for a, b in itertools.combinations(PATHS, 2))
    print(f"\n% Seeds per path and condition: {sorted({len(c) for c in common.values()})}; "
          f"BEAST-GB bit-identical on all paths: {identical}")


def sigmaTestRows(V, profiles, order):
    """ Section 3.4.2: sigma_seed of the trials `V` against sigma_test of the 60-problem `profiles`. """
    rng = np.random.default_rng(TEST_SEED)
    rows = []
    for m in order:
        profile = profiles[m]
        sigma_test = np.sqrt(profile.var(ddof=0) / len(profile))
        idx = rng.integers(0, len(profile), size=(TEST_BOOT, len(profile)))
        sigma_test_mc = profile[idx].mean(axis=1).std(ddof=1)
        sigma_seed = V[m].std(ddof=1)
        rows.append((m, sigma_seed, sigma_test, sigma_test_mc))
    return rows


def stableTables(deposit):
    """ The tables of the stable configuration (see the module docstring). """
    from cpc18_utils.stable_results import StableResults, NEURAL_MODELS

    S = StableResults(deposit)
    P = S.main()
    V = {m: P[m].to_numpy() for m in MODEL_LABELS}
    pool = np.random.default_rng(MASTER_SEED_FOR_POOL).integers(0, 2 ** 32, size=800, dtype=np.uint64).astype(np.int64)

    # Section 3.4.2: networks from the per-problem errors of the stable protocol, trees from those of version 1.0
    print("% Section 3.4.2: sigma_seed (800 trials) against sigma_test (bootstrap of the mean of the 60-problem profile "
          "averaged over the first 50 seeds), stable configuration, rows by the ratio of the larger to the smaller")
    profiles = {}
    for name, models in (("perproblem_errors.csv", ("BEAST_GB", "PsychForest")),
                         ("perproblem_errors_stable.csv", NEURAL_MODELS)):
        frame = S.depositFrame(name)
        for m in models:
            profiles[m] = frame[frame.model == m].groupby("problem_id").mean_sq_error_x100.mean().to_numpy()
    rows = sigmaTestRows(V, profiles, list(MODEL_LABELS))
    for m, sigma_seed, sigma_test, sigma_test_mc in sorted(rows, key=lambda r: -max(r[1], r[2]) / min(r[1], r[2])):
        ratio = max(sigma_seed, sigma_test) / min(sigma_seed, sigma_test)
        source = "test sampling" if sigma_test > sigma_seed else "training seed"
        print(f"{MODEL_LABELS[m]} & {sigma_seed:.3f} & {sigma_test:.3f} & {source} (${ratio:.1f}\\times$) \\\\"
              f"  % {TEST_BOOT} resamples, seed {TEST_SEED}: sigma_test {sigma_test_mc:.4f}")

    # Section 4: reporting format; every fit of the stable configuration validates
    print("\n% Section 4: reporting format (P10 of a lognormal fit; check = share of trials at or below it), "
          "stable configuration")
    features = pd.read_csv(featuresPath(deposit), float_precision="round_trip")
    test = features[features.Train_Test == "Test"]
    beast = 100 * ((test.BEASTpred - test.B_rate) ** 2).mean()
    print(f"BEAST (theory only) & --- & {beast:.3f} & {beast:.3f} & --- & {beast:.3f} & --- & "
          r"\citealp{erev2017anomalies, plonsky2024beastgb} \\")
    for m in sorted(MODEL_LABELS, key=lambda k: V[k].mean()):
        x = V[m]
        p10 = p10Lognormal(x)
        print(f"{MODEL_LABELS[m]} & {len(x)} & {x.min():.3f} & {x.mean():.3f} & {x.std(ddof=1):.3f} & "
              f"{p10:.3f} & {percent(np.mean(x <= p10))}\\% & {SOURCES[m]} \\\\")

    # Sample-size appendix: first seeds of the shared pool at N = 100, 200, 400 and 800; the ablation entries are the
    # largest factor share of each grid
    print("\n% Sample-size appendix: headline statistics at N = 100, 200, 400 and 800, stable configuration")
    sizes = (100, 200, 400, 800)
    firsts = {n: set(pool[:n].tolist()) for n in sizes}
    Pn = {n: P.loc[[s for s in P.index if s in firsts[n]]] for n in sizes}
    names = ("BEAST_GB", "PsychForest", "LinearLogistic", "ContextDepNet", "BourginMLP", "SparseMLP")
    for m in names:
        print(f"{MODEL_LABELS[m]} mean (std) & "
              + " & ".join(f"{Pn[n][m].mean():.3f} ({Pn[n][m].std(ddof=1):.3f})" for n in sizes) + " \\\\")
    for m in names:
        print(f"{MODEL_LABELS[m]} $P_{{10}}$ & " + " & ".join(f"{p10Lognormal(Pn[n][m].to_numpy()):.3f}" for n in sizes)
              + " \\\\")
    swap = [np.mean(Pn[n]["BourginMLP"].to_numpy()[:, None] > Pn[n]["PsychForest"].to_numpy()[None, :]) for n in sizes]
    print("$P(\\mathrm{swap}|N{=}1)$, BourginMLP--PsychForest & " + " & ".join(f"{v:.3f}" for v in swap) + " \\\\")
    order = list(MODEL_LABELS)
    print("ICC(3,1), six-model pool & " + " & ".join(f"{iccThreeOne(Pn[n][order].to_numpy()):.3f}" for n in sizes) + " \\\\")
    A1, A3 = S.ablation("ablation_bourginmlp"), S.ablation("ablation_sparsemlp")
    inter_share = [twoWayShares(A1[A1.seed.isin(firsts[n])], "f1", "lr", y="y")["AB"] for n in sizes]
    eps_share = [twoWayShares(A3[A3.seed.isin(firsts[n])], "f1", "lr", y="y")["A"] for n in sizes]
    print("Ablation~BourginMLP, interaction $\\eta^2$ (\\%) & " + " & ".join(f"{v:.1f}" for v in inter_share) + " \\\\")
    print("Ablation~SparseMLP, $\\varepsilon$ $\\eta^2$ (\\%) & " + " & ".join(f"{v:.1f}" for v in eps_share) + " \\\\")

    # Appendix D: training time per trial and serial total, machines of the reference path (M00 and M01)
    print("\n% Appendix D: per-trial training time (median, max) and serial total, stable configuration, machines "
          "of the reference path")
    ref = S.reference
    main_run = ref[ref.experiment == "main"]
    b1 = ref[ref.experiment == "ablation_bourginmlp"]
    rows = [(f"Main run: {MODEL_LABELS[m]}", main_run[main_run.condition == m])
            for m in ("LinearLogistic", "BourginMLP", "ContextDepNet", "SparseMLP", "PsychForest", "BEAST_GB")]
    rows += [("Ablation~BourginMLP (9 conditions)", b1[~np.isclose(b1.learning_rate, 3.16e-4)]),
             ("Grid-sensitivity column (3 conditions)", b1[np.isclose(b1.learning_rate, 3.16e-4)]),
             ("Ablation~SparseMLP (9 conditions)", ref[ref.experiment == "ablation_sparsemlp"]),
             ("Variance decomposition (4 NN)", ref[ref.experiment == "factorial_init_order"])]
    for label, d in rows:
        t = d.train_duration_s
        longest = f"{t.max():.1f}" if t.max() < 1 else f"{t.max():.0f}"
        count = f"{len(d):,}".replace(",", "{,}")
        print(f"{label} & {count} & {t.median():.1f}\\,s ({longest}\\,s) & {t.sum() / 3600:.1f}\\,h \\\\")

    # Appendix D: the cross-machine tables stay in the fragile configuration
    print("\n% Appendix D, cross-machine tables: fragile configuration (original protocol)")
    trials = S.trials[S.trials.protocol == "original"]
    printCrossMachine(trials)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deposit", default=os.path.join(_REPO_ROOT, "zenodo_export"),
                        help="folder of the data deposit (default: <repo>/zenodo_export)")
    parser.add_argument("--protocol", choices=("original", "stable"), default="original",
                        help="training protocol of the tables (cpc18_utils/protocol.py)")
    args = parser.parse_args()
    deposit = args.deposit
    if args.protocol == "stable":
        stableTables(deposit)
        return

    trials = pd.read_csv(os.path.join(deposit, "trials.csv"), float_precision="round_trip")
    if "protocol" in trials:  # deposit 1.1 and later: these tables are those of the original protocol
        trials = trials[trials.protocol == "original"]
    perproblem = pd.read_csv(os.path.join(deposit, "perproblem_errors.csv"), float_precision="round_trip")
    perproblem["model"] = perproblem.model.str.replace("BEAST-GB", "BEAST_GB")
    features = pd.read_csv(featuresPath(deposit), float_precision="round_trip")

    ref = trials[trials.machine_id == "M00"]
    main_run = ref[ref.experiment == "main"]
    P = main_run.pivot(index="seed", columns="condition", values="mse_x100")
    V = {m: P[m].to_numpy() for m in MODEL_LABELS}
    pool = np.random.default_rng(MASTER_SEED_FOR_POOL).integers(0, 2 ** 32, size=800, dtype=np.uint64).astype(np.int64)

    # Section 3.4.2: training-seed against test-sampling variance
    print("% Section 3.4.2: sigma_seed (800 trials) against sigma_test (bootstrap of the mean of the 60-problem profile "
          "averaged over the first 50 seeds)")
    rng = np.random.default_rng(TEST_SEED)
    for m in ("BEAST_GB", "PsychForest", "BourginMLP"):
        profile = perproblem[perproblem.model == m].groupby("problem_id").mean_sq_error_x100.mean().to_numpy()
        sigma_test = np.sqrt(profile.var(ddof=0) / len(profile))
        idx = rng.integers(0, len(profile), size=(TEST_BOOT, len(profile)))
        sigma_test_mc = profile[idx].mean(axis=1).std(ddof=1)
        sigma_seed = V[m].std(ddof=1)
        ratio = max(sigma_seed, sigma_test) / min(sigma_seed, sigma_test)
        source = "test sampling" if sigma_test > sigma_seed else "training seed"
        print(f"{MODEL_LABELS[m]} & {sigma_seed:.3f} & {sigma_test:.3f} & {source} (${ratio:.1f}\\times$) \\\\"
              f"  % {TEST_BOOT} resamples, seed {TEST_SEED}: sigma_test {sigma_test_mc:.4f}")

    # Section 4: reporting format
    print("\n% Section 4: reporting format (P10 of a lognormal fit; check = share of trials at or below it)")
    test = features[features.Train_Test == "Test"]
    beast = 100 * ((test.BEASTpred - test.B_rate) ** 2).mean()
    print(f"BEAST (theory only) & --- & {beast:.3f} & {beast:.3f} & --- & {beast:.3f} & --- & "
          r"\citealp{erev2017anomalies, plonsky2024beastgb} \\")
    for m in sorted(MODEL_LABELS, key=lambda k: V[k].mean()):
        x = V[m]
        p10 = p10Lognormal(x)
        cells = [f"{p10:.3f}", f"{percent(np.mean(x <= p10))}\\%"]
        if m == "BourginMLP":  # the fit does not validate: the target is printed in italics
            cells = [f"\\textit{{{c}}}" for c in cells]
        print(f"{MODEL_LABELS[m]} & {len(x)} & {x.min():.3f} & {x.mean():.3f} & {x.std(ddof=1):.3f} & "
              f"{cells[0]} & {cells[1]} & {SOURCES[m]} \\\\")

    # Appendix C: N = 400 (first seeds of the shared pool) against N = 800
    print("\n% Appendix C: N = 400 against N = 800")
    first400 = set(pool[:400].tolist())
    P4 = P.loc[[s for s in P.index if s in first400]]
    for m in ("BEAST_GB", "PsychForest", "LinearLogistic", "ContextDepNet", "BourginMLP", "SparseMLP"):
        a4, a8 = P4[m].to_numpy(), V[m]
        print(f"{MODEL_LABELS[m]} mean (std) & {a4.mean():.3f} ({a4.std(ddof=1):.3f}) & "
              f"{a8.mean():.3f} ({a8.std(ddof=1):.3f}) \\\\")
    swap = [np.mean(d["BourginMLP"].to_numpy()[:, None] > d["PsychForest"].to_numpy()[None, :]) for d in (P4, P)]
    print(f"$P(\\mathrm{{swap}}|N{{=}}1)$, BourginMLP--PsychForest & {swap[0]:.3f} & {swap[1]:.3f} \\\\")
    order = list(MODEL_LABELS)
    print(f"ICC(3,1), six-model pool & {iccThreeOne(P4[order].to_numpy()):.3f} & {iccThreeOne(P[order].to_numpy()):.3f} \\\\")
    b1 = ref[(ref.experiment == "ablation_bourginmlp") & ~ref.condition.str.endswith("000316")]
    b3 = ref[ref.experiment == "ablation_sparsemlp"]
    A1, A3 = ablationFrame(b1, "BourginMLP_B"), ablationFrame(b3, "SparseMLP_E")
    lr_share = [twoWayShares(d, "f1", "lr")["B"] for d in (A1[A1.seed.isin(first400)], A1)]
    inter_share = [twoWayShares(d, "f1", "lr")["AB"] for d in (A3[A3.seed.isin(first400)], A3)]
    print(f"Ablation~BourginMLP, learning-rate \\%SS & {lr_share[0]:.1f} & {lr_share[1]:.1f} \\\\")
    print(f"Ablation~SparseMLP, interaction \\%SS & {inter_share[0]:.1f} & {inter_share[1]:.1f} \\\\")

    # Appendix D: training time per trial and serial total, reference machine
    print("\n% Appendix D: per-trial training time (median, max) and serial total, reference machine")
    rows = [(f"Main run: {MODEL_LABELS[m]}", main_run[main_run.condition == m])
            for m in ("LinearLogistic", "BourginMLP", "ContextDepNet", "SparseMLP", "PsychForest", "BEAST_GB")]
    rows += [("Ablation~BourginMLP (9 conditions)", b1),
             ("Grid-sensitivity column (3 conditions)",
              ref[(ref.experiment == "ablation_bourginmlp") & ref.condition.str.endswith("000316")]),
             ("Ablation~SparseMLP (9 conditions)", b3),
             ("Variance decomposition (4 NN)", ref[ref.experiment == "factorial_init_order"])]
    for label, d in rows:
        t = d.train_duration_s
        longest = f"{t.max():.1f}" if t.max() < 1 else f"{t.max():.0f}"
        count = f"{len(d):,}".replace(",", "{,}")
        print(f"{label} & {count} & {t.median():.1f}\\,s ({longest}\\,s) & {t.sum() / 3600:.1f}\\,h \\\\")

    printCrossMachine(trials)


if __name__ == "__main__":
    main()
