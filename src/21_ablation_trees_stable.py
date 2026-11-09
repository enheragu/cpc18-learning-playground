#!/usr/bin/env python3
# encoding: utf-8

"""
Sequential decision trees of the two ablations of the article in the stable
configuration (cpc18_utils/protocol.py), from the data deposit (version 1.1 or
later, --deposit, by default zenodo_export/), for the 3 x 3 grids with learning
rates 1e-4, 1e-3 and 1e-2.

Each simulated ablation draws one run per cell, independently between cells
(Monte Carlo: the lognormal fitted to the 800 trials of the cell; Bootstrap: one
of those trials), and follows the sequential decision of the study on MNIST
(simulation of the MLV Tools viewer): fix one factor at a start level, choose
the best level of the other factor (lowest MSE of the sampled cells), then
choose the best level of the start factor at that level, with the same sampled
values. The global best is the cell with the lowest mean of its 800 trials. 40000
paths per method, seed 0, the values of the article on MNIST.

Output, by default in analysis_results/figures_stable/ablation_test/:

  fig_tree_bourgin_B16.tex      TikZ tree of Ablation BourginMLP from batch size 16
  fig_tree_sparse_E20.tex       TikZ tree of Ablation SparseMLP from epsilon 20
  ablation_decision_reach.csv   probability of reaching the global best: per start
                                level (one run per cell), and averaged over the six
                                starts when a cell is the mean of N runs

The node styles of the trees build on `archblock` (fig_tikz_styles.tex of the
article on MNIST). Each node gives the percentage of the paths through it, Monte
Carlo | Bootstrap; the route to the global best is green and the start level
yellow. Before writing, every tree is checked: the children of a node add up to
the node, and the leaf of the global best equals the probability of reaching it.

Usage:
    cd src && python 21_ablation_trees_stable.py [--deposit PATH] [--out PATH]
"""

import argparse
import csv
import os

import numpy as np

from cpc18_utils.stable_results import StableResults

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIALS = 40000
SEED = 0
RUNS_PER_CELL = (1, 5, 20, 100)
METHODS = ("mc", "bs")

STUDIES = (  # ablation name, experiment, first factor, name of its tree, start level of its tree
    ("Ablation BourginMLP", "ablation_bourginmlp", "batch_size", "fig_tree_bourgin_B16", 16),
    ("Ablation SparseMLP", "ablation_sparsemlp", "epsilon", "fig_tree_sparse_E20", 20),
)
LABELS = {  # factor: (name in the start node, prefix in the other nodes)
    "batch_size": ("Batch size", "Batch "),
    "epsilon": (r"Sparsity $\varepsilon$", r"$\varepsilon$ "),
    "learning_rate": ("Learning rate", "LR "),
}
COL_GAP = 2.0    # cm between the left edges of two columns
ROOT_GAP = 0.34  # cm between the start node (anchored east at x = 0) and the first column
TRUNK = 0.16     # cm from a column to the vertical segment that feeds it
ROW_GAP = 1.05   # cm between leaves


def sampleCells(cells, method, n_runs):
    """ One sampled value per cell and path: the mean of `n_runs` draws, independent between cells. Shape (3, 3, paths). """
    out = np.empty((3, 3, TRIALS))
    for (i, j), x in np.ndenumerate(cells):
        if method == "mc":
            logs = np.log(x)
            draws = np.exp(np.random.normal(logs.mean(), logs.std(), size=(TRIALS, n_runs)))
        else:
            draws = np.random.choice(x, size=(TRIALS, n_runs), replace=True)
        out[i, j] = draws.mean(axis=1)
    return out


def decide(samples, start_axis, start_level):
    """ Level of the other factor chosen first, and level of the start factor chosen then, per path. """
    paths = np.arange(samples.shape[2])
    if start_axis == 0:
        first = np.argmin(samples[start_level, :, :], axis=0)
        second = np.argmin(samples[:, first, paths], axis=0)
    else:
        first = np.argmin(samples[:, start_level, :], axis=0)
        second = np.argmin(samples[first, :, paths], axis=1)
    return first, second


def reachProbability(samples, start_axis, start_level, best):
    first, second = decide(samples, start_axis, start_level)
    rows, cols = (second, first) if start_axis == 0 else (first, second)
    return float(np.mean((rows == best[0]) & (cols == best[1])))


def label(factor, value, start=False):
    name, prefix = LABELS[factor]
    return f"{name}\n{value:g}" if start else f"{prefix}{value:g}"


def pct(p):
    return r"$<$0.1" if 0 < p < 0.0005 else f"{100 * p:.1f}"


def buildTree(samples, start_axis, start_level, levels, factors, best):
    """ Root at the start level, first column the other factor, leaves the start factor chosen again. """
    start_factor, other_factor = factors[start_axis], factors[1 - start_axis]
    start_levels, other_levels = levels[start_axis], levels[1 - start_axis]
    best_start, best_other = best[start_axis], best[1 - start_axis]
    tree = {"label": label(start_factor, start_levels[start_level], start=True), "mc": None, "bs": None, "best": False,
            "start": True, "children": []}
    decided = {m: decide(samples[m], start_axis, start_level) for m in METHODS}
    for a, other in enumerate(other_levels):
        share = {m: float(np.mean(decided[m][0] == a)) for m in METHODS}
        node = {"label": label(other_factor, other), "best": a == best_other, "children": [], **share}
        for b, value in enumerate(start_levels):
            leaf = {m: float(np.mean((decided[m][0] == a) & (decided[m][1] == b))) for m in METHODS}
            node["children"].append({"label": label(start_factor, value), "best": a == best_other and b == best_start,
                                     "children": [], **leaf})
        tree["children"].append(node)
    return tree


def checkTree(tree, reach):
    """ The children of every node add up to the node, and the leaf of the global best is the reach probability. """
    for node in tree["children"]:
        for m in METHODS:
            if abs(sum(c[m] for c in node["children"]) - node[m]) > 1e-9:
                raise SystemExit(f"children of {node['label']} do not add up")
    for m in METHODS:
        if abs(sum(n[m] for n in tree["children"]) - 1) > 1e-9:
            raise SystemExit("first level does not add up to 1")
    leaves = [c for n in tree["children"] for c in n["children"]]
    best = [leaf for leaf in leaves if leaf["best"]]
    if len(best) != 1:
        raise SystemExit("expected exactly one global-best leaf")
    for m in METHODS:
        if abs(best[0][m] - reach[m]) > 1e-9:
            raise SystemExit(f"global-best leaf ({m}) differs from the reach probability")


def layout(tree):
    """ Leaves on consecutive rows, parents centred on their children. """
    rows = [0]

    def place(n, depth):
        n["x"] = 0.0 if depth == 0 else ROOT_GAP + (depth - 1) * COL_GAP
        if not n["children"]:
            n["y"] = -rows[0] * ROW_GAP
            rows[0] += 1
            return
        for c in n["children"]:
            place(c, depth + 1)
        n["y"] = (n["children"][0]["y"] + n["children"][-1]["y"]) / 2
    place(tree, 0)


def tikz(tree):
    lines = [r"\begin{tikzpicture}[",
             r"  treenode/.style={archblock, thin, minimum width=1.5cm, minimum height=0pt, inner sep=1.5pt,",
             r"                   fill=white},",
             r"  treebest/.style={treenode, thick, fill=gGreen!35},",
             r"  treestart/.style={treenode, thick, fill=gYellow!45},",
             r"  treeedge/.style={draw=gGrey, line width=0.5pt},",
             r"  treebestedge/.style={draw=gDarkGrey, line width=1pt}]"]
    counter = [0]

    def emit(n, parent):
        counter[0] += 1
        name = f"n{counter[0]}"
        style = "treestart" if n.get("start") else ("treebest" if n["best"] else "treenode")
        text = r"\\".join(rf"\textbf{{{part}}}" for part in n["label"].split("\n"))
        if n["mc"] is not None:
            text += rf"\\MC={pct(n['mc'])}\,\%\\BS={pct(n['bs'])}\,\%"
        anchor = "east" if n.get("start") else "west"
        lines.append(rf"  \node[{style}, anchor={anchor}] ({name}) at ({n['x']:.2f},{n['y']:.3f}) {{{text}}};")
        if parent is not None:
            edge = "treebestedge" if n["best"] else "treeedge"
            lines.append(rf"  \draw[{edge}] ({parent}.east) -| ({n['x'] - TRUNK:.2f},{n['y']:.3f}) -- ({name}.west);")
        for c in n["children"]:
            emit(c, name)
    emit(tree, None)
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines) + "\n"


def simulate(cells, n_runs):
    """ Sampled cells of the two methods; each method restarts the generator, as in the article on MNIST. """
    samples = {}
    for method in METHODS:
        np.random.seed(SEED)
        samples[method] = sampleCells(cells, method, n_runs)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deposit", default=os.path.join(_REPO_ROOT, "zenodo_export"),
                        help="folder of the data deposit, version 1.1 or later (default: <repo>/zenodo_export)")
    parser.add_argument("--out", default=os.path.join(_REPO_ROOT, "analysis_results", "figures_stable", "ablation_test"),
                        help="folder of the output files")
    args = parser.parse_args()
    S = StableResults(args.deposit)
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for ablation, experiment, factor, tree_name, tree_start in STUDIES:
        long = S.ablation(experiment)
        first_levels, lrs = sorted(long.f1.unique()), sorted(long.lr.unique())
        cells = np.empty((3, 3), dtype=object)
        for i, f1 in enumerate(first_levels):
            for j, lr in enumerate(lrs):
                cells[i, j] = long[(long.f1 == f1) & (long.lr == lr)].y.to_numpy()
                assert len(cells[i, j]) == 800, (f1, lr)
        means = np.array([[cells[i, j].mean() for j in range(3)] for i in range(3)])
        best = tuple(int(k) for k in np.unravel_index(np.argmin(means), means.shape))
        levels, factors = (first_levels, lrs), (factor, "learning_rate")

        samples = simulate(cells, 1)
        single = []
        for start_axis in (0, 1):
            for start_level, value in enumerate(levels[start_axis]):
                reach = {m: reachProbability(samples[m], start_axis, start_level, best) for m in METHODS}
                tree = buildTree(samples, start_axis, start_level, levels, factors, best)
                checkTree(tree, reach)
                single.append(reach)
                rows.append({"ablation": ablation, "summary": "one run per cell, start level",
                             "start_factor": factors[start_axis], "start_value": f"{value:g}", "n_runs": 1,
                             "p_reach_mc": reach["mc"], "p_reach_bootstrap": reach["bs"]})
                if start_axis == 0 and value == tree_start:
                    layout(tree)
                    with open(os.path.join(args.out, f"{tree_name}.tex"), "w") as f:
                        f.write(tikz(tree))
                    print(f"{tree_name}: global best ({first_levels[best[0]]:g}, {lrs[best[1]]:g}), reach MC "
                          f"{100 * reach['mc']:.1f} % Bootstrap {100 * reach['bs']:.1f} %")

        for n_runs in RUNS_PER_CELL:
            if n_runs == 1:
                average = {m: float(np.mean([r[m] for r in single])) for m in METHODS}
            else:
                sampled = simulate(cells, n_runs)
                average = {m: float(np.mean([reachProbability(sampled[m], axis, level, best)
                                             for axis in (0, 1) for level in range(3)])) for m in METHODS}
            rows.append({"ablation": ablation, "summary": "mean of N runs per cell, average of the six starts",
                         "start_factor": "", "start_value": "", "n_runs": n_runs,
                         "p_reach_mc": average["mc"], "p_reach_bootstrap": average["bs"]})

    out_csv = os.path.join(args.out, "ablation_decision_reach.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nProbability of reaching the global best (%), {TRIALS} paths per method, seed {SEED}")
    print(f"{'ablation':20s} {'start':24s} {'N':>4s} {'Monte Carlo':>12s} {'Bootstrap':>10s}")
    for r in rows:
        start = f"{r['start_factor']}={r['start_value']}" if r["start_factor"] else "average of six starts"
        print(f"{r['ablation']:20s} {start:24s} {r['n_runs']:4d} {100 * r['p_reach_mc']:12.1f} "
              f"{100 * r['p_reach_bootstrap']:10.1f}")


if __name__ == "__main__":
    main()
