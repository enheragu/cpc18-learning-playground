#!/usr/bin/env python3
# encoding: utf-8

"""
    Generates the figures that populate the LaTeX manuscript at
    `/home/quique/umh/latex_docs/08_eam_variance_replication/`.

    The plotting heavy-lifting is delegated to the MNIST companion
    submodule's helpers (`utils.plot_distribution`,
    `utils.plot_sampling_graph`) so that both papers' figures share the
    same style and font conventions. The wrappers in this script handle
    only:

      * loading the per-trial YAML files,
      * converting the lower-is-better MSE into the higher-is-better
        convention expected by the MNIST sampling helpers (sign flip),
      * routing output files into the LaTeX project's media/ tree.

    Usage:
        python 05_make_paper_figures.py
"""

import os
import re
import shutil

import path_setup  # noqa: F401

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

from utils.log_utils import (
    log, bcolors, color_palette_list,
    c_blue, c_green, c_yellow, c_red, c_purple, c_grey, c_darkgrey,
)
from utils.plot_distribution import (
    plot_metric_distribution, plot_survival_function,
)
from utils.plot_sampling_graph import (
    plot_all_sampling_errors, plot_all_percentile_probabilities,
)


# `generate_interaction_plot` lives in `05_check_ablation_test.py` of the
# MNIST submodule — its filename starts with a digit so we cannot use a
# regular import. We pull it via importlib instead.
import importlib.util
_mnist_check_ablation_path = os.path.join(
    os.path.dirname(__file__), "..", "mnist-learning-playground", "src",
    "05_check_ablation_test.py",
)
_spec = importlib.util.spec_from_file_location(
    "mnist_check_ablation_test", _mnist_check_ablation_path)
_mnist_check_ablation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mnist_check_ablation)
generate_interaction_plot = _mnist_check_ablation.generate_interaction_plot

from cpc18_utils import output_path
from cpc18_utils.safe_yaml import fastLoadYaml


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3}$")

latex_root = "/home/quique/umh/latex_docs/08_eam_variance_replication"
media_path = os.path.join(latex_root, "media")


# Per-model colour palette aligned with the MNIST plot palette.
model_colours = {
    "LinearLogistic": c_grey,
    "BourginMLP":     c_red,
    "ContextDepNet":  c_purple,
    "SparseMLP":      c_darkgrey,
    "PsychForest":    c_yellow,
    "BEAST_GB":       c_blue,
}


# ---------------------------------------------------------------------
# Toggles
# ---------------------------------------------------------------------
enable_intro_distributions      = True
enable_overlap_plots            = True
enable_best_of_n                = True
enable_informed_training_panels = True
enable_ablation_grids           = True
enable_anova_interaction_plots  = True
enable_best_cell_per_seed       = True
enable_sampling_probability     = True
enable_survival_combined        = True
enable_sampling_error           = True
enable_decomp_heatmaps          = True


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def _trialEntries(data):
    return {k: v for k, v in data.items()
            if _DATE_RE.match(k) and isinstance(v, dict)}


def loadMainR2Raw(model_name):
    """ Returns the raw `{timestamp: entry}` dict (MNIST-helper format). """
    p = os.path.join(output_path, model_name,
                     "sharedpool_r2_training_metrics.yaml")
    return _trialEntries(fastLoadYaml(p))


def loadAblationRaw(model_dir):
    p = os.path.join(output_path, model_dir,
                     "sharedpool_r2_ablation_training_metrics.yaml")
    return _trialEntries(fastLoadYaml(p))


def mse100Values(raw):
    """ Returns the list of MSE x 100 floats from a raw `{ts: entry}` dict. """
    return [float(e["mse"]) * 100.0 for e in raw.values()]


def signFlippedRaw(raw):
    """
        Returns a copy of `raw` with each trial's MSE replaced by its
        sign-flipped version, stored under the same key. This converts
        the lower-is-better convention into the higher-is-better one
        expected by the MNIST sampling helpers.
    """
    out = {}
    for k, v in raw.items():
        e = dict(v)
        e["neg_mse"] = -float(e["mse"])
        out[k] = e
    return out


def _renameOutput(src_dir, src_name, dst_subdir, dst_name):
    """
        MNIST helpers write files into `analysis_path` with their own
        naming scheme. We move them under the LaTeX project's media/
        subfolders and rename them to match the manuscript's
        `\\includegraphics` calls.
    """
    src = os.path.join(src_dir, src_name)
    dst_dir = os.path.join(media_path, dst_subdir)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, dst_name)
    if os.path.exists(src):
        shutil.move(src, dst)
        log(f"[fig] wrote {dst_subdir}/{dst_name}", color=bcolors.OKGREEN)


# ---------------------------------------------------------------------
# Section 1 - introduction figures (use plot_metric_distribution)
# ---------------------------------------------------------------------

def figIntroBourginMLPMse():
    raw = loadMainR2Raw("BourginMLP")
    if not raw:
        return
    vals = mse100Values(raw)
    out_dir = os.path.join(media_path, "introduction")
    os.makedirs(out_dir, exist_ok=True)
    plot_metric_distribution(
        metrics_data={"BourginMLP": vals},
        metric_label=r"Test MSE $\times\,100$",
        analysis_path=out_dir,
        plot_filename="plot_mse_BourginMLP_r2",
        title="BourginMLP, main run",
        x_max_quantile=98.0,
        show_histogram=True,
    )
    log("[fig] wrote introduction/plot_mse_BourginMLP_r2.pdf",
        color=bcolors.OKGREEN)


def figIntroBourginMLPBestEpoch():
    raw = loadMainR2Raw("BourginMLP")
    if not raw:
        return
    epochs = []
    for v in raw.values():
        # Prefer recomputing the true best epoch from the stored per-epoch
        # test-MSE curve (argmin). The logged `best_epoch` is wrong for the
        # rare trials that hit the num_epochs cap (see base.py); the curve
        # is always correct.
        mp = v.get("mse_plot")
        if isinstance(mp, (list, tuple)) and mp:
            epochs.append(int(min(range(len(mp)), key=mp.__getitem__)))
            continue
        for k in ("best_epoch", "best_mse_epoch",
                  "epoch_best", "best_loss_epoch"):
            if k in v:
                epochs.append(int(v[k]))
                break
    if not epochs:
        return
    out_dir = os.path.join(media_path, "introduction")
    plot_metric_distribution(
        metrics_data={"BourginMLP": epochs},
        metric_label="Best epoch",
        analysis_path=out_dir,
        plot_filename="plot_best_epoch_BourginMLP_r2",
        title="BourginMLP, main run",
        x_max_quantile=99.0,
        show_histogram=True,
    )
    log("[fig] wrote introduction/plot_best_epoch_BourginMLP_r2.pdf",
        color=bcolors.OKGREEN)


def figOverlapTwoModels():
    a = loadMainR2Raw("BourginMLP")
    b = loadMainR2Raw("PsychForest")
    if not (a and b):
        return
    metrics_data = {"BourginMLP": mse100Values(a),
                    "PsychForest": mse100Values(b)}
    for subdir in ("introduction", "replication_comparison"):
        out_dir = os.path.join(media_path, subdir)
        plot_metric_distribution(
            metrics_data=metrics_data,
            metric_label=r"Test MSE $\times\,100$",
            analysis_path=out_dir,
            plot_filename="plot_mse_overlap_BourginMLP_PsychForest"
                          if subdir == "introduction"
                          else "plot_mse_BourginMLP_PsychForest",
            x_max_quantile=98.0,
            show_histogram=True,
        )
    log("[fig] wrote BourginMLP-vs-PsychForest overlap (both sections)",
        color=bcolors.OKGREEN)


def _plotOverlapBrokenAxis(metrics_data, out_path, y_break=11.0):
    """
        Six-model MSE distribution overlap with a broken y-axis.

        The submodule helper `plot_metric_distribution` cannot do a
        broken axis, and the BEAST-GB / PsychForest density spikes
        (~58 / ~36) otherwise squash the wide, low neural-network
        distributions to invisible bands. We therefore draw a custom
        two-panel figure (shared x): the top panel shows the tall
        spikes, the bottom panel zooms into the [0, y_break] band where
        the neural-network distributions live. Per model we draw the
        density histogram, a fitted-normal dashed curve and a solid mean
        line, matching the style of the submodule helper.
    """
    names = list(metrics_data.keys())
    arrays = {n: np.asarray(metrics_data[n], dtype=float) for n in names}
    pooled = np.concatenate(list(arrays.values()))
    x_lo = float(min(a.min() for a in arrays.values())) - 0.02
    x_hi = float(np.percentile(pooled, 98.0))
    bins = np.linspace(x_lo, x_hi, 60)
    grid = np.linspace(x_lo, x_hi, 400)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 6),
        gridspec_kw={"height_ratios": [1, 2], "hspace": 0.06})

    y_top_max = 0.0
    for i, n in enumerate(names):
        a = arrays[n]
        color = color_palette_list[i % len(color_palette_list)]
        mean, std = a.mean(), a.std()
        label = f"{n} (n={len(a)})"
        for ax in (ax_top, ax_bot):
            ax.hist(a, bins=bins, density=True, color=color, alpha=0.35,
                    label=label if ax is ax_bot else None)
            if std > 1e-6:
                ax.plot(grid, norm.pdf(grid, mean, std), "--",
                        color=color, linewidth=1.8)
            ax.axvline(mean, color=color, linewidth=2.0, alpha=0.9)
        # track the tallest histogram/curve for the top panel limit
        h, _ = np.histogram(a, bins=bins, density=True)
        peak = max(h.max(), norm.pdf(mean, mean, std) if std > 1e-6 else 0)
        y_top_max = max(y_top_max, peak)

    # Broken-axis limits: bottom zooms into the NN band, top shows spikes.
    ax_bot.set_ylim(0, y_break)
    ax_top.set_ylim(y_break + 1.0, y_top_max * 1.08)

    # Hide the shared spine and add diagonal break marks.
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False)
    d = 0.008
    kw = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1)
    ax_top.plot((-d, +d), (-d * 2, +d * 2), **kw)
    ax_top.plot((1 - d, 1 + d), (-d * 2, +d * 2), **kw)
    kw.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kw)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)

    ax_top.set_title(r"Test MSE $\times\,100$ distribution")
    ax_bot.set_xlabel(r"Test MSE $\times\,100$")
    fig.text(0.04, 0.5, "Density", va="center", rotation="vertical")
    ax_bot.set_xlim(x_lo, x_hi)
    # Legend in the top panel: its right half is empty (the spikes sit on
    # the left), so it does not overlap any distribution.
    handles, labels = ax_bot.get_legend_handles_labels()
    ax_top.legend(handles, labels, fontsize=8, framealpha=0.9,
                  loc="upper right")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def figOverlapSixModels():
    metrics_data = {}
    for m in ("LinearLogistic", "BourginMLP", "ContextDepNet",
              "SparseMLP", "PsychForest", "BEAST_GB"):
        raw = loadMainR2Raw(m)
        if raw:
            metrics_data[m] = mse100Values(raw)
    for subdir in ("introduction", "replication_comparison"):
        out_dir = os.path.join(media_path, subdir)
        os.makedirs(out_dir, exist_ok=True)
        fname = ("plot_mse_overlap_six_models.pdf"
                 if subdir == "introduction" else "plot_mse_six_models.pdf")
        _plotOverlapBrokenAxis(metrics_data, os.path.join(out_dir, fname))
    log("[fig] wrote six-model overlap with broken y-axis (both sections)",
        color=bcolors.OKGREEN)


def figBestOfN():
    """
        Best-of-N leaderboard curve: E[min MSE of N runs] vs N, per model.

        A practitioner reports the best of the runs they can afford. This
        figure shows how that reported value moves with the run budget N.
        BourginMLP starts last among the competitive models at N = 1 but,
        thanks to its low tail, its best-of-N value overtakes PsychForest
        at N ~ 14; BEAST-GB is never overtaken. The point is that a
        best-of-N leaderboard with heterogeneous N ranks compute budgets
        as much as it ranks models (Section~\\ref{subsec:replication_comparison}).
    """
    models = ["LinearLogistic", "BourginMLP", "ContextDepNet",
              "SparseMLP", "PsychForest", "BEAST_GB"]
    data = {m: np.asarray(mse100Values(loadMainR2Raw(m)), dtype=float)
            for m in models if loadMainR2Raw(m)}
    Ns = list(range(1, 51))
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (m, arr) in enumerate(data.items()):
        emin = [float(np.min(rng.choice(arr, size=(20000, N)), axis=1).mean())
                for N in Ns]
        ax.plot(Ns, emin, label=m,
                color=color_palette_list[i % len(color_palette_list)],
                linewidth=1.8)
    ax.set_xlabel(r"Best-of-$N$ budget (runs per model)")
    ax.set_ylabel(r"E[best of $N$ runs], MSE $\times$ 100")
    ax.set_title(r"Best-of-$N$ reported MSE by model")
    ax.legend(fontsize=8)
    out_dir = os.path.join(media_path, "replication_comparison")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "best_of_n.pdf"), bbox_inches="tight")
    plt.close(fig)
    log("[fig] wrote replication_comparison/best_of_n.pdf",
        color=bcolors.OKGREEN)


# ---------------------------------------------------------------------
# Section 3 - informed training panels
# ---------------------------------------------------------------------

def figInformedTrainingPanels():
    raw = loadMainR2Raw("BourginMLP")
    if not raw:
        return
    vals = np.asarray(mse100Values(raw))
    mu, sigma = float(vals.mean()), float(vals.std(ddof=1))
    sorted_v = np.sort(vals)
    n_sub = 100
    subsets = (
        ("right_biased",  sorted_v[-n_sub:],          "red"),
        ("center_biased", sorted_v[len(sorted_v) // 2 - n_sub // 2:
                                    len(sorted_v) // 2 + n_sub // 2],
         "yellow"),
        ("left_biased",   sorted_v[:n_sub],           "green"),
    )
    out_dir = os.path.join(media_path, "informed_training")
    for tag, subset, _bias_colour in subsets:
        plot_metric_distribution(
            metrics_data={"BourginMLP": list(subset.astype(float))},
            metric_label=r"Test MSE $\times\,100$",
            analysis_path=out_dir,
            plot_filename=f"example_BourginMLP_{tag}",
            plot_mean=mu,
            plot_std=sigma,
            x_max_quantile=99.0,
            show_histogram=True,
        )
    log("[fig] wrote informed_training panels (3)",
        color=bcolors.OKGREEN)


# ---------------------------------------------------------------------
# Section 6 - ablation grids
# ---------------------------------------------------------------------

_B1_DIRS = [
    "BourginMLP_B16_L0_0001", "BourginMLP_B16_L0_001", "BourginMLP_B16_L0_01",
    "BourginMLP_B32_L0_0001", "BourginMLP_B32_L0_001", "BourginMLP_B32_L0_01",
    "BourginMLP_B64_L0_0001", "BourginMLP_B64_L0_001", "BourginMLP_B64_L0_01",
]

_B3_DIRS = [
    "SparseMLP_E10_L0_0001", "SparseMLP_E10_L0_001", "SparseMLP_E10_L0_01",
    "SparseMLP_E20_L0_0001", "SparseMLP_E20_L0_001", "SparseMLP_E20_L0_01",
    "SparseMLP_E40_L0_0001", "SparseMLP_E40_L0_001", "SparseMLP_E40_L0_01",
]

_B1_LABELS = ["B16/$10^{-4}$", "B16/$10^{-3}$", "B16/$10^{-2}$",
              "B32/$10^{-4}$", "B32/$10^{-3}$", "B32/$10^{-2}$",
              "B64/$10^{-4}$", "B64/$10^{-3}$", "B64/$10^{-2}$"]

_B3_LABELS = [r"$\varepsilon=10$/$10^{-4}$", r"$\varepsilon=10$/$10^{-3}$",
              r"$\varepsilon=10$/$10^{-2}$",
              r"$\varepsilon=20$/$10^{-4}$", r"$\varepsilon=20$/$10^{-3}$",
              r"$\varepsilon=20$/$10^{-2}$",
              r"$\varepsilon=40$/$10^{-4}$", r"$\varepsilon=40$/$10^{-3}$",
              r"$\varepsilon=40$/$10^{-2}$"]


def figAblationB1():
    metrics_data = {}
    for d, lab in zip(_B1_DIRS, _B1_LABELS):
        raw = loadAblationRaw(d)
        if raw:
            metrics_data[lab] = mse100Values(raw)
    out_dir = os.path.join(media_path, "ablation_test")
    plot_metric_distribution(
        metrics_data=metrics_data,
        metric_label=r"Test MSE $\times\,100$",
        analysis_path=out_dir,
        plot_filename="plot_mse_B1_grid",
        x_max_quantile=99.0,
        show_histogram=True,
    )
    log("[fig] wrote ablation_test/plot_mse_B1_grid.pdf",
        color=bcolors.OKGREEN)


def figAblationB3():
    """
        Ablation SparseMLP includes one catastrophic-failure condition
        at (epsilon=40, eta=1e-2) that sits an order of magnitude above
        the others. Calling the MNIST helper with x_max_quantile=99
        clips that condition out cleanly without crushing the bulk.
    """
    metrics_data = {}
    for d, lab in zip(_B3_DIRS, _B3_LABELS):
        raw = loadAblationRaw(d)
        if raw:
            metrics_data[lab] = mse100Values(raw)
    out_dir = os.path.join(media_path, "ablation_test")
    plot_metric_distribution(
        metrics_data=metrics_data,
        metric_label=r"Test MSE $\times\,100$",
        analysis_path=out_dir,
        plot_filename="plot_mse_B3_grid",
        x_max_quantile=99.0,
        show_histogram=True,
    )
    log("[fig] wrote ablation_test/plot_mse_B3_grid.pdf",
        color=bcolors.OKGREEN)


# ---------------------------------------------------------------------
# Section 6 - ANOVA interaction plots + best-condition-per-seed bars
# (These figures are specific to this manuscript and have no MNIST
# counterpart; we keep dedicated implementations here.)
# ---------------------------------------------------------------------

def _buildAblationDataframe(dirs, factor1_name):
    """
        Long-format DataFrame (`mse`, `factor1_name`, `learning_rate`)
        for one ablation study. Used by the MNIST-style interaction
        plot helper. `factor1_name` is `batch_size` or `epsilon`.
    """
    import pandas as pd
    rows = []
    for d in dirs:
        raw = loadAblationRaw(d)
        if not raw:
            continue
        first = next(iter(raw.values()))
        bs = first.get("batch_size")
        eps = first.get("epsilon")
        f1_val = bs if bs is not None and eps is None else eps
        f2_val = first.get("learning_rate")
        for t in raw.values():
            rows.append({
                factor1_name: f1_val,
                "learning_rate": f2_val,
                "mse": float(t["mse"]) * 100,
            })
    return pd.DataFrame(rows)


def _runMnistInteractionPlot(df, x_var, y_var, hue_var, title, out_name):
    """
        Wraps MNIST's `generate_interaction_plot`. The MNIST helper
        saves to its own module-level `analysis_path`; we temporarily
        point that at our media directory and move the produced file
        to the manuscript's expected name.
    """
    target_dir = os.path.join(media_path, "statistics")
    os.makedirs(target_dir, exist_ok=True)
    _orig = getattr(_mnist_check_ablation, "analysis_path", None)
    _mnist_check_ablation.analysis_path = target_dir
    try:
        generate_interaction_plot(df, x_var, y_var, hue_var, title)
    finally:
        if _orig is not None:
            _mnist_check_ablation.analysis_path = _orig
    src = os.path.join(target_dir,
                       f"interaction_plot_{x_var}_{y_var}_{hue_var}.pdf")
    dst = os.path.join(target_dir, out_name)
    if os.path.exists(src) and src != dst:
        shutil.move(src, dst)


def _logInteractionPlot(df, x_var, title, out_name, x_label):
    """
        Log-y interaction plot. Replaces the MNIST linear-axis helper for
        the manuscript: with the divergent lr=1e-2 cells at MSE 15-20 a
        linear axis compresses the converged cells against the floor and
        hides both the batch-size sign reversal (B1) and the rank
        crossing (B3). Same visual grammar as the MNIST helper: jittered
        per-trial scatter + thick per-level mean line.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    target_dir = os.path.join(media_path, "statistics")
    os.makedirs(target_dir, exist_ok=True)

    x_levels = sorted(df[x_var].unique())
    hue_levels = sorted(df["learning_rate"].unique())
    xpos = {v: i for i, v in enumerate(x_levels)}
    rng = np.random.default_rng(20260522)

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for hue, colour in zip(hue_levels, ("#3C91BC", "#5CC47B", "#FFCB0F")):
        sub = df[df["learning_rate"] == hue]
        jitter = rng.uniform(-0.18, 0.18, size=len(sub))
        ax.scatter([xpos[v] for v in sub[x_var]] + jitter, sub["mse"],
                   s=8, alpha=0.15, color=colour, edgecolors="none",
                   zorder=1)
        means = [sub[sub[x_var] == v]["mse"].mean() for v in x_levels]
        ax.plot(range(len(x_levels)), means, "-o", color=colour,
                linewidth=2.5, markersize=7, zorder=3,
                label=f"{hue:g}")
    ax.set_yscale("log")
    ax.set_xticks(range(len(x_levels)))
    ax.set_xticklabels([str(v) for v in x_levels])
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"MSE $\times\,100$")
    ax.legend(title="learning rate", loc="center left",
              bbox_to_anchor=(1.01, 0.5), fontsize=10)
    ax.grid(axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(target_dir, out_name), bbox_inches="tight")
    plt.close(fig)


def figAnovaInteractionAblationA():
    df = _buildAblationDataframe(_B1_DIRS, "batch_size")
    if df.empty:
        return
    _logInteractionPlot(
        df, x_var="batch_size",
        title=r"Ablation BourginMLP: Batch Size $\mathit{vs}.$ MSE by Learning Rate",
        out_name="interaction_plot_ablation_A.pdf",
        x_label="Batch size",
    )
    log("[fig] wrote statistics/interaction_plot_ablation_A.pdf (log-y)",
        color=bcolors.OKGREEN)


def figAnovaInteractionAblationB():
    df = _buildAblationDataframe(_B3_DIRS, "epsilon")
    if df.empty:
        return
    _logInteractionPlot(
        df, x_var="epsilon",
        title=r"Ablation SparseMLP: $\varepsilon$ $\mathit{vs}.$ MSE by Learning Rate",
        out_name="interaction_plot_ablation_B.pdf",
        x_label=r"$\varepsilon$",
    )
    log("[fig] wrote statistics/interaction_plot_ablation_B.pdf (log-y)",
        color=bcolors.OKGREEN)


def figDecompHeatmaps():
    """
        35x35 init x order heatmaps for the manuscript's variance-
        decomposition section: BourginMLP (interaction-dominated,
        'checkerboard') vs ContextDepNet (init-dominated, horizontal
        stripes). Data: analysis_results/csv/results_factorial_decomp.csv.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    csv = os.path.join(os.path.dirname(output_path), "analysis_results",
                       "csv", "results_factorial_decomp.csv")
    if not os.path.exists(csv):
        csv = os.path.join("analysis_results", "csv",
                           "results_factorial_decomp.csv")
    df = pd.read_csv(csv)
    out_dir = os.path.join(media_path, "variance_decomposition")
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    for ax, model in zip(axes, ("BourginMLP", "ContextDepNet")):
        g = df[df.model == model].pivot_table(
            index="init_seed", columns="order_seed", values="mse_x100")
        im = ax.imshow(g.values, aspect="equal", cmap="viridis",
                       origin="lower")
        ax.set_title(model)
        ax.set_xlabel("order seed (index)")
        ax.set_ylabel("init seed (index)")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(r"MSE $\times\,100$")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "heatmap_init_order.pdf"))
    plt.close(fig)
    log("[fig] wrote variance_decomposition/heatmap_init_order.pdf",
        color=bcolors.OKGREEN)


def figBestCellPerSeed(dirs, factor1_name, factor1_label,
                        study_label, out_name):
    """
        Per-seed winner bar chart. Mirrors the block in
        `mnist-learning-playground/src/05_check_ablation_test.py` that
        produces `best_hp_combination_per_seed_bar_chart.pdf`: same data
        flow (group by seed, find winner, count combinations, plot with
        seaborn.barplot) and same visual style. Note the MNIST original
        uses `idxmax` because accuracy is higher-is-better; we use
        `idxmin` because MSE is lower-is-better.
    """
    import seaborn as sns
    df = _buildAblationDataframe(dirs, factor1_name)
    if df.empty:
        return
    # Add a seed column so we can group by it (paired-seed analysis).
    seeds = []
    for d in dirs:
        raw = loadAblationRaw(d)
        seeds.extend(int(t["seed"]) for t in raw.values())
    df["seed"] = seeds
    best_per_seed = df.loc[df.groupby("seed")["mse"].idxmin()]
    counts = (best_per_seed
              .groupby([factor1_name, "learning_rate"])
              .size().reset_index(name="count"))
    total = df["seed"].nunique()
    counts["percentage"] = counts["count"] / total * 100
    counts["combo_label"] = counts.apply(
        lambda row: f"{factor1_label}:{row[factor1_name]:g}\n"
                    f"LR:{row['learning_rate']:g}",
        axis=1,
    )
    counts = counts.sort_values(by="percentage", ascending=False)

    plt.figure(figsize=(12, 7))
    sns.barplot(data=counts, x="combo_label", y="percentage",
                palette=color_palette_list,
                hue="combo_label", legend=False)
    plt.title(f"{study_label}: percentage of seeds for which each "
              f"condition is best", fontsize=18)
    plt.xlabel("Hyperparameter combination", fontsize=16)
    plt.ylabel("Percentage of seeds (%)", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    full = os.path.join(media_path, out_name)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    plt.savefig(full, format="pdf", bbox_inches="tight")
    plt.close()
    log(f"[fig] wrote {out_name}", color=bcolors.OKGREEN)


def figBestCellAblationA():
    figBestCellPerSeed(_B1_DIRS, factor1_name="batch_size",
                       factor1_label="BS",
                       study_label="Ablation BourginMLP",
                       out_name="ablation_test/best_cell_per_seed_A.pdf")


def figBestCellAblationB():
    figBestCellPerSeed(_B3_DIRS, factor1_name="epsilon",
                       factor1_label=r"$\varepsilon$",
                       study_label="Ablation SparseMLP",
                       out_name="ablation_test/best_cell_per_seed_B.pdf")


# ---------------------------------------------------------------------
# Section 7 - sampling probability + survival + sampling error
# (Delegated to the MNIST companion helpers. MSE is sign-flipped before
# the call because those helpers assume higher-is-better semantics.)
# ---------------------------------------------------------------------

def figSamplingProbabilityP10():
    """
        Per-model probability of obtaining at least one run in the
        model's OWN top decile (P10 of MSE) within N trials. This is the
        same statistic as the MNIST paper's sampling-probability figure:
        because each model is measured against its own P10, the curve is
        ~1-0.9^N for every model and they nearly coincide. Implemented
        via MNIST's `plot_all_percentile_probabilities`, called with
        sign-flipped MSE so its higher-is-better convention picks the
        bottom 10% of MSE as "successes". Outputs analytical / bootstrap
        / montecarlo variants; we keep the bootstrap one for the
        manuscript.
    """
    flipped = {m: signFlippedRaw(loadMainR2Raw(m))
               for m in ("LinearLogistic", "BourginMLP", "ContextDepNet",
                         "SparseMLP", "PsychForest", "BEAST_GB")
               if loadMainR2Raw(m)}
    out_dir = os.path.join(media_path, "standardizing_comparisons")
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    plot_all_percentile_probabilities(
        metrics_data=flipped,
        analysis_path=out_dir,
        percentile=90,           # top 10% of -MSE = bottom 10% of MSE
        metric="neg_mse",
        color_list=color_palette_list,
    )
    # Rename the bootstrap output to the filename the manuscript expects
    # and remove the analytical / montecarlo variants we are not
    # citing in the paper.
    _renameOutput(out_dir, "sampling_probability_p90_bootstrap.pdf",
                  "standardizing_comparisons",
                  "sampling_probability_p10_bootstrap.pdf")
    for stale in ("sampling_probability_p90_analytical.pdf",
                  "sampling_probability_p90_montecarlo.pdf"):
        path = os.path.join(out_dir, stale)
        if os.path.exists(path):
            os.remove(path)


def figSurvivalCombined():
    """
        Empirical survival function (P(X > x_p) at each percentile p of
        the fitted normal). MNIST's `plot_survival_function` takes a
        per-model dict of floats; we pass MSE values directly since the
        survival semantics are symmetric under sign and the comparison
        is against the fitted normal of the same data.
    """
    metrics_data = {m: mse100Values(loadMainR2Raw(m))
                    for m in ("LinearLogistic", "BourginMLP", "ContextDepNet",
                              "SparseMLP", "PsychForest", "BEAST_GB")
                    if loadMainR2Raw(m)}
    out_dir = os.path.join(media_path, "standardizing_comparisons")
    os.makedirs(out_dir, exist_ok=True)
    plot_survival_function(
        metrics_data=metrics_data,
        analysis_path=out_dir,
        metric_label=r"Test MSE $\times\,100$",
        plot_prefix="survival_combined",
    )


def figSamplingErrorMean():
    """
        Sampling error of the mean (analytical sigma / sqrt(N)) per
        model. Delegated to MNIST's `plot_all_sampling_errors`. Sign
        does not matter for the second moment so we pass MSE as-is.
    """
    metrics_data = {m: loadMainR2Raw(m)
                    for m in ("LinearLogistic", "BourginMLP", "ContextDepNet",
                              "SparseMLP", "PsychForest", "BEAST_GB")
                    if loadMainR2Raw(m)}
    out_dir = os.path.join(media_path, "strengthening_conclusions")
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    plot_all_sampling_errors(
        metrics_data=metrics_data,
        analysis_path=out_dir,
        metric="mse",
        color_list=color_palette_list,
    )
    # The MNIST helper no longer emits the `_analytical` variant this
    # rename used to pick up (that silent no-op is how a stale
    # `sampling_error_mean_combined.pdf` with the pre-canonical BEAST-GB
    # sigma survived in the manuscript). The manuscript figure is now
    # produced directly:
    _samplingErrorCombined(metrics_data, out_dir)


def _samplingErrorCombined(metrics_data, out_dir):
    """Paper-styled sampling-error figure: per-model sigma/sqrt(N) on a
    log axis, per-model sigma in the legend."""
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    Ns = np.arange(1, 31)
    for m, raw in metrics_data.items():
        vals = np.asarray(mse100Values(raw))
        sigma = vals.std(ddof=1)
        ax.plot(Ns, sigma / np.sqrt(Ns), "-o", markersize=4,
                color=model_colours.get(m, None),
                label=rf"{m} ($\sigma = {sigma:.3f}$)")
    ax.set_yscale("log")
    ax.set_xlabel(r"Number of training runs $N$")
    ax.set_ylabel(r"Sampling error of the mean $\sigma/\sqrt{N}$"
                  "\n" r"(MSE $\times\,100$ units)")
    ax.grid(which="both", alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sampling_error_mean_combined.pdf"))
    plt.close(fig)
    log("[fig] wrote strengthening_conclusions/sampling_error_mean_combined.pdf",
        color=bcolors.OKGREEN)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def run():
    if enable_intro_distributions:
        figIntroBourginMLPMse()
        figIntroBourginMLPBestEpoch()
    if enable_overlap_plots:
        figOverlapTwoModels()
        figOverlapSixModels()
    if enable_best_of_n:
        figBestOfN()
    if enable_informed_training_panels:
        figInformedTrainingPanels()
    if enable_ablation_grids:
        figAblationB1()
        figAblationB3()
    if enable_anova_interaction_plots:
        figAnovaInteractionAblationA()
        figAnovaInteractionAblationB()
    if enable_best_cell_per_seed:
        figBestCellAblationA()
        figBestCellAblationB()
    if enable_sampling_probability:
        figSamplingProbabilityP10()
    if enable_survival_combined:
        figSurvivalCombined()
    if enable_sampling_error:
        figSamplingErrorMean()
    if enable_decomp_heatmaps:
        figDecompHeatmaps()


if __name__ == "__main__":
    run()
