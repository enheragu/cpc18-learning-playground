#!/usr/bin/env python3
# encoding: utf-8

"""
    Generates the manuscript figures into analysis_results/figures/,
    one subfolder per manuscript section.

    The plotting heavy-lifting is delegated to the MNIST companion
    submodule's helpers (`utils.plot_distribution`,
    `utils.plot_sampling_graph`) so that both papers' figures share the
    same style and font conventions. The wrappers in this script handle
    only:

      * loading the per-trial YAML files,
      * converting the lower-is-better MSE into the higher-is-better
        convention expected by the MNIST sampling helpers (sign flip),
      * routing output files into analysis_results/figures/ and, with
        --media, copying into a LaTeX project's media/ folder only the
        figures its .tex sources include.

    Usage:
        python 05_make_paper_figures.py [--media /path/to/latex_project/media]
"""

import argparse
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
    plot_metric_gammadistribution,
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
from cpc18_utils.training_logs import loadTrainingLog


_SAMPLING_SEED = 20260522  # Monte Carlo / Bootstrap of the sampling figures and their tables

figures_path = os.path.join(os.path.dirname(output_path), "analysis_results", "figures")


def _displayName(m):
    """Legend display name: BEAST_GB -> BEAST-GB (matches captions/body)."""
    return m.replace("_", "-")


# The MNIST helper `plot_metric_distribution` draws on a fixed
# 14x11-inch canvas with default 10 pt fonts; scaled to the manuscript's
# column widths that yields 2-4 pt text. These rc profiles enlarge the
# fonts in proportion to the width each figure occupies in the
# manuscript (the helper itself is not modified). Effective printed
# sizes land at roughly 5-8 pt.
_RC_HELPER = {
    # \includegraphics width = .49\textwidth (intro pair, ablation
    # nine-condition grids)
    "half":  {"font.size": 34, "axes.titlesize": 34, "axes.labelsize": 42,
              "xtick.labelsize": 36, "ytick.labelsize": 36,
              "legend.fontsize": 30},
    # width = .327\textwidth (informed-training three-panel row)
    "third": {"font.size": 44, "axes.titlesize": 44, "axes.labelsize": 56,
              "xtick.labelsize": 48, "ytick.labelsize": 48,
              "legend.fontsize": 44},
    # width = .95\textwidth (init x order heatmap row)
    "wide":  {"font.size": 22, "axes.titlesize": 22, "axes.labelsize": 26,
              "xtick.labelsize": 22, "ytick.labelsize": 22,
              "legend.fontsize": 18},
}


# Per-model colour palette, shared by every figure that shows models so
# that each model keeps its colour across the manuscript. It follows the
# order of `color_palette_list` over the six-model pool, which is what
# the six-model overlap and best-of-N figures already used.
model_colours = {
    "LinearLogistic": c_blue,
    "BourginMLP":     c_green,
    "ContextDepNet":  c_yellow,
    "SparseMLP":      c_red,
    "PsychForest":    c_purple,
    "BEAST_GB":       c_grey,
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
# The Gaussian-tail calibration plot (P(X > x_p) vs percentile) that
# `figSurvivalCombined` produces was shipped to the manuscript under a
# caption describing a trials-to-target curve; the manuscript now uses
# `figTrialsToTarget` below, which actually plots that curve.
enable_survival_combined        = False
enable_trials_to_target         = True
enable_sampling_error           = True
enable_decomp_heatmaps          = True
enable_cross_machine            = True
enable_target_family            = True


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def loadMainR2Raw(model_name):
    """ Returns the raw `{timestamp: entry}` dict (MNIST-helper format). """
    p = os.path.join(output_path, model_name,
                     "sharedpool_r2_training_metrics.yaml")
    return loadTrainingLog(p)


def loadAblationRaw(model_dir):
    p = os.path.join(output_path, model_dir,
                     "sharedpool_r2_ablation_training_metrics.yaml")
    return loadTrainingLog(p)


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
        naming scheme. We move them under figures_path/
        subfolders and rename them to match the manuscript's
        `\\includegraphics` calls.
    """
    src = os.path.join(src_dir, src_name)
    dst_dir = os.path.join(figures_path, dst_subdir)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, dst_name)
    if os.path.exists(src):
        shutil.move(src, dst)
        log(f"[fig] wrote {dst_subdir}/{dst_name}", color=bcolors.OKGREEN)


# ---------------------------------------------------------------------
# Sections 1 and 3.2 - introduction and best-of-N figures (use plot_metric_distribution)
# ---------------------------------------------------------------------

def figIntroBourginMLPMse():
    raw = loadMainR2Raw("BourginMLP")
    if not raw:
        return
    vals = np.asarray(mse100Values(raw), dtype=float)
    out_dir = os.path.join(figures_path, "introduction")
    os.makedirs(out_dir, exist_ok=True)
    # Empirical histogram over the full range on a log axis, with no
    # fitted curve: the upper tail of BourginMLP (Fisher kurtosis ~32) is
    # not described by a Gaussian or a lognormal (Section 3.1), and
    # clipping the axis would hide the range the text quotes. Bins are
    # equal-width in log space, so the y-axis shows trial counts and bar
    # area stays proportional to the number of trials. Canvas and fonts
    # match panel (b), which the MNIST helper draws.
    colour = model_colours["BourginMLP"]
    with plt.rc_context(_RC_HELPER["half"]):
        fig, ax = plt.subplots(figsize=(14, 11))
        x_lo, x_hi = 0.98 * vals.min(), 1.02 * vals.max()
        ax.hist(vals, bins=np.geomspace(x_lo, x_hi, 61), color=colour,
                alpha=0.4, edgecolor="none",
                label=f"BourginMLP   (n={len(vals)})")
        for q in (5, 25, 50, 75, 95):
            ax.axvline(np.percentile(vals, q), color=colour,
                       linewidth=2 if q == 50 else 1,
                       linestyle="solid" if q == 50 else ":")
        ax.set_xscale("log")
        ticks = [0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:.1f}" for t in ticks])
        ax.minorticks_off()
        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel(r"Test MSE $\times\,100$")
        ax.set_ylabel("Trials")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "plot_mse_BourginMLP_r2.pdf"),
                    format="pdf")
        plt.close(fig)
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
    out_dir = os.path.join(figures_path, "introduction")
    # The manuscript caption describes a Gamma fit (right-tailed,
    # positive support); the helper's default plot_func is the normal
    # fit, so it must be overridden explicitly here.
    with plt.rc_context(_RC_HELPER["half"]):
        plot_metric_distribution(
            metrics_data={"BourginMLP": epochs},
            metric_label="Best epoch",
            analysis_path=out_dir,
            plot_filename="plot_best_epoch_BourginMLP_r2",
            plot_func=plot_metric_gammadistribution,
            color_palette=[model_colours["BourginMLP"]],
            title="",
            x_max_quantile=99.0,
            show_histogram=True,
            # The auto-placement in plot_metric_distribution rejects every
            # corner here (the Gamma curve's bounding box spans nearly the
            # full width) and falls back to an outside-the-axes legend,
            # which breaks the match with panel (a). Upper-left sits over
            # the histogram's short left flank, clear of the tall bars
            # near the peak (x ~ 150-250).
            force_legend_loc="upper left",
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
    # Same renderer as the six-model panel stacked below it in the
    # manuscript, so both panels share bins, colours and font scale. The
    # MNIST helper bins each model over its own full range, which leaves
    # BourginMLP with a handful of coarse bars once the axis is clipped.
    for subdir in ("introduction", "replication_comparison"):
        out_dir = os.path.join(figures_path, subdir)
        os.makedirs(out_dir, exist_ok=True)
        fname = ("plot_mse_overlap_BourginMLP_PsychForest.pdf"
                 if subdir == "introduction"
                 else "plot_mse_BourginMLP_PsychForest.pdf")
        _plotOverlapBrokenAxis(metrics_data, os.path.join(out_dir, fname))
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
        color = model_colours[n]
        mean, std = a.mean(), a.std()
        label = f"{_displayName(n)} (n={len(a)})"
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

    ax_bot.set_xlabel(r"Test MSE $\times\,100$", fontsize=20)
    fig.text(0.02, 0.5, "Density", va="center", rotation="vertical",
             fontsize=20)
    for ax in (ax_top, ax_bot):
        ax.tick_params(labelsize=17)
    ax_bot.set_xlim(x_lo, x_hi)
    # Legend in the top panel: its right half is empty (the spikes sit on
    # the left), so it does not overlap any distribution. Two columns so
    # the six entries stay inside the (short) top panel.
    handles, labels = ax_bot.get_legend_handles_labels()
    ax_top.legend(handles, labels, ncol=2, fontsize=13.5, framealpha=0.9,
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
        out_dir = os.path.join(figures_path, subdir)
        os.makedirs(out_dir, exist_ok=True)
        fname = ("plot_mse_overlap_six_models.pdf"
                 if subdir == "introduction" else "plot_mse_six_models.pdf")
        _plotOverlapBrokenAxis(metrics_data, os.path.join(out_dir, fname))
    log("[fig] wrote six-model overlap with broken y-axis (both sections)",
        color=bcolors.OKGREEN)


def figMseRunsSixModels():
    """
        Test MSE x 100 of every run of the main experiment: one dot per run,
        one row per model, sorted by mean. The vertical position of a dot
        within its row is random (fixed seed) and only separates the dots.
        The solid line marks the mean, as in the other distribution figures,
        and the dashed line the median. Runs beyond the end of the axis are
        counted, not drawn.
    """
    axis_end = 1.8
    runs = {m: np.asarray(mse100Values(loadMainR2Raw(m))) for m in model_colours}
    order = sorted(runs, key=lambda m: runs[m].mean())
    rng = np.random.default_rng(_SAMPLING_SEED)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for row, m in enumerate(reversed(order)):
        values = runs[m]
        shown = values[values <= axis_end]
        ax.scatter(shown, row + rng.uniform(-0.3, 0.3, shown.size), s=7,
                   color=model_colours[m], alpha=0.45, linewidths=0,
                   rasterized=True)
        ax.plot([values.mean()] * 2, [row - 0.4, row + 0.4], color="black",
                linewidth=1.6, label="Mean" if row == 0 else None)
        ax.plot([np.median(values)] * 2, [row - 0.4, row + 0.4], color="black",
                linewidth=1.2, linestyle=(0, (2, 1)),
                label="Median" if row == 0 else None)
        beyond = values.size - shown.size
        if beyond:
            ax.text(axis_end + 0.02, row, f"+{beyond} runs,\nup to {values.max():.1f}",
                    va="center", fontsize=9.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_displayName(m) for m in reversed(order)])
    ax.set_xlim(0.5, axis_end + 0.2)
    ax.set_xticks(np.arange(0.6, axis_end + 0.01, 0.2))
    ax.set_xlabel(r"Test MSE $\times$ 100", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(axis="x", color="0.85", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=11)
    out_dir = os.path.join(figures_path, "introduction")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "plot_mse_runs_six_models.pdf"),
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    log("[fig] wrote introduction/plot_mse_runs_six_models.pdf",
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
        ax.plot(Ns, emin, label=_displayName(m),
                color=model_colours[m],
                linewidth=1.8)
    ax.set_xlabel(r"Best-of-$N$ budget (runs per model)", fontsize=14)
    ax.set_ylabel(r"E[best of $N$ runs], MSE $\times$ 100", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11)
    out_dir = os.path.join(figures_path, "replication_comparison")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "best_of_n.pdf"), bbox_inches="tight")
    plt.close(fig)
    log("[fig] wrote replication_comparison/best_of_n.pdf",
        color=bcolors.OKGREEN)


# ---------------------------------------------------------------------
# Informed-training panels
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
    out_dir = os.path.join(figures_path, "informed_training")
    for tag, subset, _bias_colour in subsets:
        with plt.rc_context(_RC_HELPER["third"]):
            plot_metric_distribution(
                metrics_data={"BourginMLP": list(subset.astype(float))},
                metric_label=r"Test MSE $\times\,100$",
                analysis_path=out_dir,
                plot_filename=f"example_BourginMLP_{tag}",
                plot_mean=mu,
                plot_std=sigma,
                title="",
                x_max_quantile=99.0,
                show_histogram=True,
            )
    log("[fig] wrote informed_training panels (3)",
        color=bcolors.OKGREEN)


# ---------------------------------------------------------------------
# Section 3.4 - ablation grids
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
    out_dir = os.path.join(figures_path, "ablation_test")
    with plt.rc_context(_RC_HELPER["half"]):
        plot_metric_distribution(
            metrics_data=metrics_data,
            metric_label=r"Test MSE $\times\,100$",
            analysis_path=out_dir,
            plot_filename="plot_mse_B1_grid",
            title="",
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
    out_dir = os.path.join(figures_path, "ablation_test")
    with plt.rc_context(_RC_HELPER["half"]):
        plot_metric_distribution(
            metrics_data=metrics_data,
            metric_label=r"Test MSE $\times\,100$",
            analysis_path=out_dir,
            plot_filename="plot_mse_B3_grid",
            title="",
            x_max_quantile=99.0,
            show_histogram=True,
        )
    log("[fig] wrote ablation_test/plot_mse_B3_grid.pdf",
        color=bcolors.OKGREEN)


# ---------------------------------------------------------------------
# Section 3.4 - ANOVA interaction plots + best-condition-per-seed bars
# (and the Section 3.3 init x order heatmaps)
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
    target_dir = os.path.join(figures_path, "statistics")
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

    target_dir = os.path.join(figures_path, "statistics")
    os.makedirs(target_dir, exist_ok=True)

    x_levels = sorted(df[x_var].unique())
    hue_levels = sorted(df["learning_rate"].unique())
    xpos = {v: i for i, v in enumerate(x_levels)}
    rng = np.random.default_rng(20260522)

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for hue, colour in zip(hue_levels, (c_blue, c_green, c_yellow)):   # series palette (GUI), not the LaTeX p* hex
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
    ax.set_xlabel(x_label, fontsize=21)
    ax.set_ylabel(r"MSE $\times\,100$", fontsize=21)
    ax.tick_params(labelsize=18)
    leg = ax.legend(title="learning rate", loc="center left",
                    bbox_to_anchor=(1.01, 0.5), fontsize=18)
    leg.get_title().set_fontsize(18)
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
        'checkerboard'), SparseMLP (intermediate regime), ContextDepNet
        (init-dominated, horizontal stripes). Data:
        analysis_results/csv/results_factorial_decomp.csv.

        Uses the "wide" RC profile (calibrated for the .85-.95\\textwidth
        figures elsewhere in this script) instead of the module-level
        default (font.size=24, set by the MNIST plot_distribution import
        for its own fixed 14x11in canvas): without it this hand-built
        imshow figure inherits that oversized default and prints far
        larger than the surrounding body text.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    csv = os.path.join(os.path.dirname(output_path), "analysis_results",
                       "csv", "results_factorial_decomp.csv")
    if not os.path.exists(csv):
        csv = os.path.join("analysis_results", "csv",
                           "results_factorial_decomp.csv")
    df = pd.read_csv(csv, float_precision="round_trip")
    out_dir = os.path.join(figures_path, "variance_decomposition")
    os.makedirs(out_dir, exist_ok=True)

    models = ("BourginMLP", "SparseMLP", "ContextDepNet")
    with plt.rc_context(_RC_HELPER["wide"]):
        fig, axes = plt.subplots(1, 3, figsize=(15.3, 4.8))
        for ax, model in zip(axes, models):
            g = df[df.model == model].pivot_table(
                index="init_seed", columns="order_seed", values="mse_x100")
            # Reversed map: the metric is lower-is-better, so bright must be
            # good. With plain viridis the brightest cells are the worst
            # runs, which reads backwards.
            im = ax.imshow(g.values, aspect="equal", cmap="viridis_r",
                           origin="lower")
            ax.set_title(_displayName(model))
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
        lambda row: f"{factor1_label}$=${row[factor1_name]:g}\n"
                    rf"$\alpha=10^{{{round(np.log10(row['learning_rate']))}}}$",
        axis=1,
    )
    counts = counts.sort_values(by="percentage", ascending=False)

    # The study name lives in the LaTeX caption; no in-figure title.
    plt.figure(figsize=(8.0, 5.0))
    sns.barplot(data=counts, x="combo_label", y="percentage",
                palette=color_palette_list, saturation=1,   # keep the palette hex (seaborn desaturates to 0.75 by default)
                hue="combo_label", legend=False)
    plt.xlabel("Hyperparameter combination", fontsize=18)
    plt.ylabel("Percentage of seeds (%)", fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    full = os.path.join(figures_path, out_name)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    plt.savefig(full, format="pdf", bbox_inches="tight")
    plt.close()
    log(f"[fig] wrote {out_name}", color=bcolors.OKGREEN)


def figBestCellAblationA():
    figBestCellPerSeed(_B1_DIRS, factor1_name="batch_size",
                       factor1_label="$B$",
                       study_label="Ablation BourginMLP",
                       out_name="ablation_test/best_cell_per_seed_A.pdf")


def figBestCellAblationB():
    figBestCellPerSeed(_B3_DIRS, factor1_name="epsilon",
                       factor1_label=r"$\varepsilon$",
                       study_label="Ablation SparseMLP",
                       out_name="ablation_test/best_cell_per_seed_B.pdf")


# ---------------------------------------------------------------------
# Sampling probability, survival, trials to target (Section 4) and sampling error
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
    out_dir = os.path.join(figures_path, "standardizing_comparisons")
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    np.random.seed(_SAMPLING_SEED)
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
    out_dir = os.path.join(figures_path, "standardizing_comparisons")
    os.makedirs(out_dir, exist_ok=True)
    plot_survival_function(
        metrics_data=metrics_data,
        analysis_path=out_dir,
        metric_label=r"Test MSE $\times\,100$",
        plot_prefix="survival_combined",
    )


def figTrialsToTarget():
    """
        Paper-styled fixed-target view: expected number of training
        trials needed to reach a given MSE x 100 target for the first
        time, per model, as a function of the target. With p(t) the
        empirical fraction of the model's N=800 trials at or below t,
        the first hit is geometric and E[trials] = 1/p(t). A model never
        reaches targets below its observed floor, so each curve starts
        at the model's minimum.
    """
    metrics_data = {m: mse100Values(loadMainR2Raw(m))
                    for m in ("LinearLogistic", "BourginMLP", "ContextDepNet",
                              "SparseMLP", "PsychForest", "BEAST_GB")
                    if loadMainR2Raw(m)}
    out_dir = os.path.join(figures_path, "standardizing_comparisons")
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    targets = np.linspace(0.52, 1.20, 600)
    for m, vals in metrics_data.items():
        vals = np.asarray(vals)
        p_hit = np.array([(vals <= t).mean() for t in targets])
        expected = np.full_like(targets, np.nan)
        reachable = p_hit > 0
        expected[reachable] = 1.0 / p_hit[reachable]
        ax.plot(targets, expected, "-",
                color=model_colours.get(m, None),
                label=m.replace("_", "-"))
    ax.set_yscale("log")
    ax.set_xlabel(r"Target test MSE $\times\,100$ (lower is better)",
                  fontsize=14)
    ax.set_ylabel("Expected number of trials\nto first reach the target",
                  fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(which="both", alpha=0.25)
    ax.legend(ncol=2, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "trials_to_target_combined.pdf"))
    plt.close(fig)
    log("[fig] wrote standardizing_comparisons/trials_to_target_combined.pdf",
        color=bcolors.OKGREEN)


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
    out_dir = os.path.join(figures_path, "strengthening_conclusions")
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    np.random.seed(_SAMPLING_SEED)
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
                label=rf"{_displayName(m)} ($\sigma = {sigma:.3f}$)")
    ax.set_yscale("log")
    ax.set_xlabel(r"Number of training runs $N$", fontsize=13)
    ax.set_ylabel(r"Sampling error of the mean $\sigma/\sqrt{N}$"
                  "\n" r"(MSE $\times\,100$ units)", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(which="both", alpha=0.25)
    ax.legend(ncol=2, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sampling_error_mean_combined.pdf"))
    plt.close(fig)
    log("[fig] wrote strengthening_conclusions/sampling_error_mean_combined.pdf",
        color=bcolors.OKGREEN)


# ---------------------------------------------------------------------
# Entry point

# ---------------------------------------------------------------------
# Section 3.6 and Appendix H - cross-machine replication
# ---------------------------------------------------------------------

# The five distinct arithmetic paths of the study, in the order used by
# `tab:cross_machine:reporting`. Each tag is the identifier of the machine
# in `machines.csv` of the data deposit (listed in the order of the machine
# table of Appendix H); its trials are read from `output_data_<tag>/` or from
# the deposit. The AVX512 reference row is the tree of the comparison machine
# that is bit-identical to the reference, so all five rows are loaded the same
# way and no row is a special case.
_PATHS = [
    ("M01",                 "AVX512\nIntel",     c_blue),
    ("M03",                 "AVX2\nIntel",       c_green),
    ("C02",                 "AVX2 label\nAMD Zen 2", c_red),
    ("M03-forced-DEFAULT",  "DEFAULT\nx86",      c_yellow),
    ("M06",                 "DEFAULT\nARM",      c_purple),
]
_XMACHINE_COND = "BourginMLP_B32_L0_001"
_XMACHINE_YAML = "sharedpool_r2_ablation_training_metrics.yaml"
_XMACHINE_BOOT = 5000
_XMACHINE_SEED = 20260522


def _loadTaggedTrials(tag, condition, yaml_name):
    """seed -> MSE x 100 for one condition of one tagged output tree."""
    path = os.path.join(os.path.dirname(output_path),
                        f"output_data_{tag}", condition, yaml_name)
    return {int(e["seed"]): float(e["mse"]) * 100
            for e in loadTrainingLog(path).values() if "seed" in e}


def _p10BootCi(values, n=_XMACHINE_BOOT, seed=_XMACHINE_SEED):
    """Bootstrap 95 % CI of the lognormal-fitted P10 (the reported target).

    lognorm.fit(x, floc=0) is the MLE of the log-values (mean and ddof=0
    std), so each resample is refitted in closed form.
    """
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    logs = np.log(values[idx])
    draws = np.exp(logs.mean(axis=1) + norm.ppf(_FAMILY_P) * logs.std(axis=1))
    return np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def figCrossMachine():
    """Two panels carrying the argument of Section 3.6 (figure in Appendix H).

    (a) Per-seed values do not transfer: reference against the AMD cell,
        one point per shared seed, with the identity line. A tight cloud
        on the identity line would mean the seed reproduces; it does not.
    (b) The distribution does transfer while the best run does not: the
        shift of P10 and of the best run on each path, relative to the
        reference, against the band that the finite seed sample already
        contributes at N = 800.
    """
    series = []
    for tag, label, colour in _PATHS:
        try:
            series.append((label, colour,
                           _loadTaggedTrials(tag, _XMACHINE_COND, _XMACHINE_YAML)))
        except FileNotFoundError:
            log(f"[fig] cross-machine: missing tree for {tag}", color=bcolors.WARNING)
            return
    common = sorted(set.intersection(*[set(d) for _, _, d in series]))
    if len(common) < 100:
        log("[fig] cross-machine: too few shared seeds", color=bcolors.WARNING)
        return

    ref = np.array([series[0][2][s] for s in common])
    ref_p10 = _qLognormal(ref)
    lo, hi = _p10BootCi(ref)
    band = 100 * (hi - lo) / 2 / ref_p10

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.0, 5.4))

    # --- (a) per-seed scatter, reference against the AMD cell ---------
    amd = np.array([series[2][2][s] for s in common])
    ax_a.scatter(ref, amd, s=12, alpha=0.45, color=c_red, edgecolors="none")
    lim = [0.95 * min(ref.min(), amd.min()), 1.05 * max(ref.max(), amd.max())]
    ax_a.plot(lim, lim, "-", color=c_darkgrey, linewidth=1.4, zorder=3,
              label="identical result")
    ax_a.set_xscale("log"); ax_a.set_yscale("log")
    ax_a.set_xlim(lim); ax_a.set_ylim(lim)
    ax_a.set_xlabel(r"MSE $\times\,100$, AVX512 Intel (reference)", fontsize=15)
    ax_a.set_ylabel(r"MSE $\times\,100$, AMD Zen 2", fontsize=15)
    ax_a.tick_params(labelsize=13)
    ax_a.legend(fontsize=13, loc="upper left")
    ax_a.grid(alpha=0.25, which="both")

    # --- (b) P10 and best run per path, relative to the reference -----
    xs = np.arange(len(series))
    dp10, dp10_err, dbest = [], [[], []], []
    for _, _, d in series:
        v = np.array([d[s] for s in common])
        p10 = _qLognormal(v)
        blo, bhi = _p10BootCi(v)
        dp10.append(100 * (p10 / ref_p10 - 1))
        dp10_err[0].append(100 * (p10 - blo) / ref_p10)
        dp10_err[1].append(100 * (bhi - p10) / ref_p10)
        dbest.append(100 * (v.min() / ref.min() - 1))

    ax_b.axhspan(-band, band, color=c_grey, alpha=0.18, zorder=0,
                 label=f"seed-sample noise on $P_{{10}}$ ($\\pm${band:.1f}%)")
    ax_b.axhline(0.0, color=c_darkgrey, linewidth=0.9, zorder=1)
    ax_b.errorbar(xs - 0.09, dp10, yerr=dp10_err, fmt="o", markersize=9,
                  color=c_blue, capsize=4, linewidth=1.6, zorder=3,
                  label=r"$P_{10}$")
    ax_b.plot(xs + 0.09, dbest, "^", markersize=10, color=c_red, zorder=3,
              label="best run")
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels([lab for lab, _, _ in series], fontsize=12)
    ax_b.set_ylabel("change vs reference (%)", fontsize=15)
    ax_b.tick_params(axis="y", labelsize=13)
    ax_b.legend(fontsize=12, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, framealpha=0.95)
    ax_b.grid(axis="y", alpha=0.25)
    ymax = max(max(dbest), band) * 1.35
    ymin = min(min(dbest), min(np.array(dp10) - np.array(dp10_err[0])), -band) * 1.6
    ax_b.set_ylim(ymin, ymax)

    fig.tight_layout()
    out_dir = os.path.join(figures_path, "cross_machine")
    os.makedirs(out_dir, exist_ok=True)
    full = os.path.join(out_dir, "cross_machine_transfer.pdf")
    fig.savefig(full, format="pdf", bbox_inches="tight")
    plt.close(fig)
    log(f"[fig] wrote cross_machine/cross_machine_transfer.pdf "
        f"(n={len(common)} shared seeds)", color=bcolors.OKGREEN)


# ---------------------------------------------------------------------
# Section 3.1 - choosing and validating the fitted family
# ---------------------------------------------------------------------

_FAMILY_P = 0.10
_FAMILY_REPS = 2000
_FAMILY_SEED = 20260522


def _qEmpirical(v):
    return np.percentile(v, 100 * _FAMILY_P)


def _qNormal(v):
    from scipy.stats import norm
    return v.mean() + norm.ppf(_FAMILY_P) * v.std(ddof=1)


def _qLognormal(v):
    from scipy.stats import lognorm
    sh, lo, sc = lognorm.fit(v, floc=0)
    return lognorm.ppf(_FAMILY_P, sh, lo, sc)


def figTargetFamily():
    """Two panels supporting the choice of fitted family in Section 3.1.

    (a) Error of the estimated target against its full-sample value, as a
        function of how many trials are available, for the empirical
        percentile and for two fitted families. This is the argument for
        reading the target from a fit at all, and for preferring the
        lognormal on a zero-bounded metric.
    (b) The architecture where the fit fails, with its fitted target
        marked against the observed distribution, so the failure is
        visible rather than only tabulated.
    """
    from scipy.stats import lognorm
    import warnings
    warnings.filterwarnings("ignore")

    vals = {m: np.asarray(mse100Values(loadMainR2Raw(m))) for m in
            ("ContextDepNet", "BourginMLP")}
    if not all(len(v) > 100 for v in vals.values()):
        log("[fig] target-family: not enough trials", color=bcolors.WARNING)
        return

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # --- (a) estimation error against sample size -------------------
    v = vals["ContextDepNet"]
    truth = _qEmpirical(v)
    rng = np.random.default_rng(_FAMILY_SEED)
    Ns = [5, 10, 20, 50, 100]
    series = [("empirical percentile", _qEmpirical, c_grey, "o--"),
              ("Gaussian fit", _qNormal, c_blue, "s-"),
              ("lognormal fit", _qLognormal, c_green, "^-")]
    for name, fn, colour, style in series:
        rmse = []
        for N in Ns:
            idx = rng.integers(0, len(v), size=(_FAMILY_REPS, N))
            est = np.array([fn(v[i]) for i in idx])
            rel = 100 * (est / truth - 1)
            rmse.append(np.sqrt((rel ** 2).mean()))
        ax_a.plot(Ns, rmse, style, color=colour, label=name,
                  linewidth=2.0, markersize=7)
    ax_a.set_xscale("log")
    ax_a.set_xticks(Ns)
    ax_a.set_xticklabels([str(n) for n in Ns])
    ax_a.set_xlabel("trials available", fontsize=15)
    ax_a.set_ylabel("error of the estimated target (%)", fontsize=15)
    ax_a.tick_params(labelsize=13)
    ax_a.legend(fontsize=12)
    ax_a.grid(alpha=0.25, which="both")

    # --- (b) the architecture whose fit does not validate ------------
    v = vals["BourginMLP"]
    q = _qLognormal(v)
    clip = np.percentile(v, 98)
    ax_b.hist(v[v <= clip], bins=60, color=c_grey, alpha=0.55,
              edgecolor="none", label="observed trials")
    sh, lo, sc = lognorm.fit(v, floc=0)
    xs = np.linspace(v.min() * 0.8, clip, 400)
    pdf = lognorm.pdf(xs, sh, lo, sc)
    # expected count per bin: density x total trials x bin width
    ax_b.plot(xs, pdf * len(v) * (clip - v.min()) / 60,
              color=c_green, linewidth=2.0, label="fitted lognormal")
    ax_b.axvline(q, color=c_red, linewidth=2.0, linestyle="-",
                 label=f"fitted target ({q:.3f})")
    ax_b.axvline(v.min(), color=c_darkgrey, linewidth=1.4,
                 linestyle=":", label=f"best trial ({v.min():.3f})")
    ax_b.set_xlabel(r"MSE $\times\,100$", fontsize=15)
    ax_b.set_ylabel("trials", fontsize=15)
    ax_b.tick_params(labelsize=13)
    ax_b.legend(fontsize=11)
    ax_b.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out_dir = os.path.join(figures_path, "informed_training")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "target_family.pdf"),
                format="pdf", bbox_inches="tight")
    plt.close(fig)
    log("[fig] wrote informed_training/target_family.pdf",
        color=bcolors.OKGREEN)

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Copying into a manuscript
# ---------------------------------------------------------------------

_INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{media/([^}]+)\}")


def manuscriptFigures(latex_project):
    """Figures under media/ that any .tex file of the LaTeX project includes."""
    used = set()
    for root, _, files in os.walk(latex_project):
        for name in files:
            if name.endswith(".tex"):
                with open(os.path.join(root, name), encoding="utf-8", errors="ignore") as fh:
                    used.update(_INCLUDEGRAPHICS_RE.findall(fh.read()))
    return {rel if os.path.splitext(rel)[1] else rel + ".pdf" for rel in used}


def copyUsedFigures(media_dir):
    """
        Copy from figures_path into `media_dir` only the figures that the
        LaTeX project owning `media_dir` includes, so the manuscript never
        receives unused or duplicated outputs.
    """
    media_dir = os.path.abspath(media_dir)
    used = manuscriptFigures(os.path.dirname(media_dir))
    missing = []
    for rel in sorted(used):
        src = os.path.join(figures_path, rel)
        if not os.path.exists(src):
            missing.append(rel)
            continue
        dst = os.path.join(media_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    log(f"[fig] copied {len(used) - len(missing)} of {len(used)} included figures into {media_dir}",
        color=bcolors.OKGREEN)
    for rel in missing:
        log(f"[fig] included by the manuscript but not generated here: {rel}",
            color=bcolors.WARNING)


def run():
    for subdir in ("introduction", "replication_comparison", "informed_training", "ablation_test", "statistics",
                   "variance_decomposition", "standardizing_comparisons", "strengthening_conclusions",
                   "cross_machine"):
        os.makedirs(os.path.join(figures_path, subdir), exist_ok=True)
    if enable_intro_distributions:
        figIntroBourginMLPMse()
        figIntroBourginMLPBestEpoch()
    if enable_overlap_plots:
        figOverlapTwoModels()
        figOverlapSixModels()
        figMseRunsSixModels()
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
    if enable_trials_to_target:
        figTrialsToTarget()
    if enable_sampling_error:
        figSamplingErrorMean()
    if enable_decomp_heatmaps:
        figDecompHeatmaps()
    if enable_cross_machine:
        figCrossMachine()
    if enable_target_family:
        figTargetFamily()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the manuscript figures.")
    parser.add_argument("--media", metavar="PATH",
                        help="LaTeX project media/ folder: copy there only the figures its .tex files include")
    args = parser.parse_args()
    run()
    if args.media:
        copyUsedFigures(args.media)
