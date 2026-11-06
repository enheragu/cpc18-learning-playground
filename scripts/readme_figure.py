#!/usr/bin/env python3
# encoding: utf-8

"""
Writes the figure of the README: one dot per training run of the main run, one
row per model, sorted by median. Only the jitter of the dots is random, with a
fixed seed. It needs the flat CSV of the main run, so src/09_export_csv.py runs
first:

    python scripts/readme_figure.py
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RESULTS_CSV = os.path.join(_REPO_ROOT, "analysis_results", "csv", "results_sharedpool_r2.csv")
OUT_PNG = os.path.join(_REPO_ROOT, "media", "mse_runs_six_models.png")

# Same color for each model as in the figures of the article
MODELS = {
    "BEAST_GB": ("BEAST-GB", "#769393"),
    "PsychForest": ("PsychForest", "#a66497"),
    "LinearLogistic": ("LinearLogistic", "#0171ba"),
    "ContextDepNet": ("ContextDepNet", "#f6ae2d"),
    "BourginMLP": ("BourginMLP", "#78b01c"),
    "SparseMLP": ("SparseMLP", "#f23535"),
}
AXIS_END = 1.8  # BourginMLP reaches 6.0: its few furthest runs are counted, not drawn
JITTER_SEED = 0
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"


def loadRuns():
    runs = {}
    with open(RESULTS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            runs.setdefault(row["model"], []).append(float(row["mse_x100"]))
    return {model: np.array(values) for model, values in runs.items()}


def main():
    runs = loadRuns()
    order = sorted(MODELS, key=lambda model: np.median(runs[model]))
    rng = np.random.default_rng(JITTER_SEED)

    fig, ax = plt.subplots(figsize=(8, 3.7))
    for row, model in enumerate(reversed(order)):
        values = runs[model]
        shown = values[values <= AXIS_END]
        color = MODELS[model][1]
        ax.scatter(shown, row + rng.uniform(-0.3, 0.3, shown.size), s=10, color=color, alpha=0.45, linewidths=0)
        ax.plot([np.median(values)] * 2, [row - 0.38, row + 0.38], color=INK, linewidth=1.8, solid_capstyle="round")
        beyond = values.size - shown.size
        if beyond:
            ax.text(AXIS_END + 0.02, row, f"+{beyond} runs,\nup to {values.max():.1f}", va="center", fontsize=9.5,
                    color=MUTED)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([MODELS[model][0] for model in reversed(order)])
    ax.set_xlim(0.5, AXIS_END + 0.2)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xticks(np.arange(0.6, AXIS_END + 0.01, 0.2))
    ax.set_xlabel("Test MSE × 100 (lower is better)", fontsize=12, color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="x", colors=INK, labelsize=11)
    ax.tick_params(axis="y", length=0, colors=INK, labelsize=12)

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
