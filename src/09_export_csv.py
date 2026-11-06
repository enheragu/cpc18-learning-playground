#!/usr/bin/env python3
# encoding: utf-8

"""
Export all training-result YAMLs to flat CSV files suitable for analysis
in R (or any other tool).  One CSV per experiment family:

  analysis_results/csv/results_sharedpool_r1.csv     regime 1, 4 models × 400
  analysis_results/csv/results_sharedpool_r2.csv     regime 2, 6 models × 800
  analysis_results/csv/results_ablation_b1.csv       B1: BourginMLP batch × lr, 9 × 800
  analysis_results/csv/results_ablation_b3.csv       B3: SparseMLP eps × lr, 9 × 800
  analysis_results/csv/results_ablation_b1x.csv      B1 grid-sensitivity column, 3 × 800
  analysis_results/csv/results_factorial_decomp.csv  Phase C: init × order, 4 × 1225
  analysis_results/csv/results_valsplit.csv          validation-split early stopping
  analysis_results/csv/results_split_variance.csv    split-induced variance
  analysis_results/csv/results_perproblem_errors.csv per-problem errors

Reproducibility constants (master seeds) appear as columns so
each CSV is self-contained. Without the training logs, the CSVs are built from
the data deposit (see cpc18_utils/training_logs.py): the columns that the
deposit does not hold are empty, and regime 1 is not exported.

Usage:
    cd src && python 09_export_csv.py
"""

import os

import path_setup  # noqa: F401

import pandas as pd

from utils.log_utils import log, bcolors
from cpc18_utils import output_path
from cpc18_utils.training_logs import loadTrainingLog, trainingConditions, usesDeposit, depositRows
from cpc18_models.ablation import B1_CLASSES, B1X_CLASSES, B3_CLASSES


# Reproducibility constants — must match the trainer scripts.
SPLIT_SEED            = 42
MASTER_SEED_FOR_POOL  = 20260522   # regime 1/2, B1, B3, Phase C init dimension
MASTER_SEED_ORDER     = 99999777   # Phase C order dimension only

_repo_root = os.path.dirname(output_path)
output_dir = os.path.join(_repo_root, "analysis_results", "csv")

_DECOMP_MODELS = ("BourginMLP", "LinearLogistic", "ContextDepNet", "SparseMLP")


def _load_trials(yaml_path):
    """Return list of entry dicts from a training-metrics log."""
    try:
        return list(loadTrainingLog(yaml_path).values())
    except FileNotFoundError:
        return []


def _common_fields(t, regime=None):
    row = {
        "seed":               t.get("seed"),
        "seed_mode":          t.get("seed_mode"),
        "mse":                t.get("mse"),
        "mse_x100":           t.get("mse_x100"),
        "mse_random_baseline": t.get("mse_random_baseline"),
        "batch_size":         t.get("batch_size"),
        "learning_rate":      t.get("learning_rate"),
        "best_epoch":         t.get("best_epoch"),
        "total_epochs":       t.get("total_epochs"),
        "patience":           t.get("patience"),
        "train_duration":     t.get("train_duration"),
        "master_seed_for_pool": MASTER_SEED_FOR_POOL,
    }
    if regime == 1:  # the other experiments use the canonical split
        row["split_seed"] = SPLIT_SEED
    if regime is not None:
        row["regime"] = regime
    return row


def export_sharedpool(regime: int):
    suffix = ("sharedpool_training_metrics.yaml" if regime == 1
              else "sharedpool_r2_training_metrics.yaml")
    rows = []
    for model_name in trainingConditions():
        if "_B" in model_name or "_E" in model_name:   # skip ablation dirs
            continue
        trials = _load_trials(os.path.join(output_path, model_name, suffix))
        if not trials:
            continue
        log(f"  {model_name}: {len(trials)} trials")
        for t in trials:
            rows.append({"model": model_name, **_common_fields(t, regime)})

    if not rows:
        log(f"  [WARN] no trials found for regime {regime}", color=bcolors.WARNING)
        return

    df = pd.DataFrame(rows)
    out = os.path.join(output_dir, f"results_sharedpool_r{regime}.csv")
    df.to_csv(out, index=False)
    log(f"  -> {out}  ({len(df)} rows, {df['model'].nunique()} models)",
        color=bcolors.OKGREEN)


def export_ablation(study: str):
    class_list = {"b1": B1_CLASSES, "b1x": B1X_CLASSES,
                  "b3": B3_CLASSES}[study]
    f1_name    = "epsilon" if study == "b3" else "batch_size"
    rows = []
    for cls in class_list:
        yaml_path = os.path.join(output_path, cls.__name__,
                                 "sharedpool_r2_ablation_training_metrics.yaml")
        trials = _load_trials(yaml_path)
        if not trials:
            log(f"  [WARN] no trials for {cls.__name__}", color=bcolors.WARNING)
            continue
        log(f"  {cls.__name__}: {len(trials)} trials")
        f1_val = cls.epsilon if study == "b3" else cls.batch_size
        for t in trials:
            row = {
                "model":       cls.__name__,
                f1_name:       f1_val,
                "learning_rate": cls.default_lr,
                "seed":        t.get("seed"),
                "seed_mode":   t.get("seed_mode"),
                "mse":         t.get("mse"),
                "mse_x100":    t.get("mse_x100"),
                "mse_random_baseline": t.get("mse_random_baseline"),
                "best_epoch":  t.get("best_epoch"),
                "total_epochs": t.get("total_epochs"),
                "patience":    t.get("patience"),
                "train_duration": t.get("train_duration"),
                "master_seed_for_pool": MASTER_SEED_FOR_POOL,
            }
            rows.append(row)

    if not rows:
        return

    df = pd.DataFrame(rows)
    out = os.path.join(output_dir, f"results_ablation_{study}.csv")
    df.to_csv(out, index=False)
    log(f"  -> {out}  ({len(df)} rows, {df['model'].nunique()} cells)",
        color=bcolors.OKGREEN)


def export_valsplit():
    """Sensitivity experiment (11_train_valsplit.py): internal-validation
    early stopping, 200 seeds x {BourginMLP, ContextDepNet}. Records the
    test MSE under BOTH selection rules for each trajectory."""
    rows = []
    for model_name in ("BourginMLP", "ContextDepNet"):
        yaml_path = os.path.join(output_path, model_name,
                                 "valsplit_r2_training_metrics.yaml")
        trials = _load_trials(yaml_path)
        if not trials:
            log(f"  [WARN] no trials for {model_name}", color=bcolors.WARNING)
            continue
        log(f"  {model_name}: {len(trials)} trials")
        for t in trials:
            rows.append({
                "model":              model_name,
                "seed":               t.get("seed"),
                "seed_mode":          t.get("seed_mode"),
                "protocol":           t.get("protocol"),
                "val_split_seed":     t.get("val_split_seed"),
                "n_train_problems":   t.get("n_train_problems"),
                "n_val_problems":     t.get("n_val_problems"),
                "mse":                t.get("mse"),
                "mse_x100":           t.get("mse_x100"),
                "mse_x100_testsel":   t.get("mse_x100_testsel"),
                "val_mse_at_selection": t.get("val_mse_at_selection"),
                "best_epoch":         t.get("best_epoch"),
                "best_test_epoch":    t.get("best_test_epoch"),
                "total_epochs":       t.get("total_epochs"),
                "patience":           t.get("patience"),
                "learning_rate":      t.get("learning_rate"),
                "batch_size":         t.get("batch_size"),
                "train_duration":     t.get("train_duration"),
                "master_seed_for_pool": MASTER_SEED_FOR_POOL,
            })

    if not rows:
        return

    df = pd.DataFrame(rows)
    out = os.path.join(output_dir, "results_valsplit.csv")
    df.to_csv(out, index=False)
    log(f"  -> {out}  ({len(df)} rows, {df['model'].nunique()} models)",
        color=bcolors.OKGREEN)


def export_split_variance():
    """20 random 208/60 re-splits at a fixed init seed (stable models only;
    split_variance_rerun.py). Values are MSE x100 per split."""
    import numpy as np
    npz_path = os.path.join(_repo_root, "analysis_results", "split_variance.npz")
    if usesDeposit(npz_path):
        rows = depositRows("split_variance.csv")
    else:
        d = np.load(npz_path, allow_pickle=True)
        rows = [{"model": m, "split_index": i, "mse_x100": float(v)}
                for m in d.files for i, v in enumerate(d[m])]
    if not rows:
        log("  [WARN] no split-variance results", color=bcolors.WARNING)
        return
    df = pd.DataFrame(rows)
    out = os.path.join(output_dir, "results_split_variance.csv")
    df.to_csv(out, index=False)
    log(f"  -> {out}  ({len(df)} rows, {df['model'].nunique()} models)",
        color=bcolors.OKGREEN)


def export_perproblem():
    """Per-(test problem) mean squared errors for the first 50 shared seeds
    (perproblem_rerun.py), long format."""
    import numpy as np
    npz_path = os.path.join(_repo_root, "analysis_results",
                            "perproblem_errors.npz")
    if usesDeposit(npz_path):
        rows = depositRows("perproblem_errors.csv")
    else:
        d = np.load(npz_path, allow_pickle=True)
        problems = d["problems"]
        seeds = d["seeds"]
        rows = []
        for m in d.files:
            if m in ("problems", "seeds"):
                continue
            E = d[m]   # (n_seeds, n_problems)
            for si, s in enumerate(seeds):
                for pj, p in enumerate(problems):
                    rows.append({"model": m, "seed": int(s),
                                 "problem_id": int(p),
                                 "mean_sq_error_x100": float(E[si, pj])})
    if not rows:
        log("  [WARN] no per-problem results", color=bcolors.WARNING)
        return
    df = pd.DataFrame(rows)
    out = os.path.join(output_dir, "results_perproblem_errors.csv")
    df.to_csv(out, index=False)
    log(f"  -> {out}  ({len(df)} rows, {df['model'].nunique()} models)",
        color=bcolors.OKGREEN)


def export_factorial_decomp():
    rows = []
    for model_name in _DECOMP_MODELS:
        yaml_path = os.path.join(output_path, model_name,
                                 "factorial_init_order_r2_training_metrics.yaml")
        trials = _load_trials(yaml_path)
        if not trials:
            log(f"  [WARN] no trials for {model_name}", color=bcolors.WARNING)
            continue
        log(f"  {model_name}: {len(trials)} trials")
        for t in trials:
            rows.append({
                "model":            model_name,
                "init_seed":        t.get("init_seed"),
                "order_seed":       t.get("order_seed"),
                "seed":             t.get("seed"),
                "seed_mode":        t.get("seed_mode"),
                "seed_decomposed":  t.get("seed_decomposed"),
                "mse":              t.get("mse"),
                "mse_x100":         t.get("mse_x100"),
                "mse_random_baseline": t.get("mse_random_baseline"),
                "batch_size":       t.get("batch_size"),
                "learning_rate":    t.get("learning_rate"),
                "best_epoch":       t.get("best_epoch"),
                "total_epochs":     t.get("total_epochs"),
                "patience":         t.get("patience"),
                "train_duration":   t.get("train_duration"),
                "master_seed_init": MASTER_SEED_FOR_POOL,
                "master_seed_order": MASTER_SEED_ORDER,
            })

    if not rows:
        return

    df = pd.DataFrame(rows)
    out = os.path.join(output_dir, "results_factorial_decomp.csv")
    df.to_csv(out, index=False)
    log(f"  -> {out}  ({len(df)} rows, {df['model'].nunique()} models)",
        color=bcolors.OKGREEN)


if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)

    log("Regime 1 — sharedpool (raw features)...", color=bcolors.OKCYAN)
    export_sharedpool(regime=1)

    log("Regime 2 — sharedpool_r2 (BEAST features)...", color=bcolors.OKCYAN)
    export_sharedpool(regime=2)

    log("Ablation B1 — BourginMLP (batch_size × learning_rate)...", color=bcolors.OKCYAN)
    export_ablation("b1")

    log("Ablation B3 — SparseMLP (epsilon × learning_rate)...", color=bcolors.OKCYAN)
    export_ablation("b3")

    log("Ablation B1X — BourginMLP grid-sensitivity column (lr=3.16e-4)...",
        color=bcolors.OKCYAN)
    export_ablation("b1x")

    log("Val-split sensitivity experiment...", color=bcolors.OKCYAN)
    export_valsplit()

    log("Split-variance experiment (20 re-splits)...", color=bcolors.OKCYAN)
    export_split_variance()

    log("Per-problem errors (50 seeds x 60 problems)...", color=bcolors.OKCYAN)
    export_perproblem()

    log("Phase C — factorial init × order decomposition...", color=bcolors.OKCYAN)
    export_factorial_decomp()

    log("All done.", color=bcolors.OKGREEN)
