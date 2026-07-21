#!/usr/bin/env python3
# encoding: utf-8

"""
Regime 2: train each model on BEAST-derived features (raw + naive +
psychological insight + BEASTpred), with the canonical Plonsky 2018
210/60 train/test split.

Reuses the seed-mode machinery and parallel runner of regime 1
(00_train_models.py) but:

- Loads features from `data/cpc18_features.csv` (produced once by
  `scripts/precompute_features.R`).
- Adds PsychForest (Random Forest) and BEAST_GB (XGBoost) — both
  sklearn-style, wrapped via SklearnRegressorTrainer.
- Writes YAMLs with prefix `_r2` to keep them separate from regime 1.

Pool of models = 4 (kept tight on purpose):
  - LinearLogistic   — convex baseline at the BEAST-features regime
  - BourginMLP       — Bourgin 2019 NN architecture, on enriched features
  - PsychForest      — Plonsky 2017 (AAAI)
  - BEAST_GB         — Plonsky 2018/2024 (CPC18 winner / Nature 2025)

Two model families, three published lineages, one comparable metric.
"""

import os
# Pin OpenMP / MKL to 1 thread BEFORE importing numpy / xgboost / torch.
# XGBoost-with-ProcessPoolExecutor crashes (BrokenProcessPool) when the
# parent process has OpenMP state that gets forked; pinning to 1 thread
# avoids the corrupted-state segfault. n_jobs=1 in the estimator alone
# isn't enough because XGBoost respects OMP_NUM_THREADS at startup.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import path_setup  # noqa: F401

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.log_utils import log
from utils.yaml_utils import getMetricsLogFile
from utils.set_seed import set_seed

from cpc18_utils import output_path
from cpc18_utils.features import (
    load_cpc18_features, cpc18_features_input_size,
)
from cpc18_utils.data import CPC18Dataset
from cpc18_utils.safe_yaml import atomicUpdateMetricsLogFile
from cpc18_models import (
    LinearLogistic, BourginMLP,
    ContextDepNet, SparseMLP,
    PsychForest, BEAST_GB,
)


_MODEL_REGISTRY = {
    cls.__name__: cls
    for cls in (LinearLogistic, BourginMLP, ContextDepNet, SparseMLP,
                PsychForest, BEAST_GB)
}


# ---------------------------------------------------------------------
# Configuration — change these to re-scope the experiment.
# See NOTES.md §5.A for the unified sample size table across trainers.
# ---------------------------------------------------------------------

# Seed pool master. The same value is used by 04_train_ablation.py and
# 03_train_decomp.py (init dimension) so per-seed cross-experiment
# analysis is possible. Changing this invalidates that alignment.
MASTER_SEED_FOR_POOL = 20260522

# Variance experiment configuration.
SEED_MODE = "shared_pool"      # "random" | "same" | "shared_pool"
SAME_SEED_VALUE = 42           # used only when SEED_MODE == "same"

# Per-trial hyperparameters (logged into each YAML entry for drift
# detection — see cpc18_utils/safe_yaml.py).
learning_rate = 0.001
patience = 50
num_epochs = 500
batch_size = 32

# Train/test split: canonical Plonsky 2018 (210/60 problems), shipped
# in data/cpc18_features.csv. No `split_seed` here (regime 1 uses one).

# Device selection. CPU by default; set CPC18_USE_GPU=1 to override.
# GPU is NOT faster for these tiny models, transfer overhead dominates.
device = torch.device("cuda" if (torch.cuda.is_available()
                                 and os.environ.get("CPC18_USE_GPU") == "1")
                      else "cpu")

# Sample sizes per model. CANONICAL = 800 (unified with B1/B3 ablation
# and ~2× regime 1's 400). Bump or rerun via idempotency at any time.
model_iterations = {
    LinearLogistic: 800,
    BourginMLP:     800,
    ContextDepNet:  800,
    SparseMLP:      800,
    PsychForest:    800,
    BEAST_GB:       800,
}


def _buildLoaders(splits):
    train_ds = CPC18Dataset(splits.X_train, splits.y_train)
    test_ds = CPC18Dataset(splits.X_test, splits.y_test)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


# Regime suffix keeps these YAMLs separate from regime 1 outputs.
_FILENAME_PREFIX = {
    "random":      "randomseed_r2",
    "same":        "sameseed_r2",
    "shared_pool": "sharedpool_r2",
}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3}$")


def _metricsPath(model_name: str) -> str:
    prefix = _FILENAME_PREFIX[SEED_MODE]
    p = os.path.join(output_path, model_name, f"{prefix}_training_metrics.yaml")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def _seedsAlreadyDone(metrics_log_file: str) -> set:
    existing = getMetricsLogFile(metrics_log_file) or {}
    return {entry["seed"]
            for k, entry in existing.items()
            if _DATE_RE.match(k) and isinstance(entry, dict) and "seed" in entry}


def _seedForIndex(idx: int, shared_pool) -> int:
    if SEED_MODE == "same":
        return SAME_SEED_VALUE
    if SEED_MODE == "shared_pool":
        return int(shared_pool[idx])
    rng = np.random.default_rng((MASTER_SEED_FOR_POOL, idx))
    return int(rng.integers(0, 2**32, dtype=np.uint64))


def _runOneTrial(task):
    model_name, seed, _trial_index = task

    torch.set_num_threads(1)

    metrics_log_file = _metricsPath(model_name)
    if seed in _seedsAlreadyDone(metrics_log_file):
        return f"skip {model_name}/{seed}"

    set_seed(seed)
    splits = load_cpc18_features()
    train_loader, test_loader = _buildLoaders(splits)

    ModelClass = _MODEL_REGISTRY[model_name]
    model = ModelClass(
        input_size=cpc18_features_input_size(),
        learning_rate=learning_rate,
        patience=patience,
        seed=seed,
        output_path=output_path,
        device=device,
    )
    run_metrics = model.spinTrainEval(train_loader, test_loader, num_epochs)
    for _ts, entry in run_metrics.items():
        entry["patience"] = patience
        entry["learning_rate"] = learning_rate
        entry["batch_size"] = batch_size
        entry["seed_mode"] = SEED_MODE
        entry["regime"] = 2
    atomicUpdateMetricsLogFile(run_metrics, metrics_log_file)
    return f"done {model_name}/{seed}"


def _buildTaskList(shared_pool):
    tasks = []
    for ModelClass, num_iter in model_iterations.items():
        for k in range(num_iter):
            seed = _seedForIndex(k, shared_pool)
            tasks.append((ModelClass.__name__, seed, k))
    return tasks


def main():
    if SEED_MODE not in _FILENAME_PREFIX:
        raise ValueError(f"SEED_MODE must be one of {list(_FILENAME_PREFIX)}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    input_size = cpc18_features_input_size()
    log(f"[main] Regime 2. Device: {device}. Input size: {input_size}. "
        f"Seed mode: {SEED_MODE}. Workers: {args.workers}.")

    # Touch dataset once: fail fast with a clear error if the CSV is missing.
    splits = load_cpc18_features()
    log(f"[main] Train problems: {len(splits.train_problems)} "
        f"({len(splits.y_train)} rows). "
        f"Test problems: {len(splits.test_problems)} "
        f"({len(splits.y_test)} rows).")

    max_iter = max(model_iterations.values())
    shared_pool = np.random.default_rng(MASTER_SEED_FOR_POOL).integers(
        0, 2**32, size=max_iter, dtype=np.uint64,
    )
    assert len(set(shared_pool.tolist())) == len(shared_pool), \
        "duplicate seed in shared pool (would silently drop a trial)"

    tasks = _buildTaskList(shared_pool)

    if args.workers <= 1:
        for task in tasks:
            try:
                log(f"[main] {_runOneTrial(task)}")
            except Exception:
                log(f"[main] trial {task} crashed:\n{traceback.format_exc()}")
        return

    # Default fork mp_context is fine because BEAST_GB lazy-imports
    # xgboost inside _build_estimator (i.e. inside the worker, after
    # the fork). The parent process never touches xgboost, so its
    # OpenMP state cannot be corrupted across the fork boundary.
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_runOneTrial, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                log(f"[main] {fut.result()}")
            except Exception:
                log(f"[main] task {futs[fut]} crashed:\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
