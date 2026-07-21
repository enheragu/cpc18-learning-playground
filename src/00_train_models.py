#!/usr/bin/env python3
# encoding: utf-8

"""
Train each CPC18 model for N iterations and log per-trial metrics to a
YAML file (file-lock protected so several parallel processes can append
to the same file).

Three seed modes, each writes to its own YAML:

- "random"     -> different seed per trial. Studies initialization
                  variance. Filename suffix: randomseed_*.
- "same"       -> all trials use SAME_SEED_VALUE. Sanity check: expect
                  bit-identical MSE across trials => proves all
                  randomness is captured by the seed. Filename: sameseed_*.
- "shared_pool"-> trial i of every model uses pool[i]. Enables cross-model
                  rank/correlation analysis (Spearman, ICC) extending the
                  ablation analysis of the MNIST paper to model-level
                  comparison. Filename: sharedpool_*.

IMPORTANT: Any change to patience / learning_rate / batch_size makes
existing trials in a YAML non-comparable to new ones (different
stopping criterion / optimization landscape). Delete the relevant YAML
under output_data/<Model>/ before re-running with new hyperparameters.
Per-trial hyperparameters are stored in the YAML for post-hoc
verification of drift.

Adapted from mnist-learning-playground/src/00_train_models.py.
"""

import argparse
import os
import re
import random
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

# Side-effect: adds mnist-learning-playground/src to sys.path so that the
# `utils.*` / `models.*` imports below resolve into the submodule.
import path_setup  # noqa: F401

import numpy as np
import torch
from torch.utils.data import DataLoader

# These come from the submodule (after path_setup):
from utils.log_utils import log
from utils.yaml_utils import getMetricsLogFile  # read-only use is safe
from utils.set_seed import set_seed

# Our own code:
from cpc18_utils import output_path
from cpc18_utils.data import load_cpc18_aggregate, CPC18Dataset, cpc18_input_size
from cpc18_utils.safe_yaml import atomicUpdateMetricsLogFile
from cpc18_models import LinearLogistic, MLP_1L, MLP_3L, BourginMLP


# Registry: maps a class name (string, picklable) to the class itself.
# Multiprocess workers receive only the class NAME (always picklable)
# and look up the class here in their own process.
_MODEL_REGISTRY = {
    cls.__name__: cls
    for cls in (LinearLogistic, MLP_1L, MLP_3L, BourginMLP)
}


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Seed mode: "random" | "same" | "shared_pool". See module docstring.
SEED_MODE = "shared_pool"
SAME_SEED_VALUE = 42
# Master seed used to deterministically generate the shared pool (only
# used when SEED_MODE == "shared_pool"). Changing this gives a different
# pool, so two runs with the same MASTER_SEED_FOR_POOL produce comparable
# YAMLs across models.
MASTER_SEED_FOR_POOL = 20260522

learning_rate = 0.001
patience = 50
num_epochs = 500
batch_size = 32

# Fixed train/test split seed for the laboratory case. Vary later when
# we study split-induced variance separately.
split_seed = 42

# Device. CPC18 + tiny MLPs => CPU is faster than GPU here (transfer
# overhead dominates). Set CPC18_USE_GPU=1 to override.
device = torch.device("cuda" if (torch.cuda.is_available()
                                 and os.environ.get("CPC18_USE_GPU") == "1")
                      else "cpu")

# Number of independent training trials per model. MNIST paper used 400;
# start lower for a sanity check and scale up once distributions look
# sensible.
model_iterations = {
    LinearLogistic: 400,
    MLP_1L:         400,
    MLP_3L:         400,
    BourginMLP:     400,
}


def _buildLoaders(splits):
    train_ds = CPC18Dataset(splits.X_train, splits.y_train)
    test_ds = CPC18Dataset(splits.X_test, splits.y_test)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


_FILENAME_PREFIX = {
    "random": "randomseed",
    "same": "sameseed",
    "shared_pool": "sharedpool",
}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3}$")


def _metricsPath(model_name: str) -> str:
    prefix = _FILENAME_PREFIX[SEED_MODE]
    p = os.path.join(output_path, model_name, f"{prefix}_training_metrics.yaml")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def _seedsAlreadyDone(metrics_log_file: str) -> set:
    """Return the set of seeds for which trials are already persisted."""
    existing = getMetricsLogFile(metrics_log_file) or {}
    return {entry["seed"]
            for k, entry in existing.items()
            if _DATE_RE.match(k) and isinstance(entry, dict) and "seed" in entry}


def _seedForIndex(idx: int, shared_pool) -> int:
    """Map a trial index to its seed under the current SEED_MODE."""
    if SEED_MODE == "same":
        return SAME_SEED_VALUE
    if SEED_MODE == "shared_pool":
        return int(shared_pool[idx])
    # "random": still deterministic per (idx, master) for resumability.
    # Each idx draws a separate sub-stream from MASTER_SEED_FOR_POOL.
    rng = np.random.default_rng((MASTER_SEED_FOR_POOL, idx))
    return int(rng.integers(0, 2**32, dtype=np.uint64))


def _runOneTrial(task):
    """Worker: train ONE model with ONE seed and persist results.

    Designed to be picklable -> safe for ProcessPoolExecutor. Receives
    only primitive types + the model class NAME (resolved via the
    registry below) so that pickling is robust across spawn methods.

    Idempotent: if a trial with `seed` is already in the YAML, skip.
    """
    model_name, seed, _trial_index = task

    # Each worker is its own process => set its own torch thread count
    # to 1 to avoid oversubscription on shared CPUs.
    torch.set_num_threads(1)

    metrics_log_file = _metricsPath(model_name)
    if seed in _seedsAlreadyDone(metrics_log_file):
        return f"skip {model_name}/{seed}"

    set_seed(seed)
    splits = load_cpc18_aggregate(split_seed=split_seed)
    train_loader, test_loader = _buildLoaders(splits)

    ModelClass = _MODEL_REGISTRY[model_name]
    model = ModelClass(
        input_size=cpc18_input_size(),
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
    atomicUpdateMetricsLogFile(run_metrics, metrics_log_file)
    return f"done {model_name}/{seed}"


def _buildTaskList(shared_pool):
    """Deterministic list of (model_name, seed, trial_index) tuples.

    Order is fixed so any run with the same config produces the same
    set of tasks; only the workers' completion order may vary.
    """
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
    parser.add_argument("--workers", type=int, default=1,
                        help="Process pool size. 1 = single-process (default).")
    args = parser.parse_args()

    input_size = cpc18_input_size()
    log(f"[main] Device: {device}. Input size: {input_size}. "
        f"Seed mode: {SEED_MODE}. Workers: {args.workers}.")

    # Touch the dataset once so workers hit the parquet cache without
    # racing to download/aggregate.
    splits = load_cpc18_aggregate(split_seed=split_seed)
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
                msg = _runOneTrial(task)
                log(f"[main] {msg}")
            except Exception:
                log(f"[main] trial {task} crashed:\n{traceback.format_exc()}")
        return

    # Multi-process. Pool.map distributes each task to exactly one
    # worker -> no two workers can claim the same (model, seed) pair,
    # so shared_pool reproducibility is preserved.
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_runOneTrial, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                log(f"[main] {fut.result()}")
            except Exception:
                log(f"[main] task {futs[fut]} crashed:\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
