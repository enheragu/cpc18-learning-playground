#!/usr/bin/env python3
# encoding: utf-8

"""
Variance decomposition experiment: factorial (init_seed × order_seed).

Trains BourginMLP on regime 2 features (BEAST + standardized), with the
weight initialisation and the DataLoader's batch shuffle driven by
TWO INDEPENDENT torch RNG streams:

- `init_seed`   -> the model's torch.manual_seed() before weight init
- `order_seed`  -> a torch.Generator() passed to the train DataLoader's
                   `generator=` parameter, controlling batch shuffle

Both seeds come from disjoint deterministic pools (different master
seeds). Each (init_i, order_j) cell is one trial. The resulting
N_init × N_order grid lets us partition total MSE variance into:

    SS_total = SS_init  +  SS_order  +  SS_interaction

via a two-way ANOVA-style sum-of-squares decomposition (computed offline,
read descriptively). One trial per cell would normally confound the
interaction with noise, but the pipeline is deterministic given the two
seeds, so there is no within-cell noise and the residual IS exactly the
init×order interaction.
This decoupling does NOT exist in the existing regime 1 / regime 2
main YAMLs, where `set_seed(seed)` + `DataLoader(shuffle=True)` happens
to use the same global torch state -> init and order are confounded.

YAML output: `output_data/BourginMLP/factorial_init_order_r2_training_metrics.yaml`
Per-trial metadata: `init_seed`, `order_seed`, `seed_decomposed=True`,
plus the usual `mse`, `train_duration`, etc.
"""

import os
# OpenMP threading set early so the worker children don't oversubscribe.
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
from cpc18_utils.features import load_cpc18_features, cpc18_features_input_size
from cpc18_utils.data import CPC18Dataset
from cpc18_utils.safe_yaml import atomicUpdateMetricsLogFile
from cpc18_models import BourginMLP, LinearLogistic, ContextDepNet, SparseMLP


# Restrict to torch NN models — sklearn/XGBoost models don't have a
# meaningful "DataLoader shuffle order" axis, so Phase C doesn't apply
# to PsychForest or BEAST_GB.
_DECOMP_MODEL_REGISTRY = {
    cls.__name__: cls for cls in (BourginMLP, LinearLogistic, ContextDepNet, SparseMLP)
}


# ---------------------------------------------------------------------
# Configuration — change these to re-scope the experiment.
# See NOTES.md §5.A for the unified sample size table across trainers.
# ---------------------------------------------------------------------

# Factorial grid size. CANONICAL = 35 × 35 = 1225 trials per model.
# Bumped from 25 × 25 = 625 to tighten marginal SS estimates by ~18 %.
# Increasing further (e.g. 50 × 50 = 2500) makes the interaction term
# more precise but doesn't change the qualitative findings.
N_INIT  = 35
N_ORDER = 35

# Two DISJOINT master seeds so init and order pools are statistically
# independent. The init pool aligns with the main regime 2 pool
# (MASTER_SEED_FOR_POOL = MASTER_SEED_INIT = 20260522) -> first N_INIT
# seeds match across experiments. The order pool is unique to this
# experiment (deliberate, to avoid confounding).
MASTER_SEED_INIT  = 20260522
MASTER_SEED_ORDER = 99999777

# Per-trial hyperparameters (must match the main regime 2 run for
# meaningful comparison).
learning_rate = 0.001
patience = 50
num_epochs = 500
batch_size = 32

# Device. CPU is faster for these models — see 01_train_regime2.py.
device = torch.device("cuda" if (torch.cuda.is_available()
                                 and os.environ.get("CPC18_USE_GPU") == "1")
                      else "cpu")


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3}$")
_YAML_NAME = "factorial_init_order_r2_training_metrics.yaml"

# Module-level: set by main() and read by workers (forked processes
# inherit it). Defaults to BourginMLP for back-compat with earlier runs.
_MODEL_NAME = BourginMLP.__name__


def _metricsPath(model_name=None) -> str:
    model_name = model_name or _MODEL_NAME
    p = os.path.join(output_path, model_name, _YAML_NAME)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def _pairsAlreadyDone(metrics_log_file: str) -> set:
    """Set of (init_seed, order_seed) tuples for which a trial is saved."""
    existing = getMetricsLogFile(metrics_log_file) or {}
    out = set()
    for k, entry in existing.items():
        if not _DATE_RE.match(k):
            continue
        if not isinstance(entry, dict):
            continue
        i = entry.get("init_seed")
        o = entry.get("order_seed")
        if i is not None and o is not None:
            out.add((int(i), int(o)))
    return out


def _buildLoaders(splits, order_seed: int):
    """Build train+test loaders with the train shuffle driven by a
    dedicated torch.Generator (NOT the global RNG)."""
    g = torch.Generator()
    g.manual_seed(int(order_seed))
    train_ds = CPC18Dataset(splits.X_train, splits.y_train)
    test_ds = CPC18Dataset(splits.X_test, splits.y_test)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def _runOneTrial(task):
    """Worker: train the selected model once with (init_seed, order_seed).

    The model name travels inside the task tuple so the worker does not
    rely on the inherited module-level _MODEL_NAME. This is correct under
    both the fork start method (used here) and spawn, where forked-global
    inheritance would not apply and every worker would otherwise default
    to BourginMLP regardless of --model."""
    init_seed, order_seed, model_name = task

    torch.set_num_threads(1)
    metrics_log_file = _metricsPath(model_name)
    if (init_seed, order_seed) in _pairsAlreadyDone(metrics_log_file):
        return f"skip ({init_seed}, {order_seed})"

    # set_seed fixes the global torch RNG -> used by Linear default init.
    # _initialize_weights then resets it to init_seed and draws Kaiming.
    # The DataLoader's shuffle uses its own generator (order_seed), so
    # batch order is INDEPENDENT of the global RNG.
    set_seed(init_seed)
    splits = load_cpc18_features()
    train_loader, test_loader = _buildLoaders(splits, order_seed=order_seed)

    ModelClass = _DECOMP_MODEL_REGISTRY[model_name]
    model = ModelClass(
        input_size=cpc18_features_input_size(),
        learning_rate=learning_rate,
        patience=patience,
        seed=init_seed,
        output_path=output_path,
        device=device,
    )
    run_metrics = model.spinTrainEval(train_loader, test_loader, num_epochs)
    for _ts, entry in run_metrics.items():
        entry["init_seed"] = int(init_seed)
        entry["order_seed"] = int(order_seed)
        entry["seed_decomposed"] = True
        entry["patience"] = patience
        entry["learning_rate"] = learning_rate
        entry["batch_size"] = batch_size
        entry["seed_mode"] = "factorial_init_order"
        entry["regime"] = 2
    atomicUpdateMetricsLogFile(run_metrics, metrics_log_file)
    return f"done ({init_seed}, {order_seed})"


def _build_pools():
    init_pool = np.random.default_rng(MASTER_SEED_INIT).integers(
        0, 2**32, size=N_INIT, dtype=np.uint64,
    )
    order_pool = np.random.default_rng(MASTER_SEED_ORDER).integers(
        0, 2**32, size=N_ORDER, dtype=np.uint64,
    )
    return init_pool, order_pool


def main():
    global _MODEL_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model", default="BourginMLP",
                        choices=list(_DECOMP_MODEL_REGISTRY),
                        help="Which torch NN model to run Phase C on. "
                        "PsychForest / BEAST_GB are excluded because they "
                        "don't use a DataLoader (no order axis).")
    args = parser.parse_args()
    _MODEL_NAME = args.model

    log(f"[main] Variance decomposition for {_MODEL_NAME}. Device: {device}. "
        f"Grid: {N_INIT} init x {N_ORDER} order = {N_INIT*N_ORDER} trials. "
        f"Workers: {args.workers}.")

    splits = load_cpc18_features()
    log(f"[main] Train: {len(splits.train_problems)} problems / "
        f"{len(splits.y_train)} rows. Test: {len(splits.test_problems)} / "
        f"{len(splits.y_test)} rows.")

    init_pool, order_pool = _build_pools()
    assert len(set(init_pool.tolist())) == len(init_pool), "duplicate init seed in pool"
    assert len(set(order_pool.tolist())) == len(order_pool), "duplicate order seed in pool"
    tasks = [(int(i), int(o), _MODEL_NAME)
             for i in init_pool for o in order_pool]
    log(f"[main] First few (init, order) pairs: {tasks[:3]}")

    if args.workers <= 1:
        for task in tasks:
            try:
                log(f"[main] {_runOneTrial(task)}")
            except Exception:
                log(f"[main] task {task} crashed:\n{traceback.format_exc()}")
        return

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_runOneTrial, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                log(f"[main] {fut.result()}")
            except Exception:
                log(f"[main] task {futs[fut]} crashed:\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
