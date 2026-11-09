#!/usr/bin/env python3
# encoding: utf-8

"""
Ablation trainer for B1 (BourginMLP batch_size × learning_rate) and B3
(SparseMLP epsilon × learning_rate) on regime 2 (BEAST features).

Run with:
    python 04_train_ablation.py --study b1 --workers 6
    python 04_train_ablation.py --study b1x --workers 6
    python 04_train_ablation.py --study b3 --workers 6

Add `--protocol stable` to train a grid under the stable protocol
(cpc18_utils/protocol.py), read at the cross-validated epoch of each cell;
it writes to output_data_stable/.

Each ablation condition is its own subclass (cpc18_models.ablation.py),
so each writes to `output_data/<ClassName>/sharedpool_r2_ablation_training_metrics.yaml`.
That guarantees no cross-condition YAML mixing.

Seed pool is the same MASTER_SEED_FOR_POOL=20260522 as the main regime
runs, so per-seed analysis can be cross-referenced with the rest.

PERFORMANCE NOTES
-----------------
The metrics YAMLs are ~7-15 MB at N=800 (per-epoch training curves are
stored). Parsing a 7 MB YAML with the pure-Python yaml.safe_load takes
~8 s; with the libyaml-backed CSafeLoader (enabled via
`cpc18_utils.safe_yaml`) it drops to ~1 s.

Two further optimisations applied here:

(a) Idempotency check is done ONCE per cell in the main process before
    forking workers. Workers receive a pre-filtered task list and never
    re-read the per-cell YAML. This eliminates 1 of the 3 YAML reads
    per trial.

(b) Worker processes return the trial metrics dict instead of writing
    it to disk. The main process accumulates results and flushes a
    batch of FLUSH_EVERY trials to the YAML per cell. This reduces the
    per-trial atomic read-modify-write from 1 per trial to 1 per
    FLUSH_EVERY trials. Crash window: at most FLUSH_EVERY trials per
    cell can be re-executed on next run (idempotency keeps it correct).
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import re
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import path_setup  # noqa: F401

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.log_utils import log
from utils.set_seed import set_seed

from cpc18_utils import output_path, stable_output_path
from cpc18_utils.features import load_cpc18_features, cpc18_features_input_size
from cpc18_utils.data import CPC18Dataset
from cpc18_utils.safe_yaml import addTrials, atomicUpdateMetricsLogFile, fastLoadYaml
from cpc18_utils.protocol import (
    PROTOCOLS, PROTOCOL_ORIGINAL, PROTOCOL_STABLE, STABLE_INIT,
    readoutEpoch, stablePatience, stableEntry,
)
from cpc18_models.ablation import B1_CLASSES, B1X_CLASSES, B3_CLASSES


_MODEL_REGISTRY = {cls.__name__: cls
                   for cls in (B1_CLASSES + B1X_CLASSES + B3_CLASSES)}


# ---------------------------------------------------------------------
# Configuration — change these to re-scope the experiment.
# ---------------------------------------------------------------------

# Seed pool. Must match 01_train_regime2.py and 03_train_decomp.py for
# per-seed cross-experiment analysis to work.
MASTER_SEED_FOR_POOL = 20260522

# Per-cell sample size.
N_TRIALS_BY_STUDY = {"b1": 800, "b1x": 800, "b3": 800}

# Per-trial hyperparameters. Note: `learning_rate` is OVERRIDDEN per
# cell via `cls.default_lr` baked into each ablation subclass (see
# cpc18_models/ablation.py).
# num_epochs raised from 500 to 1500 to give the low-lr cells (1e-4)
# enough room to converge before the cap. At cap=500, BourginMLP with
# lr=1e-4 hit the cap on 40-50% of trials (total_epochs > 495) — those
# trials had biased MSE because training was cut before patience could
# fire. With cap=1500 plus patience=50, even slow-convergers early-stop
# naturally.
patience = 50
num_epochs = 1500

# How many completed trials to accumulate in memory (per cell) before
# flushing them to the cell's YAML via atomicUpdateMetricsLogFile.
# Trade-off: higher = fewer expensive YAML reads, but more trials lost
# if the main process dies; lower = safer, but more YAML I/O.
FLUSH_EVERY = 25

device = torch.device("cuda" if (torch.cuda.is_available()
                                 and os.environ.get("CPC18_USE_GPU") == "1")
                      else "cpu")


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3}$")
_YAML_NAME = "sharedpool_r2_ablation_training_metrics.yaml"


def _outputRoot(protocol: str) -> str:
    return output_path if protocol == PROTOCOL_ORIGINAL else stable_output_path


def _metricsPath(model_name: str, protocol: str = PROTOCOL_ORIGINAL) -> str:
    p = os.path.join(_outputRoot(protocol), model_name, _YAML_NAME)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def _seedsAlreadyDone(metrics_log_file: str) -> set:
    """Set of seeds for which a trial entry already exists. Uses the
    fast libyaml-backed loader from cpc18_utils.safe_yaml."""
    existing = fastLoadYaml(metrics_log_file)
    return {entry["seed"]
            for k, entry in existing.items()
            if _DATE_RE.match(k) and isinstance(entry, dict) and "seed" in entry}


def _buildLoaders(splits, batch_size: int):
    train_ds = CPC18Dataset(splits.X_train, splits.y_train)
    test_ds = CPC18Dataset(splits.X_test, splits.y_test)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def _runOneTrial(task):
    """Worker: train ONE ablation cell with ONE seed and return its
    `run_metrics` dict. The main process is responsible for batching
    these returns into the YAML."""
    model_name, seed, _trial_idx, protocol = task

    torch.set_num_threads(1)
    set_seed(seed)
    splits = load_cpc18_features()

    ModelClass = _MODEL_REGISTRY[model_name]
    lr = ModelClass.default_lr
    bs = ModelClass.batch_size
    train_loader, test_loader = _buildLoaders(splits, batch_size=bs)

    if ModelClass in B1_CLASSES:
        is_b1, study = True, "B1"
    elif ModelClass in B1X_CLASSES:
        is_b1, study = True, "B1X"
    else:
        is_b1, study = False, "B3"

    stable = protocol == PROTOCOL_STABLE
    trial_patience = patience
    init_kwargs = {}
    if stable:
        base_model = "BourginMLP" if is_b1 else "SparseMLP"
        trial_patience = stablePatience(base_model)
        init_kwargs = {"init": STABLE_INIT}
        readout = readoutEpoch(base_model, bs, lr, None if is_b1 else ModelClass.epsilon)

    model = ModelClass(
        input_size=cpc18_features_input_size(),
        learning_rate=lr,
        patience=trial_patience,
        seed=seed,
        output_path=_outputRoot(protocol),
        device=device,
        **init_kwargs,
    )
    run_metrics = model.spinTrainEval(train_loader, test_loader, num_epochs)
    for _ts, entry in run_metrics.items():
        entry["patience"] = trial_patience
        entry["learning_rate"] = lr
        entry["batch_size"] = bs
        entry["seed_mode"] = "shared_pool"
        entry["regime"] = 2
        entry["ablation_study"] = study
        if not is_b1:
            entry["epsilon"] = ModelClass.epsilon
        if stable:
            stableEntry(entry, readout, model.init_name)
    return model_name, seed, run_metrics


def _buildTaskList(class_list, shared_pool, n_target, protocol):
    """Precompute the work to do: idempotency-filter once per cell
    using the fast loader, before forking any workers. Workers then
    receive only (model_name, seed) pairs for trials that are actually
    missing — no per-trial YAML read."""
    tasks = []
    total_done = 0
    total_pending = 0
    for cls in class_list:
        done = _seedsAlreadyDone(_metricsPath(cls.__name__, protocol))
        n_done_here = 0
        for k in range(n_target):
            seed = int(shared_pool[k])
            if seed in done:
                n_done_here += 1
                continue
            tasks.append((cls.__name__, seed, k, protocol))
        total_done += n_done_here
        total_pending += n_target - n_done_here
        log(f"[plan] {cls.__name__:35s} done={n_done_here:4d}/"
            f"{n_target}  pending={n_target - n_done_here:4d}")
    log(f"[plan] total: done={total_done}, pending={total_pending}")
    return tasks


def _flushBatch(model_name: str, buffer: dict, protocol: str) -> None:
    """Write the accumulated batch for one cell to its YAML and clear."""
    if not buffer:
        return
    atomicUpdateMetricsLogFile(buffer, _metricsPath(model_name, protocol))
    buffer.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=("b1", "b1x", "b3"), required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--conditions", type=str, default=None,
                         help="Comma-separated ablation class names to "
                              "restrict this run to (default: all "
                              "conditions of --study).")
    parser.add_argument("--protocol", choices=PROTOCOLS, default=PROTOCOL_ORIGINAL,
                        help="Training protocol (cpc18_utils/protocol.py). "
                             "'stable' writes to output_data_stable/.")
    args = parser.parse_args()

    class_list = {"b1": B1_CLASSES, "b1x": B1X_CLASSES,
                  "b3": B3_CLASSES}[args.study]
    if args.conditions:
        wanted = [c.strip() for c in args.conditions.split(",") if c.strip()]
        by_name = {c.__name__: c for c in class_list}
        missing = [w for w in wanted if w not in by_name]
        if missing:
            raise SystemExit(
                f"Unknown condition(s) for --study {args.study}: {missing}")
        class_list = [by_name[w] for w in wanted]
    n_target = N_TRIALS_BY_STUDY[args.study]
    log(f"[main] Ablation {args.study.upper()}. Conditions: {len(class_list)}. "
        f"Trials per condition: {n_target}. Protocol: {args.protocol}. "
        f"Workers: {args.workers}. Device: {device}. "
        f"Flush every: {FLUSH_EVERY} trials/cell.")
    log(f"[main] Conditions: {[c.__name__ for c in class_list]}")

    splits = load_cpc18_features()
    log(f"[main] Train: {len(splits.train_problems)} problems / "
        f"{len(splits.y_train)} rows. Test: {len(splits.test_problems)} / "
        f"{len(splits.y_test)} rows.")

    shared_pool = np.random.default_rng(MASTER_SEED_FOR_POOL).integers(
        0, 2**32, size=n_target, dtype=np.uint64,
    )
    assert len(set(shared_pool.tolist())) == len(shared_pool), \
        "duplicate seed in shared pool (would silently drop a trial)"

    tasks = _buildTaskList(class_list, shared_pool, n_target, args.protocol)
    if not tasks:
        log("[main] Nothing to do.")
        return
    log(f"[main] Pending trials to execute: {len(tasks)}")

    # Per-cell buffer of {timestamp: entry} accumulated until flush.
    buffers = defaultdict(dict)
    # Trial count since last flush, per cell.
    pending_count = defaultdict(int)

    def _handleResult(model_name, seed, run_metrics):
        addTrials(buffers[model_name], run_metrics)
        pending_count[model_name] += 1
        if pending_count[model_name] >= FLUSH_EVERY:
            log(f"[flush] {model_name}: writing batch of "
                f"{pending_count[model_name]} trials")
            _flushBatch(model_name, buffers[model_name], args.protocol)
            pending_count[model_name] = 0

    failed = 0
    try:
        if args.workers <= 1:
            for task in tasks:
                try:
                    model_name, seed, run_metrics = _runOneTrial(task)
                    log(f"[main] done {model_name}/{seed}")
                    _handleResult(model_name, seed, run_metrics)
                except Exception:
                    failed += 1
                    log(f"[main] task {task} crashed:\n{traceback.format_exc()}")
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(_runOneTrial, t): t for t in tasks}
                for fut in as_completed(futs):
                    try:
                        model_name, seed, run_metrics = fut.result()
                        log(f"[main] done {model_name}/{seed}")
                        _handleResult(model_name, seed, run_metrics)
                    except Exception:
                        failed += 1
                        log(f"[main] task {futs[fut]} crashed:\n{traceback.format_exc()}")
    finally:
        # Final flush — write everything still buffered, even on crash.
        for model_name, buf in list(buffers.items()):
            if buf:
                log(f"[flush] {model_name}: final flush of {len(buf)} trials")
                _flushBatch(model_name, buf, args.protocol)
    if failed:
        raise SystemExit(f"{failed} of {len(tasks)} trials failed")


if __name__ == "__main__":
    main()
