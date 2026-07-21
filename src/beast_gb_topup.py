#!/usr/bin/env python3
# encoding: utf-8

"""
Focused BEAST_GB top-up to N=800.

Why this exists: `01_train_regime2.py --workers 1` was originally used
for the same purpose, but it iterates ALL 6 models × 800 = 4800 tasks,
calling `getMetricsLogFile` (which `yaml.safe_load`s the full model
YAML, ~MB at this point) on EVERY task even when only checking
idempotency. That made the script take 6-8 hours just for the skip
checks, with the actual BEAST_GB training (only ~10 min) buried under
that.

This script reads the BEAST_GB YAML ONCE up front, computes the set
of seeds still missing, then trains them sequentially in a single
process (no fork -> no XGBoost OpenMP crash).
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import re
import sys

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
from cpc18_models import BEAST_GB


MASTER_SEED_FOR_POOL = 20260522
TARGET_TRIALS = 800
BATCH_SIZE = 32

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3}$")
_YAML_PATH = os.path.join(
    output_path, "BEAST_GB", "sharedpool_r2_training_metrics.yaml")


def main():
    torch.set_num_threads(1)

    # ONE up-front read of the BEAST_GB YAML; everything else is in-memory.
    existing = getMetricsLogFile(_YAML_PATH) or {}
    done_seeds = {
        entry["seed"] for k, entry in existing.items()
        if _DATE_RE.match(k) and isinstance(entry, dict) and "seed" in entry
    }
    log(f"[topup] YAML already has {len(done_seeds)} trials.")

    pool = np.random.default_rng(MASTER_SEED_FOR_POOL).integers(
        0, 2**32, size=TARGET_TRIALS, dtype=np.uint64)
    todo_seeds = [int(s) for s in pool if int(s) not in done_seeds]
    log(f"[topup] Target {TARGET_TRIALS}, missing {len(todo_seeds)}.")

    if not todo_seeds:
        log("[topup] Nothing to do.")
        return

    splits = load_cpc18_features()
    input_size = cpc18_features_input_size()
    log(f"[topup] Dataset: train {len(splits.y_train)} rows, "
        f"test {len(splits.y_test)} rows. input_size={input_size}")

    for i, seed in enumerate(todo_seeds, 1):
        set_seed(seed)
        train_loader = DataLoader(
            CPC18Dataset(splits.X_train, splits.y_train),
            batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(
            CPC18Dataset(splits.X_test, splits.y_test),
            batch_size=BATCH_SIZE, shuffle=False)

        model = BEAST_GB(
            input_size=input_size, seed=seed, output_path=output_path)
        run_metrics = model.spinTrainEval(train_loader, test_loader)
        for _ts, entry in run_metrics.items():
            entry["patience"] = 50
            entry["learning_rate"] = 0.001
            entry["batch_size"] = BATCH_SIZE
            entry["seed_mode"] = "shared_pool"
            entry["regime"] = 2
        atomicUpdateMetricsLogFile(run_metrics, _YAML_PATH)

        if i % 25 == 0 or i == len(todo_seeds):
            log(f"[topup] {i}/{len(todo_seeds)} done")


if __name__ == "__main__":
    main()
