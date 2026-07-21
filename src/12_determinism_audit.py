#!/usr/bin/env python3
# encoding: utf-8

"""
Determinism audit for the init x order factorial decomposition.

Re-trains three dispersed cells of the 35x35 grid for each of the four
decomposition models and compares the resulting test MSE bit-for-bit
against the value stored in
`output_data/<model>/factorial_init_order_r2_training_metrics.yaml`.

This backs the manuscript claim (variance-decomposition section) that
the pipeline is deterministic given (init_seed, order_seed): the stored
values come from a different process and session, so bit-identity here
verifies determinism across time, not only within a process.

Mirrors src/03_train_decomp.py's codepath exactly (set_seed(init) ->
load features -> DataLoader with order-seeded generator -> model with
seed=init). Checkpoints go to a throwaway temp dir; the audit writes
nothing under output_data/.

Result 2026-07-10 (torch 2.10.0, CPU, threads pinned to 1):
12 of 12 cells bit-identical to the stored values.

Run:  python src/12_determinism_audit.py
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import tempfile

import path_setup  # noqa: F401

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.set_seed import set_seed
from cpc18_utils import output_path
from cpc18_utils.features import load_cpc18_features, cpc18_features_input_size
from cpc18_utils.data import CPC18Dataset
from cpc18_utils.safe_yaml import fastLoadYaml
from cpc18_models.BourginMLP import BourginMLP
from cpc18_models.LinearLogistic import LinearLogistic
from cpc18_models.ContextDepNet import ContextDepNet
from cpc18_models.SparseMLP import SparseMLP

# Must match src/03_train_decomp.py.
MASTER_SEED_INIT = 20260522
MASTER_SEED_ORDER = 99999777
learning_rate = 0.001
patience = 50
num_epochs = 500
batch_size = 32

CELLS = [(0, 0), (17, 17), (34, 34)]   # dispersed grid indices
MODELS = (BourginMLP, LinearLogistic, ContextDepNet, SparseMLP)


def main():
    init_pool = np.random.default_rng(MASTER_SEED_INIT).integers(
        0, 2**32, size=35, dtype=np.uint64)
    order_pool = np.random.default_rng(MASTER_SEED_ORDER).integers(
        0, 2**32, size=35, dtype=np.uint64)

    device = torch.device("cpu")
    ckpt_dir = tempfile.mkdtemp(prefix="determinism_audit_")
    print("torch version:", torch.__version__, flush=True)

    ok = fail = 0
    for ModelClass in MODELS:
        name = ModelClass.__name__
        stored = fastLoadYaml(os.path.join(
            output_path, name, "factorial_init_order_r2_training_metrics.yaml"))
        by_pair = {(int(v["init_seed"]), int(v["order_seed"])): v["mse"]
                   for v in stored.values()
                   if isinstance(v, dict) and "init_seed" in v}
        for ci, co in CELLS:
            i_seed, o_seed = int(init_pool[ci]), int(order_pool[co])
            ref = by_pair.get((i_seed, o_seed))
            torch.set_num_threads(1)
            set_seed(i_seed)
            splits = load_cpc18_features()
            g = torch.Generator()
            g.manual_seed(o_seed)
            train_loader = DataLoader(
                CPC18Dataset(splits.X_train, splits.y_train),
                batch_size=batch_size, shuffle=True, generator=g)
            test_loader = DataLoader(
                CPC18Dataset(splits.X_test, splits.y_test),
                batch_size=batch_size, shuffle=False)
            model = ModelClass(input_size=cpc18_features_input_size(),
                               learning_rate=learning_rate,
                               patience=patience, seed=i_seed,
                               output_path=ckpt_dir, device=device)
            rm = model.spinTrainEval(train_loader, test_loader, num_epochs)
            mse = next(iter(rm.values()))["mse"]
            same = (ref is not None) and (repr(mse) == repr(ref))
            ok += same
            fail += (not same)
            print(f"{name:15s} cell({ci},{co}) rerun={mse!r} stored={ref!r} "
                  f"bit-identical={same}", flush=True)

    print(f"\nRESULT: {ok} identical / {fail} mismatched of {ok + fail} cells")


if __name__ == "__main__":
    main()
