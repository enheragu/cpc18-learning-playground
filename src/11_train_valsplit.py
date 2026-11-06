#!/usr/bin/env python3
# encoding: utf-8

"""
Sensitivity experiment: does test-set early stopping reshape the
across-seed MSE distribution? (Section 5.5 of the manuscript.)

The main-run protocol early-stops and selects the best epoch on the
TEST set (declared in Section 2.5 of the manuscript). Here we
re-train N seeds with an internal validation split: the 208 canonical
train problems are split BY PROBLEM into 166 train / 42 validation
(deterministic, VAL_SPLIT_SEED), early stopping monitors the validation
MSE, and the test MSE is read at the best-validation epoch.

For each trajectory we record BOTH selection rules:
  - mse / mse_x100        : test MSE at the best-VALIDATION epoch
  - mse_x100_testsel      : test MSE at the best-TEST epoch (the main
                            protocol's statistic, on this reduced train set)
so the two rules can be compared seed-by-seed on identical trajectories.

Run with:
    python src/11_train_valsplit.py --workers 2

Output: output_data/<Model>/valsplit_r2_training_metrics.yaml
Idempotent by seed (safe to resume).
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import time
import traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import path_setup  # noqa: F401

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.log_utils import log
from utils.set_seed import set_seed

from cpc18_utils import output_path, data_path
from cpc18_utils.data import CPC18Dataset
from cpc18_utils.features import (FEATURES_CSV_NAME, _featurize, _standardize)
from cpc18_utils.safe_yaml import atomicUpdateMetricsLogFile, fastLoadYaml
from cpc18_models.BourginMLP import BourginMLP
from cpc18_models.ContextDepNet import ContextDepNet


MASTER_SEED_FOR_POOL = 20260522   # same shared pool as the main run
VAL_SPLIT_SEED = 20260710         # fixed problem-level split for ALL trials
N_VAL_PROBLEMS = 42               # ~20% of the 208 train problems
N_TRIALS = 200
MODELS = (BourginMLP, ContextDepNet)

patience = 50
num_epochs = 500                  # main-run cap
batch_size = 32
learning_rate = 1e-3

_YAML_NAME = "valsplit_r2_training_metrics.yaml"

device = torch.device("cpu")


def _metricsPath(model_name: str) -> str:
    p = os.path.join(output_path, model_name, _YAML_NAME)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def _seedsAlreadyDone(metrics_log_file: str) -> set:
    existing = fastLoadYaml(metrics_log_file)
    return {entry["seed"] for entry in existing.values()
            if isinstance(entry, dict) and "seed" in entry}


def _loadValSplit():
    """208 train problems -> 166 train / 42 val (by problem, fixed split);
    canonical 60-problem test untouched. Standardization uses the REDUCED
    train-set statistics (honest protocol)."""
    df = pd.read_csv(os.path.join(data_path, FEATURES_CSV_NAME))
    train_df = df[df["Train_Test"] == "Train"]
    test_df = df[df["Train_Test"] == "Test"]

    problems = np.sort(train_df["problem_id"].unique())
    rng = np.random.default_rng(VAL_SPLIT_SEED)
    val_problems = set(rng.choice(problems, size=N_VAL_PROBLEMS,
                                  replace=False).tolist())
    tr_df = train_df[~train_df["problem_id"].isin(val_problems)]
    va_df = train_df[train_df["problem_id"].isin(val_problems)]

    X_tr, X_va, X_te = _featurize(tr_df), _featurize(va_df), _featurize(test_df)
    mean = X_tr.mean(axis=0)
    std = np.where(X_tr.std(axis=0) == 0, 1.0, X_tr.std(axis=0))
    X_tr, X_va, X_te = [(x - mean) / std for x in (X_tr, X_va, X_te)]

    y = lambda d: d["B_rate"].astype(np.float32).values
    return ((X_tr.astype(np.float32), y(tr_df)),
            (X_va.astype(np.float32), y(va_df)),
            (X_te.astype(np.float32), y(test_df)),
            len(problems) - N_VAL_PROBLEMS, N_VAL_PROBLEMS)


def _evalMSE(model, loader):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for X, yy in loader:
            X, yy = X.to(device), yy.to(device)
            preds = model(X)
            tot += torch.sum((preds - yy) ** 2).item()
            n += X.size(0)
    return tot / max(n, 1)


def _runOneTrial(task):
    model_cls_name, seed = task
    torch.set_num_threads(1)
    set_seed(seed)

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), n_tr, n_va = _loadValSplit()
    train_loader = DataLoader(CPC18Dataset(X_tr, y_tr),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CPC18Dataset(X_va, y_va),
                            batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(CPC18Dataset(X_te, y_te),
                             batch_size=batch_size, shuffle=False)

    ModelClass = {c.__name__: c for c in MODELS}[model_cls_name]
    model = ModelClass(
        input_size=X_tr.shape[1],
        learning_rate=learning_rate,
        patience=patience,
        seed=seed,
        output_path=None,      # no checkpointing: selection reads the curves
        device=device,
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    val_curve, test_curve = [], []
    best_val, no_improve = float("inf"), 0
    start = time.time()

    for epoch in range(num_epochs):
        model.train()
        for X, yy in train_loader:
            X, yy = X.to(device), yy.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), yy)
            loss.backward()
            optimizer.step()

        v = _evalMSE(model, val_loader)
        t = _evalMSE(model, test_loader)
        val_curve.append(v)
        test_curve.append(t)

        if v < best_val:
            best_val, no_improve = v, 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    duration = time.time() - start
    best_val_epoch = int(np.argmin(val_curve))
    best_test_epoch = int(np.argmin(test_curve))
    mse_valsel = test_curve[best_val_epoch]
    mse_testsel = test_curve[best_test_epoch]

    ts = (datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
          + f".{datetime.now().microsecond // 1000:03d}")
    entry = {
        "seed": seed,
        "seed_mode": "shared_pool",
        "regime": 2,
        "protocol": "val_split_early_stop",
        "val_split_seed": VAL_SPLIT_SEED,
        "n_train_problems": n_tr,
        "n_val_problems": n_va,
        "patience": patience,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "mse": mse_valsel,
        "mse_x100": mse_valsel * 100.0,
        "val_mse_at_selection": val_curve[best_val_epoch],
        "mse_x100_testsel": mse_testsel * 100.0,
        "best_epoch": best_val_epoch,
        "best_test_epoch": best_test_epoch,
        "total_epochs": len(val_curve),
        "train_duration": duration,
        "val_mse_plot": val_curve,
        "mse_plot": test_curve,
    }
    return model_cls_name, seed, {ts: entry}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    args = parser.parse_args()

    pool = np.random.default_rng(MASTER_SEED_FOR_POOL).integers(
        0, 2**32, size=args.n_trials, dtype=np.uint64)

    tasks = []
    for cls in MODELS:
        done = _seedsAlreadyDone(_metricsPath(cls.__name__))
        pending = [(cls.__name__, int(s)) for s in pool if int(s) not in done]
        log(f"[plan] {cls.__name__:15s} done={args.n_trials - len(pending):4d}"
            f"/{args.n_trials}  pending={len(pending)}")
        tasks.extend(pending)
    if not tasks:
        log("[main] Nothing to do.")
        return
    log(f"[main] Pending trials: {len(tasks)}. Workers: {args.workers}.")

    def _store(name, seed, run_metrics):
        atomicUpdateMetricsLogFile(run_metrics, _metricsPath(name))
        log(f"[main] done {name}/{seed}")

    failed = 0
    if args.workers <= 1:
        for task in tasks:
            try:
                _store(*_runOneTrial(task))
            except Exception:
                failed += 1
                log(f"[main] task {task} crashed:\n{traceback.format_exc()}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_runOneTrial, t): t for t in tasks}
            for fut in as_completed(futs):
                try:
                    _store(*fut.result())
                except Exception:
                    failed += 1
                    log(f"[main] task {futs[fut]} crashed:\n"
                        f"{traceback.format_exc()}")
    if failed:
        raise SystemExit(f"{failed} of {len(tasks)} trials failed")


if __name__ == "__main__":
    main()
