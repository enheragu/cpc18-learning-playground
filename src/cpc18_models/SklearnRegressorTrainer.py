#!/usr/bin/env python3
# encoding: utf-8

"""
SklearnRegressorTrainer: a parallel trainer for sklearn-style estimators
(RandomForestRegressor, XGBRegressor, ...) that exposes the same
`spinTrainEval` interface as RegressionBaseModelTrainer.

Variance for these models comes from the estimator's own `random_state`
(bootstrap sampling, feature subsetting, tree split randomness). We
funnel the per-trial seed into it so the existing seed-mode machinery
in 01_train_regime2.py works identically to the torch models.

Subclasses only need to implement `_build_estimator(seed)` returning a
fitted-or-fittable sklearn-style object.
"""

import os
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.log_utils import log


def _flatten_loader(loader: DataLoader):
    Xs, ys = [], []
    for X, y in loader:
        Xs.append(X.cpu().numpy())
        ys.append(y.cpu().numpy())
    X = np.vstack(Xs)
    y = np.vstack(ys).ravel()
    return X.astype(np.float32), y.astype(np.float32)


class SklearnRegressorTrainer:
    """Drop-in trainer with the same external surface as
    RegressionBaseModelTrainer (we duck-type rather than inherit, since
    these are not nn.Modules)."""

    def __init__(self, input_size, learning_rate=None, patience=None,
                 seed=42, output_path="", device=None):
        # learning_rate / patience / device are accepted but unused; they
        # exist so the caller can keep one uniform constructor signature.
        del learning_rate, patience, device
        self.input_size = input_size
        self.seed = seed
        self.model_name = type(self).__name__
        self.base_output_path = output_path
        if self.base_output_path:
            self.output_data_path = os.path.join(self.base_output_path, self.model_name)
            os.makedirs(self.output_data_path, exist_ok=True)

        self.estimator = None  # filled by _build_estimator on spinTrainEval

    # Subclasses implement.
    def _build_estimator(self, seed):
        raise NotImplementedError

    def spinTrainEval(self, train_loader, test_loader, num_epochs=None):
        del num_epochs  # not applicable
        log(f"[{self.model_name}] Init spin (seed={self.seed}).")

        X_train, y_train = _flatten_loader(train_loader)
        X_test, y_test = _flatten_loader(test_loader)

        self.estimator = self._build_estimator(self.seed)
        t0 = time.time()
        self.estimator.fit(X_train, y_train)
        train_duration = time.time() - t0

        y_pred = self.estimator.predict(X_test).astype(np.float32)
        mse = float(np.mean((y_pred - y_test) ** 2))
        mse_random = float(np.var(y_test))

        log(f"[{self.model_name}] Training took {train_duration:.2f}s. "
            f"Test MSE = {mse:.6f}")

        timestamp = (datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                     + f".{datetime.now().microsecond // 1000:03d}")
        return {timestamp: {
            "mse": mse,
            "mse_x100": mse * 100.0,
            "mse_random_baseline": mse_random,
            "train_duration": train_duration,
            # Non-iterative trainers: epoch fields not meaningful, kept
            # as 0/[] so downstream code that expects them doesn't break.
            "best_epoch": 0,
            "total_epochs": 0,
            "mse_plot": [],
            "train_mse_plot": [],
            "seed": int(self.seed),
        }}
