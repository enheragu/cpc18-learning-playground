#!/usr/bin/env python3
# encoding: utf-8

"""
Output of BourginMLP before any training under the initialization of the
original protocol, the fragile configuration of the article (Section 3.6): the
share of the test rows on which the sigmoid output is saturated (below 0.01 or
above 0.99) and the initial test MSE x 100, over the first 20 seeds of the shared
pool, against the test MSE x 100 of a constant prediction of the mean of the
training targets. It only builds the models, so it takes a few seconds and needs
neither the training logs nor the data deposit.

Usage:
    cd src && python 22_initial_saturation.py
"""

import os

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import tempfile

import path_setup  # noqa: F401

import numpy as np
import torch

from utils.set_seed import set_seed

from cpc18_models import BourginMLP
from cpc18_utils.features import load_cpc18_features, cpc18_features_input_size

MASTER_SEED_FOR_POOL = 20260522
N_SEEDS = 20


def main():
    splits = load_cpc18_features()
    X_test, y_test = torch.tensor(splits.X_test), torch.tensor(splits.y_test)
    seeds = np.random.default_rng(MASTER_SEED_FOR_POOL).integers(0, 2 ** 32, size=N_SEEDS, dtype=np.uint64).astype(np.int64)

    saturated, initial_mse = [], []
    with tempfile.TemporaryDirectory() as out:
        for seed in seeds:
            set_seed(int(seed))
            model = BourginMLP(input_size=cpc18_features_input_size(), learning_rate=1e-3, patience=50, seed=int(seed),
                               output_path=out, device=torch.device("cpu"))
            with torch.no_grad():
                output = model(X_test).squeeze()
            saturated.append(float(((output < 0.01) | (output > 0.99)).float().mean()))
            initial_mse.append(float(100 * ((output - y_test) ** 2).mean()))

    constant = 100 * float(((y_test - float(splits.y_train.mean())) ** 2).mean())
    print(f"Saturated output before training: {100 * np.mean(saturated):.0f}% of the test rows on average over {N_SEEDS} "
          f"seeds (from {100 * min(saturated):.0f} to {100 * max(saturated):.0f}%)")
    print(f"Initial test MSE x 100: median {np.median(initial_mse):.1f}; constant prediction of the training mean: "
          f"{constant:.2f}")


if __name__ == "__main__":
    main()
