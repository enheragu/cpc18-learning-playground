#!/usr/bin/env python3
# encoding: utf-8

"""
Loader for the precomputed BEAST-derived CPC18 features.

The CSV is produced by `scripts/precompute_features.R` (R, one-shot) and
contains 31 features per (problem, block) plus the canonical
Plonsky 2018 Train_Test split.

For the variance study, features are constant given a (problem, block);
they don't depend on the training seed. We compute them once in R and
reuse the CSV across all Python training runs.
"""

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import data_path
from .data import CPC18Dataset  # reuse: just wraps (X, y) arrays


FEATURES_CSV_NAME = "cpc18_features.csv"

# 12 raw lottery + 17 engineered (4 naive + 12 psychological insight + BEAST point) + 2 admin (block, Feedback)
# Of the 12 raw, two (LotShapeA / LotShapeB) are categorical and get one-hot encoded.
_NUMERIC_RAW = ["Ha", "pHa", "La", "LotNumA",
                "Hb", "pHb", "Lb", "LotNumB",
                "Amb", "Corr"]
_LOT_SHAPE_COLS = ["LotShapeA", "LotShapeB"]
_LOT_SHAPES = ("-", "Symm", "L-skew", "R-skew")
_ENGINEERED = [
    # naive
    "diffEV", "diffSDs", "diffMins", "diffMaxs",
    # psychological insight
    "diffUV", "RatioMin", "SignMax",
    "pBbet_Unbiased1", "pBbet_UnbiasedFB",
    "pBbet_Uniform", "pBbet_Sign1", "pBbet_SignFB",
    "Dom", "diffBEV0", "diffBEVfb", "diffSignEV",
    # behavioural foresight
    "BEASTpred",
]
_ADMIN = ["block", "Feedback"]


def cpc18_features_input_size() -> int:
    # 10 numeric raw + 2 * 4 one-hot + 17 engineered (16 + BEASTpred) + 2 admin = 37
    return len(_NUMERIC_RAW) + 2 * len(_LOT_SHAPES) + len(_ENGINEERED) + len(_ADMIN)


def _encode_lot_shape(series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {f"{series.name}_{shape}": (series == shape).astype(np.float32)
         for shape in _LOT_SHAPES}
    )


def _featurize(df: pd.DataFrame) -> np.ndarray:
    parts = [df[c].astype(np.float32).values.reshape(-1, 1) for c in _NUMERIC_RAW]
    for c in _LOT_SHAPE_COLS:
        parts.append(_encode_lot_shape(df[c]).values)
    parts.extend(df[c].astype(np.float32).values.reshape(-1, 1) for c in _ENGINEERED)
    parts.extend(df[c].astype(np.float32).values.reshape(-1, 1) for c in _ADMIN)
    return np.hstack(parts).astype(np.float32)


@dataclass
class CPC18FeaturesSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_problems: np.ndarray
    test_problems: np.ndarray


def _standardize(X_train: np.ndarray, X_test: np.ndarray):
    """Z-score features using train-set statistics. Mirrors Plonsky 2024
    BEAST-GB methods: 'we standardized all features by subtracting their
    mean and dividing by their standard deviation in the train set.'

    Necessary for the NN models (BourginMLP) because BEAST features mix
    scales by 2-3 orders of magnitude (diffMaxs reaches 247 while
    pBbet_* stay in [-1, 1]); unscaled inputs starve the gradient
    signal on the small-range columns.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)  # avoid div-by-zero on constant cols
    return (X_train - mean) / std, (X_test - mean) / std


def load_cpc18_features(base_dir: str = data_path,
                        standardize: bool = True) -> CPC18FeaturesSplits:
    """Load the precomputed feature CSV and split it by the canonical
    `Train_Test` column (Plonsky 2018: 210 train / 60 test problems; the file
    holds 208 train problems, without 13 and 20, as in the BEAST-GB reference).

    `standardize=True` (default) z-scores features using train-set
    statistics. This is scale-invariant for trees / XGBoost / linear-
    regression-on-MSE, but critical for the NN models."""
    path = os.path.join(base_dir, FEATURES_CSV_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run `Rscript scripts/precompute_features.R` "
            f"once to generate it (see data/README.md)."
        )

    df = pd.read_csv(path)
    if "Train_Test" not in df.columns:
        raise ValueError(f"{path} lacks 'Train_Test' column.")

    train_df = df[df["Train_Test"] == "Train"]
    test_df = df[df["Train_Test"] == "Test"]
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Empty train or test split in features CSV.")

    X_train = _featurize(train_df)
    X_test = _featurize(test_df)
    if standardize:
        X_train, X_test = _standardize(X_train, X_test)

    return CPC18FeaturesSplits(
        X_train=X_train.astype(np.float32),
        y_train=train_df["B_rate"].astype(np.float32).values,
        X_test=X_test.astype(np.float32),
        y_test=test_df["B_rate"].astype(np.float32).values,
        train_problems=train_df["problem_id"].unique(),
        test_problems=test_df["problem_id"].unique(),
    )
