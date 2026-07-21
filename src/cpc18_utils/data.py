#!/usr/bin/env python3
# encoding: utf-8

"""
CPC18 dataset loader with MNIST-style auto-download.

The official CPC18 competition kept its test set hidden, so we work
with the public Zenodo release of trial-level data
(https://zenodo.org/records/2571510, "all CPC18 raw data.csv", ~65 MB)
and aggregate it to (GameID, block) -> B_rate. Train/test split is done
by GameID (so no problem leaks across the split), 80/20 with a fixed
split seed by default.

First call downloads the raw CSV into `data/` and caches the aggregated
parquet next to it; later calls are instant.
"""

import os
import urllib.request
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import data_path


# ---------------------------------------------------------------------
# Zenodo source + on-disk paths
# ---------------------------------------------------------------------

ZENODO_RAW_URL = (
    "https://zenodo.org/api/records/2571510/files/"
    "all%20CPC18%20raw%20data.csv/content"
)
RAW_CSV_NAME = "all_CPC18_raw_data.csv"          # ~65 MB
AGG_PARQUET_NAME = "cpc18_aggregate.parquet"     # ~tens of KB after aggregation


# ---------------------------------------------------------------------
# Schema (matches CPC18 raw CSV header)
# ---------------------------------------------------------------------

RAW_FEATURE_COLS = [
    "Ha", "pHa", "La", "LotShapeA", "LotNumA",
    "Hb", "pHb", "Lb", "LotShapeB", "LotNumB",
    "Amb", "Corr",
]
BLOCK_COL = "block"
PROBLEM_ID_COL = "GameID"
CHOICE_COL = "B"            # 0/1 per trial
TARGET_COL = "B_rate"       # what we predict (aggregate)

# LotShape vocabulary observed in CPC18.
_LOT_SHAPES = ("-", "Symm", "L-skew", "R-skew")


# ---------------------------------------------------------------------
# Download + aggregation
# ---------------------------------------------------------------------

def _download_raw(dest: str) -> None:
    """Stream the raw CSV from Zenodo into `dest`."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    print(f"[cpc18_utils.data] Downloading raw CPC18 CSV from Zenodo "
          f"({ZENODO_RAW_URL}) -> {dest} ...")
    urllib.request.urlretrieve(ZENODO_RAW_URL, tmp)
    os.replace(tmp, dest)
    print(f"[cpc18_utils.data] Done ({os.path.getsize(dest) / 1e6:.1f} MB).")


def _aggregate(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trial-level data to (GameID, block) -> B_rate plus the
    lottery features (which are constant within a GameID)."""
    # Target: mean of binary B choice per (problem, block).
    grouped = (
        raw_df.groupby([PROBLEM_ID_COL, BLOCK_COL])[CHOICE_COL]
        .mean()
        .rename(TARGET_COL)
        .reset_index()
    )
    # Per-problem features: take first row per GameID (they're constant
    # within a problem by construction).
    features = (
        raw_df.groupby(PROBLEM_ID_COL)[RAW_FEATURE_COLS]
        .first()
        .reset_index()
    )
    return grouped.merge(features, on=PROBLEM_ID_COL, how="left")


def _build_or_load_aggregate(base_dir: str) -> pd.DataFrame:
    """Return the aggregated dataframe, downloading + aggregating once."""
    raw_path = os.path.join(base_dir, RAW_CSV_NAME)
    agg_path = os.path.join(base_dir, AGG_PARQUET_NAME)

    if os.path.exists(agg_path):
        return pd.read_parquet(agg_path)

    if not os.path.exists(raw_path):
        _download_raw(raw_path)

    print(f"[cpc18_utils.data] Aggregating raw CSV -> {agg_path} ...")
    raw_df = pd.read_csv(raw_path)
    agg_df = _aggregate(raw_df)
    os.makedirs(base_dir, exist_ok=True)
    agg_df.to_parquet(agg_path, index=False)
    print(f"[cpc18_utils.data] Aggregate has {len(agg_df)} rows, "
          f"{agg_df[PROBLEM_ID_COL].nunique()} unique problems.")
    return agg_df


# ---------------------------------------------------------------------
# Featurisation
# ---------------------------------------------------------------------

def _encode_lot_shape(series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {f"{series.name}_{shape}": (series == shape).astype(np.float32)
         for shape in _LOT_SHAPES}
    )


def _featurize(df: pd.DataFrame) -> np.ndarray:
    parts = []
    for col in RAW_FEATURE_COLS:
        if col in ("LotShapeA", "LotShapeB"):
            parts.append(_encode_lot_shape(df[col]).values)
        else:
            parts.append(df[col].astype(np.float32).values.reshape(-1, 1))
    parts.append(df[BLOCK_COL].astype(np.float32).values.reshape(-1, 1))
    return np.hstack(parts).astype(np.float32)


def cpc18_input_size() -> int:
    # 10 raw numeric + 2 * 4 one-hot + 1 block = 19
    return (len(RAW_FEATURE_COLS) - 2) + 2 * len(_LOT_SHAPES) + 1


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

@dataclass
class CPC18Splits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_problems: np.ndarray   # GameIDs in train
    test_problems: np.ndarray    # GameIDs in test


def load_cpc18_aggregate(
    base_dir: str = data_path,
    split_seed: int = 42,
    test_fraction: float = 0.2,
) -> CPC18Splits:
    """Auto-download (if needed), aggregate, and split CPC18.

    Split is by GameID — every block of a given problem is either in
    train or test, never both. Vary `split_seed` to study split-induced
    variance (the paper's "ya puestos, probemos varios splits" idea).
    """
    agg_df = _build_or_load_aggregate(base_dir)

    unique_problems = agg_df[PROBLEM_ID_COL].unique()
    rng = np.random.default_rng(split_seed)
    rng.shuffle(unique_problems)
    n_test = int(round(len(unique_problems) * test_fraction))
    test_problems = unique_problems[:n_test]
    train_problems = unique_problems[n_test:]

    train_df = agg_df[agg_df[PROBLEM_ID_COL].isin(train_problems)]
    test_df = agg_df[agg_df[PROBLEM_ID_COL].isin(test_problems)]

    X_train = _featurize(train_df)
    y_train = train_df[TARGET_COL].astype(np.float32).values
    X_test = _featurize(test_df)
    y_test = test_df[TARGET_COL].astype(np.float32).values

    return CPC18Splits(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        train_problems=train_problems,
        test_problems=test_problems,
    )


class CPC18Dataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
