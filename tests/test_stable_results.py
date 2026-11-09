#!/usr/bin/env python3
# encoding: utf-8
"""
Reading the stable protocol from the data deposit (cpc18_utils/stable_results.py):
the networks come from the stable rows of M00 and M01 and never from a forced
path or from the original protocol, the tree-based models from the original
protocol on M00, the frames are ordered by seed, and a factorial grid has the
initialization seeds in its rows.
"""

import csv

import pytest

from cpc18_utils.stable_results import StableResults

FIELDS = ["trial_id", "experiment", "machine_id", "condition", "model", "batch_size", "learning_rate", "epsilon",
          "seed", "init_seed", "order_seed", "mse", "mse_x100", "best_epoch", "protocol", "readout_epoch",
          "mse_x100_testmin"]


def _write(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(i, experiment, machine, condition, protocol, seed, value, **extra):
    return {"trial_id": i, "experiment": experiment, "machine_id": machine, "condition": condition,
            "model": condition, "seed": seed, "mse_x100": value, "protocol": protocol, "readout_epoch": 3,
            "mse_x100_testmin": value - 0.1, **extra}


def test_rows_of_the_stable_reference(tmp_path):
    rows = [_row(1, "main", "M00", "BourginMLP", "stable", 7, 1.0),
            _row(2, "main", "M01", "BourginMLP", "stable", 3, 2.0),
            _row(3, "main", "M01-forced-AVX2", "BourginMLP", "stable", 5, 9.0),   # a forced path
            _row(4, "main", "M00", "BourginMLP", "original", 7, 8.0),             # the original protocol
            _row(5, "main", "M00", "PsychForest", "original", 7, 0.7),
            _row(6, "main", "M01", "PsychForest", "original", 3, 0.6)]           # trees: M00 only
    _write(tmp_path / "trials.csv", rows)
    S = StableResults(str(tmp_path))
    bourgin = S.mainRows("BourginMLP")
    assert bourgin.seed.tolist() == [3, 7] and bourgin.mse_x100.tolist() == [2.0, 1.0]
    assert S.mainRows("PsychForest").mse_x100.tolist() == [0.7]


def test_factorial_grid_has_initialization_seeds_in_rows(tmp_path):
    rows, i = [], 0
    for init in range(35):
        for order in range(35):
            i += 1
            rows.append(_row(i, "factorial_init_order", "M01", "SparseMLP", "stable", 100 + init, init + order / 100,
                             init_seed=100 + init, order_seed=500 + order))
    _write(tmp_path / "trials.csv", rows)
    G = StableResults(str(tmp_path)).factorial("SparseMLP")
    assert G.index.name == "init_seed" and G.columns.name == "order_seed"
    assert G.loc[102, 504] == pytest.approx(2.04)


def test_a_deposit_without_the_stable_protocol_is_refused(tmp_path):
    _write(tmp_path / "trials.csv", [_row(1, "main", "M00", "BourginMLP", "original", 7, 1.0)])
    with pytest.raises(SystemExit):
        StableResults(str(tmp_path))
