#!/usr/bin/env python3
# encoding: utf-8

"""
Per-trial results of a training log. The analysis scripts ask for a log by its
path under output_data[_<tag>]/. When that file does not exist, the trials come
from the public data deposit of the article (the output of 16_export_zenodo.py,
the files archived in Zenodo), so that the figures and tables can be regenerated
without the training outputs. With CPC18_READ_DEPOSIT=1 the deposit is used even
when the logs exist.

The trials of the deposit keep the order of the training logs and their full
double precision, so a seeded analysis gives the same result from either source.
They hold the fields that the analysis uses, named as in the logs, and differ
from a log in three points: the keys are trial ids, because the deposit has no
timestamps; `best_epoch` is always the minimum of the curve of the selection
criterion; and the fields that the deposit does not hold are absent (per-epoch
curves, `seed_mode`, `protocol`, `seed_decomposed`, `total_epochs`, and the
batch size, learning rate and patience of the tree-based models, which use none
of them). The exploratory first regime is not in the deposit.
"""

import csv
import functools
import os
import re

from cpc18_utils import output_path, deposit_path
from cpc18_utils.safe_yaml import fastLoadYaml


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_")
_REFERENCE_MACHINE = "M00"  # machine of output_data/, the tree without a tag
_TREE_PREFIX = "output_data_"

# Experiments of the deposit held by each training log.
_LOG_EXPERIMENTS = {
    "sharedpool_r2_training_metrics.yaml": ("main",),
    "sharedpool_r2_ablation_training_metrics.yaml": ("ablation_bourginmlp", "ablation_sparsemlp"),
    "factorial_init_order_r2_training_metrics.yaml": ("factorial_init_order",),
    "valsplit_r2_training_metrics.yaml": ("validation_split",),
}

# Field of the deposit -> (field of the training logs, type)
_TRIAL_FIELDS = {
    "batch_size": ("batch_size", int),
    "learning_rate": ("learning_rate", float),
    "epsilon": ("epsilon", int),
    "seed": ("seed", int),
    "init_seed": ("init_seed", int),
    "order_seed": ("order_seed", int),
    "mse": ("mse", float),
    "mse_x100": ("mse_x100", float),
    "mse_random_baseline": ("mse_random_baseline", float),
    "best_epoch": ("best_epoch", int),
    "patience": ("patience", int),
    "train_duration_s": ("train_duration", float),
    "val_split_seed": ("val_split_seed", int),
    "n_train_problems": ("n_train_problems", int),
    "n_val_problems": ("n_val_problems", int),
    "best_test_epoch": ("best_test_epoch", int),
    "mse_x100_testsel": ("mse_x100_testsel", float),
    "val_mse_at_selection": ("val_mse_at_selection", float),
}


def _forcedDeposit():
    return os.environ.get("CPC18_READ_DEPOSIT", "") not in ("", "0")


def usesDeposit(path):
    """ Whether the results of the training output at `path` come from the deposit. """
    return _forcedDeposit() or not os.path.exists(path)


def _machineOf(tree_path):
    tree = os.path.basename(os.path.normpath(tree_path))
    return tree[len(_TREE_PREFIX):] if tree.startswith(_TREE_PREFIX) else _REFERENCE_MACHINE


@functools.lru_cache(maxsize=1)
def _depositTrials():
    """ {(machine, experiment, condition): {trial_id: trial}}, in the order of the deposit. """
    path = os.path.join(deposit_path, "trials.csv")
    trials = {}
    if not os.path.exists(path):
        return trials
    # The csv module and float() keep the exact value of every number written in the file.
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            trial = {name: cast(row[field]) for field, (name, cast) in _TRIAL_FIELDS.items() if row[field]}
            key = (row["machine_id"], row["experiment"], row["condition"])
            trials.setdefault(key, {})[row["trial_id"]] = trial
    return trials


def loadTrainingLog(log_path):
    """
        {key: trial} of the training log at `log_path` (<tree>/<condition>/<log name>),
        in the order of the log. Raises FileNotFoundError if neither the log nor the
        deposit holds its trials.
    """
    if not usesDeposit(log_path):
        return {k: v for k, v in fastLoadYaml(log_path).items()
                if _DATE_RE.match(str(k)) and isinstance(v, dict)}

    tree, condition, log_name = os.path.normpath(log_path).split(os.sep)[-3:]
    machine = _machineOf(tree)
    trials = {}
    for experiment in _LOG_EXPERIMENTS.get(log_name, ()):
        cell = _depositTrials().get((machine, experiment, condition), {})
        trials.update((key, dict(trial)) for key, trial in cell.items())
    if not trials:
        raise FileNotFoundError(f"{log_path}: no training log, and the data deposit in "
                                f"{deposit_path} does not hold these trials")
    return trials


def trainingConditions():
    """ Sorted names of the models and ablation cells with results in output_path. """
    names = set(os.listdir(output_path)) if os.path.isdir(output_path) else set()
    machine = _machineOf(output_path)
    names.update(condition for m, _, condition in _depositTrials() if m == machine)
    return sorted(names)


def depositRows(filename):
    """ Rows of a results file of the deposit, models named as in the code; [] without the file. """
    path = os.path.join(deposit_path, filename)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return [dict(row, model=row["model"].replace("-", "_")) for row in csv.DictReader(f)]
