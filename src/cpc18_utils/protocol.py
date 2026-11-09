#!/usr/bin/env python3
# encoding: utf-8

"""
Training protocols of the study (`--protocol` of the trainers).

- original: the initialization of each model (Kaiming normal in fan-out mode,
  or LinearLogistic's uniform one) and, as the result of a trial, the test MSE
  of the epoch with the lowest test MSE. Early stopping on the test curve,
  patience 50. It is the protocol of release 1.0.0 and the default.
- stable: PyTorch's default initialization of nn.Linear (`init` of the models,
  see cpc18_models/base.py), drawn without changing the data order of the
  seed, and, as the result of a trial, the test MSE at a fixed epoch per
  configuration, chosen by 5-fold cross-validation on the 208 training
  problems (19_cv_epochs.py writes data/cv_epochs.csv), never at the minimum
  of the test curve. Training still stops on the test curve, with patience 50
  (120 for LinearLogistic, so that every curve reaches its epoch 99): that only
  decides when to stop. The whole curve is kept, and its minimum is logged as a
  sensitivity readout.

Everything else (architectures, Adam with learning rate 1e-3, batch size 32,
features, canonical split, seed pools, factorial design and ablation grids) is
the same under both. The tree-based models have no initialization nor epochs:
their results serve both protocols, and they are trained only under the
original one.
"""

import csv
import functools
import os

from . import data_path


PROTOCOL_ORIGINAL = "original"
PROTOCOL_STABLE = "stable"
PROTOCOLS = (PROTOCOL_ORIGINAL, PROTOCOL_STABLE)

# Neural models whose results depend on the protocol.
NEURAL_MODELS = ("LinearLogistic", "BourginMLP", "ContextDepNet", "SparseMLP")

# Initialization of the models under the stable protocol (see cpc18_models/base.py).
STABLE_INIT = "pytorch_default"

# Early-stopping patience under the stable protocol. LinearLogistic is read at
# its epoch 99, which some of its curves do not reach with patience 50.
STABLE_PATIENCE = 50
STABLE_PATIENCE_BY_MODEL = {"LinearLogistic": 120}

# Number of epochs per configuration chosen by cross-validation (column e_full),
# written by 19_cv_epochs.py --summary.
CV_EPOCHS_CSV = os.path.join(data_path, "cv_epochs.csv")


def cvConfigKey(model, batch_size, learning_rate, epsilon=None):
    """ Name of a configuration in cv_epochs.csv, e.g. `SparseMLP_B32_lr0.001_E20`. """
    key = f"{model}_B{int(batch_size)}_lr{learning_rate:g}"
    return key + (f"_E{epsilon:g}" if epsilon is not None else "")


@functools.lru_cache(maxsize=1)
def _cvEpochs():
    if not os.path.exists(CV_EPOCHS_CSV):
        raise FileNotFoundError(f"Missing {CV_EPOCHS_CSV}: run `python 19_cv_epochs.py --summary` "
                                f"after the cross-validation (see the README).")
    with open(CV_EPOCHS_CSV, newline="") as f:
        return {row["config"]: int(row["e_full"]) for row in csv.DictReader(f)}


def fixedEpochs(model, batch_size, learning_rate, epsilon=None):
    """ Epochs of training whose test MSE the stable protocol reports for a configuration. """
    key = cvConfigKey(model, batch_size, learning_rate, epsilon)
    epochs = _cvEpochs().get(key)
    if epochs is None:
        raise KeyError(f"{key} is not in {CV_EPOCHS_CSV}: run its cross-validation with "
                       f"`python 19_cv_epochs.py --only {key}`, then `--summary`.")
    return epochs


def readoutEpoch(model, batch_size, learning_rate, epsilon=None):
    """ 0-based index, in the per-epoch curves, of the epoch the stable protocol reports. """
    return fixedEpochs(model, batch_size, learning_rate, epsilon) - 1


def stablePatience(model):
    return STABLE_PATIENCE_BY_MODEL.get(model, STABLE_PATIENCE)


def stableEntry(entry, readout_epoch, init_name):
    """
        Turns the metrics of a trial (an entry of spinTrainEval, curves in
        `mse_plot`) into those of the stable protocol: `mse` and `mse_x100` are
        the test MSE after `readout_epoch` + 1 epochs, and `mse_x100_testmin`
        the minimum of the test curve. `best_epoch` stays the epoch of that
        minimum. Modifies and returns `entry`.
    """
    curve = entry["mse_plot"]
    if readout_epoch >= len(curve):
        raise RuntimeError(f"the test curve has {len(curve)} epochs and the stable protocol reads "
                           f"its epoch {readout_epoch + 1}: train with a larger patience")
    entry["protocol"] = PROTOCOL_STABLE
    entry["init"] = init_name
    entry["readout_epoch"] = readout_epoch
    entry["mse"] = curve[readout_epoch]
    entry["mse_x100"] = curve[readout_epoch] * 100.0
    entry["mse_x100_testmin"] = min(curve) * 100.0
    return entry
