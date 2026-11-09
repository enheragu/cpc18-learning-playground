#!/usr/bin/env python3
# encoding: utf-8
"""
Split-induced variance (Section 3.4.2 of the manuscript). The canonical 210/60
split is one realisation.
Here we repartition the 268 BEAST-feature problems into N random 208/60
splits and retrain the stable models on each, with the initialisation seed
held FIXED so the only thing that varies is the split. The BEAST features are
per-problem and split-independent, so repartitioning the existing CSV is a
valid way to vary the split without re-running the R feature engineering.

Output: analysis_results/split_variance.npz + a printed sigma_split table.

With `--protocol stable` it trains the four neural models instead, under the
stable protocol (cpc18_utils/protocol.py), on the same splits and with the same
fixed seed, and writes one JSON line per model and split, with the whole test
and training curves, to output_data_stable/split_variance.jsonl.

Run:  python split_variance_rerun.py [N_SPLITS] [--protocol original|stable]
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import tempfile
import shutil

import path_setup  # noqa: F401
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.log_utils import log
from utils.set_seed import set_seed
from cpc18_utils import stable_output_path
from cpc18_utils.features import (_featurize, _standardize,
                                  cpc18_features_input_size,
                                  data_path, FEATURES_CSV_NAME)
from cpc18_utils.data import CPC18Dataset
from cpc18_utils.protocol import (PROTOCOLS, PROTOCOL_ORIGINAL, PROTOCOL_STABLE,
                                  STABLE_INIT, readoutEpoch, stablePatience, stableEntry)
from cpc18_models.BEAST_GB import BEAST_GB
from cpc18_models.PsychForest import PsychForest
from cpc18_models.LinearLogistic import LinearLogistic
from cpc18_models.BourginMLP import BourginMLP
from cpc18_models.ContextDepNet import ContextDepNet
from cpc18_models.SparseMLP import SparseMLP

MODELS = {"BEAST_GB": BEAST_GB, "PsychForest": PsychForest,
          "LinearLogistic": LinearLogistic}
# Under the stable protocol: the neural models, whose results depend on it.
STABLE_MODELS = {"LinearLogistic": LinearLogistic, "BourginMLP": BourginMLP,
                 "ContextDepNet": ContextDepNet, "SparseMLP": SparseMLP}
FIXED_SEED = 3294508113   # pool[0]; held fixed so only the split varies
N_TEST, LR, PATIENCE, NUM_EPOCHS, BATCH = 60, 0.001, 50, 500, 32

_DF = pd.read_csv(os.path.join(data_path, FEATURES_CSV_NAME))
_PROBLEMS = _DF["problem_id"].unique()   # 268 (208 train + 60 test; problems 13/20 excluded at feature stage)


def _split_trial(model_name, split_seed, protocol=PROTOCOL_ORIGINAL):
    """ Metrics of one model trained on one random split (an entry of spinTrainEval). """
    stable = protocol == PROTOCOL_STABLE
    ModelClass = (STABLE_MODELS if stable else MODELS)[model_name]
    rng = np.random.default_rng(split_seed)
    test_p = set(rng.permutation(_PROBLEMS)[:N_TEST].tolist())
    tr = _DF[~_DF["problem_id"].isin(test_p)]
    te = _DF[_DF["problem_id"].isin(test_p)]
    Xtr, Xte = _standardize(_featurize(tr), _featurize(te))
    ytr = tr["B_rate"].astype(np.float32).values
    yte = te["B_rate"].astype(np.float32).values
    set_seed(FIXED_SEED)
    tl = DataLoader(CPC18Dataset(Xtr.astype(np.float32), ytr),
                    batch_size=BATCH, shuffle=True)
    tel = DataLoader(CPC18Dataset(Xte.astype(np.float32), yte),
                     batch_size=BATCH, shuffle=False)
    patience, init_kwargs = PATIENCE, {}
    if stable:
        patience, init_kwargs = stablePatience(model_name), {"init": STABLE_INIT}
        readout = readoutEpoch(model_name, BATCH, LR, getattr(ModelClass, "epsilon", None))
    tmp = tempfile.mkdtemp()
    try:
        m = ModelClass(input_size=cpc18_features_input_size(),
                       learning_rate=LR, patience=patience,
                       seed=FIXED_SEED, output_path=tmp,
                       device=torch.device("cpu"), **init_kwargs)
        rm = m.spinTrainEval(tl, tel, NUM_EPOCHS)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    entry = next(iter(rm.values()))
    if stable:
        stableEntry(entry, readout, m.init_name)
        entry["patience"] = patience
    return entry


def _stableRecord(model_name, split, entry):
    return {"model": model_name, "split_index": split, "split_seed": 1000 + split,
            "init_seed": FIXED_SEED, "patience": entry["patience"],
            "readout_epoch": entry["readout_epoch"], "mse_x100": entry["mse_x100"],
            "mse_x100_testmin": entry["mse_x100_testmin"], "best_epoch": entry["best_epoch"],
            "n_epochs": len(entry["mse_plot"]), "test_mse": entry["mse_plot"],
            "train_mse": entry["train_mse_plot"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n_splits", nargs="?", type=int, default=20)
    parser.add_argument("--protocol", choices=PROTOCOLS, default=PROTOCOL_ORIGINAL,
                        help="Training protocol (cpc18_utils/protocol.py)")
    args = parser.parse_args()
    n = args.n_splits
    stable = args.protocol == PROTOCOL_STABLE
    models = STABLE_MODELS if stable else MODELS
    res = {mn: [] for mn in models}
    records = []
    for split in range(n):
        for mn in models:
            entry = _split_trial(mn, 1000 + split, args.protocol)
            res[mn].append(entry["mse_x100"])
            if stable:
                records.append(_stableRecord(mn, split, entry))
        log(f"[split] {split + 1}/{n} done")
    print(f"\n=== sigma across splits (MSE x100), init seed fixed, protocol {args.protocol} ===")
    for mn in models:
        a = np.array(res[mn])
        print(f"{mn:16s} mean={a.mean():.3f} std={a.std(ddof=1):.4f} "
              f"min={a.min():.3f} max={a.max():.3f}")
    if stable:
        out = os.path.join(stable_output_path, "split_variance.jsonl")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out + ".tmp", "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        os.replace(out + ".tmp", out)
    else:
        out = os.path.join(os.path.dirname(data_path), "analysis_results",
                           "split_variance.npz")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez(out, **{mn: np.array(res[mn]) for mn in models})
    log(f"[split] wrote {out}")


if __name__ == "__main__":
    main()
