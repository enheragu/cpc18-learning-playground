#!/usr/bin/env python3
# encoding: utf-8
"""
Re-run a subset of seeds saving per-(test problem) squared errors, for the
test-sampling-variance and which-problems-concentrate-variance analyses
(Section 3.4.2 of the manuscript). Aggregate test MSE is unchanged; this only records, for each trial,
the mean squared error on each of the 60 test problems (averaged over its 5
blocks). Output: analysis_results/perproblem_errors.npz.

With `--protocol stable` it trains the four neural models instead, under the
stable protocol (cpc18_utils/protocol.py): each one for its fixed number of
epochs (data/cv_epochs.csv), without early stopping, and the errors are those
of the weights after the last epoch. Output:
output_data_stable/perproblem_errors.npz, with the same arrays plus, per
model, the aggregate test MSE x 100 recomputed from the predictions
(`<model>_mse_x100`), the last value of the test curve (`<model>_curve_x100`)
and the number of epochs (`<model>_epochs`).

Run:  python perproblem_rerun.py [N_SEEDS] [WORKERS] [--protocol original|stable]
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

import path_setup  # noqa: F401
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.log_utils import log
from utils.set_seed import set_seed
from cpc18_utils import stable_output_path
from cpc18_utils.features import (load_cpc18_features, cpc18_features_input_size,
                                  data_path, FEATURES_CSV_NAME)
from cpc18_utils.data import CPC18Dataset
from cpc18_utils.protocol import PROTOCOLS, PROTOCOL_ORIGINAL, PROTOCOL_STABLE, STABLE_INIT, fixedEpochs
from cpc18_models.BourginMLP import BourginMLP
from cpc18_models.BEAST_GB import BEAST_GB
from cpc18_models.PsychForest import PsychForest
from cpc18_models.LinearLogistic import LinearLogistic
from cpc18_models.ContextDepNet import ContextDepNet
from cpc18_models.SparseMLP import SparseMLP

MASTER_SEED = 20260522
MODELS = {"BourginMLP": BourginMLP, "BEAST_GB": BEAST_GB, "PsychForest": PsychForest}
# Under the stable protocol: the neural models, whose results depend on it.
STABLE_MODELS = {"LinearLogistic": LinearLogistic, "ContextDepNet": ContextDepNet,
                 "BourginMLP": BourginMLP, "SparseMLP": SparseMLP}
LR, PATIENCE, NUM_EPOCHS, BATCH = 0.001, 50, 500, 32

_df = pd.read_csv(os.path.join(data_path, FEATURES_CSV_NAME))
_TEST_PID = _df[_df["Train_Test"] == "Test"]["problem_id"].values  # aligned with X_test
PROBLEMS = sorted(set(_TEST_PID.tolist()))


def _per_problem_se(task):
    model_name, seed = task
    torch.set_num_threads(1)
    splits = load_cpc18_features()
    inp = cpc18_features_input_size()
    set_seed(seed)
    tl = DataLoader(CPC18Dataset(splits.X_train, splits.y_train),
                    batch_size=BATCH, shuffle=True)
    te = DataLoader(CPC18Dataset(splits.X_test, splits.y_test),
                    batch_size=BATCH, shuffle=False)
    tmp = tempfile.mkdtemp()
    try:
        m = MODELS[model_name](input_size=inp, learning_rate=LR, patience=PATIENCE,
                               seed=int(seed), output_path=tmp, device=torch.device("cpu"))
        m.spinTrainEval(tl, te, NUM_EPOCHS)
        if getattr(m, "estimator", None) is not None:        # sklearn / xgboost
            preds = np.asarray(m.estimator.predict(splits.X_test)).ravel()
        else:                                                 # neural net
            m.eval()
            with torch.no_grad():
                preds = m(torch.from_numpy(splits.X_test)).cpu().numpy().ravel()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    se = (preds - splits.y_test) ** 2
    per = np.array([se[_TEST_PID == p].mean() for p in PROBLEMS]) * 100.0  # MSE x100
    return model_name, int(seed), per


def _stable_per_problem_se(task):
    """
        Stable protocol: trains for the fixed number of epochs of the model
        (patience above it, so that it never stops early) and returns the
        squared errors x 100 of the last weights, averaged per test problem,
        with their aggregate and the last value of the test curve.
    """
    model_name, seed = task
    ModelClass = STABLE_MODELS[model_name]
    epochs = fixedEpochs(model_name, BATCH, LR, getattr(ModelClass, "epsilon", None))
    torch.set_num_threads(1)
    splits = load_cpc18_features()
    set_seed(seed)
    tl = DataLoader(CPC18Dataset(splits.X_train, splits.y_train),
                    batch_size=BATCH, shuffle=True)
    te = DataLoader(CPC18Dataset(splits.X_test, splits.y_test),
                    batch_size=BATCH, shuffle=False)
    tmp = tempfile.mkdtemp()
    try:
        m = ModelClass(input_size=cpc18_features_input_size(), learning_rate=LR,
                       patience=epochs + 1, seed=int(seed), output_path=tmp,
                       device=torch.device("cpu"), init=STABLE_INIT)
        m.train_model(tl, te, epochs)
        m.eval()
        with torch.no_grad():
            preds = m(torch.from_numpy(splits.X_test)).squeeze(1).cpu().numpy()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    se = 100 * (preds - splits.y_test) ** 2
    per = pd.Series(se).groupby(_TEST_PID).mean()
    assert list(per.index) == PROBLEMS
    return (model_name, int(seed), per.to_numpy(dtype=np.float64), float(se.mean()),
            float(m.mse_each_epoch[-1] * 100), epochs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n_seeds", nargs="?", type=int, default=50)
    parser.add_argument("workers", nargs="?", type=int, default=5)
    parser.add_argument("--protocol", choices=PROTOCOLS, default=PROTOCOL_ORIGINAL,
                        help="Training protocol (cpc18_utils/protocol.py)")
    args = parser.parse_args()
    n_seeds, workers = args.n_seeds, args.workers
    stable = args.protocol == PROTOCOL_STABLE
    models = STABLE_MODELS if stable else MODELS
    pool = np.random.default_rng(MASTER_SEED).integers(
        0, 2**32, size=n_seeds, dtype=np.uint64)
    seeds = [int(s) for s in pool]
    tasks = [(mn, s) for mn in models for s in seeds]
    results = {mn: np.full((n_seeds, len(PROBLEMS)), np.nan) for mn in models}
    extra = {}
    if stable:
        for mn in models:
            extra[f"{mn}_mse_x100"] = np.full(n_seeds, np.nan)
            extra[f"{mn}_curve_x100"] = np.full(n_seeds, np.nan)
    idx = {s: i for i, s in enumerate(seeds)}
    log(f"[pp] {len(tasks)} tasks ({len(models)} models x {n_seeds} seeds), "
        f"{len(PROBLEMS)} test problems, workers={workers}, protocol={args.protocol}")
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_stable_per_problem_se if stable else _per_problem_se, t): t for t in tasks}
        for fut in as_completed(futs):
            mn, s, per, *more = fut.result()
            results[mn][idx[s]] = per
            if stable:
                extra[f"{mn}_mse_x100"][idx[s]], extra[f"{mn}_curve_x100"][idx[s]], epochs = more
                extra[f"{mn}_epochs"] = np.array(epochs)
            done += 1
            if done % 25 == 0 or done == len(tasks):
                log(f"[pp] {done}/{len(tasks)} done")
    if stable:
        out = os.path.join(stable_output_path, "perproblem_errors.npz")
    else:
        out = os.path.join(os.path.dirname(data_path), "analysis_results",
                           "perproblem_errors.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, problems=np.array(PROBLEMS), seeds=np.array(seeds), **results, **extra)
    log(f"[pp] wrote {out}")


if __name__ == "__main__":
    main()
