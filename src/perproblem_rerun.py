#!/usr/bin/env python3
# encoding: utf-8
"""
Re-run a subset of seeds saving per-(test problem) squared errors, for the
test-sampling-variance (§4.1) and which-problems-concentrate-variance (§5.7)
analyses. Aggregate test MSE is unchanged; this only records, for each trial,
the mean squared error on each of the 60 test problems (averaged over its 5
blocks). Output: analysis_results/perproblem_errors.npz.

Run:  python perproblem_rerun.py [N_SEEDS] [WORKERS]
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
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
from cpc18_utils.features import (load_cpc18_features, cpc18_features_input_size,
                                  data_path, FEATURES_CSV_NAME)
from cpc18_utils.data import CPC18Dataset
from cpc18_models.BourginMLP import BourginMLP
from cpc18_models.BEAST_GB import BEAST_GB
from cpc18_models.PsychForest import PsychForest

MASTER_SEED = 20260522
MODELS = {"BourginMLP": BourginMLP, "BEAST_GB": BEAST_GB, "PsychForest": PsychForest}
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


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    pool = np.random.default_rng(MASTER_SEED).integers(
        0, 2**32, size=n_seeds, dtype=np.uint64)
    seeds = [int(s) for s in pool]
    tasks = [(mn, s) for mn in MODELS for s in seeds]
    results = {mn: np.full((n_seeds, len(PROBLEMS)), np.nan) for mn in MODELS}
    idx = {s: i for i, s in enumerate(seeds)}
    log(f"[pp] {len(tasks)} tasks ({len(MODELS)} models x {n_seeds} seeds), "
        f"{len(PROBLEMS)} test problems, workers={workers}")
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_per_problem_se, t): t for t in tasks}
        for fut in as_completed(futs):
            mn, s, per = fut.result()
            results[mn][idx[s]] = per
            done += 1
            if done % 25 == 0 or done == len(tasks):
                log(f"[pp] {done}/{len(tasks)} done")
    out = os.path.join(os.path.dirname(data_path), "analysis_results",
                       "perproblem_errors.npz")
    np.savez(out, problems=np.array(PROBLEMS), seeds=np.array(seeds), **results)
    log(f"[pp] wrote {out}")


if __name__ == "__main__":
    main()
