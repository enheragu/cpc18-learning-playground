#!/usr/bin/env python3
# encoding: utf-8

"""
Number of training epochs of each configuration under the stable protocol
(cpc18_utils/protocol.py), chosen by 5-fold cross-validation on the 208
training problems. The test problems are never used.

The folds split the training problems (all the blocks of a problem fall in the
same fold), from a permutation drawn with seed 2026. Each configuration is
trained on four folds and evaluated on the fifth after every epoch, for each of
the five folds and the first 10 seeds of the shared pool, with PyTorch's
default initialization and without early stopping (300 epochs for learning
rates below 1e-3, 200 otherwise). One JSON line per (configuration, fold,
seed) holds its validation curve, as MSE x 100 rounded to 7 decimals.

`--summary` averages the curves of each configuration over its 50 runs: `e_cv`
is the epoch (1-based) of the minimum of that mean curve and `val_min` its
value. The training folds hold about 4/5 of the rows, so the number of epochs
on the full training set is `e_full` = round(e_cv x mean training rows / 1040),
at least 1, which is the number of epochs whose test MSE the stable protocol
reports. It writes `data/cv_epochs.csv`, read by the trainers.

Run:
    python 19_cv_epochs.py --workers 2              # cross-validation, resumable
    python 19_cv_epochs.py --summary                # data/cv_epochs.csv
    python 19_cv_epochs.py --only BourginMLP_B32_lr0.001

Configurations: LinearLogistic and ContextDepNet as in the main run; BourginMLP
at batch size {16, 32, 64} x learning rate {1e-3, 1e-2, 3.16e-4, 1e-4}; and
SparseMLP at epsilon {20, 10, 40} x learning rate {1e-3, 1e-2, 1e-4}.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import shutil
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import path_setup  # noqa: F401

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.log_utils import log
from utils.set_seed import set_seed

import cpc18_models
from cpc18_utils import stable_output_path
from cpc18_utils.features import _featurize, _standardize, cpc18_features_input_size, data_path, FEATURES_CSV_NAME
from cpc18_utils.data import CPC18Dataset
from cpc18_utils.protocol import CV_EPOCHS_CSV, STABLE_INIT, cvConfigKey


MASTER_SEED_FOR_POOL = 20260522    # shared seed pool; the first N_SEEDS seeds are used
FOLD_SEED = 2026                   # permutation of the training problems into folds
N_FOLDS = 5
N_SEEDS = 10
FULL_TRAIN_ROWS = 1040             # 208 training problems x 5 blocks

# (model, batch size, learning rate, epsilon of SparseMLP)
CONFIGS = [("LinearLogistic", 32, 1e-3, None), ("ContextDepNet", 32, 1e-3, None)]
CONFIGS += [("BourginMLP", b, lr, None) for lr in (1e-3, 1e-2, 3.16e-4, 1e-4) for b in (16, 32, 64)]
CONFIGS += [("SparseMLP", 32, lr, e) for e in (20, 10, 40) for lr in (1e-3, 1e-2, 1e-4)]


def _key(config):
    return cvConfigKey(*config)


def _epochCap(learning_rate):
    return 300 if learning_rate < 1e-3 else 200


_TRAIN = None


def _trainProblems():
    global _TRAIN
    if _TRAIN is None:
        df = pd.read_csv(os.path.join(data_path, FEATURES_CSV_NAME))
        _TRAIN = df[df.Train_Test == "Train"].reset_index(drop=True)
    return _TRAIN


def _folds():
    problems = np.array(sorted(_trainProblems().problem_id.unique()))
    perm = np.random.default_rng(FOLD_SEED).permutation(problems)
    return [set(perm[i::N_FOLDS]) for i in range(N_FOLDS)]


def _runOne(task):
    """ Validation curve (MSE x 100) of one configuration, fold and seed. """
    (model, bs, lr, eps), fold, seed = task
    torch.set_num_threads(1)
    train = _trainProblems()
    va = train.problem_id.isin(_folds()[fold]).to_numpy()
    Xa, Xv = _standardize(_featurize(train[~va]), _featurize(train[va]))
    ya, yv = train.B_rate[~va].to_numpy(np.float32), train.B_rate[va].to_numpy(np.float32)
    set_seed(seed)
    train_loader = DataLoader(CPC18Dataset(Xa.astype(np.float32), ya), batch_size=bs, shuffle=True)
    val_loader = DataLoader(CPC18Dataset(Xv.astype(np.float32), yv), batch_size=bs, shuffle=False)
    ModelClass = getattr(cpc18_models, model)
    if eps is not None:
        ModelClass = type(ModelClass.__name__, (ModelClass,), {"epsilon": eps})
    cap = _epochCap(lr)
    tmp = tempfile.mkdtemp(prefix="cv_")
    try:
        m = ModelClass(input_size=cpc18_features_input_size(), learning_rate=lr, patience=cap + 1,
                       seed=seed, output_path=tmp, device=torch.device("cpu"), init=STABLE_INIT)
        m.train_model(train_loader, val_loader, cap)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return dict(config=_key((model, bs, lr, eps)), model=model, batch_size=bs, lr=lr, epsilon=eps,
                fold=fold, seed=int(seed), n_train_rows=int((~va).sum()),
                val_curve=[round(float(x) * 100, 7) for x in m.mse_each_epoch])


def summarize(curves_path, out_path):
    """ The fixed number of epochs per configuration, from the validation curves. """
    with open(curves_path) as f:
        d = pd.DataFrame([json.loads(line) for line in f])
    rows = []
    for key, g in d.groupby("config"):
        length = min(len(c) for c in g.val_curve)
        M = np.full((len(g), length), np.nan)
        for i, c in enumerate(g.val_curve):
            M[i] = c[:length]
        mean_curve = np.nanmean(M, 0)
        e_cv = int(np.nanargmin(mean_curve)) + 1
        frac = float(np.mean(g.n_train_rows) / float(FULL_TRAIN_ROWS))
        rows.append(dict(config=key, runs=len(g), e_cv=e_cv, val_min=round(float(np.nanmin(mean_curve)), 4),
                         e_full=max(1, int(round(e_cv * frac)))))
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    summary.to_csv(out_path, index=False)
    log(f"[cv] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--curves", default=os.path.join(stable_output_path, "cv_epochs.jsonl"),
                        help="validation curves, one JSON line per run (appended, resumable)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--only", default=None, help="one configuration, e.g. BourginMLP_B32_lr0.001")
    parser.add_argument("--summary", action="store_true", help="write the summary instead of training")
    parser.add_argument("--summary-out", default=CV_EPOCHS_CSV)
    args = parser.parse_args()
    if args.summary:
        summarize(args.curves, args.summary_out)
        return

    seeds = np.random.default_rng(MASTER_SEED_FOR_POOL).integers(0, 2**32, size=args.seeds, dtype=np.uint64)
    configs = [c for c in CONFIGS if args.only is None or _key(c) == args.only]
    if not configs:
        raise SystemExit(f"unknown configuration {args.only}: one of {[_key(c) for c in CONFIGS]}")
    tasks = [(c, f, int(s)) for c in configs for f in range(N_FOLDS) for s in seeds]
    done = set()
    if os.path.exists(args.curves):
        with open(args.curves) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["config"], r["fold"], r["seed"]))
    todo = [t for t in tasks if (_key(t[0]), t[1], t[2]) not in done]
    log(f"[cv] {len(configs)} configurations, {len(tasks) - len(todo)} runs done, {len(todo)} to do")
    os.makedirs(os.path.dirname(os.path.abspath(args.curves)), exist_ok=True)

    failed = 0
    # Only this process writes the file, one line per finished run.
    with open(args.curves, "a") as out, ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_runOne, t): t for t in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                out.write(json.dumps(fut.result()) + "\n")
                out.flush()
            except Exception:
                failed += 1
                log(f"[cv] run {futs[fut]} crashed:\n{traceback.format_exc()}")
            if i % 100 == 0:
                log(f"[cv] {i}/{len(todo)}")
    if failed:
        raise SystemExit(f"{failed} of {len(todo)} runs failed")


if __name__ == "__main__":
    main()
