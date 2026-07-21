#!/usr/bin/env python3
# encoding: utf-8
"""
§5.8 split-induced variance. The canonical 210/60 split is one realisation.
Here we repartition the 268 BEAST-feature problems into N random 208/60
splits and retrain the stable models on each, with the initialisation seed
held FIXED so the only thing that varies is the split. The BEAST features are
per-problem and split-independent, so repartitioning the existing CSV is a
valid way to vary the split without re-running the R feature engineering.

Output: analysis_results/split_variance.npz + a printed sigma_split table.

Run:  python split_variance_rerun.py [N_SPLITS]
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import tempfile
import shutil

import path_setup  # noqa: F401
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.log_utils import log
from utils.set_seed import set_seed
from cpc18_utils.features import (_featurize, _standardize,
                                  cpc18_features_input_size,
                                  data_path, FEATURES_CSV_NAME)
from cpc18_utils.data import CPC18Dataset
from cpc18_models.BEAST_GB import BEAST_GB
from cpc18_models.PsychForest import PsychForest
from cpc18_models.LinearLogistic import LinearLogistic

MODELS = {"BEAST_GB": BEAST_GB, "PsychForest": PsychForest,
          "LinearLogistic": LinearLogistic}
FIXED_SEED = 3294508113   # pool[0]; held fixed so only the split varies
N_TEST, LR, PATIENCE, NUM_EPOCHS, BATCH = 60, 0.001, 50, 500, 32

_DF = pd.read_csv(os.path.join(data_path, FEATURES_CSV_NAME))
_PROBLEMS = _DF["problem_id"].unique()   # 268 (208 train + 60 test; problems 13/20 excluded at feature stage)


def _split_mse(model_name, split_seed):
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
    tmp = tempfile.mkdtemp()
    try:
        m = MODELS[model_name](input_size=cpc18_features_input_size(),
                               learning_rate=LR, patience=PATIENCE,
                               seed=FIXED_SEED, output_path=tmp,
                               device=torch.device("cpu"))
        rm = m.spinTrainEval(tl, tel, NUM_EPOCHS)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return [e["mse_x100"] for e in rm.values()][0]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    res = {mn: [] for mn in MODELS}
    for split in range(n):
        for mn in MODELS:
            res[mn].append(_split_mse(mn, 1000 + split))
        log(f"[split] {split + 1}/{n} done")
    print("\n=== sigma across splits (MSE x100), init seed fixed ===")
    for mn in MODELS:
        a = np.array(res[mn])
        print(f"{mn:16s} mean={a.mean():.3f} std={a.std(ddof=1):.4f} "
              f"min={a.min():.3f} max={a.max():.3f}")
    out = os.path.join(os.path.dirname(data_path), "analysis_results",
                       "split_variance.npz")
    np.savez(out, **{mn: np.array(res[mn]) for mn in MODELS})
    log(f"[split] wrote {out}")


if __name__ == "__main__":
    main()
