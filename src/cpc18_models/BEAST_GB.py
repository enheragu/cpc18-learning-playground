#!/usr/bin/env python3
# encoding: utf-8

"""
BEAST-GB: XGBoost over the engineered BEAST features.

CPC18 winner per Plonsky et al. 2018 (technical report) and re-evaluated
in Plonsky et al. 2024/2025 (NHB) which reports MSE x100 = 0.56 on CPC18.

We use the full feature vector (raw + naive + psychological insight +
BEASTpred). Hyperparameters are the canonical ones distributed in
Plonsky's OSF reference implementation (`BEAST_GB.R`, project vw2su),
reproduced verbatim below so this is a genuine replication of the
published CPC18 winner rather than a re-tuned variant.

Variance comes from xgboost's `seed` parameter (= `random_state`).
The canonical recipe already uses stochastic gradient boosting
(`subsample = 0.508`), so the seed has a genuine, non-engineered effect
on the row subsample drawn for each tree.
"""

from .SklearnRegressorTrainer import SklearnRegressorTrainer
from utils.log_utils import log


class BEAST_GB(SklearnRegressorTrainer):
    # Canonical OSF BEAST_GB.R hyperparameters (project vw2su), verbatim.
    n_estimators = 978                  # nrounds
    max_depth = 3
    learning_rate = 0.010878922437398755
    subsample = 0.5079640412046551
    colsample_bytree = 0.9911994607412087
    gamma = 0.012241954019987821
    reg_alpha = 0.04306269451141776
    reg_lambda = 2.9053833404234397
    n_jobs = 1

    def _build_estimator(self, seed):
        # Lazy import: keep `xgboost` out of the parent process so that
        # fork-based multiprocessing doesn't inherit a corrupted OpenMP
        # state. Each worker imports it fresh after the fork. This is
        # cheaper than using a `spawn` mp_context (no per-worker
        # interpreter startup cost).
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=int(seed),
            n_jobs=self.n_jobs,
            objective="reg:squarederror",
            verbosity=0,
        )
