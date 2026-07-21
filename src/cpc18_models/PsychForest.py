#!/usr/bin/env python3
# encoding: utf-8

"""
PsychForest: a Random Forest regressor over the engineered BEAST features,
mirroring the design of Plonsky, Erev, Hazan, Tennenholtz (2017)
"Psychological forest: Predicting human behavior" (AAAI).

The original paper uses 9 hand-picked behavioural features. Here we feed
the full feature vector (raw + naive + psychological insight + BEASTpred)
since (a) RF handles redundancy gracefully via feature_importances_ and
(b) it keeps the inputs identical to BEAST-GB for a controlled comparison.

Variance comes from sklearn's `random_state`, which seeds bootstrap
resampling, feature subset selection at splits, and tie-breaking.
"""

from sklearn.ensemble import RandomForestRegressor

from .SklearnRegressorTrainer import SklearnRegressorTrainer
from utils.log_utils import log


class PsychForest(SklearnRegressorTrainer):
    # Defaults chosen to roughly match Plonsky et al. 2017 reported setup:
    # ~500 trees, no max depth, sqrt feature subset.
    n_estimators = 500
    max_features = "sqrt"
    n_jobs = 1   # each training trial runs in a worker process; keep
                 # internal threading minimal to avoid oversubscription.

    def _build_estimator(self, seed):
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            random_state=int(seed),
            n_jobs=self.n_jobs,
            bootstrap=True,
        )
