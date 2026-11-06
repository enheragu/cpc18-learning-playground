#!/usr/bin/env python3
# encoding: utf-8
"""
Regression test for the ANOVA contrast coding.

`twoWayAnova` must use sum-to-zero contrasts so that, on a balanced design,
its type-III sum-of-squares decomposition equals the type-I decomposition.
The original code used treatment contrasts, for which type-III main-effect
SS are simple effects at the reference level of the other factor -- the bug
this test guards against.
"""

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from conftest import load_numbered_module


def _balanced_df(n_per_cell=60):
    rng = np.random.default_rng(0)
    rows = []
    for f1 in (0, 1, 2):
        for f2 in (0, 1, 2):
            mu = 1.0 + 0.5 * f1 + 0.3 * f2 + 0.4 * f1 * f2
            rows += [{"f1": f1, "f2": f2, "mse": mu + rng.normal(0, 0.1)}
                     for _ in range(n_per_cell)]
    return pd.DataFrame(rows)


def _pct_ss(formula, df, typ):
    aov = anova_lm(ols(formula, data=df).fit(), typ=typ)
    aov = aov.drop("Intercept", errors="ignore")
    ss = aov["sum_sq"]
    return (100.0 * ss / ss.sum()).values  # [f1, f2, interaction, residual]


def test_sum_contrasts_match_type_i_on_balanced_design():
    df = _balanced_df()
    mod = load_numbered_module("07_analyze_ablation.py", "ablation_analyzer")
    got = mod.twoWayAnova(df)["pct_ss"].values        # the real, fixed function
    ref = _pct_ss("mse ~ C(f1) * C(f2)", df, typ=1)   # type-I reference
    np.testing.assert_allclose(got, ref, atol=1e-6)


def test_treatment_type3_would_disagree():
    # Documents the bug: treatment-coded type-III main effects differ
    # materially from the type-I decomposition on the same balanced data.
    df = _balanced_df()
    treat3 = _pct_ss("mse ~ C(f1) * C(f2)", df, typ=3)
    typ1 = _pct_ss("mse ~ C(f1) * C(f2)", df, typ=1)
    assert abs(treat3[0] - typ1[0]) > 0.5
