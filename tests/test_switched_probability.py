#!/usr/bin/env python3
# encoding: utf-8
"""
Sanity test for the misranking helper.

For two equal-variance Gaussians the probability that a single draw reverses
the order has a closed form, P(swap | N=1) = Phi(-|dmu| / (sigma * sqrt(2))).
The Monte Carlo estimate produced by the shared submodule helper must match
it. This is an analytical check of the helper; it also pins the semantics
of the statistic the paper reports.
"""

import numpy as np
from scipy.stats import norm

from utils.compute_switched_probability import computeSwitchedProbabilityT


def test_montecarlo_swap_matches_analytical(tmp_path):
    np.random.seed(20260522)
    (tmp_path / "tables").mkdir(parents=True, exist_ok=True)  # helper writes here
    mu_a, mu_b, sigma, n = 10.0, 11.0, 1.0, 4000
    data = {
        "A": np.random.normal(mu_a, sigma, n),
        "B": np.random.normal(mu_b, sigma, n),
    }
    rows = computeSwitchedProbabilityT(
        dict_data=data, g_names=["A", "B"], n_simulations=200000,
        resampling_method="MonteCarlo", analysis_path=str(tmp_path),
    )
    p_swap_n1 = rows[1][1]  # row N=1, column "P(Switched order)"
    analytical = norm.cdf(-abs(mu_b - mu_a) / (sigma * np.sqrt(2.0)))
    assert abs(p_swap_n1 - analytical) < 0.02
