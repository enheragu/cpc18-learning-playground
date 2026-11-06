#!/usr/bin/env python3
# encoding: utf-8
"""
Analytical check for the leaderboard-ordering misranking statistic of
10_paper_tables.py. For two equal-variance Gaussian groups, the order of the
means of N draws per group is reversed with probability
Phi(-|dmu| / (sigma * sqrt(2 / N))). Both estimators must recover it.
"""

import numpy as np
import pytest
from scipy.stats import norm

from conftest import load_numbered_module


@pytest.mark.parametrize("method", ["MonteCarlo", "Bootstrap"])
def test_ordering_swap_matches_analytical(method):
    tables = load_numbered_module("10_paper_tables.py", "paper_tables")
    rng = np.random.default_rng(20260522)
    mu_a, mu_b, sigma = 10.0, 11.0, 1.0
    groups = [rng.normal(mu_a, sigma, 4000), rng.normal(mu_b, sigma, 4000)]
    probabilities = tables._orderingSwapProbability(groups, method, n_max=3)
    for n, p_swap in enumerate(probabilities, start=1):
        analytical = norm.cdf(-abs(mu_b - mu_a) / (sigma * np.sqrt(2.0 / n)))
        assert abs(p_swap - analytical) < 0.02
