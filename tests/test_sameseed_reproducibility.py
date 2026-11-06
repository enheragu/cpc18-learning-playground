#!/usr/bin/env python3
# encoding: utf-8
"""
Reproducibility guarantee the variance study relies on: a model built with a
given seed must initialise bit-identically every time, and different seeds
must give different initialisations. (The full per-trial bit-identity was
checked by hand during development; this turns it into a standing test.)
"""

import torch

from cpc18_models.LinearLogistic import LinearLogistic


def test_same_seed_identical_init(tmp_path):
    m1 = LinearLogistic(input_size=37, seed=123, output_path=str(tmp_path))
    m2 = LinearLogistic(input_size=37, seed=123, output_path=str(tmp_path))
    for (name, p1), (_, p2) in zip(m1.named_parameters(),
                                   m2.named_parameters()):
        assert torch.equal(p1, p2), f"param {name} differs across same-seed inits"


def test_different_seed_differs(tmp_path):
    m1 = LinearLogistic(input_size=37, seed=1, output_path=str(tmp_path))
    m2 = LinearLogistic(input_size=37, seed=2, output_path=str(tmp_path))
    assert any(not torch.equal(p1, p2)
               for p1, p2 in zip(m1.parameters(), m2.parameters()))
