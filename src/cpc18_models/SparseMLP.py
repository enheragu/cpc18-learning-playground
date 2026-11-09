#!/usr/bin/env python3
# encoding: utf-8

"""
SparseMLP: BourginMLP architecture (200, 275, 100, SiLU) with a STATIC
Erdős–Rényi sparse connection mask per Mocanu et al. 2018 (Nature
Communications), as used in Bourgin et al. 2019 (ICML) and Thomas et al.
2024 (NHB, DatasetBias, `src/cognitive_prior_network.py`).

We adopt only the *static* sparse mask: it is sampled once at __init__
from the same torch RNG that seeds the weight initialisation, then
held fixed. We deliberately do NOT implement the SET evolutionary
weight-rebirth step from the original Mocanu/Bourgin recipe because
that step introduces stochasticity *during training* (per-epoch random
weight rebirth) that would confound with the initialization variance
we are trying to isolate -- the same reason we keep dropout=0 on
BourginMLP. Per-step weight rebirth is parked for future work.

Mask sampling: each weight W[i, j] is kept with probability
  p = epsilon * (n_in + n_out) / (n_in * n_out)
matching the formula in Mocanu 2018 §2.2 and the
`createWeightsMask` helper in Thomas's TensorFlow port. With
epsilon=20 and the 37 -> 200 -> 275 -> 100 -> 1 layer shape we keep
~64 % / ~17 % / ~27 % of weights respectively (≈ 68 k of the 90 k dense
parameters become structural zeros).

Variance source: torch RNG seed (controls BOTH the mask sampling AND
the weight initialisation). Same seed -> same mask + same initial
weights -> deterministic training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import RegressionBaseModelTrainer, INIT_ORIGINAL
from .BourginMLP import SReLU
from utils.log_utils import log


def _erdos_renyi_mask(n_in: int, n_out: int, epsilon: float,
                      generator: torch.Generator) -> torch.Tensor:
    """Bernoulli mask of shape (n_out, n_in) — matches nn.Linear weight shape.

    Probability of keeping each weight = epsilon * (n_in + n_out) / (n_in * n_out).
    Clipped to <= 1 for very small layers."""
    p = epsilon * (n_in + n_out) / (n_in * n_out)
    p = min(p, 1.0)
    rand = torch.rand(n_out, n_in, generator=generator)
    return (rand < p).float()


class SparseMLP(RegressionBaseModelTrainer):
    hidden_sizes = (200, 275, 100)
    epsilon = 20  # Mocanu 2018 default; ≈ 8x average connectivity per neuron

    def __init__(self, input_size, learning_rate=0.001, patience=10,
                 seed=42, output_path="", device=None, init=INIT_ORIGINAL):
        super().__init__(input_size, learning_rate, patience,
                         device=device, output_path=output_path, init=init)

        h1, h2, h3 = self.hidden_sizes
        self.fc1 = nn.Linear(input_size, h1).to(self.device)
        self.fc2 = nn.Linear(h1, h2).to(self.device)
        self.fc3 = nn.Linear(h2, h3).to(self.device)
        self.fc4 = nn.Linear(h3, 1).to(self.device)
        self.act = SReLU()

        # Generate masks deterministically from the same seed used for
        # weight init. The mask draws from a SEPARATE torch.Generator(seed)
        # below, while _initialize_weights re-seeds the GLOBAL torch RNG.
        # The two streams are therefore independent (the mask does not
        # consume the global RNG that the weights draw from), yet both are
        # fully determined by `seed`, so the same seed -> same mask + same
        # initial weights -> deterministic training.
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        self.register_buffer("mask1", _erdos_renyi_mask(input_size, h1, self.epsilon, gen).to(self.device))
        self.register_buffer("mask2", _erdos_renyi_mask(h1, h2, self.epsilon, gen).to(self.device))
        self.register_buffer("mask3", _erdos_renyi_mask(h2, h3, self.epsilon, gen).to(self.device))
        # Output layer stays dense (Mocanu 2018: too few output neurons to bother sparsifying).

        self._initialize_weights(seed)
        self._applyInitScheme(seed)
        kept = [self.mask1.mean().item(), self.mask2.mean().item(), self.mask3.mean().item()]
        log(f"[{self.model_name}] Initialization complete "
            f"(hidden={self.hidden_sizes}, eps={self.epsilon}, "
            f"kept fractions per layer = "
            f"[{kept[0]:.3f}, {kept[1]:.3f}, {kept[2]:.3f}]).")

    def forward(self, x):
        # Multiply weights by mask at every forward pass: this keeps
        # the structural zeros exactly zero even under momentum / weight
        # decay, with no extra hooks.
        x = self.act(F.linear(x, self.fc1.weight * self.mask1, self.fc1.bias))
        x = self.act(F.linear(x, self.fc2.weight * self.mask2, self.fc2.bias))
        x = self.act(F.linear(x, self.fc3.weight * self.mask3, self.fc3.bias))
        return torch.sigmoid(self.fc4(x))
