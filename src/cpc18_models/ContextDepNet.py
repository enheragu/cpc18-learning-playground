#!/usr/bin/env python3
# encoding: utf-8

"""
ContextDepNet (= ContextDependantNetwork in Thomas et al. 2024,
DatasetBias, Nature Human Behaviour 8(4)).

A small fully-connected network with sigmoid activations throughout:
  Linear(d, 32) -> sigmoid -> Linear(32, 32) -> sigmoid -> Linear(32, 1) -> sigmoid

The original (their `src/context_dependant_network.py`) feeds the raw
24-dimensional Peterson 2021 format (flat (outcome, prob) pairs for
options A and B). We feed our standard 37-dim BEAST-features vector
instead so the regime 2 comparison stays apples-to-apples; the
*architecture* (small, sigmoid throughout) is what we adopt and cite.
This is the same pattern the MNIST variance paper uses ("inspired by"
rather than exact reproduction).

Lineage: Peterson et al. 2021 Science (the deeper context-dependent
model that motivated the small sigmoid stack) + Thomas et al. 2024
NHB (concrete instantiation we follow here).

Variance source: torch initialization seed only. No dropout, no
training-time stochastic regularisation.
"""

import torch
import torch.nn as nn

from .base import RegressionBaseModelTrainer
from utils.log_utils import log


class ContextDepNet(RegressionBaseModelTrainer):
    hidden_size = 32   # 32-32 stack, as in Thomas 2024 src/context_dependant_network.py

    def __init__(self, input_size, learning_rate=0.001, patience=10,
                 seed=42, output_path="", device=None):
        super().__init__(input_size, learning_rate, patience,
                         device=device, output_path=output_path)
        self.fc1 = nn.Linear(input_size, self.hidden_size).to(self.device)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size).to(self.device)
        self.fc3 = nn.Linear(self.hidden_size, 1).to(self.device)
        self._initialize_weights(seed)
        log(f"[{self.model_name}] Initialization complete "
            f"(hidden={self.hidden_size}, all-sigmoid).")

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return torch.sigmoid(self.fc3(x))
