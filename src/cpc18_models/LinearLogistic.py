#!/usr/bin/env python3
# encoding: utf-8

"""
LinearLogistic: single linear layer + sigmoid. Expected to have very low
initialization variance, so it is used as the "near-zero variance" baseline
in the paper, analogous to SimplePerceptron in MNIST.
"""

import torch
import torch.nn as nn

from .base import RegressionBaseModelTrainer, INIT_ORIGINAL
from utils.log_utils import log


class LinearLogistic(RegressionBaseModelTrainer):
    # Its own initialization below: uniform weights in [-0.1, 0.1], zero bias.
    original_init = "uniform"

    def __init__(self, input_size, learning_rate=0.001, patience=10,
                 seed=42, output_path="", device=None, init=INIT_ORIGINAL):
        super().__init__(input_size, learning_rate, patience,
                         device=device, output_path=output_path, init=init)
        self.fc1 = nn.Linear(input_size, 1).to(self.device)
        self._initialize_weights(seed)
        self._applyInitScheme(seed)
        log(f"[{self.model_name}] Initialization complete.")

    def forward(self, x):
        return torch.sigmoid(self.fc1(x))

    def _initialize_weights(self, seed):
        self.seed = seed
        torch.manual_seed(seed)
        nn.init.uniform_(self.fc1.weight, a=-0.1, b=0.1)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
