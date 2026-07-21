#!/usr/bin/env python3
# encoding: utf-8

"""
MLP_3L: 3 hidden layers, 100 units each. Analog of DNN_6L in MNIST.
"""

import torch
import torch.nn as nn

from .base import RegressionBaseModelTrainer
from utils.log_utils import log


class MLP_3L(RegressionBaseModelTrainer):
    hidden_sizes = (100, 100, 100)

    def __init__(self, input_size, learning_rate=0.001, patience=10,
                 seed=42, output_path="", device=None):
        super().__init__(input_size, learning_rate, patience,
                         device=device, output_path=output_path)
        h1, h2, h3 = self.hidden_sizes
        self.fc1 = nn.Linear(input_size, h1).to(self.device)
        self.fc2 = nn.Linear(h1, h2).to(self.device)
        self.fc3 = nn.Linear(h2, h3).to(self.device)
        self.fc4 = nn.Linear(h3, 1).to(self.device)
        self.act = nn.ReLU()
        self._initialize_weights(seed)
        log(f"[{self.model_name}] Initialization complete (hidden={self.hidden_sizes}).")

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        h = self.act(self.fc3(h))
        return torch.sigmoid(self.fc4(h))
