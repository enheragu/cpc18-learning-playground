#!/usr/bin/env python3
# encoding: utf-8

"""
MLP_1L: single hidden layer MLP. Analog of HiddenLayerPerceptron in MNIST.
Non-convex loss => meaningful initialization variance expected.
"""

import torch
import torch.nn as nn

from .base import RegressionBaseModelTrainer
from utils.log_utils import log


class MLP_1L(RegressionBaseModelTrainer):
    hidden_size = 50

    def __init__(self, input_size, learning_rate=0.001, patience=10,
                 seed=42, output_path="", device=None):
        super().__init__(input_size, learning_rate, patience,
                         device=device, output_path=output_path)
        self.fc1 = nn.Linear(input_size, self.hidden_size).to(self.device)
        self.fc2 = nn.Linear(self.hidden_size, 1).to(self.device)
        self.act = nn.ReLU()
        self._initialize_weights(seed)
        log(f"[{self.model_name}] Initialization complete (hidden={self.hidden_size}).")

    def forward(self, x):
        h = self.act(self.fc1(x))
        return torch.sigmoid(self.fc2(h))
