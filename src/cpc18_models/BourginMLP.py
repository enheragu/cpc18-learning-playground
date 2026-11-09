#!/usr/bin/env python3
# encoding: utf-8

"""
BourginMLP: replication of the MLP architecture from Bourgin et al. 2019
("Cognitive Model Priors for Predicting Human Decisions", ICML).

3 hidden layers of (200, 275, 100) units, SReLU activations, dropout 0.15,
sigmoid output. 90,576 params dense.

Acts as the "full model" for the ablation study (analog of CNN_14L in
MNIST). Dropout is kept at 0.0 by default to isolate initialization
variance — flip `dropout_rate` when reproducing Bourgin's setup.

Reference: Bourgin et al., ICML 2019, Table 2 (Choices13k) and Table 1
(CPC15/CPC18). MLP + Cognitive Prior achieved MSE x100 = 0.48 on CPC18
(but using a synth18 pre-training step we do not implement here).
"""

import torch
import torch.nn as nn

from .base import RegressionBaseModelTrainer, INIT_ORIGINAL
from utils.log_utils import log


class SReLU(nn.Module):
    """Bourgin et al. 2019 use SReLU. We substitute it by `nn.SiLU`, a smooth
    ReLU-like activation without parameters, and keep the name of the class.
    """

    def __init__(self):
        super().__init__()
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x)


class BourginMLP(RegressionBaseModelTrainer):
    hidden_sizes = (200, 275, 100)
    dropout_rate = 0.0   # set to 0.15 to follow Bourgin literally

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
        self.dropout = nn.Dropout(self.dropout_rate)
        self._initialize_weights(seed)
        self._applyInitScheme(seed)
        log(f"[{self.model_name}] Initialization complete "
            f"(hidden={self.hidden_sizes}, dropout={self.dropout_rate}).")

    def forward(self, x):
        h = self.dropout(self.act(self.fc1(x)))
        h = self.dropout(self.act(self.fc2(h)))
        h = self.dropout(self.act(self.fc3(h)))
        return torch.sigmoid(self.fc4(h))
