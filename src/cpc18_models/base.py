#!/usr/bin/env python3
# encoding: utf-8

"""
RegressionBaseModelTrainer: MSE-based counterpart of the MNIST
BaseModelTrainer for the CPC18 problem.

Inherits from the submodule's BaseModelTrainer to keep:
- spinTrainEval orchestration (timestamped metrics dict, train/eval split)
- save_architecture / load_best_model plumbing
- _initialize_weights helper

Overrides:
- train_model: MSE loss, lowest-MSE = best-model criterion
- evaluate_model / evaluate_model_with_loss: MSE on the test set
- Removes the .view(-1, 28*28) image flattening from forward callers

The target is the aggregate B-rate (a probability in [0, 1]) so we use
nn.MSELoss on a sigmoid-bounded output.

Initialization (`init`): "original" keeps the initialization of each model
(Kaiming normal in fan-out mode with zero biases, from the submodule, or the
model's own); "pytorch_default" then draws every nn.Linear again with its
`reset_parameters()` from a generator seeded with the same seed, and restores
the global torch RNG afterwards, so that the data order of a trial does not
change with the initialization.
"""

import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# These come from the mnist-learning-playground submodule (added to
# sys.path by path_setup.py at the script entry point).
from models.BaseModel import BaseModelTrainer
from utils.log_utils import log


# Initialization schemes of the networks (constructor argument `init`).
INIT_ORIGINAL = "original"
INIT_PYTORCH_DEFAULT = "pytorch_default"
INIT_SCHEMES = (INIT_ORIGINAL, INIT_PYTORCH_DEFAULT)


class RegressionBaseModelTrainer(BaseModelTrainer):
    """Base trainer for CPC18 regression models.

    Subclasses only need to implement `forward(x)` returning a single
    value per sample in [0, 1] (e.g. via sigmoid).
    """

    # CPC18 train set is tiny (~1050 (problem, block) rows). Small batches
    # default; subclasses or callers can override.
    batch_size = 32

    # Name of the initialization that the model draws by itself, as written in
    # the training logs and in the data deposit; subclasses with their own
    # _initialize_weights override it.
    original_init = "kaiming_fanout"

    def __init__(self, input_size, learning_rate=0.001, patience=10,
                 device=None, output_path="", init=INIT_ORIGINAL):
        if init not in INIT_SCHEMES:
            raise ValueError(f"init must be one of {INIT_SCHEMES}, not {init!r}")
        # num_classes is unused for regression but BaseModelTrainer
        # expects it, so we pass 1.
        super().__init__(
            input_size=input_size,
            num_classes=1,
            learning_rate=learning_rate,
            patience=patience,
            device=device,
            output_path=output_path,
        )
        self.init_scheme = init
        # Mirror the per-epoch tracking buffers but rename for MSE.
        self.mse_each_epoch = []
        self.train_mse_each_epoch = []
        # The parent class allocates accuracy_each_epoch etc.; we leave
        # them in place (empty) so any downstream code that inspects them
        # doesn't crash, but we don't populate them.

    # ---- Initialization -------------------------------------------------

    @property
    def init_name(self):
        """Initialization actually drawn, as logged with each trial."""
        return self.original_init if self.init_scheme == INIT_ORIGINAL else self.init_scheme

    def _applyInitScheme(self, seed):
        """Called by each network right after its own _initialize_weights.
        With "pytorch_default", every nn.Linear draws its weights and bias
        again with reset_parameters(), from the global generator seeded with
        `seed`. The state of that generator is saved before and restored
        after, so whatever draws from it next (the shuffle of a DataLoader
        without its own generator) sees the same state as with the original
        initialization: the data order of a seed is the same under both."""
        if self.init_scheme == INIT_ORIGINAL:
            return
        state = torch.get_rng_state()
        torch.manual_seed(seed)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        torch.set_rng_state(state)

    # ---- Training -----------------------------------------------------

    def train_model(self, train_loader, test_loader, num_epochs=500):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        epochs_without_improvement = 0
        best_mse = float("inf")

        with tqdm(range(num_epochs), desc=f"{self.model_name}", unit="epoch") as pbar:
            for epoch in pbar:
                start_time = time.time()
                self.train()

                epoch_train_loss = 0.0
                epoch_train_samples = 0

                for X, y in train_loader:
                    X, y = X.to(self.device), y.to(self.device)

                    optimizer.zero_grad()
                    preds = self(X)
                    loss = criterion(preds, y)
                    loss.backward()
                    optimizer.step()

                    epoch_train_loss += loss.item() * X.size(0)
                    epoch_train_samples += X.size(0)

                train_mse = epoch_train_loss / max(epoch_train_samples, 1)
                self.train_mse_each_epoch.append(train_mse)

                test_metrics = self.evaluate_model_with_loss(test_loader, criterion)
                test_mse = test_metrics["mse"]
                self.mse_each_epoch.append(test_mse)

                epoch_duration = time.time() - start_time
                log(f"\tEpoch [{epoch + 1}/{num_epochs}], "
                    f"Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}. "
                    f"Took {epoch_duration:.2f}s.")

                if test_mse < best_mse:
                    best_mse = test_mse
                    epochs_without_improvement = 0
                    log("\tNew best MSE, store model...")
                    if self.base_output_path is not None:
                        torch.save(self.state_dict(), self.best_trained_path)
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self.patience:
                    log(f"\tNo improvement for {self.patience} epochs. Early stop.")
                    pbar.set_postfix_str(f"Stopping early at epoch {epoch + 1}/{num_epochs}.")
                    pbar.close()
                    break

                pbar.set_postfix(mse=test_mse)

        # True best epoch = argmin of the per-epoch test MSE. This is
        # correct whether early stopping fired or the num_epochs cap was
        # hit; `epoch - patience` is only valid in the early-stop case
        # and points to the wrong epoch when the cap terminates training.
        if self.mse_each_epoch:
            best_epoch = min(range(len(self.mse_each_epoch)),
                             key=self.mse_each_epoch.__getitem__)
        else:
            best_epoch = epoch
        return best_epoch

    # ---- Evaluation ---------------------------------------------------

    def evaluate_model_with_loss(self, test_loader, criterion):
        self.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(self.device), y.to(self.device)
                preds = self(X)
                loss = criterion(preds, y)
                total_loss += loss.item() * X.size(0)
                total_samples += X.size(0)
        return {"mse": total_loss / max(total_samples, 1),
                "loss": total_loss / max(total_samples, 1)}

    def evaluate_model(self, test_loader, print_log=False):
        self.eval()
        all_preds, all_y = [], []
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(self.device), y.to(self.device)
                preds = self(X)
                all_preds.append(preds.cpu())
                all_y.append(y.cpu())
        preds_t = torch.cat(all_preds)
        y_t = torch.cat(all_y)
        mse = torch.mean((preds_t - y_t) ** 2).item()
        # Random-baseline MSE is just var(y) under a constant-mean predictor.
        mse_random = torch.var(y_t, unbiased=False).item()
        metrics = {"mse": mse,
                   "mse_x100": mse * 100.0,
                   "mse_random_baseline": mse_random}
        if print_log:
            log("\tMetrics:", metrics)
        return metrics

    # ---- spinTrainEval override (only to fix the metrics dict) --------

    def spinTrainEval(self, train_loader, test_loader, num_epochs=500):
        log(f"[{self.model_name}] Init spin in {self.device} device.")
        start_time = time.time()
        best_epoch = self.train_model(train_loader, test_loader, num_epochs)
        train_duration = time.time() - start_time

        log(f"[{self.model_name}] Training took {train_duration:.2f}s.")
        self.load_best_model()
        metrics = self.evaluate_model(test_loader, print_log=False)

        timestamp = (datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                     + f".{datetime.now().microsecond // 1000:03d}")
        metrics.update({
            "best_epoch": best_epoch,
            "train_duration": train_duration,
            "total_epochs": best_epoch + self.patience,
            "mse_plot": self.mse_each_epoch,
            "train_mse_plot": self.train_mse_each_epoch,
            "seed": self.seed,
        })
        return {timestamp: metrics}
