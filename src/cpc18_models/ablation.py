#!/usr/bin/env python3
# encoding: utf-8

"""
Ablation model classes for the CPC18 study.

We replicate the MNIST paper's pattern (`models/BatchSizeStudy.py`):
one Python subclass per (model, hyperparameter combination) so that
each lands in its own `output_data/<ClassName>/` folder and YAML, with
zero risk of accidentally mixing trials across conditions.

Two studies:

- **B1: BourginMLP × {batch_size × learning_rate}** — direct analogue of
  the ablation of the MNIST study (mnist-learning-playground), but on the
  BEAST-features regime.
- **B3: SparseMLP × {epsilon × learning_rate}** — CPC18-specific: how
  does the Erdős–Rényi sparsity parameter interact with the optimiser
  step?

`default_lr` is exposed as a CLASS attribute so the trainer can read
it and feed it into `__init__(learning_rate=...)`. `batch_size` /
`epsilon` are also class attributes inherited from the base class and
overridden here.

All ablation classes preserve the "one-shot init randomness" property
of the base models — no dropout, no SET evolution, no training-time
stochastic regularisation.
"""

from .BourginMLP import BourginMLP
from .SparseMLP import SparseMLP


# ---------------------------------------------------------------------
# Slug helper: convert numeric LR / epsilon to a filename-safe token
# ---------------------------------------------------------------------

def _lr_slug(lr: float) -> str:
    """0.0001 -> '0_0001', 0.001 -> '0_001', 0.01 -> '0_01'."""
    return f"{lr:g}".replace(".", "_").replace("-", "n")


def _make_class(base, name, **attrs):
    """Build a subclass with given class attributes; also exposes it at
    module scope so `from cpc18_models.ablation import BourginMLP_B16_L0_001`
    works."""
    cls = type(name, (base,), attrs)
    globals()[name] = cls
    return cls


# ---------------------------------------------------------------------
# B1: BourginMLP batch_size x learning_rate
# ---------------------------------------------------------------------

B1_BATCH_SIZES    = (16, 32, 64)
B1_LEARNING_RATES = (1e-4, 1e-3, 1e-2)

B1_CLASSES = []
for _b in B1_BATCH_SIZES:
    for _lr in B1_LEARNING_RATES:
        _name = f"BourginMLP_B{_b}_L{_lr_slug(_lr)}"
        B1_CLASSES.append(_make_class(
            BourginMLP, _name,
            batch_size=_b,
            default_lr=_lr,
        ))


# ---------------------------------------------------------------------
# B3: SparseMLP epsilon x learning_rate
# ---------------------------------------------------------------------

B3_EPSILONS       = (10, 20, 40)
B3_LEARNING_RATES = (1e-4, 1e-3, 1e-2)

B3_CLASSES = []
for _eps in B3_EPSILONS:
    for _lr in B3_LEARNING_RATES:
        _name = f"SparseMLP_E{_eps}_L{_lr_slug(_lr)}"
        B3_CLASSES.append(_make_class(
            SparseMLP, _name,
            epsilon=_eps,
            default_lr=_lr,
        ))


# ---------------------------------------------------------------------
# B1X: BourginMLP extension column for the grid-sensitivity analysis.
# One extra learning-rate level at the log-midpoint of the two converged
# B1 levels (10^-3.5 ~= 3.16e-4), so that {1e-4, 3.16e-4, 1e-3} x
# {16, 32, 64} forms an all-converged 3x3 grid of the same size as the
# published B1 grid ("Sensitivity to the grid", Section 3.4 of the manuscript).
# ---------------------------------------------------------------------

B1X_LEARNING_RATES = (3.16e-4,)

B1X_CLASSES = []
for _b in B1_BATCH_SIZES:
    for _lr in B1X_LEARNING_RATES:
        _name = f"BourginMLP_B{_b}_L{_lr_slug(_lr)}"
        B1X_CLASSES.append(_make_class(
            BourginMLP, _name,
            batch_size=_b,
            default_lr=_lr,
        ))


def list_b1_class_names():
    return [c.__name__ for c in B1_CLASSES]


def list_b3_class_names():
    return [c.__name__ for c in B3_CLASSES]
