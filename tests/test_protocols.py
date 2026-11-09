#!/usr/bin/env python3
# encoding: utf-8
"""
The two training protocols (cpc18_utils/protocol.py). The stable one draws the
default initialization of PyTorch from the training seed and leaves the global
generator, and with it the data order of the seed, as the original one leaves
it; the original one is unchanged. Its readout is the test curve at the
cross-validated epoch of the configuration.
"""

import pytest
import torch
import torch.nn as nn

from cpc18_models import BourginMLP, ContextDepNet, LinearLogistic, SparseMLP
from cpc18_utils.protocol import cvConfigKey, readoutEpoch, stableEntry, stablePatience

NETWORKS = (LinearLogistic, BourginMLP, ContextDepNet, SparseMLP)


def _build(cls, tmp_path, **kwargs):
    torch.manual_seed(99)   # whatever the global state before: the models seed it themselves
    model = cls(input_size=37, seed=1234, output_path=str(tmp_path), **kwargs)
    return model, torch.get_rng_state()


@pytest.mark.parametrize("cls", NETWORKS)
def test_default_init_is_the_original(cls, tmp_path):
    a, state_a = _build(cls, tmp_path)
    b, state_b = _build(cls, tmp_path, init="original")
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    assert torch.equal(state_a, state_b)
    assert a.init_name == ("uniform" if cls is LinearLogistic else "kaiming_fanout")


@pytest.mark.parametrize("cls", NETWORKS)
def test_pytorch_default_redraws_the_layers_and_keeps_the_data_order(cls, tmp_path):
    original, state_original = _build(cls, tmp_path)
    stable, state_stable = _build(cls, tmp_path, init="pytorch_default")
    # the generator after construction, which the shuffle of the training data draws from, is the same
    assert torch.equal(state_original, state_stable)
    # the layers are those of reset_parameters() from the training seed, as in the runs of the stable protocol
    expected, _ = _build(cls, tmp_path)
    state = torch.get_rng_state()
    torch.manual_seed(1234)
    for module in expected.modules():
        if isinstance(module, nn.Linear):
            module.reset_parameters()
    torch.set_rng_state(state)
    assert all(torch.equal(p, q) for p, q in zip(stable.parameters(), expected.parameters()))
    assert not all(torch.equal(p, q) for p, q in zip(stable.parameters(), original.parameters()))
    if cls is SparseMLP:   # the mask does not depend on the initialization
        assert torch.equal(stable.mask2, original.mask2)
    assert stable.init_name == "pytorch_default"


def test_unknown_init_raises(tmp_path):
    with pytest.raises(ValueError):
        BourginMLP(input_size=37, seed=1, output_path=str(tmp_path), init="glorot")


def test_fixed_epochs_of_the_cross_validation():
    assert cvConfigKey("BourginMLP", 16, 3.16e-4) == "BourginMLP_B16_lr0.000316"
    assert cvConfigKey("SparseMLP", 32, 1e-4, 10) == "SparseMLP_B32_lr0.0001_E10"
    # main configurations: epochs 99, 42, 4 and 9, read at 0-based index epochs - 1
    assert readoutEpoch("LinearLogistic", 32, 1e-3) == 98
    assert readoutEpoch("ContextDepNet", 32, 1e-3) == 41
    assert readoutEpoch("BourginMLP", 32, 1e-3) == 3
    assert readoutEpoch("SparseMLP", 32, 1e-3, 20) == 8
    with pytest.raises(KeyError):
        readoutEpoch("BourginMLP", 128, 1e-3)
    assert (stablePatience("LinearLogistic"), stablePatience("BourginMLP")) == (120, 50)


def test_stable_readout_is_the_curve_at_the_fixed_epoch():
    entry = {"mse": 0.004, "mse_x100": 0.4, "best_epoch": 2, "mse_plot": [0.009, 0.006, 0.004, 0.005, 0.007]}
    stableEntry(entry, 3, "pytorch_default")
    assert (entry["mse"], entry["mse_x100"], entry["readout_epoch"]) == (0.005, 0.005 * 100.0, 3)
    assert entry["mse_x100_testmin"] == 0.004 * 100.0 and entry["best_epoch"] == 2
    assert (entry["protocol"], entry["init"]) == ("stable", "pytorch_default")
    with pytest.raises(RuntimeError):
        stableEntry({"mse_plot": [0.01, 0.02]}, 2, "pytorch_default")
