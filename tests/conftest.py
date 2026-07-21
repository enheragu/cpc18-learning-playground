#!/usr/bin/env python3
# encoding: utf-8
"""
Shared test setup: put the repo `src/` and the mnist submodule `src/` on
sys.path so tests can import `cpc18_utils.*`, `cpc18_models.*` and the
submodule `utils.*` the same way the entry scripts do.
"""

import os
import sys
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
_SRC = os.path.join(_REPO, "src")
_MNIST_SRC = os.path.join(_REPO, "mnist-learning-playground", "src")

for _p in (_SRC, _MNIST_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_numbered_module(filename, name):
    """Import a numeric-prefixed entry script (e.g. ``07_analyze_ablation.py``),
    which a plain ``import`` cannot handle because of the leading digit."""
    path = os.path.join(_SRC, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
