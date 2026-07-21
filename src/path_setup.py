#!/usr/bin/env python3
# encoding: utf-8

"""
Side-effect module: adds the mnist-learning-playground submodule's src/ to
sys.path so that its `utils.*` and `models.*` packages are importable.

Import this module at the very top of every entry script BEFORE any
`from utils.X import ...` or `from models.X import ...` line.

Our own packages are deliberately named `cpc18_utils` and `cpc18_models` to
avoid colliding with the submodule's `utils` / `models` packages.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_MNIST_SRC = os.path.normpath(os.path.join(_REPO_ROOT, "mnist-learning-playground", "src"))

if not os.path.isdir(_MNIST_SRC):
    raise RuntimeError(
        f"mnist-learning-playground submodule not found at {_MNIST_SRC}. "
        "Did you run `git submodule update --init`?"
    )

if _MNIST_SRC not in sys.path:
    sys.path.insert(0, _MNIST_SRC)
