#!/usr/bin/env python3
# encoding: utf-8

"""
Atomic read-modify-write wrapper around the submodule's YAML store.

WHY THIS EXISTS
---------------
The submodule's `utils.yaml_utils.updateMetricsLogFile` opens TWO
separate `with FileLock(...)` blocks — one for the read+merge, one
for the write — so the lock is RELEASED between read and write. With
multiple worker processes this races as:

    A: lock -> read state S, merge S+a = Sa, release
    B: lock -> read state S, merge S+b = Sb, release
    A: lock -> write Sa, release
    B: lock -> write Sb, release        # b overwrote a's trial!

Result: ~25% of trials were silently lost in our N=100, workers=6 run.

The CLAUDE.md instructs us not to modify the submodule, so we keep
the same `FileLock` primitive but hold it across the whole
read-modify-write sequence.
"""

import os
import yaml

from utils.file_lock import FileLock
from utils.log_utils import log


# Use libyaml-backed C loader/dumper when available. About 5-10x faster
# than the pure-Python yaml.safe_load on the multi-MB metrics YAMLs we
# produce (per-trial training curves accumulate to 7-15 MB after 800
# entries). Falls back to the Python loader if libyaml is missing.
try:
    _YAMLLoader = yaml.CSafeLoader
    _YAMLDumper = yaml.CSafeDumper
except AttributeError:  # pragma: no cover -- libyaml not installed
    _YAMLLoader = yaml.SafeLoader
    _YAMLDumper = yaml.SafeDumper


def fastLoadYaml(file_path: str) -> dict:
    """Read a YAML metrics file with the C-backed loader. Returns
    `{}` for a missing or empty file."""
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r") as fh:
        return yaml.load(fh, Loader=_YAMLLoader) or {}


def atomicUpdateMetricsLogFile(metrics: dict, file_path: str) -> None:
    """Drop-in replacement for utils.yaml_utils.updateMetricsLogFile
    that holds the file lock across read+merge+write."""
    lock_file = f"{file_path}.lock"

    with FileLock(lock_file):
        if os.path.exists(file_path):
            with open(file_path, "r") as fh:
                current = yaml.load(fh, Loader=_YAMLLoader) or {}
        else:
            current = {}

        _deep_merge(current, metrics)

        # Atomic on POSIX: write to a sibling tempfile then rename.
        tmp_path = file_path + ".tmp"
        with open(tmp_path, "w") as fh:
            yaml.dump(current, fh, Dumper=_YAMLDumper,
                      default_flow_style=False, allow_unicode=True,
                      indent=4, width=5000)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, file_path)

    log(f"[atomic] Updated metrics file in: {file_path}")


def _deep_merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst
