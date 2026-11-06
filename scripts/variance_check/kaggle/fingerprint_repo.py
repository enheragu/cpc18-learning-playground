#!/usr/bin/env python3
# encoding: utf-8

"""
Fingerprint the bytes that define a CPC18 training run.

One implementation, used on both sides: make_dataset_cpc18.sh calls this when
packaging, and amd_sweep_kernel.py calls it again inside the kernel. Computing
the same digest two different ways (shell here, Python there) is how you get a
false abort -- or worse, a false pass -- from a formatting difference that has
nothing to do with the code actually changing.

Usage:  fingerprint_repo.py <repo_dir>       -> prints "key=value" lines
"""

import hashlib
import os
import sys

# Everything that can change a training result: the model and utility code, and
# the feature CSV. Deliberately NOT requirements.lock -- the kernel installs a
# pinned torch and records the resolved version in the probe, which is the fact
# that matters; the lock file's own bytes are not what the trainer reads.
CODE_ROOTS = (("src",), ("mnist-learning-playground", "src"))
FEATURES = ("data", "cpc18_features.csv")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_files(repo):
    out = []
    for root in CODE_ROOTS:
        base = os.path.join(repo, *root)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for f in filenames:
                if f.endswith(".py"):
                    out.append(os.path.join(dirpath, f))
    # Sort by repo-relative path so the digest does not depend on where the
    # tree happens to live (control machine vs /kaggle/working/repo).
    return sorted(out, key=lambda p: os.path.relpath(p, repo))


def fingerprints(repo):
    lines = "".join(f"{_sha(p)}  {os.path.relpath(p, repo)}\n" for p in code_files(repo))
    training = hashlib.sha256(lines.encode()).hexdigest()[:16]
    features = _sha(os.path.join(repo, *FEATURES))[:16]
    return {"training_fingerprint": training,
            "features_fingerprint": features,
            "n_code_files": str(len(code_files(repo)))}


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    for k, v in fingerprints(repo).items():
        print(f"{k}={v}")
