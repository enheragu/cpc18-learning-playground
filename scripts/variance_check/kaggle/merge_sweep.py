#!/usr/bin/env python3
# encoding: utf-8

"""
Merge a fetched Kaggle sweep tree into the canonical local one, de-duplicating
by seed.

Why de-duplication is needed: trial entries are keyed by TIMESTAMP, not by
seed. Two sessions that resume from the same state pick up the same pending
seeds and produce entries for them under different timestamps, so a plain
merge would silently double-count those seeds -- inflating n and biasing any
per-seed statistic.

The duplicates are not waste: within one cell the same seed must give the same
result, so every duplicate is a free cross-session bit-exactness check. This
script reports them, and a mismatch is a finding, not a nuisance -- it would
mean the cell is not a cell.

Usage:
    merge_sweep.py <fetched_tree> <canonical_tree> [--dry-run]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "src"))
import path_setup  # noqa: F401
from cpc18_utils.safe_yaml import fastLoadYaml, atomicUpdateMetricsLogFile

import re

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3}$")


def trial_entries(d):
    return {k: v for k, v in d.items()
            if DATE_RE.match(k) and isinstance(v, dict) and "seed" in v}


def merge_cell(src_yaml, dst_yaml, dry):
    src = trial_entries(fastLoadYaml(src_yaml))
    dst = trial_entries(fastLoadYaml(dst_yaml)) if os.path.exists(dst_yaml) else {}

    have = {}                 # seed -> (timestamp, mse)
    for ts, e in dst.items():
        have[e["seed"]] = (ts, e.get("mse"))

    add, dup_same, dup_diff = {}, 0, []
    for ts, e in sorted(src.items()):
        seed = e["seed"]
        if seed in have:
            prev_mse = have[seed][1]
            if prev_mse == e.get("mse"):
                dup_same += 1
            else:
                dup_diff.append((seed, prev_mse, e.get("mse")))
            continue
        if seed in {v["seed"] for v in add.values()}:
            continue          # duplicate within the incoming file itself
        add[ts] = e
        have[seed] = (ts, e.get("mse"))

    if add and not dry:
        os.makedirs(os.path.dirname(dst_yaml), exist_ok=True)
        atomicUpdateMetricsLogFile(add, dst_yaml)
    return len(src), len(add), dup_same, dup_diff, len(have)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    if len(args) != 2:
        raise SystemExit(__doc__)
    src_tree, dst_tree = args

    total_dup_diff = []
    print(f"{'cell':34s} {'incoming':>8s} {'new':>6s} {'dup=':>6s} {'dup!=':>6s} {'total':>6s}")
    for cell in sorted(os.listdir(src_tree)):
        sdir = os.path.join(src_tree, cell)
        if not os.path.isdir(sdir):
            continue
        for y in sorted(os.listdir(sdir)):
            if not y.endswith(".yaml"):
                continue
            n_in, n_new, d_same, d_diff, n_tot = merge_cell(
                os.path.join(sdir, y), os.path.join(dst_tree, cell, y), dry)
            total_dup_diff += [(cell, *d) for d in d_diff]
            print(f"{cell:34s} {n_in:8d} {n_new:6d} {d_same:6d} "
                  f"{len(d_diff):6d} {n_tot:6d}")

    if total_dup_diff:
        print("\n*** SAME SEED, DIFFERENT RESULT -- the cell is not one cell ***")
        for cell, seed, a, b in total_dup_diff[:20]:
            print(f"  {cell} seed={seed}  {a!r} vs {b!r}")
        raise SystemExit(1)
    print("\nAll repeated seeds agreed bit for bit." if not dry else "\n(dry run)")


if __name__ == "__main__":
    main()
