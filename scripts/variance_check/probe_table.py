#!/usr/bin/env python3
# encoding: utf-8

"""
Tabulate cpu_probe.py results and group machines into reproducibility cells.

A "cell" here is a set of machines whose fingerprint matches bit for bit. The
study found that a cell is an arithmetic path, set by torch's CPU capability
tier together with the BLAS dispatch and not by the CPU model or the tier
alone, so the useful output is the cross-check: does the fingerprint grouping
agree with the reported tier? A machine whose tier says AVX2 but whose
fingerprint matches nobody else at AVX2 runs a different arithmetic path under
the same tier label; its BLAS dispatch (mkl_dispatch in the probe file) is the
first thing to compare.

The probe files of the laboratory are not part of the repository. Without them,
the probes are read from probes.csv of the data deposit, which identifies each
one by its machine and has no AVX-512 column.

Usage:
    python probe_table.py [probe_results_dir]
"""

import csv
import glob
import json
import os
import sys
from collections import defaultdict


def rowsFromDeposit(path):
    with open(path, newline="") as f:
        return [{
            "tag": f"{p['machine_id']}/{p['probe_id']}",
            "tier": p["torch_tier"],
            "cpu": p["cpu_model"],
            "virt": p["hypervisor"] == "True",
            "avx512": None,
            "init": p["init_hash"],
            "trained": p["trained_hash"],
            "torch": p["torch"],
            "forced": bool(p["aten_cpu_capability"]),
            "kaggle": p["machine_id"] in ("C01", "C02"),
        } for p in csv.DictReader(f)]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "probe_results")
    files = sorted(glob.glob(os.path.join(d, "cpu_probe_*.json")))
    deposit = os.path.join(here, "..", "..", "zenodo_export", "probes.csv")
    if not files and not os.path.exists(deposit):
        raise SystemExit(f"no probes found in {d} nor in the data deposit")

    rows = [] if files else rowsFromDeposit(deposit)
    for f in files:
        p = json.load(open(f))
        c, fp = p["cpu"], p["fingerprint"]
        rows.append({
            "tag": p["tag"],
            "tier": p["cpu_capability_tier"],
            "cpu": c["cpu_model"],
            "virt": c["hypervisor"],
            "avx512": c["has_avx512f"],
            "init": fp["init_hash"],
            "trained": fp["trained_hash"],
            "torch": p["torch"],
            "forced": bool(p["forced"]["ATEN_CPU_CAPABILITY"]),
            "kaggle": p["on_kaggle"],
            "sec": fp["seconds"],
        })

    w = max(len(r["tag"]) for r in rows) + 1
    print(f"\n{'tag':{w}s}  {'tier':7s} {'avx512':6s} {'virt':5s} {'trained hash':32s}  CPU")
    print("-" * (w + 90))
    for r in sorted(rows, key=lambda r: (r["tier"], r["tag"])):
        mark = "*" if r["forced"] else (" " if not r["kaggle"] else "K")
        print(f"{r['tag']:{w}s}{mark} {r['tier']:7s} "
              f"{'-' if r['avx512'] is None else 'yes' if r['avx512'] else 'no':6s} "
              f"{'yes' if r['virt'] else 'no':5s} "
              f"{r['trained']:32s}  {r['cpu']}")
    print("  (* = tier forced by env var, K = Kaggle)\n")

    cells = defaultdict(list)
    for r in rows:
        cells[r["trained"]].append(r)

    print("Cells by fingerprint (bit-identical trained weights):")
    consistent = True
    for h, members in sorted(cells.items(), key=lambda kv: kv[1][0]["tier"]):
        tiers = {m["tier"] for m in members}
        flag = "" if len(tiers) == 1 else "  <-- MIXED TIERS, unexpected"
        if len(tiers) != 1:
            consistent = False
        print(f"  {h}  tier={'/'.join(sorted(tiers))}{flag}")
        for m in members:
            print(f"      {m['tag']}  ({m['cpu']})")

    # The inverse check matters just as much: one tier splitting across several
    # fingerprints means that the tier alone does not fix the arithmetic path.
    by_tier = defaultdict(set)
    for r in rows:
        by_tier[r["tier"]].add(r["trained"])
    print("\nTier -> number of distinct fingerprints:")
    for tier, hashes in sorted(by_tier.items()):
        flag = "" if len(hashes) == 1 else "  <-- SPLIT, tier is not one cell"
        if len(hashes) != 1:
            consistent = False
        print(f"  {tier:8s} {len(hashes)}{flag}")

    inits = defaultdict(set)
    for r in rows:
        inits[r["tier"]].add(r["init"])
    print("\nInitial weights (before any training) by tier:")
    for tier, hashes in sorted(inits.items()):
        print(f"  {tier:8s} {'  '.join(sorted(hashes))}")

    print("\nVERDICT:", "each tier is one cell in these probes"
          if consistent else "tiers and cells disagree -- see flags above; the "
          "arithmetic path (CPU capability tier and BLAS dispatch) is not set "
          "by the tier alone")


if __name__ == "__main__":
    main()
