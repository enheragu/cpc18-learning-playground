#!/usr/bin/env python3
# encoding: utf-8

"""
Controls on the two CPU dispatchers of the pipeline (Appendix H of the manuscript):
the one inside torch (ATen, set with ATEN_CPU_CAPABILITY) and the one inside MKL
(the GEMM behind nn.Linear, set with MKL_ENABLE_INSTRUCTIONS). It answers:

  - Do their effects separate when they are varied one at a time?
  - Which kernel paths does a sweep of MKL_ENABLE_INSTRUCTIONS select, and does any
    of them reach the vendor-generic path of the AMD machines?
  - Which fills change with the ATen tier?

Each case runs in its own process, because both dispatchers read their variable
when the libraries load. The fingerprint is the one of cpu_probe.py: 200 optimizer
steps on synthetic inputs, without the CPC18 data. It takes about a minute.

It needs an Intel machine with AVX-512, since a tier can only be forced downward,
and the torch build of the study (cpu_probe.py stops on a CUDA build):

    pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

The results go to dispatcher_controls.csv, next to this script, which holds no host
name. With the data deposit in zenodo_export/, the fingerprints are also compared
with those of probes.csv. Exits with status 1 if any check fails.

Usage:
    python dispatcher_controls.py [--out PATH]
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import cpu_probe  # noqa: E402  (pins the thread counts before torch loads)

_VARIABLES = ("ATEN_CPU_CAPABILITY", "MKL_ENABLE_INSTRUCTIONS")
_DEPOSIT_PROBES = os.path.normpath(os.path.join(_HERE, "..", "..", "zenodo_export", "probes.csv"))
_FILL_SIZES = {"uniform_hash": ("uniform_", 1024), "normal_15_hash": ("normal_", 15), "normal_16_hash": ("normal_", 16)}

# Case name: (ATEN_CPU_CAPABILITY, MKL_ENABLE_INSTRUCTIONS)
CASES = {
    "native": (None, None),
    "forced AVX2": ("avx2", "AVX2"),
    "forced DEFAULT": ("default", "SSE4_2"),
    "MKL AVX512 only": (None, "AVX512"),
    "MKL AVX2 only": (None, "AVX2"),
    "MKL AVX only": (None, "AVX"),
    "MKL SSE4_2 only": (None, "SSE4_2"),
    "ATen avx2 only": ("avx2", None),
    "ATen default only": ("default", None),
}
FIELDS = ["case", "aten_cpu_capability", "mkl_enable_instructions", "torch_tier", "mkl_dispatch_isa", "init_hash",
          "trained_hash", "uniform_hash", "normal_15_hash", "normal_16_hash", "cpu_model", "torch"]


def runCase():
    """ Worker of one case: prints its results as one line of JSON. """
    import torch
    if "+cu" in torch.__version__ or not torch.__version__.startswith(cpu_probe.TORCH_PIN):
        raise SystemExit(f"torch is {torch.__version__}; the controls use {cpu_probe.TORCH_PIN}+cpu")
    torch.set_num_threads(1)
    fingerprint = cpu_probe.fingerprint(torch)
    row = {"torch_tier": cpu_probe.torch_facts(torch)["cpu_capability_tier"],
           "mkl_dispatch_isa": cpu_probe.mkl_dispatch()["isa"],
           "init_hash": fingerprint["init_hash"], "trained_hash": fingerprint["trained_hash"],
           "cpu_model": cpu_probe.cpu_facts()["cpu_model"], "torch": torch.__version__}
    for field, (fill, size) in _FILL_SIZES.items():
        torch.manual_seed(cpu_probe.FP_SEED)
        tensor = getattr(torch.empty(size), fill)()
        row[field] = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()[:32]
    print(json.dumps(row))


def launchCase(name, aten, mkl):
    env = {k: v for k, v in os.environ.items() if k not in _VARIABLES}
    for variable, value in zip(_VARIABLES, (aten, mkl)):
        if value:
            env[variable] = value
    done = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker"], env=env, check=True,
                          capture_output=True, text=True)
    return {"case": name, "aten_cpu_capability": aten or "", "mkl_enable_instructions": mkl or "",
            **json.loads(done.stdout.strip().splitlines()[-1])}


def depositHashes():
    """ {machine_id: (init_hash, trained_hash)} of the probes of the deposit, if it is there. """
    if not os.path.exists(_DEPOSIT_PROBES):
        return {}
    with open(_DEPOSIT_PROBES, newline="") as f:
        return {row["machine_id"]: (row["init_hash"], row["trained_hash"]) for row in csv.DictReader(f)}


def runChecks(rows):
    r = {row["case"]: row for row in rows}
    sweep = [r[f"MKL {isa} only"] for isa in ("AVX512", "AVX2", "AVX", "SSE4_2")]
    checks = [
        ("the tier follows ATEN_CPU_CAPABILITY and not MKL_ENABLE_INSTRUCTIONS",
         r["ATen avx2 only"]["torch_tier"] == "AVX2" and r["ATen default only"]["torch_tier"] == "DEFAULT"
         and r["MKL AVX2 only"]["torch_tier"] == r["native"]["torch_tier"]),
        ("MKL AVX2 alone gives the trained fingerprint of the forced AVX2 run",
         r["MKL AVX2 only"]["trained_hash"] == r["forced AVX2"]["trained_hash"]),
        ("ATen avx2 alone leaves the trained fingerprint at its native value",
         r["ATen avx2 only"]["trained_hash"] == r["native"]["trained_hash"]),
        ("ATen default alone changes the initial weights", r["ATen default only"]["init_hash"] != r["native"]["init_hash"]),
        ("MKL alone never changes the initial weights", all(row["init_hash"] == r["native"]["init_hash"] for row in sweep)),
        ("the MKL sweep selects three trained fingerprints", len({row["trained_hash"] for row in sweep}) == 3),
        ("AVX and SSE4_2 resolve to the same fingerprint", sweep[2]["trained_hash"] == sweep[3]["trained_hash"]),
        ("the MKL sweep never reports the vendor-generic path", all(row["mkl_dispatch_isa"] != "GENERIC" for row in sweep)),
        ("ATen default leaves a uniform fill unchanged", r["ATen default only"]["uniform_hash"] == r["native"]["uniform_hash"]),
        ("ATen default leaves a normal fill of 15 elements unchanged",
         r["ATen default only"]["normal_15_hash"] == r["native"]["normal_15_hash"]),
        ("ATen default changes a normal fill of 16 elements",
         r["ATen default only"]["normal_16_hash"] != r["native"]["normal_16_hash"]),
    ]
    probes = depositHashes()
    if probes:
        checks += [
            ("the native case reproduces the native probe of M01",
             (r["native"]["init_hash"], r["native"]["trained_hash"]) == probes["M01"]),
            ("the forced AVX2 case reproduces the probe of M01-forced-AVX2",
             (r["forced AVX2"]["init_hash"], r["forced AVX2"]["trained_hash"]) == probes["M01-forced-AVX2"]),
            ("the forced DEFAULT case reproduces the probe of M01-forced-DEFAULT",
             (r["forced DEFAULT"]["init_hash"], r["forced DEFAULT"]["trained_hash"]) == probes["M01-forced-DEFAULT"]),
            ("no case of the sweep matches the fingerprint of the AMD sessions",
             all(row["trained_hash"] != probes["C02"][1] for row in sweep)),
        ]
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=os.path.join(_HERE, "dispatcher_controls.csv"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        runCase()
        return

    rows = []
    for name, (aten, mkl) in CASES.items():
        rows.append(launchCase(name, aten, mkl))
        row = rows[-1]
        print(f"{name:18s} tier {row['torch_tier']:8s} MKL {str(row['mkl_dispatch_isa']):8s} "
              f"init {row['init_hash'][:8]} trained {row['trained_hash'][:8]}", flush=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    checks = runChecks(rows)
    print()
    for text, ok in checks:
        print(f"{'OK  ' if ok else 'FAIL'} {text}")
    print(f"\n{sum(ok for _, ok in checks)} of {len(checks)} checks passed. Results in {args.out}")
    if not all(ok for _, ok in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
