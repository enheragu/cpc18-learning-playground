#!/usr/bin/env python3
# encoding: utf-8

"""
CPU / SIMD-tier probe for the cross-CPU variance study.

Answers one question about any machine, in seconds and without the CPC18
dataset: *which reproducibility cell is this?* The study found that
bit-exactness follows the arithmetic path, the kernels the machine actually
executes, which is set by torch's CPU capability tier (AVX512 / AVX2 /
DEFAULT) together with the BLAS dispatch, and not by the CPU model or the
tier alone (Appendix D of the manuscript). So the probe reports the tier, the
instruction set the BLAS library dispatches to, the hardware facts behind
them, and a weight fingerprint that is bit-exact within an arithmetic path
and diverges across paths.

The same file runs on a lab machine and inside a Kaggle kernel (it is pushed
as-is, no wrapper). That is deliberate: two copies of the fingerprint code
would be free to drift, and a drifted fingerprint silently invalidates every
comparison it feeds.

Usage
-----
    python cpu_probe.py --tag <name>                      # native tier
    ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 \
        python cpu_probe.py --tag <name>-avx2forced       # forced lower tier

Forcing a tier needs BOTH variables: ATEN_CPU_CAPABILITY only reaches ATen's
own kernels, leaving nn.Linear's GEMM on MKL's independent CPU detector.
Only downward forcing works (upward crashes).

On Kaggle the pinned torch is installed first: the preinstalled build is
2.10.0+cu128, and the probe is pinned to 2.10.0+cpu. Same version, different
build -- exactly the kind of assumption this study exists not to make.
"""

import os
import sys

# MKL reads its thread count at load time, so these must land before torch is
# imported anywhere. Same pinning the training pipeline uses.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import hashlib
import json
import platform
import re
import socket
import subprocess
import time

# Set by run_cpu_probe.sh at push time, never by hand: a stale tag silently
# files a Kaggle sample under the wrong machine name.
PROBE_TAG = ""

TORCH_PIN = "2.10.0"
CPU_WHEEL_INDEX = "https://download.pytorch.org/whl/cpu"

# Architecture of BourginMLP (src/cpc18_models/BourginMLP.py), the condition the
# study found hardware-sensitive: 37 -> 200 -> 275 -> 100 -> 1, SiLU, sigmoid
# out, kaiming_normal_(fan_out, relu) init, zero biases, Adam + MSELoss. The
# probe reuses the shape, not the data: wide dense GEMMs are what makes the
# summation order diverge between arithmetic paths, and synthetic inputs
# exercise them without shipping the dataset. It classifies a machine; it does
# not replace the real 800-trial runs.
INPUT_SIZE = 37
HIDDEN = (200, 275, 100)
FP_SEED = 20260522     # MASTER_SEED_FOR_POOL, same constant the study derives its pools from
FP_BATCH = 32
FP_STEPS = 200
FP_LR = 0.001


def on_kaggle():
    return os.path.isdir("/kaggle/working")


def sh(cmd):
    """Best-effort shell capture -- probes must not die on a missing tool."""
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return out.stdout.strip()
    except Exception as exc:
        return f"<failed: {exc}>"


def install_pinned_torch():
    """Replace Kaggle's CUDA build with the study's CPU build. Aborts on mismatch."""
    # --no-deps on purpose: the pytorch CPU index does not serve every dependency,
    # and pointing --index-url at it makes pip resolve ALL packages there. Kaggle
    # already ships torch 2.10.0+cu128, so the dependency set is identical and only
    # the torch wheel itself needs swapping. The version/build assertions below are
    # what actually guarantee we got the right thing.
    t0 = time.time()
    rc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
         "--index-url", CPU_WHEEL_INDEX, f"torch=={TORCH_PIN}"],
        capture_output=True, text=True,
    )
    if rc.returncode != 0:
        print(rc.stdout[-4000:])
        print(rc.stderr[-4000:], file=sys.stderr)
        raise SystemExit("ABORT: could not install the pinned torch build")
    return round(time.time() - t0, 1)


def cpu_facts():
    """Hardware identity: what the tier is derived from, recorded independently."""
    model, flags = None, ""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if model is None and line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                elif not flags and line.startswith("flags"):
                    flags = line.split(":", 1)[1].strip()
                if model and flags:
                    break
    except OSError:
        pass

    flag_set = set(flags.split())
    avx512 = sorted(f for f in flag_set if f.startswith("avx512"))
    return {
        "cpu_model": model,
        "vendor": sh("LC_ALL=C lscpu | awk -F: '/Vendor ID/{print $2}'").strip() or None,
        "microarch_family": sh("LC_ALL=C lscpu | awk -F: '/Model name|CPU family|^ *Model:/{print $1\"=\"$2}' | tr '\\n' ' '").strip() or None,
        "avx512_flags": avx512,
        "has_avx512f": "avx512f" in flag_set,
        "has_avx2": "avx2" in flag_set,
        "has_amx": sorted(f for f in flag_set if f.startswith("amx")),
        "hypervisor": "hypervisor" in flag_set,
        "cores_online": os.cpu_count(),
        "cores_affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "lscpu": sh("lscpu"),
    }


def torch_facts(torch):
    """The tier torch's own (ATen) kernels dispatch on; the BLAS side is
    measured by mkl_dispatch()."""
    show = torch.__config__.show()
    tier = None
    m = re.search(r"CPU capability usage:\s*(\S+)", show)
    if m:
        tier = m.group(1)
    blas = None
    m = re.search(r"BLAS_INFO=(\w+)", show)
    if m:
        blas = m.group(1)
    return {
        "torch": torch.__version__,
        "cpu_capability_tier": tier,
        "blas": blas,
        "num_threads": torch.get_num_threads(),
        "config_show": show,
    }


def mkl_dispatch():
    """What MKL itself detected, measured rather than inferred.

    The tier from torch.__config__.show() describes ATen only; MKL runs its own
    CPU detection for the GEMMs behind nn.Linear, and on non-Intel silicon the
    two can disagree. MKL_VERBOSE's banner names the ISA it actually dispatched.

    Runs in a subprocess: MKL reads MKL_VERBOSE when the library loads, and
    nothing here may perturb the state the fingerprint ran in.
    """
    env = dict(os.environ, MKL_VERBOSE="1")
    code = "import torch; a=torch.randn(64,64); b=torch.randn(64,64); (a@b).sum()"
    try:
        out = subprocess.run([sys.executable, "-c", code], env=env,
                             capture_output=True, text=True, timeout=300)
    except Exception as exc:
        return {"banner": f"<failed: {exc}>", "isa": None}

    banner = None
    for line in (out.stdout + out.stderr).splitlines():
        if line.startswith("MKL_VERBOSE") and "Product build" in line:
            banner = line.replace("MKL_VERBOSE", "", 1).strip()
            break
    isa = None
    if banner:
        for name, pat in (("AVX512", "AVX-512"), ("AVX2", "AVX2"),
                          ("AVX", "Advanced Vector Extensions ("), ("SSE4_2", "SSE4.2"),
                          # No ISA named at all means MKL took its baseline path. It
                          # prints this on non-Intel vendors: MKL checks the vendor
                          # string and declines its own AVX2 kernels on AMD even when
                          # the silicon has AVX2. Not a parse failure -- the finding.
                          ("GENERIC", "Intel(R) Architecture processors")):
            if pat in banner:
                isa = name
                break
    return {"banner": banner, "isa": isa}


def _hash_params(model):
    h = hashlib.sha256()
    for name, param in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(param.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:32]


def fingerprint(torch):
    """Init + post-training weight hashes.

    Init runs on CPU RNG and is expected to match everywhere. The trained
    hash is the discriminator: identical within an arithmetic path, divergent
    across paths.
    """
    import torch.nn as nn

    torch.set_num_threads(1)
    torch.manual_seed(FP_SEED)

    h1, h2, h3 = HIDDEN
    model = nn.Sequential(
        nn.Linear(INPUT_SIZE, h1), nn.SiLU(),
        nn.Linear(h1, h2), nn.SiLU(),
        nn.Linear(h2, h3), nn.SiLU(),
        nn.Linear(h3, 1), nn.Sigmoid(),
    )
    torch.manual_seed(FP_SEED)
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(mod.bias)
    init_hash = _hash_params(model)

    gen = torch.Generator().manual_seed(FP_SEED)
    x = torch.rand(FP_BATCH * FP_STEPS, INPUT_SIZE, generator=gen)
    y = torch.rand(FP_BATCH * FP_STEPS, 1, generator=gen)

    opt = torch.optim.Adam(model.parameters(), lr=FP_LR)
    crit = torch.nn.MSELoss()
    t0 = time.time()
    losses = []
    for step in range(FP_STEPS):
        lo, hi = step * FP_BATCH, (step + 1) * FP_BATCH
        opt.zero_grad()
        loss = crit(model(x[lo:hi]), y[lo:hi])
        loss.backward()
        opt.step()
        losses.append(loss.item())
    elapsed = time.time() - t0

    return {
        "init_hash": init_hash,
        "trained_hash": _hash_params(model),
        "final_loss_hex": float(losses[-1]).hex(),   # exact bits, no decimal rounding
        "final_loss": losses[-1],
        "steps": FP_STEPS,
        "seconds": round(elapsed, 2),
        "sec_per_step": round(elapsed / FP_STEPS, 5),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=None, help="name for this machine/run (default: PROBE_TAG or hostname)")
    ap.add_argument("--out-dir", default=None, help="where to write the JSON")
    ap.add_argument("--install", action="store_true", help="force installing the pinned torch build")
    args = ap.parse_args()

    kaggle = on_kaggle()
    report = {
        "tag": args.tag or PROBE_TAG or socket.gethostname(),
        "hostname": socket.gethostname(),
        "on_kaggle": kaggle,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "os": platform.platform(),
        "libc": " ".join(platform.libc_ver()),
        "forced": {
            # Empty on a native run; set on the forced-tier reference runs.
            "ATEN_CPU_CAPABILITY": os.environ.get("ATEN_CPU_CAPABILITY"),
            "MKL_ENABLE_INSTRUCTIONS": os.environ.get("MKL_ENABLE_INSTRUCTIONS"),
        },
        "threads_env": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }
    report["cpu"] = cpu_facts()

    if kaggle or args.install:
        print("Installing pinned torch (CPU build)...", flush=True)
        report["install_seconds"] = install_pinned_torch()

    import torch
    report.update(torch_facts(torch))

    if not torch.__version__.startswith(TORCH_PIN):
        raise SystemExit(f"ABORT: torch is {torch.__version__}, study is pinned to {TORCH_PIN}")
    if "+cu" in torch.__version__:
        raise SystemExit(f"ABORT: torch is a CUDA build ({torch.__version__}); the probe uses +cpu")

    print("Fingerprinting...", flush=True)
    report["fingerprint"] = fingerprint(torch)
    report["mkl_dispatch"] = mkl_dispatch()

    out_dir = args.out_dir or ("/kaggle/working" if kaggle else
                               os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_results"))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"cpu_probe_{report['tag']}.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)

    c, f = report["cpu"], report["fingerprint"]
    print(f"""
=== {report['tag']} ===
  CPU            : {c['cpu_model']}  ({c['vendor']}, {c['cores_online']} cores, virtualised={c['hypervisor']})
  AVX-512 flags  : {' '.join(c['avx512_flags']) or '(none)'}
  torch          : {report['torch']}  BLAS={report['blas']}
  TIER (ATen)    : {report['cpu_capability_tier']}   forced={report['forced']}
  MKL dispatch   : {report['mkl_dispatch']['isa']}
  MKL banner     : {report['mkl_dispatch']['banner']}
  init hash      : {f['init_hash']}
  trained hash   : {f['trained_hash']}
  final loss     : {f['final_loss_hex']}
  fingerprint in : {f['seconds']}s
  written to     : {path}
""")


if __name__ == "__main__":
    main()
