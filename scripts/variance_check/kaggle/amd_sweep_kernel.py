#!/usr/bin/env python3
# encoding: utf-8

"""
Run the CPC18 variance sweep on Kaggle's AMD cell.

Kaggle does not let you choose the CPU. Roughly 7% of CPU-only sessions land on
an AMD EPYC 7B12 (Zen 2); the rest land on Intel Broadwell-EP. Those are two
distinct reproducibility cells (measured 2026-09-03 over 71 probes): both report
torch tier AVX2, but MKL declines its AVX2 kernels on AuthenticAMD and falls
back to its generic path, so the trained weights differ bit for bit.

So this kernel is a lottery ticket: it checks the vendor first and exits in
seconds if it drew Intel, costing almost nothing. Re-push until it draws AMD.
Progress accumulates across sessions because the trainers are idempotent per
seed (_seedsAlreadyDone), so a 12 h session wall is not a problem -- whatever
finished is banked and the next winning session picks up from there.

Pushed by run_amd_sweep.sh, which stamps the constants below.
"""

import os
import sys

# MKL reads thread counts at load time -- must precede any torch import, here
# and in the child trainers (they inherit this environment).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import glob
import json
import shutil
import subprocess
import time

# Stamped by run_amd_sweep.sh, never edited by hand.
TAG = "kaggle-amd"
DATASET_SLUG = "eeha-cpc18-code"
WORKERS = 4
BUDGET_HOURS = 11.0            # Kaggle kills CPU sessions at 12 h; leave margin.
CPU_WHEEL_INDEX = "https://download.pytorch.org/whl/cpu"

TARGET_VENDOR = "AuthenticAMD"
# The AMD cell's fingerprint, from cpu_probe.py over 5 independent sessions.
# If a session claims to be AMD but does not reproduce this, the fleet is more
# heterogeneous than measured and the run must not be merged in with the rest
# as if it were the same cell.
EXPECTED_TRAINED_HASH = "5d99b98fa89ee7f8eb7c4c2e94f469cd"

WORK = "/kaggle/working"
REPO = os.path.join(WORK, "repo")

# The manuscript's headline statistics (partial eta^2, ICC(3,1), closest-pair
# misranking in 07_analyze_ablation.py) need the COMPLETE 3x3 grid of a study:
# closestPairMisranking picks the minimum-gap pair among the 9 cells, so a
# partial grid yields nothing at all. Hence whole studies, not single cells.
#
# B3 goes first even though B1 is the more interesting study: B3 is cheaper
# (~3 h vs ~4 h of wall time here), so if a session dies early we still come
# away with one study fully replicated on this cell instead of two half ones.
#
# B1X (the L0_000316 level) is deliberately excluded: it is a supplementary
# learning rate, not part of the headline 3x3, and its three cells alone cost
# ~39 h single-thread -- they would turn one session into four.
B3_CELLS = [f"SparseMLP_E{e}_L0_{l}"
            for e in (10, 20, 40) for l in ("01", "001", "0001")]
B1_CELLS = [f"BourginMLP_B{b}_L0_{l}"
            for b in (16, 32, 64) for l in ("01", "001", "0001")]

JOBS = [
    ("ablation", "b3", B3_CELLS),
    ("ablation", "b1", B1_CELLS),
    # Already banked from the first sweep; resume skips them in seconds. Listed
    # so the kernel states the full target of the cell rather than implying it.
    ("regime2",  None, ["ContextDepNet", "BEAST_GB"]),
]


def cpu_vendor():
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("vendor_id"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def cpu_model():
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def find_dataset():
    """Locate the packaged repo by CONTENT, not by name or path.

    Kaggle mounts datasets at a variable depth and flattens the uploaded zip,
    so there is no `repo/` directory to glob for. Matching on the content
    signature is what survives that.
    """
    for root, dirs, files in os.walk("/kaggle/input"):
        if os.path.isdir(os.path.join(root, "src")) and "requirements.lock" in files:
            return root
    print("Contents of /kaggle/input:")
    for root, dirs, files in os.walk("/kaggle/input"):
        print("   ", root, files[:5])
    raise SystemExit(f"ABORT: dataset {DATASET_SLUG} not attached to this kernel")


def verify_fingerprints(repo):
    """Recompute what the packer recorded, with the packer's own module.

    A mismatch means this session is not running the same bytes as the lab
    machines, which invalidates the whole point of the comparison.
    """
    sys.path.insert(0, os.path.join(repo, "scripts", "variance_check", "kaggle"))
    from fingerprint_repo import fingerprints

    ref = {}
    for line in open(os.path.join(repo, "FINGERPRINTS.txt")):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            ref[k] = v

    got = fingerprints(repo)
    mismatched = [k for k in ("training_fingerprint", "features_fingerprint")
                  if ref.get(k) != got.get(k)]
    return {"expected": ref, "recomputed": got, "mismatched": mismatched}


def install_stack(repo):
    """Install requirements.lock exactly, with torch from the CPU wheel index.

    Installing only torch would have been wrong: the study's methodology pins
    the WHOLE lock, and Kaggle ships its own numpy/pandas/scikit-learn/xgboost
    at different versions. Feature loading, standardisation and the seed pool
    all run through those, so an unpinned stack makes this session a different
    experiment wearing the same name.

    torch goes last and with --no-deps so nothing pulled in by the lock can
    replace the +cpu build with the CUDA one from PyPI.
    """
    lock = os.path.join(repo, "requirements.lock")
    rest = os.path.join(WORK, "requirements_notorch.txt")
    pinned = {}
    with open(lock) as fh, open(rest, "w") as out:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, _, ver = line.partition("==")
            pinned[name.lower().replace("-", "_")] = ver
            if name.lower() != "torch":
                out.write(line + "\n")

    t0 = time.time()
    for args in ([sys.executable, "-m", "pip", "install", "-q", "-r", rest],
                 [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                  "--index-url", CPU_WHEEL_INDEX, f"torch=={pinned['torch']}"]):
        rc = subprocess.run(args, capture_output=True, text=True)
        if rc.returncode != 0:
            print(rc.stdout[-3000:]); print(rc.stderr[-3000:], file=sys.stderr)
            raise SystemExit("ABORT: could not install the pinned stack")
    seconds = round(time.time() - t0, 1)

    # Verify what actually resolved, rather than trusting pip's exit code.
    import importlib.metadata as md
    resolved, wrong = {}, []
    for name, want in pinned.items():
        try:
            got = md.version(name)
        except md.PackageNotFoundError:
            got = None
        resolved[name] = got
        base = (got or "").split("+")[0]
        if base != want:
            wrong.append(f"{name}: want {want}, got {got}")
    return {"seconds": seconds, "resolved": resolved, "mismatched": wrong}


def run(cmd, env, timeout, log_name):
    t0 = time.time()
    print(f"\n$ {' '.join(cmd)}   (timeout {timeout/3600:.2f} h)", flush=True)
    try:
        p = subprocess.run(cmd, env=env, timeout=timeout)
        rc = p.returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        rc, timed_out = None, True
        print(f"  [budget] {log_name}: time budget reached, stopping this job",
              flush=True)
    return {"cmd": cmd, "rc": rc, "timed_out": timed_out,
            "seconds": round(time.time() - t0, 1)}


def main():
    started = time.time()
    summary = {"tag": TAG, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "cpu_model": cpu_model(), "vendor": cpu_vendor()}

    # ---- the lottery check: cheapest possible exit on the common case --------
    if summary["vendor"] != TARGET_VENDOR:
        summary["outcome"] = "skipped_wrong_vendor"
        json.dump(summary, open(os.path.join(WORK, "sweep_summary.json"), "w"), indent=2)
        print(f"Drew {summary['cpu_model']} ({summary['vendor']}), "
              f"not {TARGET_VENDOR}. Exiting without installing anything.")
        return

    print(f"Drew {summary['cpu_model']}. Proceeding.", flush=True)

    src = find_dataset()
    if os.path.isdir(REPO):
        shutil.rmtree(REPO)
    shutil.copytree(src, REPO)
    print(f"  dataset root: {src} -> {REPO}", flush=True)
    summary["fingerprints"] = verify_fingerprints(REPO)
    if summary["fingerprints"]["mismatched"]:
        summary["outcome"] = "abort_fingerprint_mismatch"
        json.dump(summary, open(os.path.join(WORK, "sweep_summary.json"), "w"), indent=2)
        raise SystemExit(
            f"ABORT: {summary['fingerprints']['mismatched']} do not match the "
            f"packaged fingerprints -- this session is not running the study's bytes")

    # ---- pinned stack, then prove which cell we are in ----------------------
    summary["stack"] = install_stack(REPO)
    if summary["stack"]["mismatched"]:
        summary["outcome"] = "abort_stack_mismatch"
        json.dump(summary, open(os.path.join(WORK, "sweep_summary.json"), "w"), indent=2)
        raise SystemExit(f"ABORT: stack does not match requirements.lock: "
                         f"{summary['stack']['mismatched']}")
    print(f"Stack pinned in {summary['stack']['seconds']}s", flush=True)

    # ---- prove which cell we are in, with the same code the lab ran ----------
    probe = os.path.join(REPO, "scripts", "variance_check", "cpu_probe.py")
    subprocess.run([sys.executable, probe, "--tag", f"{TAG}-probe", "--out-dir", WORK],
                   check=True)
    pj = json.load(open(os.path.join(WORK, f"cpu_probe_{TAG}-probe.json")))
    summary["probe"] = {"tier": pj["cpu_capability_tier"],
                        "mkl": (pj.get("mkl_dispatch") or {}).get("isa"),
                        "init_hash": pj["fingerprint"]["init_hash"],
                        "trained_hash": pj["fingerprint"]["trained_hash"],
                        "torch": pj["torch"]}
    if summary["probe"]["trained_hash"] != EXPECTED_TRAINED_HASH:
        summary["outcome"] = "abort_unexpected_cell"
        json.dump(summary, open(os.path.join(WORK, "sweep_summary.json"), "w"), indent=2)
        raise SystemExit(
            f"ABORT: AMD host but fingerprint {summary['probe']['trained_hash']} "
            f"!= expected {EXPECTED_TRAINED_HASH}. This is a different cell; "
            f"do not merge these results with the others.")
    print(f"Cell confirmed: tier={summary['probe']['tier']} "
          f"MKL={summary['probe']['mkl']} hash={summary['probe']['trained_hash']}",
          flush=True)

    # ---- outputs land straight in /kaggle/working via a symlink --------------
    # output_path is derived from the repo root, and a hard session kill would
    # leave results stranded inside repo/ if we copied them only at the end.
    out = os.path.join(WORK, f"output_data_{TAG}")
    os.makedirs(out, exist_ok=True)
    link = os.path.join(REPO, f"output_data_{TAG}")
    if not os.path.islink(link):
        os.symlink(out, link)

    resume = os.path.join(REPO, "resume")
    if os.path.isdir(resume):
        n = 0
        for cell in os.listdir(resume):
            s, d = os.path.join(resume, cell), os.path.join(out, cell)
            if os.path.isdir(s):
                os.makedirs(d, exist_ok=True)
                for f in os.listdir(s):
                    shutil.copy2(os.path.join(s, f), os.path.join(d, f))
                    n += 1
        print(f"Seeded {n} resume file(s) -- already-done seeds will be skipped.",
              flush=True)

    # ---- the sweep ----------------------------------------------------------
    env = dict(os.environ, CPC18_OUTPUT_TAG=TAG)
    deadline = started + BUDGET_HOURS * 3600
    summary["jobs"] = []
    for kind, study, conds in JOBS:
        left = deadline - time.time()
        if left < 120:
            print("\n[budget] out of time; remaining jobs left for the next session.",
                  flush=True)
            break
        if kind == "ablation":
            cmd = [sys.executable, os.path.join(REPO, "src", "04_train_ablation.py"),
                   "--study", study, "--workers", str(WORKERS),
                   "--conditions", ",".join(conds)]
        else:
            cmd = [sys.executable, os.path.join(REPO, "src", "01_train_regime2.py"),
                   "--workers", str(WORKERS), "--models", ",".join(conds)]
        summary["jobs"].append(run(cmd, env, left, ",".join(conds)))

    # ---- what actually got done --------------------------------------------
    counts = {}
    for cell in sorted(os.listdir(out)) if os.path.isdir(out) else []:
        y = glob.glob(os.path.join(out, cell, "*.yaml"))
        if not y:
            continue
        # Count trials without a full YAML parse: entries are timestamp keys at
        # column 0. Cheap and good enough for a progress report.
        n = sum(1 for line in open(y[0]) if line[:1].isdigit() and line.rstrip().endswith(":"))
        counts[cell] = n
    summary["trials_by_cell"] = counts
    summary["elapsed_hours"] = round((time.time() - started) / 3600, 2)
    summary["outcome"] = "ran"
    json.dump(summary, open(os.path.join(WORK, "sweep_summary.json"), "w"), indent=2)

    print("\n=== summary ===")
    for k, v in counts.items():
        print(f"  {k:32s} {v:4d}/800")
    print(f"  elapsed: {summary['elapsed_hours']} h")

    # The repo copy would otherwise be downloaded with every fetch.
    os.unlink(link)
    shutil.rmtree(REPO, ignore_errors=True)


if __name__ == "__main__":
    main()
