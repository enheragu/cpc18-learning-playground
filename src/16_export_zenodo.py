#!/usr/bin/env python3
# encoding: utf-8

"""
Build the public data deposit (Zenodo) of the CPC18 variance manuscript.

Every record is rebuilt from a whitelist of fields; no data file is copied
as is. Host names, user names, local paths, training timestamps (the keys of
the training YAMLs) and probe tags never reach the output. Machines and
probes get neutral identifiers, assigned from what each machine-information
file says (CPU, OS release, forced tier) and not from the folder name, so
this script names no host either. Before finishing, every output file is
checked: the curve files must parse as numbers only, and the text files are
scanned for the host names and tags read during the export, the local user
and host, home paths, e-mail and IP addresses and training timestamps. Any
hit aborts the export.

Output (ignored by git), by default `<repo>/zenodo_export/`:

  trials.csv                     one row per training trial, all experiments
  curves_reference.jsonl.gz      per-epoch MSE curves, reference machine
  curves_cross_machine.jsonl.gz  per-epoch MSE curves, other machines
  machines.csv                   machines and forced-tier configurations
  probes.csv                     CPU fingerprint probes
  split_variance.csv             20 random 208/60 re-splits, fixed init seed
  perproblem_errors.csv          per-test-problem errors, first 50 seeds
  cpc18_features.csv             BEAST-derived features of the CPC18 problems
  README.md                      contents, field dictionary, licence

Usage:
    cd src && python 16_export_zenodo.py [--out PATH] [--article CITATION] [--code-doi DOI]

`--article` replaces the citation of the article (by default, the manuscript
as submitted) and `--code-doi` adds the DOI of the archived code release, so
that the archived code can build the final deposit.
"""

import argparse
import csv
import getpass
import glob
import gzip
import json
import os
import re
import shutil
import socket

import path_setup  # noqa: F401

import numpy as np

from utils.log_utils import log, bcolors
from cpc18_utils import output_path
from cpc18_utils.safe_yaml import fastLoadYaml


_REPO_ROOT = os.path.dirname(output_path)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3}$")

# Default citation of the manuscript and address of the code. The final
# citation and the DOI of the archived code are given on the command line
# (--article, --code-doi); the export warns while any TODO remains.
ARTICLE_CITATION = ("Heredia-Aguado, E., Rujano, A., Valiente, D., Gil, A. "
                    "Variance-Aware Evaluation of Behavioral Choice Prediction "
                    "Models: A Replication and Extension on the Choice "
                    "Prediction Competition 2018. Manuscript submitted for "
                    "publication.")
CODE_URL = "https://github.com/enheragu/cpc18-learning-playground"
DEPOSIT_FILES = {"README.md", "trials.csv", "curves_reference.jsonl.gz", "curves_cross_machine.jsonl.gz",
                 "machines.csv", "probes.csv", "split_variance.csv", "perproblem_errors.csv",
                 "cpc18_features.csv"}

MASTER_SEED_FOR_POOL = 20260522    # shared seed pool (main runs, ablations)
MASTER_SEED_ORDER    = 99999777    # factorial design, data-order dimension


# Neutral machine identifiers, the `machine_id` of machines.csv in the data
# deposit, listed in the order of the machine table of the manuscript
# (Appendix H). An output folder is matched by the CPU model and OS release
# of its machine-information file, plus the forced tier for the forced runs.
# `native` is the torch CPU-capability tier of the hardware and `dispatch`
# the BLAS kernel path, both as listed in that table.
_MACHINES = [
    # id                    role                  cpu key           os key   forced     native     dispatch
    ("M00",                "reference",          "i7-1165G7",      None,    None,      "AVX512",  "AVX512"),
    ("M01",                "comparison",         "i7-11700",       "20.04", None,      "AVX512",  "AVX512"),
    ("M01-forced-AVX2",    "forced tier",        "i7-11700",       "20.04", "AVX2",    "AVX512",  ""),
    ("M01-forced-DEFAULT", "forced tier",        "i7-11700",       "20.04", "DEFAULT", "AVX512",  ""),
    ("M02",                "comparison",         "i9-10900X",      "20.04", None,      "AVX512",  "AVX512"),
    ("M03",                "comparison",         "i5-9400F",       "20.04", None,      "AVX2",    "AVX2"),
    ("M03-forced-DEFAULT", "forced tier",        "i5-9400F",       "20.04", "DEFAULT", "AVX2",    ""),
    ("M04",                "comparison",         "i9-14900KF",     "22.04", None,      "AVX2",    "AVX2"),
    ("M04-forced-DEFAULT", "forced tier",        "i9-14900KF",     "22.04", "DEFAULT", "AVX2",    ""),
    ("M05",                "comparison",         "i9-14900KF",     "24.04", None,      "AVX2",    "AVX2"),
    ("M06",                "comparison",         "Raspberry Pi 5", "24.04", None,      "DEFAULT", "OpenBLAS"),
    ("C01",                "cloud, probes only", "Xeon",           None,    None,      "AVX2",    "AVX2"),
    ("C02",                "cloud, full sweep",  "EPYC 7B12",      None,    None,      "AVX2",    "generic"),
]
_MACHINE_ORDER = [m[0] for m in _MACHINES]

# The reference machine wrote no machine-information file; these values
# are the ones the manuscript reports for it (Appendix H), the locked
# requirements of the repository and the Python environment its runs used,
# which holds the standard torch package (CUDA build, run on the CPU: the
# machine has no GPU torch can use) instead of the CPU-only one.
_REFERENCE_INFO = {
    "cpu_model": "Intel Core i7-1165G7 (Tiger Lake)",
    "architecture": "x86_64",
    "os": "Ubuntu 24.04",
    "kernel": "6.17",
    "python": "3.12.3",
    "torch": "2.10.0+cu128",
    "thread_pinning": "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                      "OPENBLAS_NUM_THREADS=1 torch.set_num_threads(1)",
}

_EXPERIMENT_FILES = [
    ("main",                 "sharedpool_r2_training_metrics.yaml"),
    ("ablation",             "sharedpool_r2_ablation_training_metrics.yaml"),
    ("factorial_init_order", "factorial_init_order_r2_training_metrics.yaml"),
    ("validation_split",     "valsplit_r2_training_metrics.yaml"),
]
_EXPERIMENT_ORDER = ["main", "ablation_bourginmlp", "ablation_sparsemlp",
                     "factorial_init_order", "validation_split"]
_MODEL_ORDER = ["LinearLogistic", "BourginMLP", "ContextDepNet", "SparseMLP",
                "PsychForest", "BEAST_GB"]
# sklearn-style models: the trainer logs the global batch size, learning
# rate and patience for them although they use none of the three.
_TREE_MODELS = {"PsychForest", "BEAST_GB"}
_CELL_RE = re.compile(r"^(BourginMLP|SparseMLP)_([BE])(\d+)_L(\d+)_(\d+)$")

TRIAL_FIELDS = [
    "trial_id", "experiment", "machine_id", "condition", "model",
    "batch_size", "learning_rate", "epsilon",
    "seed", "init_seed", "order_seed",
    "mse", "mse_x100", "mse_random_baseline",
    "best_epoch", "n_epochs", "patience", "train_duration_s",
    "val_split_seed", "n_train_problems", "n_val_problems",
    "best_test_epoch", "mse_x100_testsel", "val_mse_at_selection",
]
MACHINE_FIELDS = [
    "machine_id", "role", "cpu_model", "architecture", "logical_cpus",
    "avx512_flags", "native_tier", "run_tier", "aten_cpu_capability",
    "mkl_enable_instructions", "blas_dispatch", "os", "kernel", "python",
    "torch", "thread_pinning", "n_trials", "info_source",
]
PROBE_FIELDS = [
    "probe_id", "machine_id", "vendor", "cpu_model", "hypervisor",
    "logical_cpus", "os", "libc", "python", "torch", "torch_tier",
    "aten_cpu_capability", "mkl_enable_instructions", "blas",
    "mkl_dispatch_isa", "mkl_banner", "init_hash", "trained_hash",
    "final_loss", "final_loss_hex", "steps",
]


# ---------------------------------------------------------------------
# Machines
# ---------------------------------------------------------------------

def _parseMachineInfo(path):
    """`key: value` lines of a MACHINE_INFO.txt; indented lines continue
    the previous value."""
    info, key = {}, None
    with open(path) as f:
        for line in f:
            m = re.match(r"^([a-z0-9_]+):\s*(.*)$", line)
            if m:
                key = m.group(1)
                info[key] = m.group(2).strip()
            elif key and line[:1].isspace() and line.strip():
                info[key] += " " + line.strip()
    return info


def _matchMachine(cpu, os_name, forced):
    hits = [m for m in _MACHINES
            if m[0] not in ("M00", "C01", "C02") and m[2] in cpu
            and m[3] in os_name and m[4] == forced]
    if len(hits) != 1:
        raise SystemExit(f"cannot identify machine (cpu={cpu!r}, "
                         f"os={os_name!r}, forced={forced!r}): {len(hits)} matches")
    return hits[0]


def identifyMachines(sensitive):
    """
        Returns `{machine_id: folder}`, `{machine_id: info row}` and
        `{host name: machine_id}` for the native machines. Host names are
        kept in memory only, to link forced runs and laboratory probes to
        their machine, and are added to `sensitive` for the final scan.
    """
    folders = {"M00": output_path}
    rows = {"M00": dict(_REFERENCE_INFO, info_source="manuscript Appendix H, locked "
                        "requirements and Python environment of the runs "
                        "(no machine-information file)")}
    infos = {}
    for d in sorted(glob.glob(os.path.join(_REPO_ROOT, "output_data_*"))):
        tag = os.path.basename(d)[len("output_data_"):]
        if tag not in _MACHINE_ORDER:
            sensitive.add(tag)
        p = os.path.join(d, "MACHINE_INFO.txt")
        infos[d] = _parseMachineInfo(p) if os.path.exists(p) else None

    natives = {i["hostname"]: i for i in infos.values()
               if i and "forced_tier" not in i}
    host2id = {}
    for d, info in infos.items():
        if info is None:
            # Only the cloud AMD sweep was fetched without an information
            # file; its hardware is described from the probes instead.
            if "amd_sweep" not in os.path.realpath(d) or "C02" in folders:
                raise SystemExit(f"unexpected folder without machine info: {d}")
            folders["C02"] = d
            continue
        sensitive.add(info["hostname"])
        forced = info.get("forced_tier")
        hw = natives[info["hostname"]] if forced else info
        cpu = hw.get("cpu_model") or hw.get("board", "")
        mid, role, _, _, _, native, dispatch = _matchMachine(cpu, hw["os"], forced)
        if forced is None:
            host2id[info["hostname"]] = mid
            tier = hw.get("torch_cpu_capability")
            if tier and tier != native:
                raise SystemExit(f"{mid}: tier {tier} differs from inventory {native}")
        else:
            if f"CPU capability usage: {forced}" not in info.get("verification", ""):
                raise SystemExit(f"{mid}: forced tier not verified in machine info")
        env = dict(kv.split("=", 1) for kv in info.get("env", "").split() if "=" in kv)
        if hw.get("board"):
            cpu = f"{hw['board']}, {hw.get('cpu_part', '').split(',')[0]}"
        folders[mid] = d
        rows[mid] = {
            "cpu_model": cpu,
            "architecture": hw.get("arch", "x86_64"),
            "logical_cpus": hw.get("cores", ""),
            "avx512_flags": "" if hw.get("avx512_flags") == "(none)" else hw.get("avx512_flags", ""),
            "aten_cpu_capability": env.get("ATEN_CPU_CAPABILITY", ""),
            "mkl_enable_instructions": env.get("MKL_ENABLE_INSTRUCTIONS", ""),
            "os": hw.get("os", ""), "kernel": hw.get("kernel", ""),
            "python": hw.get("python", ""), "torch": hw.get("torch", ""),
            "thread_pinning": hw.get("thread_pinning", ""),
            "info_source": "machine-information file" + (
                " of the forced run and of the native machine" if forced else ""),
        }
    missing = [m[0] for m in _MACHINES if m[0] != "C01" and m[0] not in folders]
    if missing:
        raise SystemExit(f"no output folder found for {missing}")
    return folders, rows, host2id


# ---------------------------------------------------------------------
# Trials and curves
# ---------------------------------------------------------------------

def _conditionKey(condition):
    m = _CELL_RE.match(condition)
    if m:
        return (_MODEL_ORDER.index(m.group(1)), int(m.group(3)),
                float(f"{m.group(4)}.{m.group(5)}"))
    return (_MODEL_ORDER.index(condition), 0, 0.0)


def _argmin(curve):
    return int(np.argmin(curve)) if curve else None


def _trialRow(experiment, machine_id, condition, e, stats):
    model = _CELL_RE.match(condition).group(1) if _CELL_RE.match(condition) else condition
    tree = model in _TREE_MODELS
    test_curve = e.get("mse_plot") or []
    if experiment == "validation_split":
        best = _argmin(e.get("val_mse_plot") or [])
    else:
        best = _argmin(test_curve)
        if test_curve:
            rel = abs(e["mse"] - min(test_curve)) / e["mse"]
            stats["max_rel_mse_vs_curve"] = max(stats["max_rel_mse_vs_curve"], rel)
    if best is not None and best != e.get("best_epoch"):
        stats["best_epoch_recomputed"] += 1

    cell = _CELL_RE.match(condition)
    if cell:
        f1, lr = int(cell.group(3)), float(f"{cell.group(4)}.{cell.group(5)}")
        f1_rec = e.get("batch_size") if cell.group(2) == "B" else e.get("epsilon")
        if f1_rec != f1 or abs(e["learning_rate"] - lr) > 0.02 * lr:
            raise SystemExit(f"{machine_id}/{condition}: record does not match cell")
    if experiment == "factorial_init_order" and e["seed"] != e["init_seed"]:
        raise SystemExit(f"{machine_id}/{condition}: seed != init_seed")

    return {
        "experiment": experiment, "machine_id": machine_id,
        "condition": condition, "model": model.replace("_", "-"),
        "batch_size": None if tree else e.get("batch_size"),
        "learning_rate": None if tree else e.get("learning_rate"),
        "epsilon": e.get("epsilon"),
        "seed": e.get("seed"), "init_seed": e.get("init_seed"),
        "order_seed": e.get("order_seed"),
        "mse": e["mse"], "mse_x100": e["mse_x100"],
        "mse_random_baseline": e.get("mse_random_baseline"),
        "best_epoch": best, "n_epochs": len(test_curve) or None,
        "patience": None if tree else e.get("patience"),
        "train_duration_s": e.get("train_duration"),
        "val_split_seed": e.get("val_split_seed"),
        "n_train_problems": e.get("n_train_problems"),
        "n_val_problems": e.get("n_val_problems"),
        "best_test_epoch": e.get("best_test_epoch"),
        "mse_x100_testsel": e.get("mse_x100_testsel"),
        "val_mse_at_selection": e.get("val_mse_at_selection"),
    }


def _curveRecord(trial_id, e):
    rec = {"trial_id": trial_id}
    for key, name in (("mse_plot", "test_mse"), ("train_mse_plot", "train_mse"),
                      ("val_mse_plot", "val_mse")):
        if e.get(key):
            rec[name] = [float(v) for v in e[key]]
    return rec if len(rec) > 1 else None


def exportTrials(out_dir, folders, stats):
    """ trials.csv and the two curve files. Returns trials per machine. """
    sources = []
    for mid in _MACHINE_ORDER:
        if mid not in folders:
            continue
        for kind, fname in _EXPERIMENT_FILES:
            for p in glob.glob(os.path.join(folders[mid], "*", fname)):
                condition = os.path.basename(os.path.dirname(p))
                experiment = kind
                if kind == "ablation":
                    experiment = ("ablation_bourginmlp"
                                  if condition.startswith("BourginMLP")
                                  else "ablation_sparsemlp")
                sources.append((_MACHINE_ORDER.index(mid),
                                _EXPERIMENT_ORDER.index(experiment),
                                _conditionKey(condition), mid, experiment,
                                condition, p))
    sources.sort()

    per_machine = {}
    trial_id = 0
    curve_files = {
        "reference": gzip.open(os.path.join(out_dir, "curves_reference.jsonl.gz"), "wt"),
        "cross": gzip.open(os.path.join(out_dir, "curves_cross_machine.jsonl.gz"), "wt"),
    }
    with open(os.path.join(out_dir, "trials.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRIAL_FIELDS)
        w.writeheader()
        for _, _, _, mid, experiment, condition, p in sources:
            data = fastLoadYaml(p)
            extra = [k for k in data if not _DATE_RE.match(str(k))]
            if extra:
                raise SystemExit(f"{mid}/{condition}: unexpected top-level keys {extra}")
            n = 0
            for e in data.values():
                trial_id += 1
                n += 1
                row = _trialRow(experiment, mid, condition, e, stats)
                row["trial_id"] = trial_id
                w.writerow(row)
                rec = _curveRecord(trial_id, e)
                if rec:
                    out = curve_files["reference" if mid == "M00" else "cross"]
                    out.write(json.dumps(rec, separators=(",", ":")) + "\n")
            per_machine[mid] = per_machine.get(mid, 0) + n
            stats["counts"][(mid, experiment)] = stats["counts"].get((mid, experiment), 0) + n
            log(f"  {mid:20s} {experiment:22s} {condition:26s} {n:5d}")
    for fh in curve_files.values():
        fh.close()
    log(f"  -> trials.csv ({trial_id} trials)", color=bcolors.OKGREEN)
    return per_machine


# ---------------------------------------------------------------------
# Probes, machine table, auxiliary re-runs, features
# ---------------------------------------------------------------------

def exportProbes(out_dir, host2id, sensitive):
    files = sorted(glob.glob(os.path.join(_REPO_ROOT, "scripts", "variance_check",
                                          "probe_results", "cpu_probe_*.json")))
    probes = []
    for fpath in files:
        with open(fpath) as f:
            p = json.load(f)
        sensitive.update({p["tag"], p["hostname"]})
        c, fp = p["cpu"], p["fingerprint"]
        aten = p["forced"].get("ATEN_CPU_CAPABILITY")
        if p["on_kaggle"]:
            mid = "C02" if c["vendor"] == "AuthenticAMD" else "C01"
        else:
            base = host2id[p["hostname"]]
            mid = base if aten is None else f"{base}-forced-{aten.upper()}"
            if mid not in _MACHINE_ORDER:
                raise SystemExit(f"laboratory probe with unknown configuration {mid}")
        if not re.search(_MACHINES[_MACHINE_ORDER.index(mid)][2], c["cpu_model"]):
            raise SystemExit(f"probe CPU {c['cpu_model']!r} does not match {mid}")
        disp = p.get("mkl_dispatch") or {}
        probes.append(((_MACHINE_ORDER.index(mid), p["utc"]), {
            "machine_id": mid, "vendor": c["vendor"],
            "cpu_model": c["cpu_model"], "hypervisor": c["hypervisor"],
            "logical_cpus": c["cores_online"], "os": p["os"],
            "libc": p["libc"], "python": p["python"], "torch": p["torch"],
            "torch_tier": p["cpu_capability_tier"],
            "aten_cpu_capability": aten or "",
            "mkl_enable_instructions": p["forced"].get("MKL_ENABLE_INSTRUCTIONS") or "",
            "blas": p["blas"], "mkl_dispatch_isa": disp.get("isa", ""),
            "mkl_banner": disp.get("banner", ""),
            "init_hash": fp["init_hash"], "trained_hash": fp["trained_hash"],
            "final_loss": fp["final_loss"], "final_loss_hex": fp["final_loss_hex"],
            "steps": fp["steps"],
        }))
    probes.sort(key=lambda t: t[0])
    with open(os.path.join(out_dir, "probes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PROBE_FIELDS)
        w.writeheader()
        for i, (_, row) in enumerate(probes, 1):
            w.writerow({"probe_id": f"P{i:02d}", **row})
    log(f"  -> probes.csv ({len(probes)} probes)", color=bcolors.OKGREEN)
    return [row for _, row in probes]


def exportMachines(out_dir, rows, per_machine, probes):
    """ machines.csv; the two cloud groups are described from their probes. """
    for mid in ("C01", "C02"):
        group = [p for p in probes if p["machine_id"] == mid]
        uniform = {k: {p[k] for p in group} for k in
                   ("cpu_model", "logical_cpus", "os", "libc", "python", "torch")}
        if any(len(v) != 1 for v in uniform.values()):
            raise SystemExit(f"{mid}: probes disagree on {uniform}")
        u = {k: next(iter(v)) for k, v in uniform.items()}
        rows[mid] = {"cpu_model": u["cpu_model"], "architecture": "x86_64",
                     "logical_cpus": u["logical_cpus"],
                     "os": f"{u['os']} ({u['libc']})", "python": u["python"],
                     "torch": u["torch"],
                     "info_source": f"CPU fingerprint probes ({len(group)} sessions)"}
    with open(os.path.join(out_dir, "machines.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MACHINE_FIELDS)
        w.writeheader()
        for mid, role, _, _, forced, native, dispatch in _MACHINES:
            w.writerow({**rows[mid], "machine_id": mid, "role": role,
                        "native_tier": native, "run_tier": forced or native,
                        "blas_dispatch": dispatch,
                        "n_trials": per_machine.get(mid, 0)})
    log("  -> machines.csv", color=bcolors.OKGREEN)


def exportReruns(out_dir):
    """ The two auxiliary re-runs of Section 3.5, from their npz outputs. """
    res = os.path.join(_REPO_ROOT, "analysis_results")
    d = np.load(os.path.join(res, "split_variance.npz"))
    with open(os.path.join(out_dir, "split_variance.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "split_index", "mse_x100"])
        for m in _MODEL_ORDER:
            if m in d.files:
                for i, v in enumerate(d[m]):
                    w.writerow([m.replace("_", "-"), i, float(v)])
    d = np.load(os.path.join(res, "perproblem_errors.npz"))
    with open(os.path.join(out_dir, "perproblem_errors.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "seed", "problem_id", "mean_sq_error_x100"])
        for m in _MODEL_ORDER:
            if m not in d.files:
                continue
            for si, s in enumerate(d["seeds"]):
                for pj, pid in enumerate(d["problems"]):
                    w.writerow([m.replace("_", "-"), int(s), int(pid),
                                float(d[m][si, pj])])
    log("  -> split_variance.csv, perproblem_errors.csv", color=bcolors.OKGREEN)


def exportFeatures(out_dir):
    # Written at the root of the deposit: Zenodo does not keep folders.
    src = os.path.join(_REPO_ROOT, "data", "cpc18_features.csv")
    with open(src, newline="") as fi, \
         open(os.path.join(out_dir, "cpc18_features.csv"), "w", newline="") as fo:
        rows = list(csv.reader(fi))
        csv.writer(fo).writerows(rows)
    log(f"  -> cpc18_features.csv ({len(rows) - 1} rows)", color=bcolors.OKGREEN)
    return len(rows) - 1


# ---------------------------------------------------------------------
# README
# ---------------------------------------------------------------------

_README = """# Per-trial results: variance-aware evaluation of CPC18 choice prediction models

This deposit holds the per-trial results behind the article:

> {article}

Each training trial run for the article is one row of `trials.csv`. The
other files hold the per-epoch learning curves, the machines and their
configurations, the CPU fingerprint probes, two auxiliary re-runs and the
input features. The code that trains the models and recomputes the tables
and figures of the article is archived separately: {code}.

## Files

| File | Content | Records |
|---|---|---|
| `trials.csv` | One row per training trial, all experiments and machines | {n_trials} |
| `curves_reference.jsonl.gz` | Per-epoch MSE curves of the trials on the reference machine `M00` | {n_curves_ref} |
| `curves_cross_machine.jsonl.gz` | Per-epoch MSE curves of the trials on the other machines | {n_curves_cross} |
| `machines.csv` | Machines and forced-tier configurations | {n_machines} |
| `probes.csv` | CPU fingerprint probes | {n_probes} |
| `split_variance.csv` | Test MSE of LinearLogistic, PsychForest and BEAST-GB over 20 random 208/60 re-splits of the problems, initialization seed fixed | {n_split} |
| `perproblem_errors.csv` | Squared error per test problem (mean over its 5 blocks) of BourginMLP, PsychForest and BEAST-GB for the first 50 shared seeds | {n_perproblem} |
| `cpc18_features.csv` | BEAST-derived features and observed choice rates of the CPC18 problems | {n_features} |

## Experiments

The `experiment` column of `trials.csv` takes these values. Rows of
machines other than `M00` re-run a subset of the `main` and `ablation_*`
conditions on other hardware (Section 3.6 and Appendix H).

| `experiment` | Design | Article |
|---|---|---|
| `main` | Six models on the canonical CPC18 split (208 of its 210 training problems and its 60 test problems), 800 shared seeds each | Sections 1.1, 3.1, 3.2, 3.5, 4 |
| `ablation_bourginmlp` | BourginMLP, batch size {{16, 32, 64}} x learning rate {{1e-4, 1e-3, 1e-2}}, 800 seeds per cell; the cells at learning rate 3.16e-4 extend the grid ("Sensitivity to the grid") | Section 3.4 |
| `ablation_sparsemlp` | SparseMLP, target sparsity epsilon {{10, 20, 40}} x learning rate {{1e-4, 1e-3, 1e-2}}, 800 seeds per cell | Section 3.4 |
| `factorial_init_order` | 35 initialization seeds x 35 data-order seeds, four models | Section 3.3 |
| `validation_split` | Early stopping on an internal validation split (166/42 of the training problems), 200 seeds, BourginMLP and ContextDepNet | Section 5.5 |

Trials per machine and experiment:

{count_table}

## Fields of `trials.csv`

An empty cell means that the field does not apply to the row or was not
logged for it.

| Field | Meaning |
|---|---|
| `trial_id` | Integer identifier, also the key of the curve files |
| `experiment` | See above |
| `machine_id` | Machine or configuration, see `machines.csv` |
| `condition` | Model or ablation cell as named in the code, e.g. `BourginMLP_B16_L0_0001` |
| `model` | Model name as used in the article |
| `batch_size`, `learning_rate`, `epsilon` | Training hyperparameters logged with the trial. `epsilon` (target sparsity) is logged only in the SparseMLP ablation; elsewhere SparseMLP uses its default of 20. Empty for the tree-based models (PsychForest, BEAST-GB), which use none of the three; their own settings are in the code |
| `seed` | Training seed. In `main`, the ablations and `validation_split` it is drawn from the shared seed pool (generated from master seed {pool_seed}) and drives both the weight initialization and the data-shuffle order; equal seeds therefore pair trials across models, cells and machines. In `factorial_init_order` it equals `init_seed` |
| `init_seed`, `order_seed` | `factorial_init_order` only: seed of the weight initialization and of the data-shuffle order (order seeds generated from master seed {order_seed}) |
| `mse` | Test MSE of the selected model, recomputed after training on the 60 test problems |
| `mse_x100` | `mse` x 100, the unit used in the article |
| `mse_random_baseline` | Variance of the test targets, i.e. the test MSE of a constant prediction equal to their mean (field name as in the code) |
| `best_epoch` | 0-based epoch whose weights were selected: minimum of the test curve, or of the validation curve in `validation_split`. Empty for the tree-based models |
| `n_epochs` | Number of training epochs run (length of the curves). Training stops `patience` epochs after the best one, or at the epoch cap: 500 epochs, 1500 in the ablations |
| `patience` | Early-stopping patience in epochs |
| `train_duration_s` | Wall-clock training time in seconds; depends on the machine and its load |
| `val_split_seed`, `n_train_problems`, `n_val_problems` | `validation_split` only: seed and size of the internal split |
| `best_test_epoch`, `mse_x100_testsel` | `validation_split` only: epoch of the test-curve minimum and the corresponding test MSE x 100, i.e. the main protocol's selection applied to the same trajectory |
| `val_mse_at_selection` | `validation_split` only: validation MSE at `best_epoch` |

Numbers are written with full double precision, so equality between
machines can be checked directly on `mse`.

## Curve files

One JSON object per line, for every trial of a neural model (the
tree-based models have no epochs): `trial_id`, `test_mse` (test MSE after
each epoch) and `train_mse` (training MSE after each epoch), or `val_mse`
instead of `train_mse` in `validation_split`. The values are MSE, not MSE x 100. The
minimum of `test_mse` and `mse` in `trials.csv` agree to a relative
difference of at most {max_rel:.1e}; `mse` is computed again after
restoring the selected weights.

## Fields of `machines.csv`

The reference machine, the comparison machines, the four runs whose torch
CPU-capability tier was forced below the native one, and the two groups of
public cloud sessions. An empty cell means that the value was not recorded or
does not apply.

| Field | Meaning |
|---|---|
| `machine_id` | Machine or configuration, as used in `trials.csv` and `probes.csv` |
| `role` | `reference`, `comparison`, `forced tier`, `cloud, probes only` or `cloud, full sweep` |
| `cpu_model`, `architecture`, `logical_cpus` | Hardware |
| `avx512_flags` | AVX-512 flags of the CPU |
| `native_tier` | Torch CPU-capability tier selected on the hardware |
| `run_tier` | Torch CPU-capability tier used in the run |
| `aten_cpu_capability`, `mkl_enable_instructions` | The two environment variables that force the tier |
| `blas_dispatch` | Kernel path of the linear-algebra library, as reported in Appendix H |
| `os`, `kernel`, `python`, `torch` | Software of the run |
| `thread_pinning` | Thread settings of the run |
| `n_trials` | Rows of `trials.csv` from this machine |
| `info_source` | Where the description of the machine comes from |

## Fields of `probes.csv`

The CPU fingerprint probes of Appendix H, one row per session. Each probe
trains a network with the layers of BourginMLP on seeded random data. An empty
cell means that the probe did not record the value or that it does not apply.

| Field | Meaning |
|---|---|
| `probe_id` | Identifier of the probe |
| `machine_id` | Machine or configuration, see `machines.csv` |
| `vendor`, `cpu_model`, `hypervisor`, `logical_cpus` | CPU; `hypervisor` tells whether the CPU reports running under one |
| `os`, `libc`, `python`, `torch` | Software of the session |
| `torch_tier` | Torch CPU-capability tier reported in the session |
| `aten_cpu_capability`, `mkl_enable_instructions` | The two environment variables that force the tier |
| `blas` | Linear-algebra library used by torch |
| `mkl_dispatch_isa`, `mkl_banner` | Instruction set announced by the linear-algebra library at load time, and the line that announces it |
| `init_hash` | Hash of the weights after seeded initialization |
| `trained_hash` | Hash of the weights after the training run |
| `final_loss`, `final_loss_hex` | Loss of the last training step, in decimal and as an exact hexadecimal float |
| `steps` | Number of training steps |

## Fields of the re-runs

`split_variance.csv` has one row per model and re-split: `model`,
`split_index` (0 to 19) and `mse_x100`. `perproblem_errors.csv` has one row per
model, seed and test problem: `model`, `seed`, `problem_id` (as in
`cpc18_features.csv`) and `mean_sq_error_x100` (squared error averaged over the
5 blocks of the problem, x 100).

## Fields of `cpc18_features.csv`

One row per problem and block. The features are those of the BEAST-GB
reference code. Every difference is option B minus option A.

| Field | Meaning |
|---|---|
| `Dataset` | Always `CPC18` |
| `problem_id` | Position of the problem in the CPC18 data of the reference code. Problems 13 and 20 are left out, as in that code |
| `Train_Test` | Canonical split of the problem, `Train` or `Test` |
| `block` | Block of trials, 1 to 5 |
| `Feedback` | 0 in block 1 and 1 in the blocks with feedback |
| `B_rate` | Observed share of choices of option B, the target of the models |
| `Ha`, `pHa`, `La` | Option A: high outcome, its probability and low outcome |
| `LotShapeA`, `LotNumA` | Option A: shape of the lottery (`-`, `Symm`, `L-skew` or `R-skew`) and its number of outcomes |
| `Hb`, `pHb`, `Lb`, `LotShapeB`, `LotNumB` | The same for option B |
| `Amb` | 1 when the probabilities of option B are ambiguous, 0 otherwise |
| `Corr` | Correlation between the outcomes of the two options: -1, 0 or 1 |
| `diffEV`, `diffSDs`, `diffMins`, `diffMaxs` | Difference in expected value, standard deviation, minimum and maximum |
| `diffUV` | Difference in the mean of the outcomes, each with the same weight |
| `RatioMin` | Smaller absolute minimum of the two options over the larger: 1 when the minima are equal, 0 when their signs differ |
| `SignMax` | Sign of the largest outcome of the two options |
| `pBbet_Unbiased1`, `pBbet_UnbiasedFB` | Probability that option B gives a strictly larger outcome than A, minus the reverse: before feedback, with an estimate of the probabilities when ambiguous, and with feedback, with the true ones and `Corr` |
| `pBbet_Uniform` | The same with all the outcomes of each option equally likely |
| `pBbet_Sign1`, `pBbet_SignFB` | As the first two, comparing only the signs of the outcomes |
| `Dom` | 1 when option B stochastically dominates A, -1 when A dominates B, 0 otherwise |
| `diffBEV0`, `diffBEVfb` | Difference in the estimate of the expected value before feedback, and after it (the mean of that estimate and the true value) |
| `diffSignEV` | Difference in the expected sign of the outcome, multiplied by the range of the outcomes of both options |
| `BEASTpred` | Prediction of the BEAST model, from the reference code |

## Reproducing

Each experiment is produced by a script of the code repository:
`01_train_regime2.py` (`main`), `04_train_ablation.py` (ablations),
`03_train_decomp.py` (`factorial_init_order`), `11_train_valsplit.py`
(`validation_split`), `split_variance_rerun.py`, `perproblem_rerun.py`
and `scripts/variance_check/cpu_probe.py`. The seed pools are regenerated
from the master seeds above, identically on every platform.

The analysis scripts of the repository regenerate the figures and tables of
the article from these files alone: copy them into `zenodo_export/` at the
root of the repository and run the analysis as its README describes.

## Licence and attribution

This deposit is released under the Creative Commons Attribution 4.0
International licence (CC BY 4.0). `cpc18_features.csv` is derived
from the CPC18 data of Plonsky, Erev and Ert, "All raw data for CPC18:
Choice prediction competition 2018", Zenodo,
https://doi.org/10.5281/zenodo.2571510 (CC BY 4.0), with the feature
definitions of BEAST-GB (Plonsky, "Predicting Human Decisions with
Behavioral Theories and Machine Learning", OSF,
https://doi.org/10.17605/OSF.IO/VW2SU), computed by
`scripts/precompute_features.R` of the code repository. Please cite the
article and the CPC18 data when using these files.
"""


def writeReadme(out_dir, stats, per_machine, n, article, code):
    exps = [e for e in _EXPERIMENT_ORDER]
    lines = ["| Machine | " + " | ".join(f"`{e}`" for e in exps) + " |",
             "|---|" + "---|" * len(exps)]
    for mid in _MACHINE_ORDER:
        if per_machine.get(mid):
            cells = [str(stats["counts"].get((mid, e), "")) for e in exps]
            lines.append(f"| `{mid}` | " + " | ".join(cells) + " |")
    text = _README.format(
        article=article, code=code,
        n_trials=sum(per_machine.values()),
        n_curves_ref=n["curves_reference"], n_curves_cross=n["curves_cross"],
        n_machines=len(_MACHINES), n_probes=n["probes"], n_split=n["split"],
        n_perproblem=n["perproblem"], n_features=n["features"],
        count_table="\n".join(lines), pool_seed=MASTER_SEED_FOR_POOL,
        order_seed=MASTER_SEED_ORDER, max_rel=stats["max_rel_mse_vs_curve"])
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(text)
    log("  -> README.md", color=bcolors.OKGREEN)


# ---------------------------------------------------------------------
# Final checks
# ---------------------------------------------------------------------

def _countLines(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return sum(1 for _ in f)


def checkOutput(out_dir, sensitive):
    """
        The curve files must contain only trial ids and lists of numbers;
        every other file is scanned as text for anything that identifies a
        machine, a person or a local path. Any hit aborts the export.
    """
    allowed = {"trial_id", "test_mse", "train_mse", "val_mse"}
    for name in ("curves_reference.jsonl.gz", "curves_cross_machine.jsonl.gz"):
        with gzip.open(os.path.join(out_dir, name), "rt") as f:
            for line in f:
                rec = json.loads(line)
                if set(rec) - allowed or not isinstance(rec["trial_id"], int) or \
                        any(not all(isinstance(v, float) for v in rec[k])
                            for k in rec if k != "trial_id"):
                    raise SystemExit(f"{name}: non-numeric content in {rec.get('trial_id')}")

    terms = {t for t in sensitive | {getpass.getuser(), socket.gethostname()} if t}
    patterns = [re.escape(t) if len(t) >= 6 else rf"\b{re.escape(t)}\b"
                for t in sorted(terms, key=len, reverse=True)]
    patterns += [r"/home/", r"~/", r"[\w.+-]+@[\w-]+\.[\w.-]+",
                 r"\b\d{1,3}(?:\.\d{1,3}){3}\b",
                 r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", r"(?i:kaggle)"]
    forbidden = re.compile("|".join(patterns))
    todos = []
    for dirpath, _, files in os.walk(out_dir):
        for fn in files:
            if fn.endswith(".gz"):
                continue
            path = os.path.join(dirpath, fn)
            with open(path) as f:
                for i, line in enumerate(f, 1):
                    m = forbidden.search(line)
                    if m:
                        raise SystemExit(f"forbidden content in {os.path.relpath(path, out_dir)}"
                                         f":{i}: {m.group(0)!r}")
                    if "TODO" in line:
                        todos.append(f"{os.path.relpath(path, out_dir)}:{i}")
    log(f"  output check passed ({len(terms)} host/tag terms and generic patterns)",
        color=bcolors.OKGREEN)
    for t in todos:
        log(f"  [WARN] placeholder to fill before upload: {t}", color=bcolors.WARNING)


def removeGzipTimestamps(out_dir):
    """gzip stores the time of the export in its header: without it, two exports of the same data are byte-identical."""
    for path in glob.glob(os.path.join(out_dir, "*.gz")):
        with gzip.open(path, "rb") as f:
            content = f.read()
        with open(path, "wb") as f:
            f.write(gzip.compress(content, mtime=0))


def export(out_dir, article, code):
    os.makedirs(out_dir)
    sensitive = set()
    stats = {"counts": {}, "best_epoch_recomputed": 0, "max_rel_mse_vs_curve": 0.0}
    log("Identifying machines")
    folders, rows, host2id = identifyMachines(sensitive)
    log("Exporting trials and curves")
    per_machine = exportTrials(out_dir, folders, stats)
    empty = [mid for mid, role, *_ in _MACHINES if role != "cloud, probes only" and not per_machine.get(mid)]
    if empty:
        raise SystemExit(f"no trials found for {empty}")
    log(f"  best_epoch differs from the logged field in {stats['best_epoch_recomputed']} trials "
        f"(the curve minimum is exported); max relative |mse - min(test curve)| = "
        f"{stats['max_rel_mse_vs_curve']:.2e}")
    log("Exporting probes, machines, re-runs and features")
    probes = exportProbes(out_dir, host2id, sensitive)
    exportMachines(out_dir, rows, per_machine, probes)
    exportReruns(out_dir)
    n_features = exportFeatures(out_dir)
    n = {"curves_reference": _countLines(os.path.join(out_dir, "curves_reference.jsonl.gz")),
         "curves_cross": _countLines(os.path.join(out_dir, "curves_cross_machine.jsonl.gz")),
         "probes": len(probes),
         "split": _countLines(os.path.join(out_dir, "split_variance.csv")) - 1,
         "perproblem": _countLines(os.path.join(out_dir, "perproblem_errors.csv")) - 1,
         "features": n_features}
    writeReadme(out_dir, stats, per_machine, n, article, code)
    removeGzipTimestamps(out_dir)
    log("Checking the output")
    checkOutput(out_dir, sensitive)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=os.path.join(_REPO_ROOT, "zenodo_export"))
    parser.add_argument("--article", default=ARTICLE_CITATION,
                        help="citation of the article (default: the manuscript as submitted)")
    parser.add_argument("--code-doi", default=None,
                        help="DOI of the archived code release, e.g. 10.5281/zenodo.NNNNNNN")
    args = parser.parse_args()
    code = (f"{CODE_URL} (archived release: https://doi.org/{args.code_doi})" if args.code_doi
            else f"{CODE_URL} [TODO: archived release DOI]")

    # --out may point to any folder: only a previous export is ever replaced
    out_dir = os.path.realpath(args.out)
    if os.path.exists(out_dir):
        if not os.path.isdir(out_dir) or set(os.listdir(out_dir)) - DEPOSIT_FILES:
            raise SystemExit(f"{out_dir} is not a previous export: choose another --out")

    # Built aside, so that a failure leaves the previous export as it was
    build_dir = out_dir + ".building"
    shutil.rmtree(build_dir, ignore_errors=True)
    try:
        export(build_dir, args.article, code)
    except BaseException:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise
    shutil.rmtree(out_dir, ignore_errors=True)
    os.rename(build_dir, out_dir)
    log(f"Done: {out_dir}", color=bcolors.OKGREEN)


if __name__ == "__main__":
    main()
