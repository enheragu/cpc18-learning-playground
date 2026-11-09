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

Version 1.1 of the deposit adds the stable protocol (cpc18_utils/protocol.py)
and two sensitivity checks of the original one (patience 300, and the reference
machine forced to the AVX2 tier). Those runs were logged by the
runners of the audit of the article, one JSON object per trial with its whole
curves, and are read from `--added-runs` through the inventory `_addedRuns()`.
The rows of version 1.0 come first, from the training logs, with the same
identifiers and values; the added rows follow. The machine of an added trial is
told by the host it logged, matched with the machine-information files (the
host of the export is the reference machine), and by its CPU-capability tier and
linear-algebra setting, also logged with each trial. Trees of the stable
protocol under output_data_stable*/ are not exported.

Output (ignored by git), by default `<repo>/zenodo_export/`:

  trials.csv                         one row per training trial, all experiments and protocols
  curves_reference.jsonl.gz          per-epoch MSE curves, reference machine (rows of version 1.0)
  curves_cross_machine.jsonl.gz      per-epoch MSE curves, other machines (rows of version 1.0)
  curves_added_v1.1.jsonl.gz         per-epoch MSE x 100 curves of the rows added in version 1.1
  machines.csv                       machines and forced-tier configurations
  probes.csv                         CPU fingerprint probes
  split_variance.csv                 20 random 208/60 re-splits, fixed init seed
  split_variance_stable.csv          the same re-splits under the stable protocol
  split_variance_stable_curves.jsonl.gz  their test curves
  perproblem_errors.csv              per-test-problem errors, first 50 seeds
  perproblem_errors_stable.csv       the same under the stable protocol
  cv_epochs.csv                      fixed number of epochs per configuration (stable protocol)
  cv_curves.jsonl.gz                 validation curves of the cross-validation behind it
  cpc18_features.csv                 BEAST-derived features of the CPC18 problems
  README.md                          contents, field dictionary, licence

Usage:
    cd src && python 16_export_zenodo.py --added-runs DIR [--out PATH] [--article CITATION] [--code-doi DOI]

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
import cpc18_models
from cpc18_models.ablation import B1_CLASSES, B1X_CLASSES, B3_CLASSES, _lr_slug
from cpc18_utils import output_path, data_path
from cpc18_utils.protocol import CV_EPOCHS_CSV, NEURAL_MODELS, readoutEpoch, stablePatience
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
                 "curves_added_v1.1.jsonl.gz", "machines.csv", "probes.csv", "split_variance.csv",
                 "split_variance_stable.csv", "split_variance_stable_curves.jsonl.gz", "perproblem_errors.csv",
                 "perproblem_errors_stable.csv", "cv_epochs.csv", "cv_curves.jsonl.gz", "cpc18_features.csv"}

MASTER_SEED_FOR_POOL = 20260522    # shared seed pool (main runs, ablations)
MASTER_SEED_ORDER    = 99999777    # factorial design, data-order dimension


# Neutral machine identifiers, the `machine_id` of machines.csv in the data
# deposit, listed in the order of the machine table of the manuscript
# (Appendix D). An output folder is matched by the CPU model and OS release
# of its machine-information file, plus the forced tier for the forced runs.
# `native` is the torch CPU-capability tier of the hardware and `dispatch`
# the BLAS kernel path, both as listed in that table.
_MACHINES = [
    # id                    role                  cpu key           os key   forced     native     dispatch
    ("M00",                "reference",          "i7-1165G7",      None,    None,      "AVX512",  "AVX512"),
    ("M00-forced-AVX2",    "forced tier",        "i7-1165G7",      None,    "AVX2",    "AVX512",  ""),
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
# Configurations with trials only in the added runs: no output folder nor
# machine-information file, described from the reference machine and from the
# environment logged with each trial.
_ADDED_ONLY_MACHINES = {"M00-forced-AVX2": {"aten_cpu_capability": "avx2", "mkl_enable_instructions": "AVX2"}}

# The reference machine wrote no machine-information file; these values
# are the ones the manuscript reports for it (Appendix D), the locked
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
    # Version 1.1: the columns above keep their values in the rows of 1.0
    "protocol", "init", "readout_epoch", "mse_x100_testmin",
]
# Protocols of the rows; the added rows follow the rows of version 1.0 in this order.
PROTOCOL_ORIGINAL, PROTOCOL_STABLE, PROTOCOL_PATIENCE300 = "original", "stable", "original_patience300"
_ADDED_PROTOCOL_ORDER = [PROTOCOL_STABLE, PROTOCOL_PATIENCE300, PROTOCOL_ORIGINAL]
# Initialization drawn by each neural model by itself (the original one).
_ORIGINAL_INIT = {name: getattr(cpc18_models, name).original_init for name in NEURAL_MODELS}
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
    rows = {"M00": dict(_REFERENCE_INFO, info_source="manuscript Appendix D, locked "
                        "requirements and Python environment of the runs "
                        "(no machine-information file)")}
    infos = {}
    for d in sorted(glob.glob(os.path.join(_REPO_ROOT, "output_data_*"))):
        tag = os.path.basename(d)[len("output_data_"):]
        if tag == "stable" or tag.startswith("stable_"):
            continue  # logs of the stable protocol, see the module docstring
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
    missing = [m[0] for m in _MACHINES
               if m[0] != "C01" and m[0] not in _ADDED_ONLY_MACHINES and m[0] not in folders]
    if missing:
        raise SystemExit(f"no output folder found for {missing}")
    for mid, env in _ADDED_ONLY_MACHINES.items():
        rows[mid] = dict(_REFERENCE_INFO, **env, info_source=(
            "description of M00, and the CPU-capability tier and linear-algebra setting "
            "logged with each trial of the forced run"))
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
        # The original protocol reports the epoch of the minimum of its
        # selection curve, which is also the readout epoch.
        "protocol": PROTOCOL_ORIGINAL,
        "init": None if tree else _ORIGINAL_INIT[model],
        "readout_epoch": best,
        "mse_x100_testmin": min(test_curve) * 100.0 if test_curve else None,
    }


def _curveRecord(trial_id, e):
    rec = {"trial_id": trial_id}
    for key, name in (("mse_plot", "test_mse"), ("train_mse_plot", "train_mse"),
                      ("val_mse_plot", "val_mse")):
        if e.get(key):
            rec[name] = [float(v) for v in e[key]]
    return rec if len(rec) > 1 else None


def exportTrials(out_dir, folders, stats, added):
    """
        trials.csv and the three curve files: the rows of version 1.0 from the
        training logs, then the `added` rows of version 1.1 (loadAddedRuns).
        Returns trials per machine.
    """
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
    same_seed = {"M00": {}, "M01": {}}    # the README states that M00 and M01 agree
    curve_files = {
        "reference": gzip.open(os.path.join(out_dir, "curves_reference.jsonl.gz"), "wt"),
        "cross": gzip.open(os.path.join(out_dir, "curves_cross_machine.jsonl.gz"), "wt"),
        "added": gzip.open(os.path.join(out_dir, "curves_added_v1.1.jsonl.gz"), "wt"),
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
                if mid in ("M00", "M01"):
                    same_seed[mid][(experiment, condition, row["seed"], row["order_seed"])] = row["mse"]
                rec = _curveRecord(trial_id, e)
                if rec:
                    out = curve_files["reference" if mid == "M00" else "cross"]
                    out.write(json.dumps(rec, separators=(",", ":")) + "\n")
            per_machine[mid] = per_machine.get(mid, 0) + n
            key = (mid, PROTOCOL_ORIGINAL, experiment)
            stats["counts"][key] = stats["counts"].get(key, 0) + n
            log(f"  {mid:20s} {experiment:22s} {condition:26s} {n:5d}")
        shared = set(same_seed["M00"]) & set(same_seed["M01"])
        if len(shared) != len(same_seed["M01"]) or any(
                same_seed["M00"][k] != same_seed["M01"][k] for k in shared):
            raise SystemExit("M00 and M01 differ on a seed that both ran")
        stats["m01_shared"] = len(shared)
        for row, curves in added:
            trial_id += 1
            row["trial_id"] = trial_id
            w.writerow(row)
            curve_files["added"].write(json.dumps({"trial_id": trial_id, **curves}, separators=(",", ":")) + "\n")
            mid = row["machine_id"]
            per_machine[mid] = per_machine.get(mid, 0) + 1
            key = (mid, row["protocol"], row["experiment"])
            stats["counts"][key] = stats["counts"].get(key, 0) + 1
        log(f"  {len(added)} rows added in version 1.1")
    for fh in curve_files.values():
        fh.close()
    log(f"  -> trials.csv ({trial_id} trials)", color=bcolors.OKGREEN)
    return per_machine


# ---------------------------------------------------------------------
# Runs added in version 1.1
# ---------------------------------------------------------------------

_ABLATION_CLASSES = {cls.__name__: cls for cls in B1_CLASSES + B1X_CLASSES + B3_CLASSES}
# Hyperparameters of the main run and of the factorial design (01_train_regime2.py, 03_train_decomp.py).
_MAIN_BATCH_SIZE, _MAIN_LEARNING_RATE = 32, 0.001
# CPU-capability tier and linear-algebra setting logged with a trial -> forced tier
_FORCED_TIERS = {("AVX2", "AVX2"): "AVX2", ("DEFAULT", "SSE4_2"): "DEFAULT"}


def _addedRuns():
    """
        Inventory of the runs added in version 1.1, files under --added-runs as
        the runners wrote them: (file, experiment, condition, protocol). Under the
        stable protocol, LinearLogistic comes from its runs with patience 120
        (`_p120`), and the learning-rate 1e-4 column of the BourginMLP grid from
        `abl_portatil/`, where it was completed to 800 seeds.
    """
    runs = []
    for suffix in ("", "_avx2", "_DEFAULT"):   # the native tier, then the two forced ones
        folder = "" if not suffix else "datos_4090/"
        for m in ("BourginMLP", "ContextDepNet", "SparseMLP"):
            runs.append((f"{folder}main_{m}_default{suffix}.jsonl", "main", m, PROTOCOL_STABLE))
        runs.append((f"{folder}main_LinearLogistic_default{suffix}_p120.jsonl", "main", "LinearLogistic",
                     PROTOCOL_STABLE))
    for fname, m in (("fact_LinearLogistic_default_p120.jsonl", "LinearLogistic"),
                     ("datos_4090/fact_BourginMLP_default.jsonl", "BourginMLP"),
                     ("fact_ContextDepNet_default.jsonl", "ContextDepNet"),
                     ("datos_4090/fact_SparseMLP_default.jsonl", "SparseMLP"),
                     ("datos_4090/fact_BourginMLP_default_avx2.jsonl", "BourginMLP")):
        runs.append((fname, "factorial_init_order", m, PROTOCOL_STABLE))
    for lr, tag in ((1e-4, "0.0001"), (3.16e-4, "3.16e-4"), (1e-3, "0.001"), (1e-2, "0.01")):
        folder = "abl_portatil" if lr == 1e-4 else "datos_4090"
        for b in (16, 32, 64):
            runs.append((f"{folder}/abl_B{b}_lr{tag}_default.jsonl", "ablation_bourginmlp",
                         f"BourginMLP_B{b}_L{_lr_slug(lr)}", PROTOCOL_STABLE))
    for e in (10, 20, 40):
        for lr, tag in ((1e-4, "0.0001"), (1e-3, "0.001"), (1e-2, "0.01")):
            runs.append((f"ablS_E{e}_lr{tag}_default.jsonl", "ablation_sparsemlp",
                         f"SparseMLP_E{e}_L{_lr_slug(lr)}", PROTOCOL_STABLE))
    runs += [("datos_4090/main_BourginMLP_p300.jsonl", "main", "BourginMLP", PROTOCOL_PATIENCE300),
             ("fact_BourginMLP_p300.jsonl", "factorial_init_order", "BourginMLP", PROTOCOL_PATIENCE300)]
    for m in NEURAL_MODELS:
        runs.append((f"fact_{m}_avx2.jsonl", "factorial_init_order", m, PROTOCOL_ORIGINAL))
    return runs


def _conditionSettings(condition):
    """ (model, batch size, learning rate, epsilon of the SparseMLP cells) of a condition. """
    cls = _ABLATION_CLASSES.get(condition)
    if cls is None:
        return condition, _MAIN_BATCH_SIZE, _MAIN_LEARNING_RATE, None
    model = _CELL_RE.match(condition).group(1)
    return model, cls.batch_size, cls.default_lr, cls.epsilon if model == "SparseMLP" else None


def _expectedSettings(experiment, protocol, model):
    """ (patience, epoch cap) of the trials of an added run. """
    if protocol == PROTOCOL_PATIENCE300:
        return 300, 1500
    patience = stablePatience(model) if protocol == PROTOCOL_STABLE else 50
    return patience, 1500 if experiment.startswith("ablation") else 500


def _poolPositions():
    """ {seed: position} of the shared pool and of the order pool of the factorial design. """
    pool = np.random.default_rng(MASTER_SEED_FOR_POOL).integers(0, 2**32, size=800, dtype=np.uint64)
    order = np.random.default_rng(MASTER_SEED_ORDER).integers(0, 2**32, size=35, dtype=np.uint64)
    return {int(v): i for i, v in enumerate(pool)}, {int(v): i for i, v in enumerate(order)}


def _addedMachine(rec, host2id):
    """ Machine of an added trial, from its host and its logged tier and linear-algebra setting. """
    base = host2id.get(rec["host"])
    if base is None:
        raise SystemExit("an added trial was logged on a host without a machine-information file")
    native = _MACHINES[_MACHINE_ORDER.index(base)][5]
    if rec["cpu_cap"] == native and rec["mkl"] == "":
        return base
    forced = _FORCED_TIERS.get((rec["cpu_cap"], rec["mkl"]))
    mid = f"{base}-forced-{forced}" if forced else None
    if mid not in _MACHINE_ORDER:
        raise SystemExit(f"{base}: added trial with tier {rec['cpu_cap']} and linear-algebra "
                         f"setting {rec['mkl']!r}, a configuration without machine id")
    return mid


def _checkAddedRecord(r, where, design, model, bs, lr, eps, init, patience, cap):
    """ The record must be the trial its file stands for. """
    bad = [k for k, ok in (
        ("model", r["model"] == model), ("design", r["design"] == design), ("init", r["init"] == init),
        ("patience", r["patience"] == patience), ("cap", r["cap"] == cap),
        # The first runs logged neither the batch size nor the learning rate:
        # they ran at the defaults of their runner, 32 and 1e-3.
        ("batch_size", r.get("batch_size") in (None, bs)),
        ("lr", r.get("lr") is None or abs(r["lr"] - lr) <= 1e-12 * lr),
        ("epsilon", r.get("epsilon") is None or r["epsilon"] == eps),
        ("order_seed", (r["order_seed"] is None) == (design == "main")),
        ("curves", len(r["test_curve"]) == len(r["train_curve"]) == r["n_epochs"] <= cap),
        # `best` is the minimum of the curve with full precision, the curve is rounded
        ("best", r["test_curve"][r["best_epoch"]] == round(r["best"], 7) == min(r["test_curve"])),
        # early stopping: the run ends `patience` epochs after its best epoch, or at the cap
        ("stop", r["n_epochs"] == cap or r["n_epochs"] - 1 - r["best_epoch"] == patience),
    ) if not ok]
    if bad:
        raise SystemExit(f"{where}: record of seed {r['seed']} does not match its run ({bad})")


def loadAddedRuns(runs_dir, host2id, sensitive):
    """
        Rows of trials.csv and curves of the runs added in version 1.1, in their
        order: by protocol, machine, experiment, condition and position of the
        seeds in their pools. The host names only link a trial to its machine
        and are added to `sensitive`.
    """
    host2id = dict(host2id)
    host2id.setdefault(socket.gethostname(), "M00")   # the export runs on the reference machine
    seed_pos, order_pos = _poolPositions()
    rows = []
    for fname, experiment, condition, protocol in _addedRuns():
        model, bs, lr, eps = _conditionSettings(condition)
        patience, cap = _expectedSettings(experiment, protocol, model)
        factorial = experiment == "factorial_init_order"
        stable = protocol == PROTOCOL_STABLE
        if stable:
            readout = readoutEpoch(model, bs, lr, getattr(getattr(cpc18_models, model), "epsilon", None)
                                   if eps is None else eps)
        with open(os.path.join(runs_dir, fname)) as f:
            records = [json.loads(line) for line in f]
        if len(records) != (1225 if factorial else 800):
            raise SystemExit(f"{fname}: {len(records)} trials")
        keys = set()
        for r in records:
            sensitive.add(r["host"])
            _checkAddedRecord(r, fname, "factorial" if factorial else "main", model, bs, lr, eps,
                              "default" if stable else "kaiming", patience, cap)
            key = (r["seed"], r["order_seed"])
            if key in keys:
                raise SystemExit(f"{fname}: seed {key} twice")
            keys.add(key)
            pos = (seed_pos[r["seed"]], order_pos[r["order_seed"]]) if factorial else (seed_pos[r["seed"]], 0)
            if factorial and pos[0] >= 35:
                raise SystemExit(f"{fname}: initialization seed out of the 35 of the design")
            curve = r["test_curve"]
            if stable and len(curve) <= readout:
                raise SystemExit(f"{fname}: seed {key} stops before its readout epoch {readout}")
            mid = _addedMachine(r, host2id)
            row = {
                "experiment": experiment, "machine_id": mid, "condition": condition, "model": model,
                "batch_size": bs, "learning_rate": lr,
                "epsilon": eps if experiment == "ablation_sparsemlp" else None,
                "seed": r["seed"], "init_seed": r["seed"] if factorial else None,
                "order_seed": r["order_seed"],
                # Logged as MSE x 100 only: `mse` is left empty.
                "mse": None, "mse_x100": curve[readout] if stable else r["best"],
                "mse_random_baseline": None,
                "best_epoch": r["best_epoch"], "n_epochs": r["n_epochs"], "patience": r["patience"],
                "train_duration_s": r["seconds"],
                "protocol": protocol, "init": "pytorch_default" if stable else _ORIGINAL_INIT[model],
                "readout_epoch": readout if stable else r["best_epoch"],
                "mse_x100_testmin": r["best"],
            }
            order = (_ADDED_PROTOCOL_ORDER.index(protocol), _MACHINE_ORDER.index(mid),
                     _EXPERIMENT_ORDER.index(experiment), _conditionKey(condition), pos)
            rows.append((order, row, {"test_mse_x100": curve, "train_mse_x100": r["train_curve"]}))
        log(f"  {fname:52s} {experiment:22s} {condition:26s} {protocol:21s} {len(records):5d}")
    rows.sort(key=lambda t: t[0])
    return [(row, curves) for _, row, curves in rows]


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
    """ The two auxiliary re-runs of Section 3.4.2, from their npz outputs. """
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


def _mainReadout(model):
    return readoutEpoch(model, _MAIN_BATCH_SIZE, _MAIN_LEARNING_RATE,
                        getattr(getattr(cpc18_models, model), "epsilon", None))


def exportStableReruns(out_dir, runs_dir):
    """
        The two re-runs of Section 3.4.2 and the cross-validation under the
        stable protocol, from the files of the added runs. Returns their numbers
        of records.
    """
    n = {}
    # Random re-splits: the four neural models, read at their fixed epoch.
    # Each record logs its patience (the stable one: 120 for LinearLogistic); the first ones, without it, used 50.
    with open(os.path.join(runs_dir, "split_default.jsonl")) as f:
        recs = [json.loads(line) for line in f]
    by_key = {(r["model"], r["split"]): r for r in recs}
    if len(by_key) != len(recs) or set(by_key) != {(m, i) for m in NEURAL_MODELS for i in range(20)}:
        raise SystemExit("split_default.jsonl: not one record per neural model and split")
    with open(os.path.join(out_dir, "split_variance_stable.csv"), "w", newline="") as f, \
            gzip.open(os.path.join(out_dir, "split_variance_stable_curves.jsonl.gz"), "wt") as fc:
        w = csv.writer(f)
        w.writerow(["model", "split_index", "readout_epoch", "mse_x100", "mse_x100_testmin",
                    "best_epoch", "n_epochs", "patience"])
        for m in NEURAL_MODELS:
            readout = _mainReadout(m)
            for i in range(20):
                c = by_key[(m, i)]["test_curve"]
                if len(c) <= readout or by_key[(m, i)]["best"] != min(c):
                    raise SystemExit(f"split_default.jsonl: {m} split {i} inconsistent")
                w.writerow([m, i, readout, c[readout], min(c), int(np.argmin(c)), len(c),
                            by_key[(m, i)].get("patience", 50)])
                fc.write(json.dumps({"model": m, "split_index": i, "test_mse_x100": c},
                                    separators=(",", ":")) + "\n")
    n["split_stable"] = len(recs)

    # Per-problem errors: the four neural models trained for their fixed epochs, first 50 seeds.
    seeds = [int(v) for v in np.random.default_rng(MASTER_SEED_FOR_POOL).integers(
        0, 2**32, size=50, dtype=np.uint64)]
    with open(os.path.join(data_path, "cpc18_features.csv"), newline="") as f:
        problems = sorted({int(r["problem_id"]) for r in csv.DictReader(f) if r["Train_Test"] == "Test"})
    with open(os.path.join(runs_dir, "perproblem_default.jsonl")) as f:
        recs = [json.loads(line) for line in f]
    by_key = {(r["model"], r["seed"]): r for r in recs}
    if len(by_key) != len(recs) or set(by_key) != {(m, s) for m in NEURAL_MODELS for s in seeds}:
        raise SystemExit("perproblem_default.jsonl: not one record per neural model and seed")
    with open(os.path.join(out_dir, "perproblem_errors_stable.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "seed", "problem_id", "mean_sq_error_x100"])
        for m in NEURAL_MODELS:
            for sd in seeds:
                r = by_key[(m, sd)]
                if r["epochs"] != _mainReadout(m) + 1 or r["problem_id"] != problems:
                    raise SystemExit(f"perproblem_default.jsonl: {m} seed {sd} inconsistent")
                for pid, v in zip(r["problem_id"], r["sq_error_x100"]):
                    w.writerow([m, sd, pid, v])
    n["perproblem_stable"] = len(recs) * len(problems)

    # Cross-validation: its validation curves and the epochs chosen from them.
    seed_pos, _ = _poolPositions()
    with open(os.path.join(runs_dir, "cv.jsonl")) as f:
        recs = [json.loads(line) for line in f]
    recs.sort(key=lambda r: (r["config"], r["fold"], seed_pos[r["seed"]]))
    if len({(r["config"], r["fold"], r["seed"]) for r in recs}) != len(recs):
        raise SystemExit("cv.jsonl: a run appears twice")
    with gzip.open(os.path.join(out_dir, "cv_curves.jsonl.gz"), "wt") as fc:
        for r in recs:
            fc.write(json.dumps({"config": r["config"], "model": r["model"], "batch_size": r["batch_size"],
                                 "learning_rate": r["lr"], "epsilon": r["epsilon"], "fold": r["fold"],
                                 "seed": r["seed"], "n_train_rows": r["n_train_rows"],
                                 "val_mse_x100": r["val_curve"]}, separators=(",", ":")) + "\n")
    with open(CV_EPOCHS_CSV, newline="") as f:
        rows = list(csv.reader(f))
    runs = {}
    for r in recs:
        runs[r["config"]] = runs.get(r["config"], 0) + 1
    if {(row[0], int(row[1])) for row in rows[1:]} != set(runs.items()):
        raise SystemExit(f"{CV_EPOCHS_CSV} does not summarize the curves of cv.jsonl")
    with open(os.path.join(out_dir, "cv_epochs.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)
    n["cv_curves"], n["cv_configs"] = len(recs), len(rows) - 1
    log("  -> split_variance_stable.csv, perproblem_errors_stable.csv, cv_epochs.csv and their curves",
        color=bcolors.OKGREEN)
    return n


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
configurations, the CPU fingerprint probes, two auxiliary re-runs, the
cross-validation of the stable protocol and the input features. The code that
trains the models and recomputes the tables and figures of the article is
archived separately: {code}.

Version 1.1 adds the stable training protocol (see Protocols) and two
sensitivity checks of the original one: a patience of 300 epochs, and the
factorial design on the reference machine forced to the AVX2 tier. The rows of
version 1.0 keep their identifiers and values, with four new columns; the added
rows follow them.

## Files

| File | Content | Records |
|---|---|---|
| `trials.csv` | One row per training trial, all experiments, protocols and machines | {n_trials} |
| `curves_reference.jsonl.gz` | Per-epoch MSE curves of the trials of version 1.0 on the reference machine `M00` | {n_curves_ref} |
| `curves_cross_machine.jsonl.gz` | Per-epoch MSE curves of the trials of version 1.0 on the other machines | {n_curves_cross} |
| `curves_added_v1.1.jsonl.gz` | Per-epoch MSE x 100 curves of the trials added in version 1.1 | {n_curves_added} |
| `machines.csv` | Machines and forced-tier configurations | {n_machines} |
| `probes.csv` | CPU fingerprint probes | {n_probes} |
| `split_variance.csv` | Test MSE of LinearLogistic, PsychForest and BEAST-GB over 20 random 208/60 re-splits of the problems, initialization seed fixed | {n_split} |
| `split_variance_stable.csv` | The same re-splits for the four neural models under the stable protocol | {n_split_stable} |
| `split_variance_stable_curves.jsonl.gz` | Their per-epoch test curves | {n_split_stable} |
| `perproblem_errors.csv` | Squared error per test problem (mean over its 5 blocks) of BourginMLP, PsychForest and BEAST-GB for the first 50 shared seeds | {n_perproblem} |
| `perproblem_errors_stable.csv` | The same for the four neural models under the stable protocol | {n_perproblem_stable} |
| `cv_epochs.csv` | Number of epochs of each configuration under the stable protocol, chosen by cross-validation | {n_cv_configs} |
| `cv_curves.jsonl.gz` | Validation curves of that cross-validation | {n_cv_curves} |
| `cpc18_features.csv` | BEAST-derived features and observed choice rates of the CPC18 problems | {n_features} |

## Protocols

The `protocol` column of `trials.csv` tells how a trial was trained and read.
The article calls `stable` its stable configuration and `original` its fragile
configuration:

| `protocol` | Initialization of the networks (`init`) | Reported test MSE (`mse_x100`) | Early stopping on the test curve |
|---|---|---|---|
| `original` | Kaiming normal in fan-out mode with zero biases (`kaiming_fanout`); LinearLogistic, uniform in [-0.1, 0.1] with zero bias (`uniform`) | At the epoch of the minimum of the test curve | Patience 50 |
| `stable` | The default of PyTorch for linear layers (`pytorch_default`), drawn from the training seed without changing the data order of that seed | At a fixed epoch per configuration, chosen by 5-fold cross-validation on the 208 training problems (`cv_epochs.csv`), never at the minimum of the test curve | Patience 50, and 120 for LinearLogistic so that every curve reaches its epoch 99; it only decides when to stop |
| `original_patience300` | As `original` | As `original` | Patience 300, with a cap of 1500 epochs: sensitivity of the original protocol to the patience |

Everything else is the same under every protocol: architectures, Adam,
batch size and learning rate (32 and 1e-3 out of the ablation grids),
features, canonical split, seed pools, factorial design and ablation grids.
PsychForest and BEAST-GB have no initialization nor epochs: they are trained
the same way under every protocol, so their rows (protocol `original`) also
complete the comparisons of the stable protocol. Every row of a neural model
also holds the minimum of its test curve (`mse_x100_testmin`), so that a
stable trial can also be read the way of the original protocol.

The stable trials of a condition were run on `M00` or on `M01`, and the
condition is complete only with the rows of both: `M00` holds the main run,
the SparseMLP ablation, the factorial design of LinearLogistic and
ContextDepNet and 600 of the 800 seeds of the BourginMLP cells at learning
rate 1e-4; `M01` holds the rest. The two machines give bit-identical results
for the same seed: each of the {n_m01_shared} trials of version 1.0 that both ran
has the same `mse` on both.

## Experiments

The `experiment` column of `trials.csv` takes these values. Under the
original protocol, the rows of machines other than `M00` re-run a subset of the
`main` and `ablation_*` conditions on other hardware (Section 3.5 and Appendix
D), and `M00-forced-AVX2` the factorial design. Under the stable protocol, the
rows of the forced tiers of `M01` re-run the main run and the factorial design
of BourginMLP.

| `experiment` | Design | Article |
|---|---|---|
| `main` | Six models on the canonical CPC18 split (208 of its 210 training problems and its 60 test problems), 800 shared seeds each | Sections 1, 3.1, 3.2, 3.4.2, 3.5, 3.6, 4, Appendices A, C, D |
| `ablation_bourginmlp` | BourginMLP, batch size {{16, 32, 64}} x learning rate {{1e-4, 1e-3, 1e-2}}, 800 seeds per cell; the cells at learning rate 3.16e-4 extend the grid to check that the decomposition does not depend on its levels | Sections 3.3, 3.5, 3.6, Appendices C, D |
| `ablation_sparsemlp` | SparseMLP, target sparsity epsilon {{10, 20, 40}} x learning rate {{1e-4, 1e-3, 1e-2}}, 800 seeds per cell | Sections 3.3, 3.5, 3.6, Appendices C, D |
| `factorial_init_order` | 35 initialization seeds x 35 data-order seeds, four models | Sections 3.4.1, 3.5, 3.6, Appendix B |
| `validation_split` | Early stopping on an internal validation split (166/42 of the training problems), 200 seeds, BourginMLP and ContextDepNet | Not reported in the current version of the article |

Trials per machine, protocol and experiment:

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
| `batch_size`, `learning_rate`, `epsilon` | Training hyperparameters of the trial. `epsilon` (target sparsity) is given only in the SparseMLP ablation; elsewhere SparseMLP uses its default of 20. Empty for the tree-based models (PsychForest, BEAST-GB), which use none of the three; their own settings are in the code |
| `seed` | Training seed. In `main`, the ablations and `validation_split` it is drawn from the shared seed pool (generated from master seed {pool_seed}) and drives both the weight initialization and the data-shuffle order; equal seeds therefore pair trials across models, cells, protocols and machines. In `factorial_init_order` it equals `init_seed` |
| `init_seed`, `order_seed` | `factorial_init_order` only: seed of the weight initialization and of the data-shuffle order (order seeds generated from master seed {order_seed}) |
| `mse` | Test MSE of the selected model, recomputed after training on the 60 test problems. Empty in the rows added in version 1.1, which were logged as MSE x 100 only |
| `mse_x100` | Test MSE x 100, the unit used in the article, as reported under the protocol of the row: `mse` x 100 in the rows of version 1.0, and the logged test curve at `readout_epoch` in the added rows |
| `mse_random_baseline` | Variance of the test targets, i.e. the test MSE of a constant prediction equal to their mean (field name as in the code). Not logged in the added rows |
| `best_epoch` | 0-based epoch of the minimum of the test curve, or of the validation curve in `validation_split`; under the original protocols, the epoch whose weights were selected. Empty for the tree-based models |
| `n_epochs` | Number of training epochs run (length of the curves). Training stops `patience` epochs after the best one, or at the epoch cap: 500 epochs, 1500 in the ablations and in the `original_patience300` runs |
| `patience` | Early-stopping patience in epochs |
| `train_duration_s` | Wall-clock training time in seconds; depends on the machine and its load. Rounded to 0.1 s in the added rows |
| `val_split_seed`, `n_train_problems`, `n_val_problems` | `validation_split` only: seed and size of the internal split |
| `best_test_epoch`, `mse_x100_testsel` | `validation_split` only: epoch of the test-curve minimum and the corresponding test MSE x 100, i.e. the main protocol's selection applied to the same trajectory |
| `val_mse_at_selection` | `validation_split` only: validation MSE at `best_epoch` |
| `protocol` | `original`, `stable` or `original_patience300`, see Protocols |
| `init` | Initialization of the weights: `kaiming_fanout`, `uniform` or `pytorch_default`, see Protocols. Empty for the tree-based models |
| `readout_epoch` | 0-based epoch whose test MSE is `mse_x100`: `best_epoch` under the original protocols, and under the stable one the fixed epoch of the configuration, `e_full` of `cv_epochs.csv` minus one. Empty for the tree-based models |
| `mse_x100_testmin` | Minimum of the test curve x 100, the readout of the original protocols, for every row of a neural model. Empty for the tree-based models |

The rows of version 1.0 are written with full double precision, so equality
between machines can be checked directly on `mse`. The added rows were logged
as MSE x 100: their curves rounded to 7 decimals, and the minimum of the test
curve with full precision (`mse_x100_testmin`, also `mse_x100` under the
original protocols). Under the stable protocol, `mse_x100` is an element of the
logged curve, with 7 decimals.

## Curve files

One JSON object per line, for every trial of a neural model (the
tree-based models have no epochs). In `curves_reference.jsonl.gz` and
`curves_cross_machine.jsonl.gz`: `trial_id`, `test_mse` (test MSE after
each epoch) and `train_mse` (training MSE after each epoch), or `val_mse`
instead of `train_mse` in `validation_split`. The values are MSE, not MSE x 100. The
minimum of `test_mse` and `mse` in `trials.csv` agree to a relative
difference of at most {max_rel:.1e}; `mse` is computed again after
restoring the selected weights, and `mse_x100_testmin` is the minimum of
`test_mse` x 100. In `curves_added_v1.1.jsonl.gz`, for the rows added in
version 1.1: `trial_id`, `test_mse_x100` and `train_mse_x100`, MSE x 100 after
each epoch rounded to 7 decimals, as logged. Under the stable protocol
`mse_x100` is the element `readout_epoch` (0-based) of `test_mse_x100`; in
every added row, `mse_x100_testmin` rounded to 7 decimals is its minimum.

## Fields of `machines.csv`

The reference machine, the comparison machines, the five runs whose torch
CPU-capability tier was forced below the native one, and the two groups of
public cloud sessions. `M00-forced-AVX2` is the reference machine forced to the
AVX2 tier, which ran only rows of version 1.1 and wrote no machine-information
file: its row repeats the description of `M00` with the two forcing variables,
and the tier logged with each of its trials confirms them. An empty cell means
that the value was not recorded or does not apply.

| Field | Meaning |
|---|---|
| `machine_id` | Machine or configuration, as used in `trials.csv` and `probes.csv` |
| `role` | `reference`, `comparison`, `forced tier`, `cloud, probes only` or `cloud, full sweep` |
| `cpu_model`, `architecture`, `logical_cpus` | Hardware |
| `avx512_flags` | AVX-512 flags of the CPU |
| `native_tier` | Torch CPU-capability tier selected on the hardware |
| `run_tier` | Torch CPU-capability tier used in the run |
| `aten_cpu_capability`, `mkl_enable_instructions` | The two environment variables that force the tier |
| `blas_dispatch` | Kernel path of the linear-algebra library, as reported in Appendix D |
| `os`, `kernel`, `python`, `torch` | Software of the run |
| `thread_pinning` | Thread settings of the run |
| `n_trials` | Rows of `trials.csv` from this machine |
| `info_source` | Where the description of the machine comes from |

## Fields of `probes.csv`

The CPU fingerprint probes of Appendix D, one row per session. Each probe
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

Under the stable protocol, on the reference machine: `split_variance_stable.csv`
has the same 20 re-splits (split seeds 1000 to 1019, initialization seed held
fixed) for the four neural models, one row per model and re-split: `model`,
`split_index`, `readout_epoch`, `mse_x100` (at the readout epoch, as in
`trials.csv`), `mse_x100_testmin`, `best_epoch`, `n_epochs` and `patience` (that
of the stable protocol; every curve reaches its readout epoch).
`split_variance_stable_curves.jsonl.gz` holds their test curves, one JSON
object per row: `model`, `split_index` and `test_mse_x100` (MSE x 100 after each
epoch, rounded to 7 decimals). `perproblem_errors_stable.csv` has the columns of
`perproblem_errors.csv` for the four neural models, each trained for its fixed
number of epochs without early stopping and evaluated with the weights of its
last epoch; the values are rounded to 7 decimals.

## Cross-validation of the stable protocol

`cv_epochs.csv` has one row per configuration: `config` (model, batch size `B`,
learning rate `lr` and, for SparseMLP, epsilon `E`), `runs` (5 folds x 10
seeds), `e_cv` (1-based epoch of the minimum of the validation curve averaged
over the runs), `val_min` (that minimum, MSE x 100, 4 decimals) and `e_full`,
the number of epochs on the 208 training problems: `e_cv` x the mean number of
training rows of the folds / 1040, rounded, at least 1. The stable protocol
reports the test MSE after `e_full` epochs. `cv_curves.jsonl.gz` holds the
validation curves behind it, one JSON object per configuration, fold and seed:
`config`, `model`, `batch_size`, `learning_rate`, `epsilon`, `fold` (0 to 4),
`seed` (the first 10 of the shared pool), `n_train_rows` and `val_mse_x100` (MSE
x 100 after each epoch, rounded to 7 decimals). The folds split the training
problems, from a permutation drawn with seed 2026. The runs use the default
initialization of PyTorch and no early stopping: 300 epochs for learning rates
below 1e-3 and 200 otherwise. The test problems are never used.

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
and `scripts/variance_check/cpu_probe.py`. The same trainers and re-runs with
`--protocol stable` train under the stable protocol, and `19_cv_epochs.py` runs
its cross-validation. The seed pools are regenerated from the master seeds
above, identically on every platform.

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
    lines = ["| Machine | Protocol | " + " | ".join(f"`{e}`" for e in exps) + " |",
             "|---|---|" + "---|" * len(exps)]
    for mid in _MACHINE_ORDER:
        for protocol in (PROTOCOL_ORIGINAL, PROTOCOL_STABLE, PROTOCOL_PATIENCE300):
            cells = [str(stats["counts"].get((mid, protocol, e), "")) for e in exps]
            if any(cells):
                lines.append(f"| `{mid}` | `{protocol}` | " + " | ".join(cells) + " |")
    text = _README.format(
        article=article, code=code,
        n_trials=sum(per_machine.values()),
        n_curves_ref=n["curves_reference"], n_curves_cross=n["curves_cross"],
        n_curves_added=n["curves_added"],
        n_machines=len(_MACHINES), n_probes=n["probes"], n_split=n["split"],
        n_split_stable=n["split_stable"], n_perproblem=n["perproblem"],
        n_perproblem_stable=n["perproblem_stable"], n_cv_configs=n["cv_configs"],
        n_cv_curves=n["cv_curves"], n_features=n["features"],
        n_m01_shared=f"{stats['m01_shared']:,}",
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
    # Curve files: their curves (lists of floats) and their identifying fields only.
    curve_files = {
        "curves_reference.jsonl.gz": ({"trial_id": int}, {"test_mse", "train_mse", "val_mse"}),
        "curves_cross_machine.jsonl.gz": ({"trial_id": int}, {"test_mse", "train_mse", "val_mse"}),
        "curves_added_v1.1.jsonl.gz": ({"trial_id": int}, {"test_mse_x100", "train_mse_x100"}),
        "split_variance_stable_curves.jsonl.gz": ({"model": str, "split_index": int}, {"test_mse_x100"}),
        "cv_curves.jsonl.gz": ({"config": str, "model": str, "batch_size": int, "learning_rate": float,
                                "epsilon": (int, type(None)), "fold": int, "seed": int, "n_train_rows": int},
                               {"val_mse_x100"}),
    }
    identifiers = re.compile(r"^[A-Za-z0-9_.]+$")
    for name, (fields, curves) in curve_files.items():
        with gzip.open(os.path.join(out_dir, name), "rt") as f:
            for line in f:
                rec = json.loads(line)
                if set(rec) - set(fields) - curves or set(fields) - set(rec) or \
                        any(not isinstance(rec[k], t) or (t is str and not identifiers.match(rec[k]))
                            for k, t in fields.items()) or \
                        any(not all(isinstance(v, float) for v in rec[k]) for k in curves if k in rec):
                    raise SystemExit(f"{name}: unexpected content in a record")

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


def export(out_dir, article, code, runs_dir):
    os.makedirs(out_dir)
    sensitive = set()
    stats = {"counts": {}, "best_epoch_recomputed": 0, "max_rel_mse_vs_curve": 0.0}
    log("Identifying machines")
    folders, rows, host2id = identifyMachines(sensitive)
    log("Reading the runs added in version 1.1")
    added = loadAddedRuns(runs_dir, host2id, sensitive)
    log("Exporting trials and curves")
    per_machine = exportTrials(out_dir, folders, stats, added)
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
    n = exportStableReruns(out_dir, runs_dir)
    n_features = exportFeatures(out_dir)
    n.update({
        "curves_reference": _countLines(os.path.join(out_dir, "curves_reference.jsonl.gz")),
        "curves_cross": _countLines(os.path.join(out_dir, "curves_cross_machine.jsonl.gz")),
        "curves_added": _countLines(os.path.join(out_dir, "curves_added_v1.1.jsonl.gz")),
        "probes": len(probes),
        "split": _countLines(os.path.join(out_dir, "split_variance.csv")) - 1,
        "perproblem": _countLines(os.path.join(out_dir, "perproblem_errors.csv")) - 1,
        "features": n_features})
    writeReadme(out_dir, stats, per_machine, n, article, code)
    removeGzipTimestamps(out_dir)
    log("Checking the output")
    checkOutput(out_dir, sensitive)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--added-runs", required=True, metavar="DIR",
                        help="folder of the runs added in version 1.1, as their runners logged them")
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
        export(build_dir, args.article, code, os.path.realpath(args.added_runs))
    except BaseException:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise
    shutil.rmtree(out_dir, ignore_errors=True)
    os.rename(build_dir, out_dir)
    log(f"Done: {out_dir}", color=bcolors.OKGREEN)


if __name__ == "__main__":
    main()
