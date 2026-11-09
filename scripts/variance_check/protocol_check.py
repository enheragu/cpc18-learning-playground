#!/usr/bin/env python3
# encoding: utf-8

"""
Checks of the two training protocols (src/cpc18_utils/protocol.py) and of
version 1.1 of the data deposit.

  train    Retrains a sample of trials with the trainers of the repository and
           compares them with the results they must reproduce. Under the stable
           protocol, the runs added in version 1.1 (--runs, one JSON line per
           trial, as their runners logged them): the test MSE at the fixed epoch,
           the common part of the test and training curves and, with the same
           patience, the minimum of the test curve and the number of epochs.
           Under the original protocol, trials of the deposit of version 1.0
           (--base), bit for bit, with their curves.
  export   Checks an export of version 1.1 (--deposit) against version 1.0
           (--base) and the runs (--runs): the rows of 1.0 unchanged, the rows
           per experiment, condition, protocol and machine, no seed twice in a
           condition, and every added value equal to its logged run. The runs
           are matched from their own fields, not from the inventory of the
           export script.

Run from the root of the repository, on the reference machine (its host name
identifies the trials it logged; those of the other machines are identified
from their machine-information files):

    python scripts/variance_check/protocol_check.py train  --runs DIR [--workers 3]
    python scripts/variance_check/protocol_check.py export --runs DIR [--deposit zenodo_export_v1.1]

Both end with an error when a check fails. Training runs on the CPU, one thread
per worker; the sample takes about a minute with 3 workers.
"""

import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import csv
import glob
import gzip
import hashlib
import importlib.util
import json
import multiprocessing as mp
import re
import shutil
import socket
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

_REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_SRC = os.path.join(_REPO, "src")
sys.path.insert(0, _SRC)
os.chdir(_SRC)
import path_setup  # noqa: F401,E402

import numpy as np  # noqa: E402

from cpc18_models.ablation import _lr_slug  # noqa: E402
from cpc18_utils.protocol import NEURAL_MODELS, readoutEpoch  # noqa: E402


MASTER_SEED_FOR_POOL = 20260522
MASTER_SEED_ORDER = 99999777
POOL = [int(v) for v in np.random.default_rng(MASTER_SEED_FOR_POOL).integers(0, 2**32, size=800, dtype=np.uint64)]
ORDER_POOL = [int(v) for v in np.random.default_rng(MASTER_SEED_ORDER).integers(0, 2**32, size=35, dtype=np.uint64)]
N_V10_COLUMNS = 24                      # columns of trials.csv in version 1.0
NEW_COLUMNS = ["protocol", "init", "readout_epoch", "mse_x100_testmin"]
TREES = {"PsychForest", "BEAST-GB"}


def _load(filename, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_SRC, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _readJsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def _r7(x):
    """ A value of a curve of the trainers (MSE) as the runners logged it (MSE x 100, 7 decimals). """
    return round(float(x) * 100, 7)


# ---------------------------------------------------------------------
# train: the sample
# ---------------------------------------------------------------------

# (group, file under --runs, pool position of the seed, or (init, order) positions in the factorial)
STABLE_SAMPLE = [
    ("main", "main_BourginMLP_default.jsonl", 0),
    ("main", "main_BourginMLP_default.jsonl", 517),
    ("main", "main_ContextDepNet_default.jsonl", 250),
    ("main", "main_SparseMLP_default.jsonl", 799),
    ("factorial", "fact_ContextDepNet_default.jsonl", (3, 30)),
    ("factorial", "datos_4090/fact_BourginMLP_default.jsonl", (17, 5)),
    ("factorial", "datos_4090/fact_SparseMLP_default.jsonl", (34, 12)),
    ("ablation BourginMLP", "datos_4090/abl_B16_lr0.01_default.jsonl", 100),
    ("ablation BourginMLP", "datos_4090/abl_B64_lr3.16e-4_default.jsonl", 640),
    ("ablation BourginMLP", "abl_portatil/abl_B32_lr0.0001_default.jsonl", 10),
    ("ablation BourginMLP", "abl_portatil/abl_B32_lr0.0001_default.jsonl", 700),
    ("ablation SparseMLP", "ablS_E10_lr0.0001_default.jsonl", 5),
    ("ablation SparseMLP", "ablS_E20_lr0.01_default.jsonl", 300),
    ("ablation SparseMLP", "ablS_E40_lr0.001_default.jsonl", 600),
    ("LinearLogistic", "main_LinearLogistic_default_p120.jsonl", 0),
    ("LinearLogistic", "main_LinearLogistic_default_p120.jsonl", 400),
    ("LinearLogistic", "fact_LinearLogistic_default_p120.jsonl", (8, 21)),
]
# per-problem errors: (model, pool position); split variance: (model, split index)
PERPROBLEM_SAMPLE = [("ContextDepNet", 0), ("SparseMLP", 49)]
SPLIT_SAMPLE = [("BourginMLP", 3), ("LinearLogistic", 11)]
# Trials of the deposit of version 1.0 (machine M00): (experiment, condition, pool position or (init, order))
ORIGINAL_SAMPLE = [("main", "BourginMLP", 0), ("main", "LinearLogistic", 1),
                   ("factorial_init_order", "SparseMLP", (2, 3)), ("ablation_sparsemlp", "SparseMLP_E10_L0_01", 0)]


def _cellOf(rec):
    """ Ablation class of a logged run of the grids (its cap is 1500 and its patience 50). """
    if rec["model"] == "BourginMLP":
        return f"BourginMLP_B{rec['batch_size']}_L{_lr_slug(rec['lr'])}"
    return f"SparseMLP_E{int(rec['epsilon'])}_L{_lr_slug(rec['lr'])}"


def _trainTask(task):
    """ Runs one trial with the code of the repository; returns what is compared. """
    kind, args = task[0], task[1:]
    tmp = tempfile.mkdtemp(prefix="protocol_check_")
    try:
        if kind == "perproblem":
            pp = _load("perproblem_rerun.py", "pp")
            model, seed, per, mse_x100, curve_last, epochs = pp._stable_per_problem_se(args)
            return {"per": per.tolist(), "mse_x100": mse_x100, "curve_last": curve_last, "epochs": epochs}
        if kind == "split":
            sv = _load("split_variance_rerun.py", "sv")
            model, split = args
            return sv._split_trial(model, 1000 + split, "stable")
        module, protocol, trial = args
        mod = _load(module, "trainer")
        mod.output_path, mod.stable_output_path = os.path.join(tmp, "original"), os.path.join(tmp, "stable")
        if module == "04_train_ablation.py":
            return next(iter(mod._runOneTrial(trial)[2].values()))
        mod._runOneTrial(trial)
        name = trial[0] if module == "01_train_regime2.py" else trial[2]
        from cpc18_utils.safe_yaml import fastLoadYaml
        entries = list(fastLoadYaml(mod._metricsPath(name, protocol)).values())
        assert len(entries) == 1
        return entries[0]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _stableTask(rec):
    seed = rec["seed"]
    if rec["design"] == "factorial":
        return ("trial", "03_train_decomp.py", "stable", (seed, rec["order_seed"], rec["model"], "stable"))
    if rec["cap"] == 1500:
        return ("trial", "04_train_ablation.py", "stable", (_cellOf(rec), seed, POOL.index(seed), "stable"))
    return ("trial", "01_train_regime2.py", "stable", (rec["model"], seed, POOL.index(seed), "stable"))


def _pick(records, pos):
    if isinstance(pos, tuple):
        key = (POOL[pos[0]], ORDER_POOL[pos[1]])
        hits = [r for r in records if (r["seed"], r["order_seed"]) == key]
    else:
        hits = [r for r in records if r["seed"] == POOL[pos]]
    assert len(hits) == 1, pos
    return hits[0]


def _compareCurves(mine, ref):
    """ Length of the common part of two curves and the largest difference on it (MSE x 100). """
    n = min(len(mine), len(ref))
    return n, max(abs(_r7(a) - b) for a, b in zip(mine[:n], ref[:n]))


def checkTrain(args):
    runs = args.runs
    jobs, rows = [], []
    for group, fname, pos in STABLE_SAMPLE:
        rec = _pick(_readJsonl(os.path.join(runs, fname)), pos)
        jobs.append(_stableTask(rec))
        rows.append(("stable", group, fname, rec))
    pp_ref = {(r["model"], r["seed"]): r for r in _readJsonl(os.path.join(runs, "perproblem_default.jsonl"))}
    for model, pos in PERPROBLEM_SAMPLE:
        jobs.append(("perproblem", model, POOL[pos]))
        rows.append(("stable", "per-problem", "perproblem_default.jsonl", pp_ref[(model, POOL[pos])]))
    sp_ref = {(r["model"], r["split"]): r for r in _readJsonl(os.path.join(runs, "split_default.jsonl"))}
    for model, split in SPLIT_SAMPLE:
        jobs.append(("split", model, split))
        rows.append(("stable", "split", "split_default.jsonl", sp_ref[(model, split)]))
    with open(os.path.join(args.base, "trials.csv"), newline="") as f:
        base = [r for r in csv.DictReader(f) if r["machine_id"] == "M00"]
    for experiment, condition, pos in ORIGINAL_SAMPLE:
        if isinstance(pos, tuple):
            hit = [r for r in base if r["experiment"] == experiment and r["condition"] == condition and
                   int(r["init_seed"]) == POOL[pos[0]] and int(r["order_seed"]) == ORDER_POOL[pos[1]]]
        else:
            hit = [r for r in base if r["experiment"] == experiment and r["condition"] == condition and
                   int(r["seed"]) == POOL[pos]]
        assert len(hit) == 1
        r = hit[0]
        seed = int(r["seed"])
        if experiment == "main":
            jobs.append(("trial", "01_train_regime2.py", "original", (condition, seed, POOL.index(seed), "original")))
        elif experiment == "factorial_init_order":
            jobs.append(("trial", "03_train_decomp.py", "original", (seed, int(r["order_seed"]), condition, "original")))
        else:
            jobs.append(("trial", "04_train_ablation.py", "original", (condition, seed, POOL.index(seed), "original")))
        rows.append(("original", experiment, "trials.csv of version 1.0", r))
    print(f"Training {len(jobs)} trials with {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as ex:
        results = list(ex.map(_trainTask, jobs))

    # Curves of the trials of version 1.0 in the sample
    wanted = {int(r["trial_id"]) for p, _, _, r in rows if p == "original"}
    base_curves = {}
    with gzip.open(os.path.join(args.base, "curves_reference.jsonl.gz"), "rt") as f:
        for line in f:
            if int(line[12:line.index(",")]) in wanted:
                rec = json.loads(line)
                base_curves[rec["trial_id"]] = rec

    failures = 0
    print("\n| Protocol | Group | Reference | Seed | Readout epoch | Reference value | Repository value | "
          "Common epochs | Max diff on curves | Same patience: min, epochs | Result |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (protocol, group, source, ref), job, mine in zip(rows, jobs, results):
        checks, readout, ref_v, my_v, common, diff, full = [], "", "", "", "", "", ""
        if protocol == "original":
            curve = base_curves[int(ref["trial_id"])]
            seed = ref["seed"] if ref["experiment"] != "factorial_init_order" else f"{ref['seed']}/{ref['order_seed']}"
            readout, ref_v, my_v = ref["best_epoch"], ref["mse"], repr(mine["mse"])
            same_curves = mine["mse_plot"] == curve["test_mse"] and mine["train_mse_plot"] == curve["train_mse"]
            checks = [repr(mine["mse"]) == ref["mse"], same_curves,
                      int(np.argmin(mine["mse_plot"])) == int(ref["best_epoch"]),
                      len(mine["mse_plot"]) == int(ref["n_epochs"])]
            common, diff, full = len(curve["test_mse"]), 0.0 if same_curves else "differ", "yes, yes"
        elif group == "per-problem":
            seed = ref["seed"]
            readout, ref_v, my_v = ref["epochs"] - 1, ref["mse_x100"], mine["mse_x100"]
            per = [round(v, 7) for v in mine["per"]]
            checks = [mine["epochs"] == ref["epochs"], per == ref["sq_error_x100"],
                      mine["mse_x100"] == ref["mse_x100"], mine["curve_last"] == ref["curve_last"]]
            common, diff = "60 problems", max(abs(a - b) for a, b in zip(per, ref["sq_error_x100"]))
            full = "yes (no early stopping)"
        else:
            curve, ref_curve = mine["mse_plot"], ref["test_curve"]
            readout = mine["readout_epoch"]
            ref_v, my_v = ref_curve[readout], mine["mse_x100"]
            seed = ref.get("seed", f"split {ref.get('split')}")
            if ref.get("order_seed") is not None:
                seed = f"{ref['seed']}/{ref['order_seed']}"
            n, d_test = _compareCurves(curve, ref_curve)
            checks = [readout < n, _r7(curve[readout]) == ref_curve[readout], abs(my_v - ref_v) <= 1e-6,
                      d_test == 0.0]
            if "train_curve" in ref:
                n2, d_train = _compareCurves(mine["train_mse_plot"], ref["train_curve"])
                checks.append(d_train == 0.0)
                d_test = max(d_test, d_train)
            common, diff = n, d_test
            ref_patience = ref.get("patience", 50)   # the split runs of the reference used 50
            if mine["patience"] == ref_patience:
                # `best` of the trials has full precision; that of the splits was rounded like their curve
                same_min = (mine["mse_x100_testmin"] == ref["best"] if group != "split"
                            else round(mine["mse_x100_testmin"], 7) == ref["best"])
                same_len = len(curve) == len(ref_curve)
                checks += [same_min, same_len]
                full = f"{'yes' if same_min else 'NO'}, {'yes' if same_len else 'NO'}"
            else:
                full = f"patience {mine['patience']} against {ref_patience}: not compared"
        ok = all(checks)
        failures += not ok
        print(f"| {protocol} | {group} | {source} | {seed} | {readout} | {ref_v} | {my_v} | {common} | {diff} | "
              f"{full} | {'pass' if ok else 'FAIL'} |")
    print(f"\n{len(rows) - failures} of {len(rows)} trials reproduced")
    if failures:
        raise SystemExit(f"{failures} trials not reproduced")


# ---------------------------------------------------------------------
# export
# ---------------------------------------------------------------------

def _hostMachines():
    """ {host: machine} of the native machines with a machine-information file, and this host as M00. """
    hosts = {}
    for p in glob.glob(os.path.join(_REPO, "output_data_*", "MACHINE_INFO.txt")):
        with open(p) as f:
            info = dict(re.match(r"^([a-z_]+):\s*(.*)$", line).groups() for line in f
                        if re.match(r"^([a-z_]+):\s*(.*)$", line))
        if "forced_tier" not in info:
            hosts[info["hostname"].strip()] = os.path.basename(os.path.dirname(p))[len("output_data_"):]
    hosts[socket.gethostname()] = "M00"
    return hosts


def _derivedKey(rec, hosts):
    """ (protocol, experiment, condition, machine) of a logged run, from its own fields. """
    protocol = ("stable" if rec["init"] == "default" else
                "original_patience300" if rec["patience"] == 300 else "original")
    if rec["design"] == "factorial":
        experiment, condition = "factorial_init_order", rec["model"]
    elif rec["cap"] == 1500 and protocol != "original_patience300":
        experiment = "ablation_bourginmlp" if rec["model"] == "BourginMLP" else "ablation_sparsemlp"
        condition = _cellOf(rec)
    else:
        experiment, condition = "main", rec["model"]
    tier = {("AVX512", ""): "", ("AVX2", "AVX2"): "-forced-AVX2", ("DEFAULT", "SSE4_2"): "-forced-DEFAULT"}
    machine = hosts[rec["host"]] + tier[(rec["cpu_cap"], rec["mkl"])]
    return protocol, experiment, condition, machine


def _includedRuns(runs):
    """ Files of the runs that version 1.1 holds, chosen by name as the inventory of the runs says. """
    files = [f for folder in ("", "datos_4090", "abl_portatil")
             for pattern in ("*_default*.jsonl", "*_p300.jsonl", "fact_*_avx2.jsonl")
             for f in glob.glob(os.path.join(runs, folder, pattern))]
    out = set()
    for f in files:
        name, rel = os.path.basename(f), os.path.relpath(f, runs)
        if name.startswith(("check_", "smoke_", "v5_", "v6_", "v7_", "perproblem", "split")):
            continue
        if "LinearLogistic" in name and "_default" in name and "_p120" not in name:
            continue  # LinearLogistic under the stable protocol: the runs with patience 120
        if name.startswith("abl_B") and "lr0.0001" in name and not rel.startswith("abl_portatil"):
            continue  # the 1e-4 column of the BourginMLP grid: abl_portatil/
        out.add(rel)
    return sorted(out)


def _expectedCounts():
    exp = {}
    for m in NEURAL_MODELS:
        exp["stable", "main", m, "M00"] = 800
        for tier in ("M01-forced-AVX2", "M01-forced-DEFAULT"):
            exp["stable", "main", m, tier] = 800
        exp["original", "factorial_init_order", m, "M00-forced-AVX2"] = 1225
    for m, machine in (("LinearLogistic", "M00"), ("ContextDepNet", "M00"), ("BourginMLP", "M01"),
                       ("SparseMLP", "M01"), ("BourginMLP", "M01-forced-AVX2")):
        exp["stable", "factorial_init_order", m, machine] = 1225
    for b in (16, 32, 64):
        for lr in ("0_01", "0_001", "0_000316"):
            exp["stable", "ablation_bourginmlp", f"BourginMLP_B{b}_L{lr}", "M01"] = 800
        exp["stable", "ablation_bourginmlp", f"BourginMLP_B{b}_L0_0001", "M01"] = 200
        exp["stable", "ablation_bourginmlp", f"BourginMLP_B{b}_L0_0001", "M00"] = 600
    for e in (10, 20, 40):
        for lr in ("0_0001", "0_001", "0_01"):
            exp["stable", "ablation_sparsemlp", f"SparseMLP_E{e}_L{lr}", "M00"] = 800
    exp["original_patience300", "main", "BourginMLP", "M01"] = 800
    exp["original_patience300", "factorial_init_order", "BourginMLP", "M00"] = 1225
    return exp


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _Report:
    def __init__(self):
        self.failed = 0

    def __call__(self, ok, what):
        self.failed += not ok
        print(f"  [{'pass' if ok else 'FAIL'}] {what}", flush=True)


def checkExport(args):
    dep, base, runs = args.deposit, args.base, args.runs
    check = _Report()
    print("== Files of version 1.0 ==")
    for name in ("curves_reference.jsonl.gz", "curves_cross_machine.jsonl.gz", "probes.csv", "split_variance.csv",
                 "perproblem_errors.csv", "cpc18_features.csv"):
        check(_sha(os.path.join(dep, name)) == _sha(os.path.join(base, name)), f"{name} byte-identical to version 1.0")

    with open(os.path.join(base, "trials.csv"), newline="") as f:
        old = list(csv.reader(f))
    with open(os.path.join(dep, "trials.csv"), newline="") as f:
        new = list(csv.reader(f))
    header, rows = new[0], new[1:]
    col = {c: i for i, c in enumerate(header)}
    n0 = len(old) - 1
    print("== Rows of version 1.0 ==")
    check(header == old[0] + NEW_COLUMNS, f"columns of version 1.0 in their order, then {NEW_COLUMNS}")
    check(all(rows[i][:N_V10_COLUMNS] == old[i + 1] for i in range(n0)),
          f"the {n0} rows of version 1.0 keep every value, in their order")
    init_of = {"LinearLogistic": "uniform"}
    bad_new = 0
    for r in rows[:n0]:
        tree = r[col["model"]] in TREES
        bad_new += not (r[col["protocol"]] == "original"
                        and r[col["init"]] == ("" if tree else init_of.get(r[col["model"]], "kaiming_fanout"))
                        and r[col["readout_epoch"]] == r[col["best_epoch"]]
                        and (r[col["mse_x100_testmin"]] == "") == tree)
    check(bad_new == 0, "rows of version 1.0: protocol original, init by model, readout = best epoch, "
                        "test minimum for every neural row")
    mins = {}
    for name in ("curves_reference.jsonl.gz", "curves_cross_machine.jsonl.gz"):
        with gzip.open(os.path.join(dep, name), "rt") as f:
            for line in f:
                rec = json.loads(line)
                mins[rec["trial_id"]] = min(rec["test_mse"]) * 100.0
    neural = [r for r in rows[:n0] if r[col["model"]] not in TREES]
    check(all(float(r[col["mse_x100_testmin"]]) == mins[int(r[col["trial_id"]])] for r in neural),
          f"mse_x100_testmin of the {len(neural)} neural rows of 1.0 = 100 x minimum of their curve")
    rel = max(abs(float(r[col["mse_x100_testmin"]]) - float(r[col["mse_x100"]])) / float(r[col["mse_x100"]])
              for r in neural if r[col["experiment"]] != "validation_split")
    check(rel < 3e-7, f"and agrees with mse_x100 (the recomputed minimum) to {rel:.1e} relative")

    print("== Rows added in version 1.1 ==")
    added = rows[n0:]
    check([int(r[col["trial_id"]]) for r in added] == list(range(n0 + 1, n0 + 1 + len(added))),
          f"{len(added)} added rows with consecutive ids after {n0}")
    counts = Counter((r[col["protocol"]], r[col["experiment"]], r[col["condition"]], r[col["machine_id"]])
                     for r in added)
    expected = _expectedCounts()
    check(counts == Counter(expected), f"rows per protocol, experiment, condition and machine as expected "
                                      f"({len(expected)} groups, {sum(expected.values())} rows)")
    for k in sorted(set(counts) | set(expected)):
        if counts.get(k) != expected.get(k):
            print(f"      {k}: {counts.get(k)} rows, {expected.get(k)} expected")
    keys = Counter((r[col["protocol"]], r[col["experiment"]], r[col["condition"]], r[col["machine_id"]],
                    r[col["seed"]], r[col["order_seed"]]) for r in rows)
    check(max(keys.values()) == 1, "no (seed, order seed) twice within a protocol, experiment, condition and machine")
    union = defaultdict(Counter)
    for r in added:
        if r[col["protocol"]] == "stable" and r[col["machine_id"]] in ("M00", "M01"):
            union[(r[col["experiment"]], r[col["condition"]])][(r[col["seed"]], r[col["order_seed"]])] += 1
    check(all(max(c.values()) == 1 and len(c) == (1225 if e == "factorial_init_order" else 800)
              for (e, _), c in union.items()),
          f"stable protocol: each of the {len(union)} conditions complete and without repeated seeds over M00 + M01")

    # Added values against the logged runs, matched by their own fields
    hosts = _hostMachines()
    included = _includedRuns(runs)
    logged = {}
    for rel_path in included:
        for rec in _readJsonl(os.path.join(runs, rel_path)):
            k = _derivedKey(rec, hosts) + (str(rec["seed"]), "" if rec["order_seed"] is None else str(rec["order_seed"]))
            assert k not in logged, k
            logged[k] = rec
    check(len(logged) == len(added), f"{len(included)} files of runs, {len(logged)} logged trials for "
                                     f"{len(added)} added rows")
    curves = {}
    with gzip.open(os.path.join(dep, "curves_added_v1.1.jsonl.gz"), "rt") as f:
        for line in f:
            rec = json.loads(line)
            curves[rec.pop("trial_id")] = rec
    mismatches = Counter()
    for r in added:
        k = (r[col["protocol"]], r[col["experiment"]], r[col["condition"]], r[col["machine_id"]],
             r[col["seed"]], r[col["order_seed"]])
        rec = logged.get(k)
        if rec is None:
            mismatches["no logged trial"] += 1
            continue
        stable = r[col["protocol"]] == "stable"
        model = r[col["model"]]
        epsilon = int(rec["epsilon"]) if rec.get("epsilon") else (20 if model == "SparseMLP" else None)
        readout = (readoutEpoch(model, rec.get("batch_size", 32), rec.get("lr", 1e-3), epsilon)
                   if stable else rec["best_epoch"])
        value = rec["test_curve"][readout] if stable else rec["best"]
        c = curves[int(r[col["trial_id"]])]
        for name, ok in (
                ("mse_x100", float(r[col["mse_x100"]]) == value),
                ("readout_epoch", int(r[col["readout_epoch"]]) == readout),
                ("mse_x100_testmin", float(r[col["mse_x100_testmin"]]) == rec["best"]),
                ("best_epoch", int(r[col["best_epoch"]]) == rec["best_epoch"]),
                ("n_epochs", int(r[col["n_epochs"]]) == rec["n_epochs"]),
                ("patience", int(r[col["patience"]]) == rec["patience"]),
                ("train_duration_s", float(r[col["train_duration_s"]]) == rec["seconds"]),
                ("init", r[col["init"]] == ("pytorch_default" if stable else init_of.get(model, "kaiming_fanout"))),
                ("empty mse", r[col["mse"]] == "" and r[col["mse_random_baseline"]] == ""),
                ("curves", c["test_mse_x100"] == rec["test_curve"] and c["train_mse_x100"] == rec["train_curve"])):
            mismatches[name] += not ok
    check(sum(mismatches.values()) == 0, "every added row equals its logged trial: mse_x100 at the readout epoch "
                                         "(stable) or the minimum (original), test minimum, epochs, patience, "
                                         f"duration, init and curves {dict(mismatches) or ''}")

    print("== Machines ==")
    with open(os.path.join(dep, "machines.csv"), newline="") as f:
        machines = {r["machine_id"]: r for r in csv.DictReader(f)}
    with open(os.path.join(base, "machines.csv"), newline="") as f:
        machines0 = {r["machine_id"]: r for r in csv.DictReader(f)}
    per_machine = Counter(r[col["machine_id"]] for r in rows)
    check(all(int(m["n_trials"]) == per_machine.get(mid, 0) for mid, m in machines.items()),
          "n_trials of machines.csv = rows of trials.csv per machine")
    # info_source only points to the part of the article that lists the machine
    check(all({k: v for k, v in machines[mid].items() if k not in ("n_trials", "info_source")} ==
              {k: v for k, v in m.items() if k not in ("n_trials", "info_source")} for mid, m in machines0.items()),
          "machines of version 1.0 described as before")
    new_m = set(machines) - set(machines0)
    check(new_m == {"M00-forced-AVX2"} and machines["M00-forced-AVX2"]["run_tier"] == "AVX2" and
          machines["M00-forced-AVX2"]["aten_cpu_capability"] == "avx2" and
          machines["M00-forced-AVX2"]["mkl_enable_instructions"] == "AVX2", "one new machine, M00-forced-AVX2")

    print("== Stable re-runs and cross-validation ==")
    sp = {(r["model"], r["split"]): r for r in _readJsonl(os.path.join(runs, "split_default.jsonl"))}
    with open(os.path.join(dep, "split_variance_stable.csv"), newline="") as f:
        srows = list(csv.DictReader(f))
    ok = len(srows) == len(sp) == 80
    for r in srows:
        ref = sp[(r["model"], int(r["split_index"]))]["test_curve"]
        e = readoutEpoch(r["model"], 32, 1e-3, 20 if r["model"] == "SparseMLP" else None)
        ok &= (int(r["readout_epoch"]) == e and float(r["mse_x100"]) == ref[e] and
               float(r["mse_x100_testmin"]) == min(ref) and int(r["n_epochs"]) == len(ref) and
               int(r["patience"]) == sp[(r["model"], int(r["split_index"]))].get("patience", 50))
    with gzip.open(os.path.join(dep, "split_variance_stable_curves.jsonl.gz"), "rt") as f:
        ok &= all(sp[(c["model"], c["split_index"])]["test_curve"] == c["test_mse_x100"]
                  for c in map(json.loads, f))
    check(ok, "split_variance_stable.csv and its curves equal the 80 logged re-splits")
    pp = {(r["model"], r["seed"]): r for r in _readJsonl(os.path.join(runs, "perproblem_default.jsonl"))}
    with open(os.path.join(dep, "perproblem_errors_stable.csv"), newline="") as f:
        prow = list(csv.DictReader(f))
    vals = defaultdict(list)
    for r in prow:
        vals[(r["model"], int(r["seed"]))].append((int(r["problem_id"]), float(r["mean_sq_error_x100"])))
    check(len(prow) == 200 * 60 and set(vals) == set(pp) and
          all(vals[k] == list(zip(pp[k]["problem_id"], pp[k]["sq_error_x100"])) for k in pp),
          "perproblem_errors_stable.csv equals the 200 logged trials, 60 problems each")
    cv = {(r["config"], r["fold"], r["seed"]): r for r in _readJsonl(os.path.join(runs, "cv.jsonl"))}
    with gzip.open(os.path.join(dep, "cv_curves.jsonl.gz"), "rt") as f:
        dcv = [json.loads(line) for line in f]
    check(len(dcv) == len(cv) and all(cv[(c["config"], c["fold"], c["seed"])]["val_curve"] == c["val_mse_x100"]
                                      for c in dcv), f"cv_curves.jsonl.gz equals the {len(cv)} logged runs")
    cvmod = _load("19_cv_epochs.py", "cv_epochs")
    tmp = tempfile.mkdtemp(prefix="protocol_check_")
    try:
        with open(os.path.join(tmp, "curves.jsonl"), "w") as f:
            for c in dcv:
                f.write(json.dumps({"config": c["config"], "n_train_rows": c["n_train_rows"],
                                    "val_curve": c["val_mse_x100"]}) + "\n")
        cvmod.summarize(os.path.join(tmp, "curves.jsonl"), os.path.join(tmp, "summary.csv"))
        with open(os.path.join(tmp, "summary.csv")) as f:
            summary = f.read()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    with open(os.path.join(_REPO, "data", "cv_epochs.csv")) as f:
        repo_cv = f.read()
    with open(os.path.join(dep, "cv_epochs.csv"), newline="") as f:
        dep_cv = [line.rstrip("\r\n") for line in f]
    check(summary == repo_cv and dep_cv == repo_cv.splitlines(),
          "cv_epochs.csv of the deposit = data/cv_epochs.csv = summary of the deposited curves by 19_cv_epochs.py")

    print("== Host names ==")
    terms = {rec["host"] for rec in logged.values()}
    hits = []
    for name in os.listdir(dep):
        if name.endswith(".gz"):
            continue
        with open(os.path.join(dep, name)) as f:
            text = f.read()
        hits += [(name, t) for t in terms if t in text]
    check(not hits, f"no host name in the text files {hits or ''}")

    if check.failed:
        raise SystemExit(f"{check.failed} checks failed")
    print("\nAll checks passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "export"):
        p = sub.add_parser(name)
        p.add_argument("--runs", required=True, help="folder of the runs added in version 1.1")
        p.add_argument("--base", default=os.path.join(_REPO, "zenodo_export"), help="deposit of version 1.0")
        if name == "train":
            p.add_argument("--workers", type=int, default=3)
        else:
            p.add_argument("--deposit", default=os.path.join(_REPO, "zenodo_export_v1.1"),
                           help="export of version 1.1")
    args = parser.parse_args()
    for attr in ("runs", "base", "deposit"):
        if getattr(args, attr, None):
            setattr(args, attr, os.path.realpath(os.path.join(_REPO, getattr(args, attr))))
    checkTrain(args) if args.command == "train" else checkExport(args)


if __name__ == "__main__":
    main()
