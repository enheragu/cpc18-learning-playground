#!/usr/bin/env python3
# encoding: utf-8
"""
Reading the per-trial results from the data deposit: a log that does not exist
is served from trials.csv under the names of the training logs, with the exact
values and the order of the file, and an existing log still has priority.
"""

import csv
import importlib

import pytest
import yaml


FIELDS = ["trial_id", "experiment", "machine_id", "condition", "batch_size", "learning_rate", "epsilon",
          "seed", "init_seed", "order_seed", "mse", "mse_x100", "mse_random_baseline", "best_epoch", "patience",
          "train_duration_s", "val_split_seed", "n_train_problems", "n_val_problems", "best_test_epoch",
          "mse_x100_testsel", "val_mse_at_selection"]
LOG_MAIN = "sharedpool_r2_training_metrics.yaml"
LOG_ABLATION = "sharedpool_r2_ablation_training_metrics.yaml"
LOG_REGIME1 = "sharedpool_training_metrics.yaml"


def _row(trial_id, experiment, machine, condition, **values):
    return {"trial_id": trial_id, "experiment": experiment, "machine_id": machine, "condition": condition, **values}


@pytest.fixture
def logs(tmp_path, monkeypatch):
    """ The module on a temporary repository with a small deposit and no training logs. """
    deposit = tmp_path / "zenodo_export"
    deposit.mkdir()
    rows = [_row(1, "main", "M00", "BourginMLP", seed=7, mse="0.009276810102164745", best_epoch=12, patience=50,
                 learning_rate="0.001", batch_size=32, train_duration_s="1.5"),
            _row(2, "main", "M00", "BourginMLP", seed=3, mse="0.1", best_epoch=4),
            _row(3, "main", "M00", "BEAST_GB", seed=7, mse="0.2"),
            _row(4, "main", "M03", "BourginMLP", seed=7, mse="0.3"),
            _row(5, "ablation_sparsemlp", "M03", "SparseMLP_E20_L0_001", seed=7, mse="0.4", epsilon=20)]
    with open(deposit / "trials.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with open(deposit / "split_variance.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "split_index", "mse_x100"])
        writer.writeheader()
        writer.writerow({"model": "BEAST-GB", "split_index": 0, "mse_x100": "0.8"})

    import cpc18_utils
    import cpc18_utils.training_logs as module
    monkeypatch.setattr(cpc18_utils, "output_path", str(tmp_path / "output_data"))
    monkeypatch.setattr(cpc18_utils, "deposit_path", str(deposit))
    monkeypatch.delenv("CPC18_READ_DEPOSIT", raising=False)
    importlib.reload(module)
    yield module, tmp_path
    monkeypatch.undo()
    importlib.reload(module)


def test_trials_come_from_the_deposit_with_log_names(logs):
    module, root = logs
    trials = module.loadTrainingLog(str(root / "output_data" / "BourginMLP" / LOG_MAIN))
    assert list(trials) == ["1", "2"]
    first = trials["1"]
    assert first["mse"] == 0.009276810102164745
    assert (first["seed"], first["best_epoch"], first["batch_size"]) == (7, 12, 32)
    assert first["train_duration"] == 1.5
    assert "init_seed" not in first


def test_machine_comes_from_the_tag_of_the_tree(logs):
    module, root = logs
    assert list(module.loadTrainingLog(str(root / "output_data_M03" / "BourginMLP" / LOG_MAIN))) == ["4"]
    cell = module.loadTrainingLog(str(root / "output_data_M03" / "SparseMLP_E20_L0_001" / LOG_ABLATION))
    assert cell["5"]["epsilon"] == 20


@pytest.mark.parametrize("tree, condition, log_name", [
    ("output_data", "BourginMLP", LOG_REGIME1),     # the exploratory regime is not in the deposit
    ("output_data", "ContextDepNet", LOG_MAIN),     # no trials of that condition
    ("output_data_NEW", "BourginMLP", LOG_MAIN),    # a machine that is not in the deposit
])
def test_missing_results_raise(logs, tree, condition, log_name):
    module, root = logs
    with pytest.raises(FileNotFoundError):
        module.loadTrainingLog(str(root / tree / condition / log_name))


def test_returned_trials_are_copies(logs):
    module, root = logs
    path = str(root / "output_data" / "BourginMLP" / LOG_MAIN)
    module.loadTrainingLog(path)["1"]["mse"] = -1.0
    assert module.loadTrainingLog(path)["1"]["mse"] == 0.009276810102164745


def test_existing_log_has_priority_unless_the_deposit_is_forced(logs, monkeypatch):
    module, root = logs
    log_file = root / "output_data" / "BourginMLP" / LOG_MAIN
    log_file.parent.mkdir(parents=True)
    log_file.write_text(yaml.safe_dump({"2026-01-01_10-00-00.000": {"seed": 1, "mse": 0.5}, "version": 3}))
    assert list(module.loadTrainingLog(str(log_file))) == ["2026-01-01_10-00-00.000"]
    monkeypatch.setenv("CPC18_READ_DEPOSIT", "1")
    assert list(module.loadTrainingLog(str(log_file))) == ["1", "2"]


def test_conditions_of_the_reference_machine_sorted(logs):
    module, _ = logs
    assert module.trainingConditions() == ["BEAST_GB", "BourginMLP"]


def test_result_files_use_the_model_names_of_the_code(logs):
    module, _ = logs
    assert module.depositRows("split_variance.csv") == [{"model": "BEAST_GB", "split_index": "0", "mse_x100": "0.8"}]
    assert module.depositRows("not_in_the_deposit.csv") == []


def test_trials_ending_in_the_same_millisecond_are_kept(tmp_path):
    from cpc18_utils.safe_yaml import addTrials, atomicUpdateMetricsLogFile, fastLoadYaml

    key = "2026-07-22_18-13-00.999"
    path = str(tmp_path / "metrics.yaml")
    atomicUpdateMetricsLogFile({key: {"seed": 1, "mse": 0.5}}, path)
    atomicUpdateMetricsLogFile({key: {"seed": 2, "mse": 0.6}}, path)
    atomicUpdateMetricsLogFile({key: {"seed": 2, "mse": 0.6}}, path)  # the same trial again, not a new one
    stored = fastLoadYaml(path)
    assert sorted(entry["seed"] for entry in stored.values()) == [1, 2, 2]
    assert "2026-07-22_18-13-01.000" in stored

    assert len(addTrials({key: {"seed": 1}}, {key: {"seed": 1}})) == 1
