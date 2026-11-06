# Experiments and analysis scripts

This folder holds the training and analysis scripts of the study. The main [README](../README.md) explains how to run
them.

- `cpc18_utils/`: data and feature loaders, atomic YAML wrapper, and the reader of the training logs and of the deposit.
- `cpc18_models/`: the models and the ablation classes, described in [their own README](cpc18_models/README.md).
- `path_setup.py`: adds the `src/` folder of the MNIST submodule to `sys.path`.

## Experiments

All experiments share the seed pool generated from master seed `20260522` (`MASTER_SEED_FOR_POOL`), so trials with the
same seed are paired across models, ablation cells and machines. The factorial design adds a second, disjoint pool for
the data-order dimension (master seed `99999777`).

| Experiment | Script | Design | Article |
|---|---|---|---|
| Main run | `01_train_regime2.py` | Six models (BourginMLP, LinearLogistic, ContextDepNet, SparseMLP, PsychForest, BEAST-GB) on the BEAST feature set and the canonical CPC18 train/test split, 800 shared seeds each | Sections 1.1, 3.1, 3.2, 3.5, 4 |
| Ablation B1 | `04_train_ablation.py --study b1` | BourginMLP, batch size {16, 32, 64} x learning rate {1e-4, 1e-3, 1e-2}, 800 seeds per cell; `--study b1x` adds the learning rate 3.16e-4 column of the grid-sensitivity check | Section 3.4 |
| Ablation B3 | `04_train_ablation.py --study b3` | SparseMLP, target sparsity epsilon {10, 20, 40} x learning rate {1e-4, 1e-3, 1e-2}, 800 seeds per cell | Section 3.4 |
| Init x order factorial | `03_train_decomp.py` | Weight initialization and data-shuffle order seeded from two independent RNG streams, 35 x 35 grid per neural model | Section 3.3 |
| Validation split | `11_train_valsplit.py` | Early stopping on an internal validation split of the training problems instead of the test set, 200 seeds, BourginMLP and ContextDepNet | Section 5.5 |
| Split variance | `split_variance_rerun.py` | 20 random re-partitions of the problems, initialization seed fixed | Section 3.5 |
| Per-problem errors | `perproblem_rerun.py` | Squared error per test problem for the first 50 shared seeds | Section 3.5 |
| Cross-machine | the trainers above, [`scripts/variance_check/`](../scripts/variance_check/README.md) | A subset of the main-run and ablation conditions re-run on other machines and forced CPU tiers | Section 3.6, Appendix H |

`beast_gb_topup.py` completes the BEAST-GB trials of the main run without re-checking the other five models.
`00_train_models.py` trains four models on the raw lottery parameters only (Regime 1); it is an exploratory experiment
not reported in the article.

## Analysis scripts

| Script | Output | Article |
|---|---|---|
| `02_analyze.py` | Cross-model Monte Carlo / Bootstrap misranking, ICC and Spearman | Section 3.2 |
| `05_make_paper_figures.py` | Figures; `--media <latex>/media` copies only the ones the manuscript includes | all |
| `06_analyze_decomp.py` | Sum-of-squares decomposition of the factorial and its slicing modes | Section 3.3, Appendix C |
| `07_analyze_ablation.py` | Two-way ANOVA, ICC and misranking of the ablation grids | Section 3.4 |
| `08_power_analysis.py` | Sample size and statistical power of the headline statistics, and the ICC with its exact interval | Appendix E |
| `09_export_csv.py` | Flat CSVs of all the results (see [Flat CSVs](#flat-csvs)) | |
| `10_paper_tables.py` | Summary and reporting format (lognormal P10 and its check), misranking, normality, ICC, paired-win, ANOVA and decomposition tables, from the flat CSVs only | Sections 3.1 to 3.4 and 4, Appendix A |
| `12_determinism_audit.py` | Re-trains dispersed cells of the factorial and compares the test MSE bit for bit | Section 3.3 |
| `13_variance_components.py` | Random-effects variance components of the factorial with bootstrap CIs | Section 3.3 |
| `14_gaussian_calibration_audit.py` | Lower-tail calibration of the Gaussian fit and single-run swap probabilities | Section 3.1, Appendix A |
| `15_parametric_family_audit.py` | Normal, lognormal, gamma and mixture fits of BourginMLP | Section 3.1, Appendix A |
| `16_export_zenodo.py` | The data deposit; `--article` and `--code-doi` fill its citation and the DOI of the code | |
| `17_article_tables.py` | Training-seed against test-sampling variance, reporting format, statistics at N = 400 against N = 800, training times, cross-machine summary per arithmetic path and resolution floor of P10, from the data deposit only (`--deposit`, by default `zenodo_export/`) | Sections 3.5 and 4, Appendices F and H |
| `18_text_numbers.py` | Numbers quoted in the text of the article that no other script prints, one block per section, from the data deposit only (`--deposit`, by default `zenodo_export/`) | all |

## Flat CSVs

`09_export_csv.py` writes one flat CSV per experiment to `analysis_results/csv/`:

| File | Contents |
|---|---|
| `results_sharedpool_r2.csv` | Main run, 6 models x 800 trials |
| `results_ablation_b1.csv`, `results_ablation_b1x.csv` | Ablation B1 and its grid-sensitivity column |
| `results_ablation_b3.csv` | Ablation B3 |
| `results_factorial_decomp.csv` | Init x order factorial, 4 models x 35 x 35 |
| `results_valsplit.csv` | Validation-split early stopping |
| `results_split_variance.csv` | Split-induced variance |
| `results_perproblem_errors.csv` | Per-problem errors |
| `results_sharedpool_r1.csv` | Regime 1 (exploratory) |

Each row of the trial files is one training trial (one model trained with one seed); the MSE is the aggregate over all
test rows, and variance is studied at this level across seeds. The main columns are `model`, `seed`, `mse` and
`mse_x100`, the training hyperparameters, `best_epoch`, `total_epochs`, `patience`, `train_duration` and, in the
factorial file, `init_seed` and `order_seed`. Only `results_perproblem_errors.csv` has a problem-level column: one row
per (model, seed, test problem), with `problem_id`. To add problem-level context, join it on `problem_id` with
`data/cpc18_features.csv`, which holds one row per (problem, block) with the lottery parameters, the BEAST-derived
features and the canonical `Train_Test` split column.

Built from the data deposit, the columns that it does not hold are empty (`seed_mode`, `protocol`, `seed_decomposed`,
`total_epochs`, and `batch_size`, `learning_rate` and `patience` of the tree-based models) and `best_epoch` is the
minimum of the curve of the selection criterion. Integer columns with empty cells are written as floats (`217.0`), and
the two re-run files list their models in another order, with the same values. No analysis script reads these columns or
depends on that order.
