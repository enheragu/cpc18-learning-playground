# CPC18 Learning Playground

Variance-aware study of model performance on the
[CPC18 Choice Prediction Competition 2018](https://cpc-18.com/) dataset.
Mirrors the methodology of
[mnist-learning-playground](https://github.com/enheragu/mnist-learning-playground)
on a behavioural-decision problem instead of image classification.
The MNIST repo is included as a git submodule and reused for shared
infrastructure (logging, YAML metrics persistence with file locking,
seed control, Monte Carlo / Bootstrap analysis pipeline).

Companion interactive browser for the per-trial metrics, distribution
plots, sampling curves and cross-model misranking tables produced by
this repository along with the ones from the MNIST playground:

> Heredia-Aguado, E., Rujano, A., Valiente, D., & Gil, A. (2026).
> *MLV Tools: Interactive Web Applications for Machine Learning Variance
> Analysis*. Zenodo. [https://doi.org/10.5281/zenodo.20268237](https://doi.org/10.5281/zenodo.20268237)
> — live at [enheragu.github.io/mlv-tools](https://enheragu.github.io/mlv-tools).


## Experiment families

Five experiments, each isolating a different source of performance variance.
Full design rationale and empirical findings are reported in the accompanying
manuscript.

**Regime 1 — raw features** (`00_train_models.py`)
Laboratory stress test. Four models trained on the 12 raw lottery parameters
plus block indicator, with no behavioural priors. Each model is trained 400
times with different random seeds. The question is whether the seed-to-seed
MSE variance is large enough to affect model ranking — a controlled setting
analogous to the MNIST case in the parent paper.

**Regime 2 — BEAST features** (`01_train_regime2.py`)
Bibliography-grounded regime. Six published model families (two neural,
one sparse neural, one random forest, one gradient-boosted tree) trained
on the full BEAST feature set (raw + naive + psychological insight + BEAST
prediction). Each model trained 800 times. Same question as Regime 1 but
at the level of the CPC18 leaderboard, allowing comparison against published
point estimates.

**Ablation B1 — BourginMLP hyperparameters** (`04_train_ablation.py --study b1`)
Two-way grid: 3 batch sizes × 3 learning rates = 9 conditions, 800 trials
each. Varies the training hyperparameters of BourginMLP on BEAST features.
The question is whether seed-induced variance within a condition is large
enough to cause misranking across conditions — i.e., whether a single run
is sufficient to select the best hyperparameter setting.

**Ablation B3 — SparseMLP hyperparameters** (`04_train_ablation.py --study b3`)
Same design as B1 but for SparseMLP: 3 epsilon (sparsity) × 3 learning
rates = 9 conditions, 800 trials each. Contrasts with B1: whereas B1 is
dominated by the learning-rate main effect, B3 is dominated by the
epsilon × learning-rate interaction — a different variance regime.

**Phase C — init × order factorial** (`03_train_decomp.py`)
Variance decomposition experiment. Weight-initialisation seed and
batch-shuffle-order seed are decoupled into two independent RNG streams,
giving a 35 × 35 = 1 225-trial factorial grid per model. Decomposes total
MSE variance into init contribution, order contribution, and their
interaction. Applied to the four neural models (tree-based models have no
batch-order axis). Shows that the relative importance of each source is
architecture-dependent and not knowable without measurement.

---

All five families share the same master seed (`MASTER_SEED_FOR_POOL = 20260522`)
for the init dimension, so per-seed cross-experiment analysis is possible.
Phase C adds a second disjoint pool for the order dimension
(`MASTER_SEED_ORDER = 99999777`).

## Layout

```
cpc18-learning-playground/
├── mnist-learning-playground/    # git submodule (library-style reuse)
├── data/                         # raw + aggregated CPC18 (regenerable, gitignored)
├── deps/                         # external baselines (BEAST-GB R code from OSF, gitignored)
├── output_data/                  # per-model YAML training logs (regenerable, gitignored)
├── analysis_results/             # tables, plots; csv/ exports are committed
└── src/
    ├── path_setup.py             # adds submodule src/ to sys.path
    ├── 00_train_models.py        # regime 1 trainer (raw features, 3 seed modes)
    ├── 01_train_regime2.py       # regime 2 trainer (BEAST features)
    ├── 02_analyze.py             # cross-model Monte Carlo / Bootstrap / ICC / Spearman
    ├── 03_train_decomp.py        # Phase C: factorial init × order grid
    ├── 04_train_ablation.py      # ablation B1 / B3 trainer
    ├── 05_make_paper_figures.py  # figure generation
    ├── 06_analyze_decomp.py      # Phase C SS decomposition + slicing modes
    ├── 07_analyze_ablation.py    # B1 / B3 ANOVA + ICC + misranking
    ├── 08_power_analysis.py      # sample-size / power analysis
    ├── 09_export_csv.py          # export all YAMLs to flat CSVs for R
    ├── 10_paper_tables.py        # regenerate all paper tables from the committed CSVs
    ├── cpc18_utils/              # data loader, feature loader, atomic YAML wrapper
    └── cpc18_models/             # model definitions + ablation subclasses
tests/                           # pytest: ANOVA contrasts, misranking vs analytic, sameseed
```

## Setup

```sh
git submodule update --init --recursive
source /path/to/venv/bin/activate    # or: python3 -m venv venv && source venv/bin/activate
pip install -r requirements
```

CPU is the default; set `CPC18_USE_GPU=1` to opt in (not faster on this
problem size).

Features (`data/cpc18_features.csv`) are pre-committed. To regenerate:
`Rscript scripts/precompute_features.R` (requires R + dplyr + xgboost).

## Running

Training writes one YAML per (model, seed-mode) under `output_data/`,
file-lock protected so several worker processes can append safely.

```sh
cd src

# Regime 1 — raw features (400 trials × 4 models)
python 00_train_models.py --workers 6

# Regime 2 — BEAST features (800 trials × 6 models)
python 01_train_regime2.py --workers 6

# Phase C — factorial init × order (35 × 35 per model)
python 03_train_decomp.py --model BourginMLP --workers 6
python 03_train_decomp.py --model LinearLogistic --workers 6
python 03_train_decomp.py --model ContextDepNet --workers 6
python 03_train_decomp.py --model SparseMLP --workers 6

# Ablation studies
python 04_train_ablation.py --study b1 --workers 6   # BourginMLP batch × lr
python 04_train_ablation.py --study b3 --workers 6   # SparseMLP epsilon × lr
```

All trainers are idempotent: re-running only adds missing seeds.

Analysis outputs land in `analysis_results/`:

```sh
python 02_analyze.py --mode shared_pool --regime 2   # cross-model ICC / Spearman
python 06_analyze_decomp.py                          # Phase C SS decomposition
python 07_analyze_ablation.py                        # B1 / B3 ANOVA + misranking
```

To export flat CSVs for R (written to `analysis_results/csv/`):

```sh
python 09_export_csv.py
```

## CSV exports 

Running `09_export_csv.py` writes five files to `analysis_results/csv/`:

| File | Rows | Contents |
|---|---|---|
| `results_sharedpool_r1.csv` | 1 600 | Regime 1 — 4 models × 400 trials on raw features |
| `results_sharedpool_r2.csv` | 4 800 | Regime 2 — 6 models × 800 trials on BEAST features |
| `results_ablation_b1.csv` | 7 200 | B1 ablation — BourginMLP, 9 conditions (batch × lr) × 800 trials |
| `results_ablation_b3.csv` | 7 200 | B3 ablation — SparseMLP, 9 conditions (ε × lr) × 800 trials |
| `results_factorial_decomp.csv` | 4 899 | Phase C — 4 models × 35 × 35 factorial (init × order seeds) |

Each row is one training trial (one model trained with one seed).
The mean squared error (MSE) reported is the aggregate over all test-set
rows — variance is studied at this level across seeds, not at the
individual-problem level.

| Column | Present in | Description |
|---|---|---|
| `model` | all | model class name |
| `seed` | all | random seed for this trial |
| `seed_mode` | all | pool type (`shared_pool`, `factorial_init_order`) |
| `mse` / `mse_x100` | all | test-set mean squared error (raw and ×100) |
| `mse_random_baseline` | all | MSE of a random predictor on the same split (reference) |
| `batch_size`, `learning_rate` | all | training hyperparameters |
| `best_epoch`, `total_epochs`, `patience` | all | convergence info |
| `train_duration` | all | wall-clock seconds |
| `regime` | sharedpool only | 1 = raw features, 2 = BEAST features |
| `batch_size` / `epsilon` | ablation B1 / B3 | factor-level value for that condition |
| `init_seed`, `order_seed`, `seed_decomposed` | factorial only | decoupled seed axes |
| `split_seed` | all | 42 — seed used to generate the train/test split (regime 1) |
| `master_seed_for_pool` | all | 20260522 — master for the shared seed pool |
| `master_seed_order` | factorial only | 99999777 — master for the order pool |

**Joining with problem-level features**: `data/cpc18_features.csv` contains
one row per (problem, block) with lottery parameters, BEAST-derived features,
and the canonical `Train_Test` split column (Plonsky 2018: 210 train /
60 test). Join on `problem_id` = `GameID` to add per-problem context.
Regime 1 uses a different split (80/20 by GameID, `split_seed=42`); see
`cpc18_utils/data.py:load_cpc18_aggregate` for the exact procedure.
