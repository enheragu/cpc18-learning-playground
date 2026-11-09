# Data

`cpc18_features.csv` is the feature table used by every experiment: one row per (problem, block) of
the Choice Prediction Competition 2018 (CPC18), with the lottery parameters, the observed choice
rate `B_rate`, the BEAST-derived features, the BEAST prediction and the canonical `Train_Test` split
column.

`cv_epochs.csv` holds the number of epochs whose test MSE the stable training protocol reports, one row per configuration
(model, batch size, learning rate and, for SparseMLP, sparsity epsilon). `src/19_cv_epochs.py` chooses them by 5-fold
cross-validation on the 208 training problems, without the test problems, and the trainers read them; its columns are
described in that script and in the README of the data deposit, which also holds the validation curves behind it.

## License

Unlike the code of this repository (MIT License, see [`../LICENSE`](../LICENSE)), `cpc18_features.csv` is
distributed under the Creative Commons Attribution 4.0 International licence (CC BY 4.0). It is
derived from the CPC18 data of Plonsky, Erev and Ert, "All raw data for CPC18: Choice prediction
competition 2018", Zenodo,
[https://doi.org/10.5281/zenodo.2571510](https://doi.org/10.5281/zenodo.2571510) (CC BY 4.0), with
the feature definitions of the BEAST-GB reference implementation (Plonsky, "Predicting Human
Decisions with Behavioral Theories and Machine Learning", OSF,
[https://doi.org/10.17605/OSF.IO/VW2SU](https://doi.org/10.17605/OSF.IO/VW2SU)). Please cite the
CPC18 data when using it.

## How it is generated

`scripts/precompute_features.R` runs the feature-engineering code of the BEAST-GB reference
implementation, fetched from OSF by `deps/download.sh`, on the CPC18 data included in that download,
and writes this file (requires R, and takes about 20 minutes):

```sh
deps/download.sh
Rscript scripts/precompute_features.R
```

The other files of this folder (`all_CPC18_raw_data.csv`, `cpc18_aggregate.parquet`) are downloaded
or derived locally by `src/cpc18_utils/data.py` and are not part of the repository.
