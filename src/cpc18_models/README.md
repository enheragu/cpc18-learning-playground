# Models

Six model families that predict the choices of CPC18, from a single linear layer to the winner of the competition. The
neural models are representative rather than exact reproductions of prior work, so each section says what differs from
the published version.

The task is a regression. Each problem offers two lotteries, and the target is the share of people who chose option B in
that problem and block. The inputs describe the lotteries and add the BEAST-derived features (`data/cpc18_features.csv`);
no input describes a person. The error is the test MSE, lower is better, usually reported multiplied by 100.

| Model | Family | Size |
|:------|:-------|:-----|
| [LinearLogistic](LinearLogistic.py) | Neural, baseline | 38 parameters |
| [ContextDepNet](ContextDepNet.py) | Neural | 2,305 parameters |
| [BourginMLP](BourginMLP.py) | Neural | 90,576 parameters |
| [SparseMLP](SparseMLP.py) | Neural, sparse | about 22,400 active of 90,576 parameters |
| [PsychForest](PsychForest.py) | Random forest | 500 trees |
| [BEAST_GB](BEAST_GB.py) | Gradient-boosted trees | 978 trees |

## How they are trained

- **Neural models** ([base.py](base.py)): Adam, MSE loss on a sigmoid output, learning rate 0.001, batch size 32, patience
  50 and up to 500 epochs in the main run (1,500 in the ablations). Early stopping and the selection of the best epoch
  use the test MSE. The article describes this choice as part of its fragile configuration, and `11_train_valsplit.py`
  measures its effect with a validation split. This is the original protocol; under the stable one, the test MSE is read at a fixed epoch chosen without the
  test set (see the main README).
- **Initialization** (argument `init` of the neural models): `"original"`, the default, keeps the initialization of each
  model, Kaiming normal in fan-out mode with zero biases, or uniform weights in [-0.1, 0.1] for LinearLogistic.
  `"pytorch_default"` draws every linear layer again with its `reset_parameters()`, from the global generator seeded with
  the training seed, and then restores that generator, so that the data order of the seed is the same as with the
  original initialization. The stable protocol uses it.
- **Tree-based models** ([SklearnRegressorTrainer.py](SklearnRegressorTrainer.py)): the same interface and logs, with the
  seed passed as the random state of the estimator.
- **Ablations** ([ablation.py](ablation.py)): one subclass per cell of each grid, named after its hyperparameters, such
  as `BourginMLP_B32_L0_001` or `SparseMLP_E20_L0_001`.

## LinearLogistic

One linear layer and a sigmoid. It is the baseline of the study, with a low initialization variance, like
SimplePerceptron in the MNIST study.

## ContextDepNet

The architecture of `ContextDependantNetwork` in Thomas et al. 2024 (*Nature Human Behaviour* 8(4)): two hidden layers of
32 units, with sigmoid activations throughout.

- It takes the feature vector of the other models, not the original 24-dimensional input.
- It is the network alone, without the noise and guessing wrapper of that paper.

## BourginMLP

The architecture of the MLP in Bourgin et al. 2019 (ICML): three hidden layers of 200, 275 and 100 units, and a sigmoid
output. The published setup differs in four points:

- SiLU replaces SReLU as the activation. The class in the code is still named `SReLU`.
- There is no dropout (0.15 in the paper), to isolate the variance of the initialization.
- It uses Adam instead of RMSProp.
- There is no pretraining on synthetic data. The published MSE x 100 of 0.48 needs that step, so it is not comparable
  with these runs.

## SparseMLP

BourginMLP with a static Erdős-Rényi mask on its first three layers (Mocanu et al. 2018, *Nature Communications*), as
used by Bourgin et al. 2019 and Thomas et al. 2024. With the seed fixing both the mask and the initial weights, sparsity
epsilon 20 keeps about 64%, 17% and 27% of the weights of those layers.

The evolutionary weight rebirth of the original recipe is left out, because it would add randomness during training.

## PsychForest

A random forest after Plonsky et al. 2017 (AAAI): 500 trees, no maximum depth and a square-root feature subset, which
roughly match the setup of that paper. It takes the full feature vector instead of its 9 hand-picked features, so that
the inputs are the same as in BEAST-GB.

## BEAST_GB

XGBoost with the hyperparameters of the reference implementation (`BEAST_GB.R`, OSF project
[vw2su](https://doi.org/10.17605/OSF.IO/VW2SU)), copied verbatim. This model won CPC18 (Plonsky et al., *Nature Human
Behaviour*, 2025). Its 800 runs go from 0.551 to 0.593 and contain the published values of 0.56 and 0.57.

## Exploratory first regime

`MLP_1L` (one hidden layer of 50 units) and `MLP_3L` (three hidden layers of 100 units) only appear in
`00_train_models.py`, the exploratory regime on the raw lottery parameters, which the article does not report.
