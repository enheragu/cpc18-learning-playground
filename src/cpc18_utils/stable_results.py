#!/usr/bin/env python3
# encoding: utf-8

"""
Results of the stable protocol (cpc18_utils/protocol.py) from the data deposit,
version 1.1 or later, in the form the figures, tables and text numbers of the
article read them.

- Neural models: rows of protocol `stable` on the reference arithmetic path,
  machines M00 and M01, which give bit-identical results for the same seed (the
  stable trials of a condition are split between them). The value of a trial is
  `mse_x100`, its test MSE x 100 at the fixed epoch of its configuration.
- Tree-based models: rows of the original protocol on M00. They have no
  initialization nor epochs and are trained the same way under both protocols.

The frames are ordered by seed (and by order seed in the factorial design), so
that a seeded resampling of the same data always draws the same trials.
"""

import gzip
import json
import os

import numpy as np
import pandas as pd


REFERENCE_MACHINES = ("M00", "M01")
NEURAL_MODELS = ("LinearLogistic", "BourginMLP", "ContextDepNet", "SparseMLP")
TREE_MODELS = ("PsychForest", "BEAST_GB")
MODELS = NEURAL_MODELS + TREE_MODELS


class StableResults:
    """ Reader of the stable-protocol results of one deposit folder. """

    def __init__(self, deposit):
        self.deposit = deposit
        path = os.path.join(deposit, "trials.csv")
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found: --deposit must point to the data deposit")
        trials = pd.read_csv(path, float_precision="round_trip", low_memory=False)
        if "protocol" not in trials or not (trials.protocol == "stable").any():
            raise SystemExit(f"{deposit} holds no trials of the stable protocol (version 1.1 of the deposit "
                             f"or later is needed)")
        trials["condition"] = trials.condition.astype(str)
        self.trials = trials
        stable = trials[(trials.protocol == "stable") & trials.machine_id.isin(REFERENCE_MACHINES)]
        trees = trials[(trials.protocol == "original") & (trials.machine_id == "M00")
                       & (trials.experiment == "main") & trials.condition.isin(TREE_MODELS)]
        self.reference = pd.concat([stable, trees])
        self._curves = None

    # ---- Main run --------------------------------------------------------

    def mainRows(self, model):
        r = self.reference
        return r[(r.experiment == "main") & (r.condition == model)].sort_values("seed")

    def main(self, models=MODELS):
        """ MSE x 100 of the main run, one column per model, one row per seed (sorted). """
        P = pd.DataFrame({m: self.mainRows(m).set_index("seed").mse_x100 for m in models})
        if P.isna().any().any() or len(P) != 800:
            raise SystemExit("the main run of the stable protocol is not complete in the deposit")
        return P

    # ---- Ablation grids --------------------------------------------------

    def ablation(self, experiment, include_316=False):
        """
            Long frame of an ablation grid: `f1` (batch size or epsilon), `lr`, `seed`, `y`
            (MSE x 100 at the fixed epoch of the cell), `readout_epoch` and `trial_id`. The
            3.16e-4 column of the BourginMLP grid only with `include_316`.
        """
        r = self.reference
        d = r[r.experiment == experiment]
        if not include_316:
            d = d[~np.isclose(d.learning_rate, 3.16e-4)]
        f1 = "batch_size" if experiment == "ablation_bourginmlp" else "epsilon"
        out = pd.DataFrame({"f1": d[f1].astype(int).to_numpy(), "lr": d.learning_rate.to_numpy(),
                            "seed": d.seed.astype(np.int64).to_numpy(), "y": d.mse_x100.to_numpy(),
                            "readout_epoch": d.readout_epoch.astype(int).to_numpy(),
                            "trial_id": d.trial_id.to_numpy()})
        return out.sort_values(["f1", "lr", "seed"]).reset_index(drop=True)

    @staticmethod
    def cells(long):
        """ seed x (f1, lr) matrix of a long ablation frame. """
        return long.pivot_table(index="seed", columns=["f1", "lr"], values="y")

    # ---- Factorial design -----------------------------------------------

    def factorial(self, model):
        """ 35 x 35 grid of MSE x 100: initialization seeds in rows, order seeds in columns (both sorted). """
        r = self.reference
        d = r[(r.experiment == "factorial_init_order") & (r.condition == model)]
        G = d.pivot(index="init_seed", columns="order_seed", values="mse_x100")
        if G.shape != (35, 35) or G.isna().any().any():
            raise SystemExit(f"the factorial design of {model} is not complete in the deposit")
        return G

    # ---- Curves ---------------------------------------------------------

    def curves(self):
        """ {trial_id: test curve as MSE x 100} of the rows added in version 1.1 of the deposit. """
        if self._curves is None:
            self._curves = {}
            with gzip.open(os.path.join(self.deposit, "curves_added_v1.1.jsonl.gz"), "rt") as f:
                for line in f:
                    rec = json.loads(line)
                    self._curves[rec["trial_id"]] = rec["test_mse_x100"]
        return self._curves

    # ---- Other files ----------------------------------------------------

    def depositFrame(self, name):
        frame = pd.read_csv(os.path.join(self.deposit, name), float_precision="round_trip", low_memory=False)
        if "model" in frame:
            frame["model"] = frame.model.str.replace("BEAST-GB", "BEAST_GB")
        return frame

    def cvEpochs(self):
        """ {configuration: fixed number of epochs} from cv_epochs.csv of the deposit. """
        return self.depositFrame("cv_epochs.csv").set_index("config").e_full.astype(int).to_dict()
