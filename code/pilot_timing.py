"""Timing pilot (amendment A2, CPU only).

For each (d, n) build a GP on n Sobol points of a rotated ellipsoid and time, with one
thread: (a) one hyperparameter fit from default initial values (cold), (b) one refit
after 5 more points, warm-started from the previous hyperparameters, (c) one LogEI
acquisition optimisation. The BO wrapper refits every iteration while n <= 100 and every
5 iterations afterwards, so the average cost per iteration at larger n is
t_acq + t_fit / 5. Three instance seeds per cell.

Writes results/summary/pilot_timing.csv.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import qmc

from methods.bo import build_model
from problems import ellipsoid

torch.set_num_threads(1)
warnings.simplefilter("ignore")

OUT = Path(__file__).resolve().parent.parent / "results" / "summary" / "pilot_timing.csv"
DIMS = [2, 10, 20]
SIZES = [100, 300, 600, 1000]


SEEDS = [0, 1, 2]
REFIT_EVERY = 5


def time_one(d, n, seed):
    """Time a cold fit, a warm-started refit after 5 new points, and one acquisition."""
    f = ellipsoid(d, kappa=1e4, rotated=True, seed=seed)
    x = qmc.Sobol(d, scramble=True, seed=seed).random(n + REFIT_EVERY)
    y = -np.array([f(u) for u in x])

    def make(m):
        return build_model(torch.tensor(x[:m]), torch.tensor(y[:m]).unsqueeze(-1),
                           ard=True, nu=2.5)

    model = make(n)
    t = time.perf_counter()
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    t_fit_cold = time.perf_counter() - t

    bounds = torch.stack([torch.zeros(d), torch.ones(d)])
    t = time.perf_counter()
    optimize_acqf(LogExpectedImprovement(model, best_f=torch.tensor(y[:n]).max()), bounds,
                  q=1, num_restarts=10, raw_samples=512)
    t_acq = time.perf_counter() - t

    warm = make(n + REFIT_EVERY)                  # same data plus 5 new points
    warm.covar_module.load_state_dict(model.covar_module.state_dict())
    warm.likelihood.load_state_dict(model.likelihood.state_dict())
    t = time.perf_counter()
    fit_gpytorch_mll(ExactMarginalLogLikelihood(warm.likelihood, warm))
    t_fit_warm = time.perf_counter() - t
    return t_fit_cold, t_fit_warm, t_acq


if __name__ == "__main__":
    rows = []
    for d in DIMS:
        for n in SIZES:
            for seed in SEEDS:
                cold, warm, acq = time_one(d, n, seed)
                rows.append({"d": d, "n": n, "seed": seed, "t_fit_cold_s": cold,
                             "t_fit_warm_s": warm, "t_acq_s": acq,
                             "t_iter_cold_s": acq + cold / REFIT_EVERY,
                             "t_iter_warm_s": acq + warm / REFIT_EVERY, "device": "cpu"})
                print(rows[-1], flush=True)
                OUT.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(OUT, index=False)
