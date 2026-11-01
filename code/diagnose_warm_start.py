"""Why is a warm-started GP refit worse on the rotated ellipsoid? (follow-up to A4)

Take the evaluated points of a real cold-start BO run and replay the fits: for every n,
fit the GP from default values (cold) and along a chain that starts each fit at the
previous fit's hyperparameters (warm). Compare the maximised marginal log-likelihood.
If warm lands in a worse optimum, its log-likelihood is lower than cold's at the same data.

Writes results/summary/a4_warm_start_diagnostic.csv.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from methods.bo import build_model, fit, lengthscales, marginal_log_likelihood
from runner import RAW

OUT = Path(__file__).resolve().parent.parent / "results" / "summary" / "a4_warm_start_diagnostic.csv"
SEEDS, N_START, N_END = [1000, 1001, 1002, 1003], 12, 100


def load_run(seed):
    for path in (RAW / "a4_refit_check").glob("*.json"):
        r = json.loads(path.read_text())
        s = r["spec"]
        if (s["method"]["name"] == "bo_cold_every5" and s["seed"] == seed
                and s["problem"]["name"] == "ell_d5_k100_rotated"):
            return np.array(r["X"]), np.array(r["F"])
    raise FileNotFoundError(seed)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    rows = []
    for seed in SEEDS:
        X, F = load_run(seed)
        d, saved = X.shape[1], None
        for n in range(N_START, N_END + 1):
            x, y = torch.tensor(X[:n]), torch.tensor(-F[:n]).unsqueeze(-1)
            cold = build_model(x, y, ard=True, nu=2.5)
            fit_gpytorch_mll(ExactMarginalLogLikelihood(cold.likelihood, cold))
            warm = build_model(x, y, ard=True, nu=2.5)
            saved = fit(warm, saved, warm=True)           # chain of warm fits
            rows.append({"seed": seed, "n": n, "mll_cold": marginal_log_likelihood(cold),
                         "mll_warm": marginal_log_likelihood(warm),
                         "ls_min_cold": lengthscales(cold, d).min(),
                         "ls_max_cold": lengthscales(cold, d).max(),
                         "ls_min_warm": lengthscales(warm, d).min(),
                         "ls_max_warm": lengthscales(warm, d).max()})
    df = pd.DataFrame(rows)
    df["mll_gap"] = df.mll_warm - df.mll_cold          # negative: warm is a worse optimum
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    for seed, g in df.groupby("seed"):
        last = g[g.n == N_END].iloc[0]
        print(f"seed {seed}: warm worse (gap < -0.5) at {(g.mll_gap < -0.5).mean():.0%} of n; "
              f"median gap {g.mll_gap.median():+.2f}; at n={N_END}: "
              f"mll cold {last.mll_cold:.1f} warm {last.mll_warm:.1f}; "
              f"lengthscale range cold [{last.ls_min_cold:.2f}, {last.ls_max_cold:.2f}] "
              f"warm [{last.ls_min_warm:.2f}, {last.ls_max_warm:.2f}]")
