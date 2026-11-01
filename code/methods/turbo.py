"""TuRBO-1: Bayesian optimisation inside a single adaptive trust region.

Follows Eriksson et al. (2019) and BoTorch's TuRBO-1 tutorial:
  * a GP with an ARD Matern kernel is fitted to the data of the current run;
  * the trust region is a box around the best point, with side length L scaled per
    coordinate by the ARD lengthscales (long lengthscale -> wider side);
  * the next point is a Thompson sample over Sobol candidates in the box, where each
    candidate only perturbs a random subset of coordinates (probability min(20/d, 1));
  * L doubles after `succ_tol` consecutive successes and halves after `fail_tol`
    consecutive failures; when L < L_MIN the run restarts from a fresh Sobol design.
Tolerances follow the tutorial's defaults (succ_tol=10, fail_tol=max(4, d)). The GP uses
the same kernel, prior and refit schedule as methods/bo.py.
"""

import math
import warnings

import numpy as np
import torch
from scipy.stats import qmc

from methods.base import Tracker
from methods.bo import build_model, fit, load

L_INIT, L_MIN, L_MAX = 0.8, 0.5**7, 1.6


def trust_region(model, x_center, length, d):
    ls = model.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
    w = ls / ls.mean()
    w = w / np.prod(w ** (1.0 / d))             # keep the volume independent of the weights
    lb = np.clip(x_center - w * length / 2.0, 0.0, 1.0)
    ub = np.clip(x_center + w * length / 2.0, 0.0, 1.0)
    return lb, ub


def run(problem, budget, seed, nu=2.5, n_init_factor=2.0, refit_after=100, refit_every=5,
        n_candidates=None, succ_tol=10, fail_tol=None, **_):
    d = problem.dim
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    tracker = Tracker(problem, budget)
    n_init = max(2, int(round(n_init_factor * d)))
    n_candidates = n_candidates or min(5000, max(2000, 200 * d))
    fail_tol = fail_tol or max(4, d)
    prob_perturb = min(20.0 / d, 1.0)
    sobol_seed = int(rng.integers(0, 2**31 - 1))
    restarts = 0

    while not tracker.done:
        # (Re)start: fresh space-filling design and a fresh trust region.
        start = tracker.n
        length, succ, fail, saved, last_fit_n = L_INIT, 0, 0, None, 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            init = qmc.Sobol(d, scramble=True, seed=sobol_seed + restarts).random(n_init)
        for u in init:
            if tracker.done:
                break
            tracker(u)

        while not tracker.done and length >= L_MIN:
            x = np.array(tracker.X[start:])
            f = np.array(tracker.F[start:])
            model = build_model(torch.tensor(x), torch.tensor(-f).unsqueeze(-1),
                                ard=True, nu=nu)
            if saved is None or len(f) <= refit_after or len(f) - last_fit_n >= refit_every:
                saved = fit(model, saved)
                last_fit_n = len(f)
            else:
                load(model, saved)

            center = x[int(np.argmin(f))]
            lb, ub = trust_region(model, center, length, d)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                u = qmc.Sobol(d, scramble=True, seed=int(rng.integers(0, 2**31 - 1))
                              ).random(n_candidates)
            cand = lb + (ub - lb) * u
            mask = rng.random((n_candidates, d)) <= prob_perturb
            mask[~mask.any(axis=1), rng.integers(0, d, size=(~mask.any(axis=1)).sum())] = True
            cand = np.where(mask, cand, center)

            with torch.no_grad():
                sample = model.posterior(torch.tensor(cand)).rsample().squeeze()
            f_new = tracker(cand[int(torch.argmax(sample))])

            # Success if the new value improves the run's best by a small margin.
            if f_new < f.min() - 1e-3 * abs(f.min()):
                succ, fail = succ + 1, 0
            else:
                succ, fail = 0, fail + 1
            if succ == succ_tol:
                length, succ = min(2.0 * length, L_MAX), 0
            elif fail == fail_tol:
                length, fail = length / 2.0, 0
        restarts += 1

    return tracker.result(method="turbo", nu=nu, n_init=n_init, restarts=restarts - 1,
                          n_candidates=n_candidates)
