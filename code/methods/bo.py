"""Bayesian optimisation with a BoTorch GP.

Inputs are already in the unit cube; outputs are standardised by the model. f is
minimised, so the GP models y = -f (optionally y = -log f, used by the E0 invariance
check). The kernel is built explicitly so that the ARD and isotropic variants differ
only in the number of lengthscales. The lengthscale prior is the dimension-scaled
log-normal prior that BoTorch uses by default (Hvarfner et al., 2024).

Refit schedule (amendment A4): the GP hyperparameters are refitted at every iteration
while n <= refit_after, then every refit_every iterations. Between refits the previous
hyperparameters are reused with the new data.

Diagnostics (E2): at each n in `checkpoints` the model is refitted and we record the
lengthscales and the Spearman rank correlation between the GP posterior mean and the
true f on fresh points around the incumbent. Those extra evaluations of f are not
counted in the budget and are not shown to the optimiser.
"""

import copy
import math
import warnings

import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior
from scipy.stats import qmc, spearmanr

from methods.base import Tracker

torch.set_default_dtype(torch.float64)


def build_model(train_x, train_y, ard, nu):
    d = train_x.shape[-1]
    prior = LogNormalPrior(loc=math.sqrt(2) + 0.5 * math.log(d), scale=math.sqrt(3))
    kernel = ScaleKernel(
        MaternKernel(nu=nu, ard_num_dims=d if ard else None, lengthscale_prior=prior)
    )
    return SingleTaskGP(train_x, train_y, covar_module=kernel)


def lengthscales(model, d):
    """Lengthscale vector of length d (an isotropic kernel's scalar is repeated)."""
    ls = model.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
    return np.full(d, ls[0]) if ls.size == 1 else ls


def rank_correlation(model, problem, x_inc, rng, m=200, radius=0.1):
    """Spearman correlation of the GP mean with the true f near the incumbent."""
    pts = np.clip(x_inc + radius * rng.standard_normal((m, x_inc.size)), 0.0, 1.0)
    with torch.no_grad():
        mean = model.posterior(torch.tensor(pts)).mean.numpy().ravel()
    true_f = np.array([problem(p) for p in pts])
    return float(spearmanr(mean, -true_f).statistic)    # mean models -f


def run(problem, budget, seed, acq="logei", ard=True, nu=2.5, n_init_factor=2.0,
        beta=4.0, refit_after=100, refit_every=5, log_y=False, raw_samples=512,
        num_restarts=10, checkpoints=(), **_):
    d = problem.dim
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    tracker = Tracker(problem, budget)
    bounds = torch.stack([torch.zeros(d), torch.ones(d)])

    # Initial space-filling design; these points count towards the budget.
    n_init = min(budget, max(2, int(round(n_init_factor * d))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for u in qmc.Sobol(d, scramble=True, seed=seed).random(n_init):
            tracker(u)

    saved = None            # hyperparameters from the last refit
    last_fit_n = 0
    pending = sorted(checkpoints)
    diag = {"checkpoint_n": [], "lengthscales": [], "rank_corr": []}
    while not tracker.done:
        train_x = torch.tensor(np.array(tracker.X))
        f = np.array(tracker.F)
        y = -np.log(np.maximum(f, 1e-300)) if log_y else -f
        train_y = torch.tensor(y).unsqueeze(-1)
        model = build_model(train_x, train_y, ard, nu)

        at_checkpoint = bool(pending) and tracker.n >= pending[0]
        refit = (saved is None or tracker.n <= refit_after
                 or tracker.n - last_fit_n >= refit_every or at_checkpoint)
        if refit:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
                except Exception:       # keep going with the last good hyperparameters
                    if saved is not None:
                        model.covar_module.load_state_dict(saved[0])
                        model.likelihood.load_state_dict(saved[1])
            saved = (copy.deepcopy(model.covar_module.state_dict()),
                     copy.deepcopy(model.likelihood.state_dict()))
            last_fit_n = tracker.n
        else:
            model.covar_module.load_state_dict(saved[0])
            model.likelihood.load_state_dict(saved[1])

        if at_checkpoint:
            pending.pop(0)
            x_inc = tracker.X[int(np.argmin(f))]
            diag["checkpoint_n"].append(tracker.n)
            diag["lengthscales"].append(lengthscales(model, d).tolist())
            diag["rank_corr"].append(rank_correlation(model, problem, x_inc, rng))

        if acq == "logei":
            acqf = LogExpectedImprovement(model, best_f=train_y.max())
        elif acq == "ucb":
            acqf = UpperConfidenceBound(model, beta=beta)
        else:
            raise ValueError(f"unknown acquisition {acq!r}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cand, _ = optimize_acqf(acqf, bounds, q=1, num_restarts=num_restarts,
                                    raw_samples=raw_samples)
        tracker(cand[0].numpy())

    return tracker.result(method="bo", acq=acq, ard=ard, nu=nu, n_init=n_init,
                          log_y=log_y, refit_after=refit_after, refit_every=refit_every,
                          diagnostics=diag)
