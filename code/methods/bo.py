"""Bayesian optimisation with a BoTorch GP.

Inputs are already in the unit cube; outputs are standardised by the model. f is
minimised, so the GP models y = -f (optionally y = -log f, used by the E0 invariance
check). The kernel is built explicitly so that the ARD and isotropic variants differ
only in the number of lengthscales. The lengthscale prior is the dimension-scaled
log-normal prior that BoTorch uses by default (Hvarfner et al., 2024).

Refit schedule (amendment A4): the GP hyperparameters are refitted at every iteration
while n <= refit_after, then every refit_every iterations. Between refits the previous
hyperparameters are reused with the new data. Each refit starts from BoTorch's defaults
(cold); warm-started refits are optional (warm_start=True) but changed results, see A4.

Diagnostics (E2): at each n in `checkpoints` a cold fit of the current data is recorded
(lengthscales, marginal likelihood) together with the Spearman rank correlation between
the GP posterior mean and the true f on fresh points around the incumbent. With
multistart=K, K further fits from random starting lengthscales are recorded to show how
stable the lengthscales (and quantities derived from them) are. The extra evaluations of f
are not counted in the budget and are not shown to the optimiser, and diagnostics never
change the optimiser's path (they use their own fits and restore the torch random state).
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


def fit(model, saved, warm=False):
    """Fit hyperparameters by maximum marginal likelihood.

    Cold start (default): the optimiser begins at BoTorch's default values. With warm=True
    it begins at the previously saved hyperparameters, which is about 3x cheaper but gave
    worse results on the rotated ellipsoid (check A4, results/summary/a4_refit_check.csv),
    so it is off. If the optimiser fails, the saved values are kept. Returns the (kernel,
    likelihood) state to reuse at the iterations between refits.
    """
    if saved is not None and warm:
        load(model, saved)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        except Exception:
            if saved is not None:
                load(model, saved)
    return (copy.deepcopy(model.covar_module.state_dict()),
            copy.deepcopy(model.likelihood.state_dict()))


def load(model, saved):
    """Reuse saved hyperparameters with the model's current data."""
    model.covar_module.load_state_dict(saved[0])
    model.likelihood.load_state_dict(saved[1])


def lengthscales(model, d):
    """Lengthscale vector of length d (an isotropic kernel's scalar is repeated)."""
    ls = model.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
    return np.full(d, ls[0]) if ls.size == 1 else ls


def marginal_log_likelihood(model):
    """Marginal log-likelihood of the model's training data at its current hyperparameters."""
    model.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with torch.no_grad():
        return float(mll(model(*model.train_inputs), model.train_targets))


def multistart_fits(train_x, train_y, ard, nu, n_starts, rng):
    """Fit the GP from several random starting lengthscales and return every result.

    The marginal likelihood can be flat along a ridge (several lengthscale settings with
    the same likelihood), so one fit does not show how stable a quantity such as the
    relevance ratio is. Starting lengthscales are log-uniform in [0.1, 10] (unit-cube
    units). Returns a list of {"mll": ..., "lengthscales": [...]}; failed fits are skipped.
    """
    d = train_x.shape[-1]
    out = []
    for _ in range(n_starts):
        model = build_model(train_x, train_y, ard, nu)
        base = model.covar_module.base_kernel
        start = np.exp(rng.uniform(np.log(0.1), np.log(10.0), size=base.lengthscale.shape))
        base.initialize(lengthscale=torch.tensor(start))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            except Exception:
                continue
        out.append({"mll": marginal_log_likelihood(model),
                    "lengthscales": lengthscales(model, d).tolist()})
    return out


def rank_correlation(model, problem, x_inc, rng, m=200, radius=0.1):
    """Spearman correlation of the GP mean with the true f near the incumbent."""
    pts = np.clip(x_inc + radius * rng.standard_normal((m, x_inc.size)), 0.0, 1.0)
    with torch.no_grad():
        mean = model.posterior(torch.tensor(pts)).mean.numpy().ravel()
    true_f = np.array([problem(p) for p in pts])
    return float(spearmanr(mean, -true_f).statistic)    # mean models -f


def run(problem, budget, seed, acq="logei", ard=True, nu=2.5, n_init_factor=2.0,
        beta=4.0, refit_after=100, refit_every=5, warm_start=False, log_y=False,
        raw_samples=512, num_restarts=10, checkpoints=(), multistart=0, **_):
    if any(c >= budget for c in checkpoints):
        # Diagnostics are taken before choosing the next point, so a checkpoint at or beyond
        # the budget would be skipped silently.
        raise ValueError("every checkpoint must be smaller than the budget")
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
    diag = {"checkpoint_n": [], "lengthscales": [], "mll": [], "rank_corr": [],
            "multistart": []}
    while not tracker.done:
        train_x = torch.tensor(np.array(tracker.X))
        f = np.array(tracker.F)
        y = -np.log(np.maximum(f, 1e-300)) if log_y else -f
        train_y = torch.tensor(y).unsqueeze(-1)
        model = build_model(train_x, train_y, ard, nu)

        at_checkpoint = bool(pending) and tracker.n >= pending[0]
        refit = (saved is None or tracker.n <= refit_after
                 or tracker.n - last_fit_n >= refit_every)
        if refit:
            saved = fit(model, saved, warm_start)
            last_fit_n = tracker.n
        else:
            load(model, saved)

        if at_checkpoint:
            # Diagnostics only observe: they use their own cold fit of the current data and
            # restore the torch random state, so the optimiser's path is unchanged.
            pending.pop(0)
            torch_state = torch.get_rng_state()
            diag_model = model
            if not refit or warm_start:
                diag_model = build_model(train_x, train_y, ard, nu)
                fit(diag_model, None)
            x_inc = tracker.X[int(np.argmin(f))]
            diag["checkpoint_n"].append(tracker.n)
            diag["lengthscales"].append(lengthscales(diag_model, d).tolist())
            diag["mll"].append(marginal_log_likelihood(diag_model))
            diag["rank_corr"].append(rank_correlation(diag_model, problem, x_inc, rng))
            diag["multistart"].append(
                multistart_fits(train_x, train_y, ard, nu, multistart, rng))
            torch.set_rng_state(torch_state)

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
                          warm_start=warm_start, diagnostics=diag)
