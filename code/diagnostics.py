"""Diagnostics computed from the quantities BO records at its checkpoints.

Relevance ratio. An ARD kernel measures distance with M = diag(1 / l_i^2): a large entry
means the GP believes f changes quickly along coordinate i. For a function that depends on
x only through a d_eff-dimensional subspace, with orthogonal projector P onto that subspace,

    rho = tr(P M) / tr(M) = sum_i P_ii / l_i^2  /  sum_i 1 / l_i^2.

rho is the share of the kernel's total sensitivity that falls in the relevant directions.
An isotropic kernel gives exactly tr(P) / d = d_eff / d. A diagonal M can only reach a
rho above d_eff / d when coordinate axes line up with the relevant subspace.

Stability. The marginal likelihood can be flat along a ridge, so different fits can have
the same likelihood and different lengthscales. rho_stability reports rho for the main
fit and for every multi-start fit at a checkpoint, and the spread (max - min) over all
fits and over the near-optimal ones (marginal likelihood within `tol` nats of the best).
"""

import numpy as np
import pandas as pd


def relevance_ratio(lengthscales, P):
    """rho = tr(P M) / tr(M) for M = diag(1 / lengthscales^2); P is d x d."""
    m = 1.0 / np.asarray(lengthscales, dtype=float) ** 2
    return float(np.sum(np.diag(P) * m) / np.sum(m))


def rho_stability(diag, P, tol=1.0):
    """One row per checkpoint: rho of the main fit and its spread across starting points."""
    rows = []
    for k, n in enumerate(diag["checkpoint_n"]):
        fits = [{"mll": diag["mll"][k], "lengthscales": diag["lengthscales"][k]}]
        fits += diag["multistart"][k]
        rho = np.array([relevance_ratio(f["lengthscales"], P) for f in fits])
        mll = np.array([f["mll"] for f in fits])
        near = mll >= mll.max() - tol
        rows.append({"n": n, "rho_main": rho[0], "n_fits": len(fits), "n_near_optimal": int(near.sum()),
                     "rho_min": rho.min(), "rho_max": rho.max(), "spread_all": rho.max() - rho.min(),
                     "spread_near_optimal": rho[near].max() - rho[near].min()})
    return pd.DataFrame(rows)
