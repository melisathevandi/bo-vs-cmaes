"""Invariance checks (E0).

Exact, path-by-path invariance holds for monotone transforms of f, because CMA-ES uses
only the ranking of the candidates. Rotation invariance holds only in distribution (the
same noise vector z would have to be rotated too), so it is tested statistically in the
E0 experiment script, not here. BO is not invariant to monotone transforms: the GP sees
the values, so f and log f give different runs.
"""

import numpy as np

from methods import bo, cmaes
from problems import ellipsoid


class Transformed:
    """Problem wrapper that applies a strictly increasing function g to f."""

    def __init__(self, problem, g):
        self.problem, self.g, self.dim = problem, g, problem.dim

    def __call__(self, u):
        return self.g(self.problem(u))


def test_cmaes_exactly_invariant_to_monotone_transform():
    f = ellipsoid(5, kappa=1e2, rotated=True, seed=0)
    base = cmaes.run(f, 150, seed=3)
    for g in (np.log1p, lambda v: v ** 3, lambda v: 7.0 * v + 2.0):
        other = cmaes.run(Transformed(f, g), 150, seed=3)
        np.testing.assert_array_equal(base["X"], other["X"])


def test_bo_is_not_invariant_to_monotone_transform():
    f = ellipsoid(2, kappa=1e2, rotated=True, seed=0)
    plain = bo.run(f, 12, seed=3, num_restarts=3, raw_samples=64)
    logged = bo.run(f, 12, seed=3, num_restarts=3, raw_samples=64, log_y=True)
    assert not np.allclose(plain["X"][4:], logged["X"][4:])   # after the Sobol start
