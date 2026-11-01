"""Instance randomness and method randomness must come from independent streams."""

import numpy as np

from methods.base import method_seed
from problems import ellipsoid


def test_method_seed_is_deterministic_and_distinct():
    assert method_seed(3) == method_seed(3)
    assert len({method_seed(s) for s in range(1000)}) == 1000


def test_method_start_is_independent_of_optimum_location():
    """Using the raw seed would make x0 a function of x*; method_seed must break that."""
    xstar, x0_raw, x0_method = [], [], []
    for s in range(300):
        xstar.append(ellipsoid(5, seed=s).u_star[0])
        x0_raw.append(np.random.default_rng(s).uniform(0, 1, 5)[0])
        x0_method.append(np.random.default_rng(method_seed(s)).uniform(0, 1, 5)[0])
    assert np.corrcoef(xstar, x0_raw)[0, 1] > 0.99          # the flaw this guards against
    assert abs(np.corrcoef(xstar, x0_method)[0, 1]) < 0.2
