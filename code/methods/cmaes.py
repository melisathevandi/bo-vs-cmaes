"""CMA-ES via pycma: full covariance, separable (diagonal), and IPOP restarts.

The search runs in the unit cube with box bounds. x0 is drawn uniformly in the cube
(not the centre) and sigma0 is a fraction of the box width (default 0.3, as recommended
in Hansen's tutorial). All other settings are pycma defaults.
"""

import cma
import numpy as np

from methods.base import Tracker


def run(problem, budget, seed, sigma0=0.3, popsize_factor=1.0, diagonal=False,
        ipop=False, **_):
    d = problem.dim
    rng = np.random.default_rng(seed)
    tracker = Tracker(problem, budget)
    restarts = 0
    while not tracker.done:
        opts = {
            "bounds": [0, 1],
            "seed": int(rng.integers(1, 2**31 - 1)),
            "verbose": -9,
            # IPOP doubles the population size at every restart
            "popsize_factor": popsize_factor * 2**restarts,
        }
        if diagonal:
            opts["CMA_diagonal"] = True
        es = cma.CMAEvolutionStrategy(rng.uniform(0, 1, d), sigma0, opts)
        while not tracker.done and not es.stop():
            candidates = es.ask()
            values = []
            for x in candidates:
                if tracker.done:
                    break
                values.append(tracker(x))
            if len(values) < len(candidates):    # budget ran out mid-generation
                break
            es.tell(candidates, values)
        if not ipop:
            break                                 # plain CMA-ES stops when it terminates
        restarts += 1
    return tracker.result(method="cmaes", sigma0=sigma0, popsize_factor=popsize_factor,
                          diagonal=diagonal, ipop=ipop, restarts=restarts)
