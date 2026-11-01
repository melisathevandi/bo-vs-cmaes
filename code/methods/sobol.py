"""Scrambled Sobol sampling: the "no model" reference line in every figure."""

import warnings

from scipy.stats import qmc

from methods.base import Tracker


def run(problem, budget, seed, **_):
    tracker = Tracker(problem, budget)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # scipy warns when budget is not a power of 2
        points = qmc.Sobol(problem.dim, scramble=True, seed=seed).random(budget)
    for u in points:
        tracker(u)
    return tracker.result(method="sobol")
