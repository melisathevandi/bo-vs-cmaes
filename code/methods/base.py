"""Shared bookkeeping for every optimiser.

Each method exposes run(problem, budget, seed, **cfg) -> dict and evaluates the problem
only through a Tracker. The Tracker enforces the budget, records every evaluation, and
splits wall-clock time into time spent evaluating f and time spent by the method itself.
"""

import time

import numpy as np


class Tracker:
    def __init__(self, problem, budget):
        self.problem = problem
        self.budget = budget
        self.X, self.F = [], []
        self.t_method = []          # cumulative method overhead (s) after each evaluation
        self.t_eval = 0.0           # total time spent inside problem(u)
        self._start = time.perf_counter()

    @property
    def n(self):
        return len(self.F)

    @property
    def done(self):
        return self.n >= self.budget

    def __call__(self, u):
        """Evaluate f at u (clipped to the unit cube) and record the result."""
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        t = time.perf_counter()
        f = self.problem(u)
        self.t_eval += time.perf_counter() - t
        self.X.append(u)
        self.F.append(f)
        self.t_method.append(time.perf_counter() - self._start - self.t_eval)
        return f

    def result(self, **meta):
        """Trajectory in plain arrays. The best-so-far point is the recommendation."""
        return {
            "X": np.array(self.X),
            "F": np.array(self.F),
            "t_method": np.array(self.t_method),
            "t_eval": self.t_eval,
            "device": "cpu",
            "meta": meta,
        }
