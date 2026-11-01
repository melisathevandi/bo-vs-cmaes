"""Correctness anchor (amendment A8a): do the wrappers behave like the reference?

1. CMA-ES wrapper vs the library: on the same bounded problem, start and seed, the
   wrapper must evaluate exactly the same points as `cma.fmin2`.
2. BO on a standard benchmark: Branin (d=2, f* = 0.397887) is a smooth, low-dimensional
   function that a working GP-BO should solve in a few dozen evaluations. The pass
   threshold (median gap < 0.01 after 40 evaluations) was fixed before running; Sobol is
   shown as the "no model" reference.

Writes results/summary/anchor.csv and exits non-zero if a check fails.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import warnings
from pathlib import Path

import cma
import numpy as np
import pandas as pd

from methods import bo, cmaes, sobol, turbo
from methods.base import method_seed
from problems import ellipsoid

OUT = Path(__file__).resolve().parent.parent / "results" / "summary" / "anchor.csv"
BRANIN_MIN = 0.397887


class Branin:
    """Branin on the unit cube: x1 = -5 + 15 u1, x2 = 15 u2."""
    dim, f_star = 2, BRANIN_MIN

    def __call__(self, u):
        x1, x2 = -5.0 + 15.0 * u[0], 15.0 * u[1]
        a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5.0 / np.pi
        r, s, t = 6.0, 10.0, 1.0 / (8 * np.pi)
        return float(a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s)


def check_cma_matches_library(n_seeds=10, budget=200):
    ok = []
    for s in range(n_seeds):
        f = ellipsoid(8, kappa=1e3, rotated=True, seed=s)
        seed = method_seed(s)
        ours = cmaes.run(f, budget, seed)["F"]

        # Reproduce the wrapper's random draws, then hand the same start to the library.
        rng = np.random.default_rng(seed)
        opts = {"bounds": [0, 1], "seed": int(rng.integers(1, 2**31 - 1)),
                "verbose": -9, "popsize_factor": 1.0}
        x0 = rng.uniform(0, 1, f.dim)
        seen = []
        cma.fmin2(lambda x: seen.append(f(np.clip(x, 0, 1))) or seen[-1], x0, 0.3, opts)
        m = min(len(ours), len(seen), budget)
        ok.append(bool(np.array_equal(ours[:m], np.array(seen)[:m])) and m >= 100)
    return {"check": "cma_wrapper_equals_library", "passed": int(sum(ok)), "total": n_seeds}


def branin_gap(method, budget, seeds=range(10), **cfg):
    gaps = [np.min(method.run(Branin(), budget, method_seed(s), **cfg)["F"]) - BRANIN_MIN
            for s in seeds]
    return float(np.median(gaps)), float(np.max(gaps))


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    rows = [check_cma_matches_library()]
    rows[0]["pass"] = rows[0]["passed"] == rows[0]["total"]
    for name, method, budget, cfg in [("bo_ard_logei", bo, 40, {}),
                                      ("turbo", turbo, 40, {"n_candidates": 500}),
                                      ("sobol", sobol, 40, {}), ("sobol", sobol, 400, {})]:
        med, worst = branin_gap(method, budget, **cfg)
        rows.append({"check": f"branin_{name}_n{budget}", "median_gap": med, "max_gap": worst,
                     "pass": (med < 0.01) if name != "sobol" else None})
    for r in rows:
        print(r, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    sys.exit(0 if all(r["pass"] in (True, None) for r in rows) else 1)
