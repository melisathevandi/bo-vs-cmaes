"""E0: is CMA-ES rotation-invariant in practice, with and without the box?

For each condition number and seed, CMA-ES is run on three instances:
  aligned           ellipsoid with the axes along the coordinates
  rotated           the same ellipsoid (same optimum x*) after a random rotation
  aligned replicate a second, independent aligned instance (different optimum)
and compare final log10 gaps, paired by seed.

  Delta_rot = gap(rotated)  - gap(aligned)            effect of rotation
  Delta_rep = gap(replicate) - gap(aligned)           noise floor (same kind of instance)

Rotation invariance holds in distribution, so Delta_rot should look like Delta_rep when the
box does not interfere. "bounded" runs use the box [0,1]^d as in all main experiments
(methods/cmaes.py); "unbounded" runs use plain pycma with no bounds on the same start.

Writes results/summary/e0_rotation.csv.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import warnings
from pathlib import Path

import cma
import numpy as np
import pandas as pd

from analysis import bootstrap_ci, log_gap
from methods import cmaes
from methods.base import method_seed
from problems import ellipsoid

OUT = Path(__file__).resolve().parent.parent / "results" / "summary" / "e0_rotation.csv"
D, BUDGET, SEEDS, REPLICATE_OFFSET = 10, 1000, range(50), 5000


def run_unbounded(f, budget, seed, sigma0=0.3):
    """Plain pycma without bounds, same start distribution as the bounded wrapper."""
    rng = np.random.default_rng(seed)
    es = cma.CMAEvolutionStrategy(rng.uniform(0, 1, f.dim), sigma0,
                                  {"seed": int(rng.integers(1, 2**31 - 1)), "verbose": -9})
    values = []
    while len(values) < budget and not es.stop():
        cand = es.ask()
        fs = [f(x) for x in cand[: budget - len(values)]]
        values += fs
        if len(fs) < len(cand):
            break
        es.tell(cand, fs)
    return np.array(values)


def final_gap(mode, f, seed):
    seed = method_seed(seed)      # independent of the instance seed (see methods/base.py)
    if mode == "bounded":
        F = cmaes.run(f, BUDGET, seed)["F"]
    else:
        F = run_unbounded(f, BUDGET, seed)
    return log_gap(F, f.f_star, BUDGET)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    rows = []
    for kappa in (1e2, 1e4):
        for mode in ("bounded", "unbounded"):
            g = {"aligned": [], "rotated": [], "replicate": []}
            for s in SEEDS:
                g["aligned"].append(final_gap(mode, ellipsoid(D, kappa, False, s), s))
                g["rotated"].append(final_gap(mode, ellipsoid(D, kappa, True, s), s))
                g["replicate"].append(
                    final_gap(mode, ellipsoid(D, kappa, False, s + REPLICATE_OFFSET), s))
            g = {k: np.array(v) for k, v in g.items()}
            for name, diff in (("delta_rot", g["rotated"] - g["aligned"]),
                               ("delta_rep", g["replicate"] - g["aligned"])):
                med, lo, hi = bootstrap_ci(diff)
                rows.append({"kappa": kappa, "mode": mode, "quantity": name, "n_pairs": len(diff),
                             "median": med, "ci_lo": lo, "ci_hi": hi,
                             "median_gap_aligned": float(np.median(g["aligned"])),
                             "median_gap_rotated": float(np.median(g["rotated"]))})
                print(rows[-1], flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
