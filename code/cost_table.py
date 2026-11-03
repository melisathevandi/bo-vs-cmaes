"""E4: break-even evaluation cost c* between BO and CMA-ES, from the E1 logs.

    uv run python code/cost_table.py [--experiments e1_d2 e1_d3 ...]

For a target gap eps, a run "reaches" it at the first evaluation n whose best-so-far gap is <= eps.
Its total time is the method's own overhead up to n (cumulative t_method) plus n * c, where c is the
cost of one function evaluation. BO needs fewer evaluations n_BO < n_CMA but more overhead, so the two
total times are equal at

    c* = (T_BO - T_CMA) / (n_CMA - n_BO),    T = method overhead at the evaluation that reaches eps.

c* is computed per seed (same instance for both methods) when both reach eps and n_BO < n_CMA, and
summarised by its median. BO is worth its overhead when one evaluation costs more than c*.
Writes results/summary/cost_table.csv (one row per problem, d and target).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import load_results

SUMMARY = Path(__file__).resolve().parent.parent / "results" / "summary"
TARGETS = (1e-1, 1e-2, 1e-4)


def reach(F, f_star, eps):
    """Index (0-based) of the first evaluation whose best-so-far gap is <= eps, or None."""
    gap = np.minimum.accumulate(np.asarray(F, dtype=float)) - f_star
    hit = np.flatnonzero(gap <= eps)
    return int(hit[0]) if len(hit) else None


def break_even(run_bo, run_cma, eps):
    """c* for one pair of runs, or None when BO does not reach eps first (fewer evaluations)."""
    nb, nc = reach(run_bo["F"], run_bo["f_star"], eps), reach(run_cma["F"], run_cma["f_star"], eps)
    if nb is None or nc is None or nb >= nc:
        return None
    return float((run_bo["t_method"][nb] - run_cma["t_method"][nc]) / (nc - nb))


def table(df, bo="bo_ard", cma="cma", targets=TARGETS):
    rows = []
    for problem, g in df.groupby("problem"):
        by = {(r.method, r.seed): r for r in g.itertuples()}
        seeds = sorted({s for m, s in by if m == bo} & {s for m, s in by if m == cma})
        for eps in targets:
            cs, n_reach = [], 0
            for s in seeds:
                a, b = by[(bo, s)], by[(cma, s)]
                nb, nc = (reach(a.F, a.f_star, eps), reach(b.F, b.f_star, eps))
                n_reach += nb is not None
                c = break_even({"F": a.F, "f_star": a.f_star, "t_method": a.t_method},
                               {"F": b.F, "f_star": b.f_star, "t_method": b.t_method}, eps)
                if c is not None:
                    cs.append(c)
            rows.append({"problem": problem, "target_gap": eps, "n_seeds": len(seeds),
                         "bo_reaches": n_reach, "bo_first_and_defined": len(cs),
                         "median_c_star_s": float(np.median(cs)) if cs else np.nan})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", default=["e1_d2", "e1_d3", "e1_d5", "e1_d8", "e1_d10"])
    args = ap.parse_args()
    frames = []
    for exp in args.experiments:
        df = load_results(exp)
        if df.empty:
            continue
        t = table(df)
        t.insert(0, "experiment", exp)
        frames.append(t)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(SUMMARY / "cost_table.csv", index=False)
    print(out.round(4).to_string(index=False))
