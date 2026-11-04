"""Reach analysis of the supplementary CMA-ES long runs, as specified in analysis_addendum_e2_cma_long.md.

    uv run python code/analyze_cma_long.py

For each instance (aligned, rotated) and BO variant (ARD-BO, iso-BO):
  reach  target = median final log10 gap of that variant at n = 500 (e2_primary); n_reach = first evaluation
         (1-based) at which a CMA-ES long run's best-so-far gap is at or below the target; fraction reached,
         median and IQR of n_reach over runs that reach it, censored runs split into budget exhausted vs
         terminated by pycma before reaching it.
  cstar  break-even cost per seed (cost_table.break_even): target = that BO run's own final gap at n = 500,
         c* = (T_BO - T_CMA) / (n_CMA - n_BO) when CMA-ES needs more evaluations; median and IQR over defined seeds.
Writes results/summary/e2_cma_long_reach.csv. Descriptive only; no test. Wall-clock times of the long runs were
taken while other jobs were running (see the lab notebook).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from analysis import load_results
from cost_table import break_even, reach

SUMMARY = Path(__file__).resolve().parent.parent / "results" / "summary"
INSTANCES = {"aligned": "ell_d10_k1e4_aligned", "rotated": "ell_d10_k1e4_rotated"}
VARIANTS = ["bo_ard", "bo_iso"]
N_BO = 500


def final_gap(run, n=N_BO):
    return float(np.min(run.F[:n]) - run.f_star)


def summarise(primary, long):
    rows = []
    for inst, prob in INSTANCES.items():
        cma = {r.seed: r for r in long[long.problem == prob].itertuples()}
        for v in VARIANTS:
            bo = {r.seed: r for r in primary[(primary.problem == prob) & (primary.method == v)].itertuples()}
            target = 10 ** float(np.median([np.log10(max(final_gap(r), 1e-12)) for r in bo.values()]))
            n_reach, budget_cens, term_cens = [], 0, 0
            for r in cma.values():
                i = reach(r.F, r.f_star, target)
                if i is not None:
                    n_reach.append(i + 1)
                elif len(r.F) >= r.budget:
                    budget_cens += 1
                else:
                    term_cens += 1
            q = np.quantile(n_reach, [0.25, 0.5, 0.75]) if n_reach else [np.nan] * 3
            rows.append({"quantity": "reach", "instance": inst, "bo_variant": v, "target_gap": target,
                         "n_runs": len(cma), "n_reached": len(n_reach), "censored_budget": budget_cens,
                         "censored_terminated": term_cens, "median": q[1], "q1": q[0], "q3": q[2]})
            cs = []
            for s, b in bo.items():
                c = break_even({"F": b.F, "f_star": b.f_star, "t_method": b.t_method},
                               {"F": cma[s].F, "f_star": cma[s].f_star, "t_method": cma[s].t_method},
                               final_gap(b))
                if c is not None:
                    cs.append(c)
            q = np.quantile(cs, [0.25, 0.5, 0.75]) if cs else [np.nan] * 3
            rows.append({"quantity": "cstar_s", "instance": inst, "bo_variant": v, "target_gap": np.nan,
                         "n_runs": len(bo), "n_reached": len(cs), "censored_budget": np.nan,
                         "censored_terminated": np.nan, "median": q[1], "q1": q[0], "q3": q[2]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    primary = load_results("e2_primary")
    long = load_results("e2_cma_long")
    long["budget"] = long["budget"].astype(int)
    out = summarise(primary, long)
    out.to_csv(SUMMARY / "e2_cma_long_reach.csv", index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
