"""E1 analysis as fixed in analysis_addendum_e1.md (regime map, H1 statistic, c* by median target, H4, overheads).

    uv run python code/cost_table.py          # per-seed c* and censored seeds (own target + absolute levels)
    uv run python code/analyze_e1.py          # this file

Writes results/summary/e1_regime_map.csv, e1_h1.csv, e1_cost_median_target.csv, e1_h4.csv, e1_overhead.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis import bootstrap_ci, gap_table, load_results, paired_difference
from cost_table import _d, break_even, final_gap, reach

SUMMARY = Path(__file__).resolve().parent.parent / "results" / "summary"
DIMS = (2, 3, 5, 8, 10)
BUDGET = {2: 200, 3: 200, 5: 400, 8: 640, 10: 1000}
GRID_K = (10, 20, 40, 60)               # evaluations per dimension (addendum)
PROBLEMS = ("ell_k100_rotated", "rosenbrock")


def regime_map():
    rows, diffs = [], {}
    for d in DIMS:
        df = load_results(f"e1_d{d}")
        for n in [k * d for k in GRID_K] + [BUDGET[d]]:
            t = gap_table(df, n)
            for prob in PROBLEMS:
                for other in ("cma", "sobol"):
                    r = paired_difference(t, "bo_ard", other, prob)
                    rows.append({"d": d, "problem": prob, "n": n, "n_over_d": n / d,
                                 "final_budget": n == BUDGET[d], "comparison": f"bo_ard-{other}", **r})
                if n == 20 * d:
                    diffs[(d, prob)] = (t.loc[prob]["bo_ard"] - t.loc[prob]["cma"]).dropna().to_numpy()
    return pd.DataFrame(rows), diffs


def h1(diffs):
    rows = []
    for prob in PROBLEMS:
        for d in DIMS:
            med, lo, hi = bootstrap_ci(diffs[(d, prob)])
            rows.append({"problem": prob, "statistic": f"D(d={d})", "estimate": med, "ci_lo": lo, "ci_hi": hi})
        a, b = diffs[(10, prob)], diffs[(2, prob)]
        rng = np.random.default_rng(0)
        boot = np.median(a[rng.integers(0, len(a), (10_000, len(a)))], axis=1) \
            - np.median(b[rng.integers(0, len(b), (10_000, len(b)))], axis=1)
        s = float(np.median(a) - np.median(b))
        lo, hi = np.quantile(boot, [0.025, 0.975])
        verdict = "S>=0.5: size supported" if s >= 0.5 else ("S<=0: refuted" if s <= 0 else "0<S<0.5: direction only")
        rows.append({"problem": prob, "statistic": "S=D(10)-D(2)", "estimate": s, "ci_lo": float(lo),
                     "ci_hi": float(hi), "verdict_rule": verdict})
    return pd.DataFrame(rows)


def cost_median_target():
    """Addendum target (ii): median over seeds of BO's final gap at the E1 budget."""
    rows = []
    for d in DIMS:
        bo, lg = load_results(f"e1_d{d}"), load_results(f"e1_d{d}_cma_long")
        for prob in PROBLEMS:
            B = {r.seed: r for r in bo[(bo.problem == prob) & (bo.method == "bo_ard")].itertuples()}
            C = {r.seed: r for r in lg[(lg.problem == prob) & (lg.method == "cma")].itertuples()}
            seeds = sorted(set(B) & set(C))
            eps = float(np.median([final_gap(B[s], BUDGET[d]) for s in seeds]))
            nb = {s: reach(B[s].F, B[s].f_star, eps) for s in seeds}
            nc = {s: reach(C[s].F, C[s].f_star, eps) for s in seeds}
            cs = [c for s in seeds if (c := break_even(_d(B[s]), _d(C[s]), eps)) is not None]
            q = np.quantile(cs, [0.25, 0.5, 0.75]) if cs else [np.nan] * 3
            rows.append({"d": d, "N": BUDGET[d], "problem": prob, "target_gap": eps, "n_seeds": len(seeds),
                         "bo_reaches": sum(v is not None for v in nb.values()),
                         "cma_reaches": sum(v is not None for v in nc.values()),
                         "median_n_bo": float(np.median([v + 1 for v in nb.values() if v is not None])),
                         "median_n_cma": float(np.median([v + 1 for v in nc.values() if v is not None]))
                         if any(v is not None for v in nc.values()) else np.nan,
                         "c_star_defined": len(cs), "median_c_star_s": q[1], "q1_c_star_s": q[0], "q3_c_star_s": q[2]})
    return pd.DataFrame(rows)


def h4(cost):
    ct = pd.read_csv(SUMMARY / "cost_table.csv")
    ct = ct[ct.target == "own"].copy()
    ct["d"] = ct.experiment.str.extract(r"e1_d(\d+)").astype(int)
    ct["N"] = ct.d.map(BUDGET)
    rows = []
    for source, tab in (("own target (per seed)", ct), ("median target", cost)):
        for prob in PROBLEMS:
            t = tab[tab.problem == prob].sort_values("N")
            rho = spearmanr(t.N, t.median_c_star_s).statistic
            rows.append({"source": source, "problem": prob,
                         "median_c_star_by_N": ";".join(f"{n}:{c:.3f}" for n, c in zip(t.N, t.median_c_star_s)),
                         "defined_seeds_by_N": ";".join(str(int(x)) for x in t.c_star_defined),
                         "spearman_c_star_vs_N": float(rho),
                         "sign": "increases" if rho > 0 else ("decreases" if rho < 0 else "flat")})
    return pd.DataFrame(rows)


def overhead():
    rows = []
    for d in DIMS:
        df = load_results(f"e1_d{d}")
        for (prob, m), g in df.groupby(["problem", "method"]):
            per = [r.t_method[-1] / len(r.F) for r in g.itertuples()]
            rows.append({"d": d, "problem": prob, "method": m, "n_runs": len(g),
                         "median_overhead_s_per_eval": float(np.median(per)),
                         "q1": float(np.quantile(per, 0.25)), "q3": float(np.quantile(per, 0.75))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    rm, diffs = regime_map()
    rm.round(6).to_csv(SUMMARY / "e1_regime_map.csv", index=False)
    h1(diffs).round(6).to_csv(SUMMARY / "e1_h1.csv", index=False)
    cm = cost_median_target()
    cm.round(6).to_csv(SUMMARY / "e1_cost_median_target.csv", index=False)
    h4(cm).round(6).to_csv(SUMMARY / "e1_h4.csv", index=False)
    overhead().round(9).to_csv(SUMMARY / "e1_overhead.csv", index=False)
    print("wrote e1_regime_map, e1_h1, e1_cost_median_target, e1_h4, e1_overhead")
