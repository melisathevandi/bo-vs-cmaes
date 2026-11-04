"""E4: break-even evaluation cost c* between BO and CMA-ES, from the E1 logs.

    uv run python code/cost_table.py [--experiments e1_d2 e1_d3 ...]

For a target gap eps, a run "reaches" it at the first evaluation n whose best-so-far gap is <= eps.
Its total time is the method's own overhead up to n (cumulative t_method) plus n * c, where c is the
cost of one function evaluation. BO needs fewer evaluations n_BO < n_CMA but more overhead, so the two
total times are equal at

    c* = (T_BO - T_CMA) / (n_CMA - n_BO),    T = method overhead at the evaluation that reaches eps.

c* is computed per seed (same instance for both methods) when both reach eps and n_BO < n_CMA, and
summarised by its median. BO is worth its overhead when one evaluation costs more than c*.
The CMA-ES side is read from the supplementary long runs `<experiment>_cma_long` (analysis_addendum_e1_cma_long.md)
when they exist, so CMA-ES can run past the BO budget; otherwise from the CMA-ES runs of the experiment itself (column
cma_source). Targets (addenda, unchanged): "own" = each BO run's own final gap at its budget (for c*), and the absolute
gap levels TARGETS. Writes results/summary/cost_table.csv (one row per problem and target, with the CMA-ES seeds that
never reach the target) and results/summary/cost_censored.csv (one row per such seed, with BO's final gap on it).

    uv run python code/cost_table.py --check-e2     # reproduce the E2 c* values of e2_cma_long_reach.csv
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


def final_gap(run, n):
    """Best-so-far gap of a run after its first n evaluations."""
    return float(np.min(run.F[:n]) - run.f_star)


def _d(r):
    return {"F": r.F, "f_star": r.f_star, "t_method": r.t_method}


def cstar_seeds(bo_runs, cma_runs, budget):
    """Per-seed c* with target = the BO run's own final gap at `budget`. Returns {seed: c* or None}."""
    return {s: break_even(_d(b), _d(cma_runs[s]), final_gap(b, budget)) for s, b in bo_runs.items()}


def _runs(df, problem, method):
    return {r.seed: r for r in df[(df.problem == problem) & (df.method == method)].itertuples()}


def table(bo_df, cma_df, bo="bo_ard", cma="cma", targets=TARGETS, cma_source="long"):
    rows, censored = [], []
    budget = int(bo_df["budget"].max())
    for problem in sorted(bo_df.problem.unique()):
        B, C = _runs(bo_df, problem, bo), _runs(cma_df, problem, cma)
        seeds = sorted(set(B) & set(C))
        for label, eps_of in [("own", lambda s: final_gap(B[s], budget))] + [(f"{e:g}", lambda s, e=e: e) for e in targets]:
            cs, nb_l, nc_l, cens = [], [], [], []
            for s in seeds:
                eps = eps_of(s)
                nb, nc = reach(B[s].F, B[s].f_star, eps), reach(C[s].F, C[s].f_star, eps)
                nb_l += [nb + 1] if nb is not None else []
                if nc is None:
                    cens.append(s)
                else:
                    nc_l.append(nc + 1)
                c = break_even(_d(B[s]), _d(C[s]), eps)
                if c is not None:
                    cs.append(c)
            q = np.quantile(cs, [0.25, 0.5, 0.75]) if cs else [np.nan] * 3
            rows.append({"problem": problem, "target": label, "n_seeds": len(seeds), "bo_reaches": len(nb_l),
                         "cma_reaches": len(nc_l), "cma_censored": len(cens),
                         "cma_censored_seeds": ";".join(map(str, cens)),
                         "median_n_bo": float(np.median(nb_l)) if nb_l else np.nan,
                         "median_n_cma": float(np.median(nc_l)) if nc_l else np.nan,
                         "c_star_defined": len(cs), "median_c_star_s": q[1], "q1_c_star_s": q[0],
                         "q3_c_star_s": q[2], "cma_source": cma_source})
            for s in cens:
                gaps = {e: reach(B[s].F, B[s].f_star, e) is not None for e in targets}
                censored.append({"problem": problem, "target": label, "seed": s,
                                 "cma_final_gap": final_gap(C[s], len(C[s].F)), "cma_n_evals": len(C[s].F),
                                 "cma_cause": "budget" if len(C[s].F) >= int(cma_df["budget"].max()) else "terminated",
                                 "bo_final_gap": final_gap(B[s], budget),
                                 **{f"bo_reaches_{e:g}": v for e, v in gaps.items()}})
    return pd.DataFrame(rows), pd.DataFrame(censored)


def check_e2():
    """Reproduce the E2 c* values of results/summary/e2_cma_long_reach.csv with the shared functions."""
    primary, long = load_results("e2_primary"), load_results("e2_cma_long")
    ref = pd.read_csv(SUMMARY / "e2_cma_long_reach.csv")
    ref = ref[ref.quantity == "cstar_s"].set_index(["instance", "bo_variant"])
    worst = 0.0
    for inst, prob in {"aligned": "ell_d10_k1e4_aligned", "rotated": "ell_d10_k1e4_rotated"}.items():
        for v in ("bo_ard", "bo_iso"):
            c = [x for x in cstar_seeds(_runs(primary, prob, v), _runs(long, prob, "cma"), 500).values() if x is not None]
            r = ref.loc[(inst, v)]
            diff = max(abs(np.median(c) - r["median"]), abs(np.quantile(c, 0.25) - r["q1"]),
                       abs(np.quantile(c, 0.75) - r["q3"]), abs(len(c) - r["n_reached"]))
            worst = max(worst, diff)
            print(f"{inst:8s} {v:7s} defined {len(c)}  median c* {np.median(c):.4f}  (reference {r['median']:.4f})")
    print(f"largest absolute difference to e2_cma_long_reach.csv: {worst:.2e}")
    return worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", default=["e1_d2", "e1_d3", "e1_d5", "e1_d8", "e1_d10"])
    ap.add_argument("--check-e2", action="store_true")
    args = ap.parse_args()
    if args.check_e2:
        raise SystemExit(0 if check_e2() < 1e-9 else 1)
    frames, cens = [], []
    for exp in args.experiments:
        bo_df = load_results(exp)
        if bo_df.empty:
            continue
        long = load_results(f"{exp}_cma_long")
        cma_df, src = (long, "long") if not long.empty else (bo_df, "same_budget")
        t, c = table(bo_df, cma_df, cma_source=src)
        t.insert(0, "experiment", exp)
        c.insert(0, "experiment", exp)
        frames.append(t)
        cens.append(c)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(SUMMARY / "cost_table.csv", index=False)
    pd.concat(cens, ignore_index=True).to_csv(SUMMARY / "cost_censored.csv", index=False)
    print(out.round(4).to_string(index=False))
