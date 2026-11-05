"""E2 low-effective dimension: relevance ratio rho and GP rank correlation against the H2 thresholds (predictions.md).

    uv run python code/analyze_lowdim.py

At the checkpoint n = 200 (one value per run, 20 seeds per instance): rho of the main cold fit of ARD-BO, the GP rank
correlation of ARD-BO and iso-BO, the median and IQR over seeds, the fraction of runs on the right side of the threshold
(rho >= 0.5 aligned, rho <= 0.2 rotated; isotropic baseline d_eff/d = 0.1), the spread of rho across the 5 multi-start fits,
and how many lengthscales are very large. The lengthscale has no upper bound in the model (Positive() constraint, upper bound
inf), so "at the bound" is reported as the fraction of lengthscales above 10 and above 100 box widths. For the aligned
instances the irrelevant coordinates are those with diag(P) = 0; for the rotated ones no coordinate is irrelevant, so all
coordinates are counted. Writes results/summary/e2_lowdim_rho.csv (n = 200) and e2_lowdim_checkpoints.csv (all checkpoints).
Descriptive; nothing here is a test.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from analysis import load_results
from diagnostics import relevance_ratio, rho_stability
from runner import make_problem

SUMMARY = Path(__file__).resolve().parent.parent / "results" / "summary"
RAW = Path(__file__).resolve().parent.parent / "results" / "raw" / "e2_lowdim"
N_CHECK = 200
THRESHOLD = {"aligned": ("rho >= 0.5", lambda r: r >= 0.5), "rotated": ("rho <= 0.2", lambda r: r <= 0.2)}
BIG = (10.0, 100.0)


def _q(x):
    return np.quantile(x, [0.25, 0.5, 0.75]) if len(x) else [np.nan] * 3


def collect(df):
    import json
    rows = []
    for r in df[df.method.isin(["bo_ard", "bo_iso"])].itertuples():
        d = r.meta["diagnostics"]
        if not d["checkpoint_n"]:
            continue
        spec = json.loads((RAW / f"{r.job_id}.json").read_text())["spec"]
        P = make_problem(spec["problem"], spec["instance_seed"]).P
        inst = "aligned" if "aligned" in r.problem else "rotated"
        irrelevant = np.diag(P) < 1e-9
        stab = rho_stability(d, P) if r.method == "bo_ard" else None
        for i, n in enumerate(d["checkpoint_n"]):
            ls = np.array(d["lengthscales"][i])
            row = {"instance": inst, "method": r.method, "seed": r.seed, "n": n, "rank_corr": d["rank_corr"][i],
                   "rho": relevance_ratio(ls, P)}
            if stab is not None:
                s = stab.iloc[i]
                row.update(rho_spread_all=s.spread_all, rho_spread_near_optimal=s.spread_near_optimal)
                pool = ls[irrelevant] if (inst == "aligned") else ls
                row.update(max_ls=float(ls.max()), n_pool=len(pool),
                           **{f"frac_pool_gt{int(b)}": float(np.mean(pool > b)) for b in BIG},
                           **{f"all_pool_gt{int(b)}": bool(np.all(pool > b)) for b in BIG},
                           median_relevant_ls=float(np.median(ls[~irrelevant])) if inst == "aligned" else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(c):
    rows = []
    at = c[c.n == N_CHECK]
    for inst in ("aligned", "rotated"):
        a = at[(at.instance == inst) & (at.method == "bo_ard")]
        label, ok = THRESHOLD[inst]
        q1, med, q3 = _q(a.rho)
        out = {"instance": inst, "n": N_CHECK, "n_runs": len(a), "rho_median": med, "rho_q1": q1, "rho_q3": q3,
               "threshold": label, "frac_runs_meeting": float(np.mean([ok(x) for x in a.rho])),
               "median_meets": bool(ok(med)), "iso_baseline_rho": 0.1}
        for m in ("bo_ard", "bo_iso"):
            b = at[(at.instance == inst) & (at.method == m)]
            q1, med, q3 = _q(b.rank_corr)
            out.update({f"rank_corr_{m}_median": med, f"rank_corr_{m}_q1": q1, f"rank_corr_{m}_q3": q3})
        out.update(rho_spread_all_median=float(a.rho_spread_all.median()),
                   rho_spread_near_optimal_median=float(a.rho_spread_near_optimal.median()),
                   max_lengthscale_max=float(a.max_ls.max()), median_relevant_ls=float(a.median_relevant_ls.median()))
        for b in BIG:
            k = f"gt{int(b)}"
            out[f"frac_runs_all_pool_{k}"] = float(a[f"all_pool_{k}"].mean())
            out[f"frac_pool_{k}_mean"] = float(a[f"frac_pool_{k}"].mean())
        rows.append(out)
    return pd.DataFrame(rows)


def by_checkpoint(c):
    g = c.groupby(["instance", "method", "n"])
    out = g.agg(n_runs=("seed", "size"), rho_median=("rho", "median"), rho_q1=("rho", lambda x: x.quantile(.25)),
                rho_q3=("rho", lambda x: x.quantile(.75)), rank_corr_median=("rank_corr", "median"),
                rank_corr_q1=("rank_corr", lambda x: x.quantile(.25)),
                rank_corr_q3=("rank_corr", lambda x: x.quantile(.75))).reset_index()
    return out


if __name__ == "__main__":
    c = collect(load_results("e2_lowdim"))
    s = summarise(c)
    s.to_csv(SUMMARY / "e2_lowdim_rho.csv", index=False)
    by_checkpoint(c).to_csv(SUMMARY / "e2_lowdim_checkpoints.csv", index=False)
    print(s.T.to_string(float_format=lambda x: f"{x:.4g}"))
