"""kappa = 1e4 supplement for H1, exactly as in analysis_addendum_e1_kappa1e4.md.

    uv run python code/analyze_e1_kappa1e4.py   -> results/summary/e1_kappa1e4_h1.csv
d = 2, 5 from e1k4_d2 / e1k4_d5 (20 seeds), d = 10 from the rotated ellipsoid of e2_primary (30 seeds) read at n = 200.
D(d) = median over seeds of log10 gap_BO - log10 gap_CMA at n = 20 d; S = D(10) - D(2), seeds resampled independently.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from analysis import bootstrap_ci, gap_table, load_results

SUMMARY = Path(__file__).resolve().parent.parent / "results" / "summary"
SOURCES = {2: ("e1k4_d2", "ell_k1e4_rotated"), 5: ("e1k4_d5", "ell_k1e4_rotated"),
           10: ("e2_primary", "ell_d10_k1e4_rotated")}
GRID_K = (10, 20, 40)


def diffs_at(d, n):
    exp, prob = SOURCES[d]
    t = gap_table(load_results(exp), n)
    return (t.loc[prob]["bo_ard"] - t.loc[prob]["cma"]).dropna().to_numpy()


def main():
    rows, at20 = [], {}
    for d in SOURCES:
        for k in GRID_K:
            x = diffs_at(d, k * d)
            m, lo, hi = bootstrap_ci(x)
            win = float(np.mean(x < 0))
            rows.append({"statistic": f"D(d={d}) at {k} evaluations per dimension", "d": d, "n": k * d,
                         "n_seeds": len(x), "estimate": m, "ci_lo": lo, "ci_hi": hi, "bo_win_rate": win})
            if k == 20:
                at20[d] = x
    a, b = at20[10], at20[2]
    rng = np.random.default_rng(0)
    boot = np.median(a[rng.integers(0, len(a), (10_000, len(a)))], axis=1) \
        - np.median(b[rng.integers(0, len(b), (10_000, len(b)))], axis=1)
    s = float(np.median(a) - np.median(b))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    verdict = "S>=0.5: size supported" if s >= 0.5 else ("S<=0: refuted" if s <= 0 else "0<S<0.5: direction only")
    rows.append({"statistic": "S=D(10)-D(2) at 20 evaluations per dimension", "d": np.nan, "n": np.nan,
                 "n_seeds": np.nan, "estimate": s, "ci_lo": float(lo), "ci_hi": float(hi), "verdict_rule": verdict})
    out = pd.DataFrame(rows)
    out.round(6).to_csv(SUMMARY / "e1_kappa1e4_h1.csv", index=False)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(main().round(3).to_string(index=False))
