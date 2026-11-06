"""E3 (decoy basin, d = 2) exactly as fixed in analysis_addendum_e3.md.

    uv run python code/analyze_e3.py     -> results/summary/e3_success.csv, e3_tests.csv
Success at n: best value among the first n evaluations < F_DECOY (= 1).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from analysis import bootstrap_ci, load_results
from problems import F_DECOY

SUMMARY = Path(__file__).resolve().parent.parent / "results" / "summary"
WIDTHS = [0.2, 0.1, 0.05, 0.025, 0.0125]
NAME = {w: "decoy_w" + str(w).replace(".", "p") for w in WIDTHS}


def wilson(k, n, level=0.95):
    z = norm.ppf(0.5 + level / 2)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(p), float(c - h), float(c + h)


def success_table(df, n):
    """Wide table: index (problem, seed), columns method, values 0/1."""
    s = [float(np.min(f[:n]) < F_DECOY) for f in df.F]
    return df.assign(s=s).pivot_table(index=["problem", "seed"], columns="method", values="s")


def main():
    short, long = load_results("e3_d2"), load_results("e3_d2_long")
    tabs = {"short200": success_table(short, 200), "long2000": success_table(long, 2000),
            "long200": success_table(long, 200)}
    rows = []
    for key, t in tabs.items():
        n = 2000 if key == "long2000" else 200
        for w in WIDTHS:
            for m in t.columns:
                x = t.loc[NAME[w]][m].dropna()
                p, lo, hi = wilson(int(x.sum()), len(x))
                row = {"n": n, "source": key, "width": w, "method": m, "n_seeds": len(x),
                       "successes": int(x.sum()), "p_success": p, "wilson_lo": lo, "wilson_hi": hi}
                if m == "sobol":
                    row["sobol_expected"] = 1 - (1 - np.pi * w * w) ** n
                rows.append(row)
    success = pd.DataFrame(rows)
    success.round(4).to_csv(SUMMARY / "e3_success.csv", index=False)

    def diff(t, a, b, w):
        return (t.loc[NAME[w]][a] - t.loc[NAME[w]][b]).dropna().to_numpy()

    tests = []

    def add(name, x):
        m, lo, hi = bootstrap_ci(x, np.mean)
        tests.append({"test": name, "estimate": m, "ci_lo": lo, "ci_hi": hi, "n_pairs": len(x)})

    t200, t2000 = tabs["short200"], tabs["long2000"]
    for label, t, n in (("n=200", t200, 200), ("n=2000", t2000, 2000)):
        for m in t.columns:
            add(f"width dependence {label}: P(w=0.2)-P(w=0.0125), {m}",
                (t.loc[NAME[0.2]][m] - t.loc[NAME[0.0125]][m]).dropna().to_numpy())
    for w in WIDTHS:
        add(f"UCB - LogEI, w={w}, n=200", diff(t200, "bo_ucb", "bo_logei", w))
    for w in WIDTHS:
        add(f"IPOP - CMA-ES, w={w}, n=2000", diff(t2000, "cma_ipop", "cma", w))
    for w in WIDTHS:
        add(f"IPOP - CMA-ES, w={w}, n=200 (descriptive)", diff(t200, "cma_ipop", "cma", w))

    def pooled(t):      # per seed, mean over the five widths (the widths share the centres)
        return np.mean([diff(t, "cma_ipop", "cma", w) for w in WIDTHS], axis=0)

    add("IPOP - CMA-ES pooled over widths, n=2000", pooled(t2000))
    add("IPOP - CMA-ES pooled over widths, n=200 (descriptive)", pooled(t200))
    for w in WIDTHS:
        for m in ("bo_logei", "bo_ucb"):
            add(f"{m} - Sobol, w={w}, n=200 (descriptive)", diff(t200, m, "sobol", w))
    tests = pd.DataFrame(tests)
    tests.round(4).to_csv(SUMMARY / "e3_tests.csv", index=False)
    return success, tests


if __name__ == "__main__":
    s, t = main()
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
    print(s[s.source == "short200"].pivot_table(index="width", columns="method", values="p_success").round(2))
    print(s[s.source == "long2000"].pivot_table(index="width", columns="method", values="p_success").round(2))
    print(t.round(2).to_string(index=False))
