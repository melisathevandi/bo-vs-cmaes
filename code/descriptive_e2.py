"""Descriptive (not pre-registered) summary of E2 primary: median log10 gap and IQR at the final budget.

    uv run python code/descriptive_e2.py

One row per (instance, method) over the 30 seeds, from results/raw/e2_primary. It adds no test and
does not touch the pre-registered analysis (code/analyze_e2.py). Writes
results/summary/e2_primary_descriptive.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from analysis import gap_table, load_results

SUMMARY = Path(__file__).resolve().parent.parent / "results" / "summary"
INSTANCES = {"aligned": "ell_d10_k1e4_aligned", "rotated": "ell_d10_k1e4_rotated"}
METHODS = ["bo_ard", "bo_iso", "turbo", "cma", "sep_cma", "sobol"]


def summarise(df):
    n = int(df["budget"].max())
    t = gap_table(df, n)
    rows = []
    for inst, prob in INSTANCES.items():
        for m in METHODS:
            x = t.loc[prob][m].dropna().to_numpy()
            q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
            rows.append({"budget": n, "instance": inst, "method": m, "n_seeds": len(x),
                         "median_log10_gap": med, "q1": q1, "q3": q3})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = summarise(load_results("e2_primary"))
    out.to_csv(SUMMARY / "e2_primary_descriptive.csv", index=False)
    print(out.round(2).to_string(index=False))
