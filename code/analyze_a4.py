"""A4: does warm-starting the GP refit, or refitting every 10 iterations, change BO's results?

Reference: cold-start refit every 5 iterations. Paired by seed on held-out seeds 1000-1009.
For each problem and budget we report the median paired difference in log10 gap
(variant - reference; negative = variant better) with its 95% bootstrap CI, and the
median wall-clock time of a run.

Writes results/summary/a4_refit_check.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from analysis import gap_table, load_results, paired_difference

OUT = Path(__file__).resolve().parent.parent / "results" / "summary" / "a4_refit_check.csv"
REFERENCE = "bo_cold_every5"
VARIANTS = ["bo_warm_every5", "bo_warm_every10", "bo_cold_every10"]
BUDGETS = [100, 200, 300]


if __name__ == "__main__":
    df = load_results("a4_refit_check")
    wall = df.groupby(["problem", "method"])["wall_s"].median()
    rows = []
    for n in BUDGETS:
        table = gap_table(df, n)
        for problem in sorted(df["problem"].unique()):
            for variant in VARIANTS:
                r = paired_difference(table, variant, REFERENCE, problem)
                r.update({"n": n,
                          "median_wall_s_variant": float(wall[(problem, variant)]),
                          "median_wall_s_reference": float(wall[(problem, REFERENCE)]),
                          "ci_contains_0": bool(r["ci_lo"] <= 0 <= r["ci_hi"])})
                rows.append(r)
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    pd.set_option("display.width", 220)
    cols = ["problem", "a", "n", "n_pairs", "median_diff", "ci_lo", "ci_hi", "ci_contains_0",
            "median_wall_s_variant", "median_wall_s_reference"]
    print(out[cols].round(3).to_string(index=False))
