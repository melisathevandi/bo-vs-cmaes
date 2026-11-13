"""Sensitivity grids (CMA-ES: sigma0 x population; BO: n_init x nu) and the pre-set decision rule.

    uv run python code/analyze_sensitivity.py

Each alternative configuration is compared with the default on the two pilot problems, paired by
seed (held-out seeds 1000+), at the final budget of the run. Rule (fixed before the grids were run): keep the
default unless an alternative is better on BOTH pilot problems with its 95% paired bootstrap CI
excluding 0 (median difference in log10 gap, alternative minus default, CI upper bound < 0).
Writes results/summary/sensitivity_<method>.csv and exits with status 3 if any alternative wins,
so that the experiment chain in code/run_core.sh stops before the main runs.
"""

import sys
from pathlib import Path

import pandas as pd

from analysis import gap_table, load_results, paired_difference

SUMMARY = Path(__file__).resolve().parent.parent / "results" / "summary"
GRIDS = {"cmaes": ("sens_cmaes", "cma_s0.3_p1"), "bo": ("sens_bo", "bo_n2d_nu2.5")}


if __name__ == "__main__":
    winners = []
    pd.set_option("display.width", 200)
    for key, (experiment, default) in GRIDS.items():
        df = load_results(experiment)
        rows = []
        for problem in sorted(df["problem"].unique()):
            budget = int(df[df.problem == problem]["budget"].max())
            table = gap_table(df[df.problem == problem], budget)
            for method in sorted(set(df["method"]) - {default}):
                r = paired_difference(table, method, default, problem)
                r["budget"] = budget
                rows.append(r)
        out = pd.DataFrame(rows)
        out["better_ci_excludes_0"] = out["ci_hi"] < 0
        out.to_csv(SUMMARY / f"sensitivity_{key}.csv", index=False)
        print(f"\n== {experiment}: alternative minus default ({default}); negative = alternative better")
        print(out[["problem", "a", "budget", "n_pairs", "median_diff", "ci_lo", "ci_hi", "win_rate",
                   "better_ci_excludes_0"]].round(3).to_string(index=False))
        ok = out.groupby("a")["better_ci_excludes_0"].all()
        n_prob = out["problem"].nunique()
        for name in ok[ok].index:
            if (out["a"] == name).sum() == n_prob:
                winners.append((experiment, name))
    if winners:
        print("\nNON-DEFAULT WINNERS:", winners)
        sys.exit(3)
    print("\nNo alternative beats the default on both problems: keep all defaults.")
