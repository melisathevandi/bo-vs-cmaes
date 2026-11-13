"""A4: does the GP refit schedule change BO's results?

Reference vs variants, paired by seed on held-out seeds. For each problem and budget the
script reports the median paired difference in log10 gap (variant - reference; negative = variant
better) with its 95% bootstrap CI, and the median wall-clock time of a run.

    uv run python code/analyze_a4.py                      # first check (d=5, n <= 300)
    uv run python code/analyze_a4.py --experiment a4_d10_check --reference bo_cold_every5 \
        --variants bo_cold_every10 --budgets 100 300 500 --rule-budget 500 \
        --out results/summary/a4_d10_check.csv

With --rule-budget the decision rule of configs/a4_d10_check.toml is applied at that budget:
adopt the variant only if the CI contains 0 and its half-width is at most 0.35 decades.
"""

import argparse
from pathlib import Path

import pandas as pd

from analysis import gap_table, load_results, paired_difference

SUMMARY = Path(__file__).resolve().parent.parent / "results" / "summary"
HALF_WIDTH_MAX = 0.35

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="a4_refit_check")
    ap.add_argument("--reference", default="bo_cold_every5")
    ap.add_argument("--variants", nargs="+",
                    default=["bo_warm_every5", "bo_warm_every10", "bo_cold_every10"])
    ap.add_argument("--budgets", nargs="+", type=int, default=[100, 200, 300])
    ap.add_argument("--rule-budget", type=int, default=None)
    ap.add_argument("--out", default=str(SUMMARY / "a4_refit_check.csv"))
    args = ap.parse_args()

    df = load_results(args.experiment)
    wall = df.groupby(["problem", "method"])["wall_s"].median()
    rows = []
    for n in args.budgets:
        table = gap_table(df, n)
        for problem in sorted(df["problem"].unique()):
            for variant in args.variants:
                r = paired_difference(table, variant, args.reference, problem)
                r.update({"n": n, "median_wall_s_variant": float(wall[(problem, variant)]),
                          "median_wall_s_reference": float(wall[(problem, args.reference)]),
                          "ci_contains_0": bool(r["ci_lo"] <= 0 <= r["ci_hi"]),
                          "ci_half_width": (r["ci_hi"] - r["ci_lo"]) / 2})
                rows.append(r)
    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    pd.set_option("display.width", 220)
    cols = ["problem", "a", "n", "n_pairs", "median_diff", "ci_lo", "ci_hi", "ci_contains_0",
            "ci_half_width", "median_wall_s_variant", "median_wall_s_reference"]
    print(out[cols].round(3).to_string(index=False))
    if args.rule_budget is not None:
        for _, r in out[out.n == args.rule_budget].iterrows():
            adopt = r.ci_contains_0 and r.ci_half_width <= HALF_WIDTH_MAX
            print(f"RULE at n={args.rule_budget}: {r.a} on {r.problem}: CI contains 0 = {r.ci_contains_0}, "
                  f"half-width = {r.ci_half_width:.3f} (max {HALF_WIDTH_MAX}) -> "
                  f"{'ADOPT' if adopt else 'KEEP refit every 5'}")
