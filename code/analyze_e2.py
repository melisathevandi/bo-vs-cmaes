"""E2 primary analysis, as fixed in analysis_addendum_e2.md (written before any E2 result was read).

    uv run python code/analyze_e2.py

Reads results/raw/e2_primary (and e2_cma_sigma015 for the declared secondary variant) at the final
budget and writes results/summary/e2_primary.csv with one row per statistic:
  noise_floor   aligned replicate minus aligned, per method, 99% CI, and the pass/flag verdict
  delta_rot     rotated minus aligned, per method, 95% CI
  interaction   Delta_rot(axis-aligned method) - Delta_rot(its rotation-insensitive counterpart), 95% CI
  secondary_*   the same quantities for CMA-ES with sigma0 = 0.15 (never enters the main rows)
All differences are in log10 gap, paired by seed; bootstrap seed 0 (analysis.bootstrap_ci).
"""

from pathlib import Path

import pandas as pd

from analysis import bootstrap_ci, gap_table, load_results

SUMMARY = Path(__file__).resolve().parent.parent / "results" / "summary"
ALIGNED, ROTATED, REPLICATE = "ell_d10_k1e4_aligned", "ell_d10_k1e4_rotated", "ell_d10_k1e4_aligned_rep"
MAIN_METHODS = ["bo_ard", "bo_iso", "turbo", "cma", "sep_cma", "sobol"]
INTERACTIONS = {"ard_minus_iso": ("bo_ard", "bo_iso"), "sepcma_minus_cma": ("sep_cma", "cma")}
NOISE_LEVEL, TOLERANCE = 0.99, 0.15       # fixed in analysis_addendum_e2.md
EFFECT_LEVEL = 0.95


def paired(table, method, problem_a, problem_b):
    """Per-seed difference gap(problem_a) - gap(problem_b) for one method."""
    return (table.loc[problem_a][method] - table.loc[problem_b][method]).dropna()


def noise_floor(table, methods, tol=TOLERANCE, level=NOISE_LEVEL):
    """Aligned replicate minus aligned. A method passes if its CI contains 0 or |median| <= tol."""
    rows = []
    for m in methods:
        d = paired(table, m, REPLICATE, ALIGNED)
        med, lo, hi = bootstrap_ci(d.to_numpy(), level=level)
        contains0 = lo <= 0 <= hi
        rows.append({"quantity": "noise_floor", "method": m, "n_pairs": len(d), "median": med, "ci_lo": lo,
                     "ci_hi": hi, "level": level, "ci_contains_0": contains0,
                     "verdict": "pass" if contains0 or abs(med) <= tol else "FLAG"})
    return rows


def delta_rot(table, methods, level=EFFECT_LEVEL, quantity="delta_rot"):
    rows = []
    for m in methods:
        d = paired(table, m, ROTATED, ALIGNED)
        med, lo, hi = bootstrap_ci(d.to_numpy(), level=level)
        rows.append({"quantity": quantity, "method": m, "n_pairs": len(d), "median": med, "ci_lo": lo,
                     "ci_hi": hi, "level": level})
    return rows


def interactions(table, pairs=INTERACTIONS, level=EFFECT_LEVEL, prefix=""):
    rows = []
    for name, (a, b) in pairs.items():
        d = (paired(table, a, ROTATED, ALIGNED) - paired(table, b, ROTATED, ALIGNED)).dropna()
        med, lo, hi = bootstrap_ci(d.to_numpy(), level=level)
        rows.append({"quantity": prefix + "interaction", "method": name, "n_pairs": len(d), "median": med,
                     "ci_lo": lo, "ci_hi": hi, "level": level, "supports_h2": bool(lo > 0)})
    return rows


if __name__ == "__main__":
    df = load_results("e2_primary")
    n = int(df["budget"].max())
    table = gap_table(df, n)
    methods = [m for m in MAIN_METHODS if m in table.columns]
    rows = noise_floor(table, methods) + delta_rot(table, methods) + interactions(table)
    try:
        sec = load_results("e2_cma_sigma015")
    except Exception:
        sec = pd.DataFrame()
    if not sec.empty:
        both = pd.concat([df, sec])
        t2 = gap_table(both, n)
        rows += [dict(r, quantity="secondary_noise_floor") for r in noise_floor(t2, ["cma_s015"])]
        rows += [dict(r, quantity="secondary_delta_rot") for r in delta_rot(t2, ["cma_s015"])]
        rows += interactions(t2, {"sepcma_minus_cma_s015": ("sep_cma", "cma_s015")}, prefix="secondary_")
    out = pd.DataFrame(rows)
    out.insert(0, "budget", n)
    out.to_csv(SUMMARY / "e2_primary.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.round(3).to_string(index=False))
    flagged = out[(out.quantity == "noise_floor") & (out.verdict == "FLAG")]
    print("\nNoise-floor flags:", ", ".join(flagged.method) if len(flagged) else "none")
