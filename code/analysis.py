"""Load saved runs and compute gaps, paired differences and bootstrap intervals.

Gap = f(best point so far) - f*, evaluated on the true noiseless objective. It is floored
at GAP_FLOOR so that log10 stays finite once a method reaches machine precision.
All comparisons are paired by seed: every method sees the same instance for a given seed.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "results" / "raw"
GAP_FLOOR = 1e-12


def load_results(experiment, raw_dir=RAW):
    """One row per saved run, with the trajectory kept as a numpy array in column `F`."""
    rows = []
    for path in sorted((Path(raw_dir) / experiment).glob("*.json")):
        r = json.loads(path.read_text())
        spec = r["spec"]
        rows.append({
            "problem": spec["problem"].get("name", spec["problem"]["family"]),
            "method": spec["method"]["name"], "seed": spec["seed"],
            "instance_seed": spec["instance_seed"], "budget": spec["budget"],
            "F": np.array(r["F"]), "f_star": r["f_star"],
            "t_method": np.array(r["t_method"]), "t_eval": r["t_eval"],
            "wall_s": r["wall_s"], "meta": r["meta"], "job_id": r["job_id"],
        })
    return pd.DataFrame(rows)


def log_gap(F, f_star, n):
    """log10 gap of the best value among the first n evaluations (forward-filled)."""
    best = np.min(F[:n]) if len(F) else np.inf
    return float(np.log10(max(best - f_star, GAP_FLOOR)))


def gap_table(df, n):
    """Wide table: rows (problem, seed), one column per method, values = log10 gap at n."""
    d = df.assign(gap=[log_gap(f, s, n) for f, s in zip(df["F"], df["f_star"])])
    return d.pivot_table(index=["problem", "seed"], columns="method", values="gap")


def bootstrap_ci(x, stat=np.median, n_boot=10_000, level=0.95, seed=0):
    """Percentile bootstrap CI of stat(x), resampling the paired differences."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = stat(x[idx], axis=1)
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(stat(x)), float(lo), float(hi)


def paired_difference(table, a, b, problem, seed=0):
    """Median of (gap_a - gap_b) over seeds with its CI, and the win rate P(a better)."""
    t = table.loc[problem]
    diff = (t[a] - t[b]).dropna().to_numpy()
    med, lo, hi = bootstrap_ci(diff, np.median, seed=seed)
    wins, w_lo, w_hi = bootstrap_ci((diff < 0).astype(float), np.mean, seed=seed)
    return {"problem": problem, "a": a, "b": b, "n_pairs": len(diff),
            "median_diff": med, "ci_lo": lo, "ci_hi": hi,
            "win_rate": wins, "win_lo": w_lo, "win_hi": w_hi}
