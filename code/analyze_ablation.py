"""TuRBO rotation-penalty ablation, exactly as fixed in predictions.md (follow-up section and its amendments).

    uv run python code/analyze_ablation.py     -> results/summary/turbo_ablation.csv, turbo_ablation_verdict.csv

Default TuRBO is the `turbo` method of e2_primary (not repeated); the variants are in turbo_ablation. Gap at n = 500.
Delta_rot = log10 gap(rotated) - log10 gap(aligned), paired by seed; noise floor = aligned replicate - aligned.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from analysis import bootstrap_ci, gap_table, load_results

SUMMARY = Path(__file__).resolve().parent.parent / "results" / "summary"
A, R, REP = "ell_d10_k1e4_aligned", "ell_d10_k1e4_rotated", "ell_d10_k1e4_aligned_rep"
N = 500


def per_method():
    e2, ab = load_results("e2_primary"), load_results("turbo_ablation")
    ta, tb = gap_table(e2, N), gap_table(ab, N)
    cols = {"turbo (default)": ta["turbo"], "turbo_cubic_ard": tb["turbo_cubic_ard"], "turbo_iso": tb["turbo_iso"],
            "bo_ard (E2)": ta["bo_ard"], "bo_iso (E2)": ta["bo_iso"], "sobol (E2)": ta["sobol"]}
    rows, arrays = [], {}
    for name, s in cols.items():
        a, r, p = (s.loc[x].dropna() for x in (A, R, REP))
        seeds = sorted(set(a.index) & set(r.index) & set(p.index))
        d_rot = (r.loc[seeds] - a.loc[seeds]).to_numpy()
        floor = (p.loc[seeds] - a.loc[seeds]).to_numpy()
        arrays[name] = d_rot
        row = {"method": name, "n_seeds": len(seeds)}
        for lab, x in (("aligned", a), ("rotated", r), ("replicate", p)):
            q = np.quantile(x.loc[seeds], [0.25, 0.5, 0.75])
            row.update({f"median_{lab}": q[1], f"q1_{lab}": q[0], f"q3_{lab}": q[2]})
        for lab, x in (("delta_rot", d_rot), ("noise_floor", floor)):
            m, lo, hi = bootstrap_ci(x)
            row.update({lab: m, f"{lab}_lo": lo, f"{lab}_hi": hi})
        row["delta_rot_minus_floor"] = row["delta_rot"] - row["noise_floor"]
        rows.append(row)
    return pd.DataFrame(rows), arrays


def contrast(x, y, n_boot=10_000, seed=0):
    """median(x) - median(y), seeds resampled independently for the two arrays (as written in the prediction)."""
    rng = np.random.default_rng(seed)
    bx = np.median(x[rng.integers(0, len(x), (n_boot, len(x)))], axis=1)
    by = np.median(y[rng.integers(0, len(y), (n_boot, len(y)))], axis=1)
    lo, hi = np.quantile(bx - by, [0.025, 0.975])
    return float(np.median(x) - np.median(y)), float(lo), float(hi)


def verdicts(t, arrays):
    g = t.set_index("method")
    c, i = g.loc["turbo_cubic_ard"], g.loc["turbo_iso"]
    ct, ct_lo, ct_hi = contrast(arrays["turbo_cubic_ard"], arrays["turbo (default)"])
    ic, ic_lo, ic_hi = contrast(arrays["turbo_iso"], arrays["turbo_cubic_ard"])
    band = bool(0.45 <= c.delta_rot <= 1.25 and c.delta_rot_lo > 0)
    smaller = bool(ct < 0 and ct_hi < 0)
    iso_ok = bool(abs(i.delta_rot) <= 0.3 and i.delta_rot_lo <= 0 <= i.delta_rot_hi)
    rows = [
        {"item": "cubic ARD: delta_rot in [0.45,1.25] and CI excludes 0", "value": c.delta_rot,
         "ci_lo": c.delta_rot_lo, "ci_hi": c.delta_rot_hi, "met": band},
        {"item": "cubic ARD - default: delta_rot contrast < 0, CI excludes 0", "value": ct, "ci_lo": ct_lo,
         "ci_hi": ct_hi, "met": smaller},
        {"item": "cubic ARD prediction (both of the above)", "value": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
         "met": band and smaller},
        {"item": "isotropic: |delta_rot| <= 0.3 and CI contains 0", "value": i.delta_rot, "ci_lo": i.delta_rot_lo,
         "ci_hi": i.delta_rot_hi, "met": iso_ok},
        {"item": "isotropic: delta_rot minus its noise floor (|.| <= 0.3)", "value": i.delta_rot_minus_floor,
         "ci_lo": np.nan, "ci_hi": np.nan, "met": bool(abs(i.delta_rot_minus_floor) <= 0.3)},
        {"item": "isotropic - cubic ARD: delta_rot contrast (reported)", "value": ic, "ci_lo": ic_lo, "ci_hi": ic_hi,
         "met": np.nan},
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t, arrays = per_method()
    t.round(4).to_csv(SUMMARY / "turbo_ablation.csv", index=False)
    v = verdicts(t, arrays)
    v.round(4).to_csv(SUMMARY / "turbo_ablation_verdict.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(t.round(2).to_string(index=False))
    print(v.round(2).to_string(index=False))
