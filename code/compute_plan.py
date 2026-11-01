"""Planning estimate of the compute each experiment needs (not a result of the study).

BO cost per run is integrated from the timing pilot (cold GP fits, results/summary/
pilot_timing.csv): per iteration, acquisition cost + (fit cost when a refit is due). The
refit schedule is every iteration while n <= 100, then every `every` iterations. Beyond
n = 1000 the pilot has no data, so fit cost is extrapolated as n^3 and acquisition as n^2
(very uncertain). A factor SLOWDOWN accounts for parallel load. TuRBO per-run cost is taken
from the TuRBO pilot (results/raw/pilot_turbo) where measured. CMA-ES and Sobol cost is
negligible (about 0.1 ms per evaluation) and listed as 0.

Prints a markdown table and writes notes/compute_plan.md.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WORKERS, SLOWDOWN = 7, 1.3
LOW, HIGH = 0.7, 1.6                 # judgement band around the central estimate

_pilot = pd.read_csv(ROOT / "results" / "summary" / "pilot_timing.csv")
_g = _pilot.groupby(["d", "n"])[["t_fit_cold_s", "t_acq_s"]].mean()


def _curve(d, col, n):
    sub = _g.loc[d][col]
    ns, ts = np.array(sub.index, float), sub.values
    if n < ns[0]:
        return ts[0] * n / ns[0]
    if n > ns[-1]:
        return ts[-1] * (n / ns[-1]) ** (3 if col == "t_fit_cold_s" else 2)
    return float(np.interp(n, ns, ts))


CAL = 1.0      # calibration factor (measured / estimated), set from the BO pilot in calibrate()


def bo_minutes(d, N, every):
    """Minutes for one BO run of N evaluations in dimension d (timings of the nearest pilot d)."""
    dd = min((2, 10, 20), key=lambda x: abs(x - d)) if d not in (2, 10, 20) else d
    total = 0.0
    for n in range(2 * d, N):
        refit = n <= 100 or (n - 100) % every == 0
        total += _curve(dd, "t_acq_s", n) + (_curve(dd, "t_fit_cold_s", n) if refit else 0.0)
    return total / 60 * SLOWDOWN * CAL


def calibrate():
    """Set CAL from the real BO runs of the pilot (d=10, N=500, cold, refit every 5)."""
    global CAL
    walls = [json.loads(p.read_text())["wall_s"] / 60
             for p in (ROOT / "results" / "raw" / "pilot_bo").glob("*.json")]
    CAL = 1.0
    if walls:
        CAL = float(np.median(walls)) / bo_minutes(10, 500, 5)
    return CAL, walls


def turbo_minutes(d, N):
    """Measured median wall time (min) of the TuRBO pilot, scaled linearly in N if unmeasured."""
    runs = {}
    for p in (ROOT / "results" / "raw" / "pilot_turbo").glob("*.json"):
        r = json.loads(p.read_text())
        s = r["spec"]
        runs.setdefault((s["problem"]["d"], s["budget"]), []).append(r["wall_s"] / 60)
    if (d, N) in runs:
        return float(np.median(runs[(d, N)])), "measured"
    for (dm, Nm), v in runs.items():
        if dm == d:
            return float(np.median(v)) * N / Nm, "scaled from N=%d" % Nm
    dm = min({k[0] for k in runs}, key=lambda x: abs(x - d))
    Nm = max(k[1] for k in runs if k[0] == dm)
    return float(np.median(runs[(dm, Nm)])) * N / Nm, f"scaled from d={dm}, N={Nm}"


def rows(every):
    """(experiment, note, cut-order mark, list of (n_runs, minutes_per_run))."""
    b = lambda d, N: bo_minutes(d, N, every)
    t = lambda d, N: turbo_minutes(d, N)[0]
    R = []
    R.append(("Sensitivity grids", "BO: 6 cfgs x 2 problems x 10 seeds; CMA-ES grid negligible", "never cut",
              [(60, b(10, 500)), (60, b(5, 400))]))
    R.append(("E0 invariance (done)", "CMA-ES only", "done", []))
    for d, N in [(2, 200), (3, 200), (5, 400), (8, 640), (10, 1000), (20, 1000)]:
        mark = "cut 4th (d=20 in E1)" if d == 20 else "core"
        R.append((f"E1 d={d}, N={N}", "BO ARD x {ellipsoid, Rosenbrock} x 20 seeds", mark, [(40, b(d, N))]))
    R.append(("E2 primary (N=500)", "k=1e4: aligned, rotated, replicate x 30 seeds x {ARD, iso, TuRBO}",
              "never cut", [(180, b(10, 500)), (90, t(10, 500))]))
    R.append(("E2 secondary N=1000 (optional)", "k=1e4 aligned+rotated x 10 seeds x {ARD, iso, TuRBO}",
              "optional", [(40, b(10, 1000)), (20, t(10, 1000))]))
    R.append(("E2 other kappa (N=500)", "k=1 aligned; k=1e2 aligned+rotated x 20 seeds x {ARD, iso, TuRBO}",
              "core", [(120, b(10, 500)), (60, t(10, 500))]))
    R.append(("E2 low-eff-dim (d=20, N=500)", "aligned+rotated x 20 seeds x {ARD, iso, TuRBO}; ARD adds ~3 min diagnostics",
              "never cut", [(80, b(20, 500) + 1.5), (40, t(20, 500))]))
    R.append(("E3 d=2 (N=200)", "5 widths x 30 seeds x {LogEI, UCB}", "UCB cut 1st (kept in E3)", [(300, b(2, 200))]))
    R.append(("E3 d=5 (N=400)", "5 widths x 30 seeds x {LogEI, UCB}", "UCB cut 1st (kept in E3)", [(300, b(5, 400))]))
    R.append(("H5 CMA-ES runs", "11 dims x 20 seeds, cap 100 d^2", "core", [(220, 0.0)]))
    R.append(("A8c noise (E5)", "d=5 N=300, 2 noise levels x 20 seeds, BO ARD", "cut 1st (A8 c-e)", [(40, b(5, 300))]))
    R.append(("A8d long-budget probe", "d=10 k=1e4 rotated, N=2000, 10 seeds, BO ARD (extrapolated beyond n=1000)",
              "cut 1st (A8 c-e)", [(10, b(10, 2000))]))
    R.append(("A8e second family", "4 BBOB-style functions, d=10, N=500, 10 seeds, BO ARD", "cut 1st (A8 c-e)",
              [(40, b(10, 500))]))
    return R


def table(every):
    lines = [f"| Experiment | Runs (BO-type) | Min/run | CPU-h | Wall-h on {WORKERS} workers | Cut-order mark |",
             "|---|---|---|---|---|---|"]
    tot = 0.0
    for name, note, mark, parts in rows(every):
        n_runs = sum(n for n, _ in parts)
        cpu_h = sum(n * m for n, m in parts) / 60
        tot += cpu_h
        per = (sum(n * m for n, m in parts) / n_runs) if n_runs else 0.0
        lines.append(f"| {name}: {note} | {n_runs} | {per:.1f} | {cpu_h:.1f} | {cpu_h / WORKERS:.1f} | {mark} |")
    lines.append(f"| **Total** | | | **{tot:.0f}** | **{tot / WORKERS:.0f}** "
                 f"(range {LOW * tot / WORKERS:.0f} to {HIGH * tot / WORKERS:.0f}) | |")
    return "\n".join(lines), tot


if __name__ == "__main__":
    # Check the estimator against wall times measured in A4 (d=5, N=300, cold; 6 parallel workers).
    a4 = pd.read_csv(ROOT / "results" / "summary" / "a4_refit_check.csv")
    print("Estimator check (A4 measured, d=5 N=300, estimate uses d=10 timings so it is an upper bound):")
    for every, col in ((5, "median_wall_s_reference"), (10, "median_wall_s_variant")):
        sub = a4[(a4.a == ("bo_cold_every10" if every == 10 else "bo_warm_every5")) & (a4.n == 300)]
        meas = sub.groupby("problem")[col].first() / 60
        print(f"  every {every}: estimate {bo_minutes(5, 300, every):.1f} min; measured "
              + ", ".join(f"{k} {v:.1f}" for k, v in meas.items()) + " min")
    pilot_rows = []
    for exp in ("pilot_turbo", "pilot_bo"):
        for p in sorted((ROOT / "results" / "raw" / exp).glob("*.json")):
            r = json.loads(p.read_text())
            pilot_rows.append({"experiment": exp, "problem": r["spec"]["problem"]["name"],
                               "method": r["spec"]["method"]["name"], "seed": r["spec"]["seed"],
                               "wall_min": r["wall_s"] / 60, "restarts": r["meta"].get("restarts")})
    pd.DataFrame(pilot_rows).to_csv(ROOT / "results" / "summary" / "pilot_run_timing.csv", index=False)
    cal, walls = calibrate()
    print(f"\nCalibration (real BO runs, d=10 N=500, cold, every 5): measured {len(walls)} runs "
          f"{', '.join(f'{w:.1f}' for w in walls)} min; calibration factor {cal:.2f}")
    print("\nTuRBO per-run minutes:")
    for d, N in [(10, 500), (10, 1000), (20, 500), (20, 1000)]:
        m, how = turbo_minutes(d, N)
        print(f"  d={d} N={N}: {m:.1f} ({how})")
    out = []
    totals = {}
    for every in (5, 10):
        tab, tot = table(every)
        totals[every] = tot
        out.append(f"### BO refit every {every} iterations (n > 100)\n\n{tab}\n")
    saving = (totals[5] - totals[10]) / WORKERS
    out.append(f"Refit every 5 costs {totals[5] - totals[10]:.0f} CPU-h more than every 10 "
               f"({saving:.0f} wall-h on {WORKERS} workers).\n")
    text = "\n".join(out)
    (ROOT / "notes" / "compute_plan.md").write_text(
        "# Compute plan (planning estimate, not a result)\n\n"
        f"BO estimates are the timing-pilot integral times {SLOWDOWN} (parallel load) times a "
        f"calibration factor of {cal:.2f} (measured / estimated at d=10, N=500).\n\n" + text,
        encoding="utf-8")
    print("\n" + text)
