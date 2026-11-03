"""Planning estimate of the compute each experiment needs (not a result of the study).

BO cost per run is integrated from the timing pilot (cold GP fits, results/summary/
pilot_timing.csv): per iteration, acquisition cost + (fit cost when a refit is due). The
refit schedule is every iteration while n <= 100, then every `every` iterations. Beyond
n = 1000 the pilot has no data, so fit cost is extrapolated as n^3 and acquisition as n^2
(very uncertain).

Pilot-to-real factor FACTOR = measured / integral at the E2 primary cell (d=10, N=500,
cold, refit every 5; 3 real runs, results/summary/pilot_run_timing.csv). It folds together
(i) real trajectories having harder GP fits than the pilot's Sobol points and (ii) the
parallel load during the calibration run (about 6 busy processes). LOAD7 is an assumed
extra factor for a full 7-worker pool; it is checked against the wall times of the A4
check at d=10, which runs on 7 workers. TuRBO per-run cost is the measured median of the
TuRBO pilot where available. CMA-ES and Sobol cost is negligible (about 0.1 ms per
evaluation).

BO refit schedule: cold refit every 10 iterations after n = 100 (A4 outcome, see lab notebook).
The plan is the lean core (grids, E2 primary, E2 low-effective-dimension, H5, E1 d <= 10, reserved
follow-up); E3, E2 other kappa, E1 d=20 and A8 c-e are optional and run only if the core is ahead
of schedule on 6 Nov. Each row has a key matching the chain step names in logs/chain_times.csv
(code/run_core.sh), so actual times can be set against these estimates.

Prints a markdown table and writes notes/compute_plan.md.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WORKERS, LOAD7, CONTINGENCY = 7, 1.25, 0.15
LOW, HIGH = 0.7, 1.6                 # judgement band around the central estimate
HARD_STOP = datetime(2026, 11, 8, 23, 59)
EVERY = 10                           # BO refit interval after n = 100 (A4 outcome)
START = datetime(2026, 11, 4, 0, 52)  # launch of the sensitivity grids
GATE = datetime(2026, 11, 6, 23, 59)  # optional experiments only if the core is ahead here
FOLLOW_UP_WALL_H = 20.0              # reserved follow-up experiment (given)

_pilot = pd.read_csv(ROOT / "results" / "summary" / "pilot_timing.csv")
_g = _pilot.groupby(["d", "n"])[["t_fit_cold_s", "t_acq_s"]].mean()
FACTOR = 1.0


def _curve(d, col, n):
    sub = _g.loc[d][col]
    ns, ts = np.array(sub.index, float), sub.values
    if n < ns[0]:
        return ts[0] * n / ns[0]
    if n > ns[-1]:
        return ts[-1] * (n / ns[-1]) ** (3 if col == "t_fit_cold_s" else 2)
    return float(np.interp(n, ns, ts))


def _integral_minutes(d, N, every):
    dd = min((2, 10, 20), key=lambda x: abs(x - d)) if d not in (2, 10, 20) else d
    total = 0.0
    for n in range(2 * d, N):
        refit = n <= 100 or (n - 100) % every == 0
        total += _curve(dd, "t_acq_s", n) + (_curve(dd, "t_fit_cold_s", n) if refit else 0.0)
    return total / 60


def bo_minutes(d, N, every):
    return _integral_minutes(d, N, every) * FACTOR * LOAD7


def calibrate():
    """FACTOR = measured / integral at the E2 primary cell, from the real BO pilot runs."""
    global FACTOR
    walls = [json.loads(p.read_text())["wall_s"] / 60
             for p in (ROOT / "results" / "raw" / "pilot_bo").glob("*.json")]
    FACTOR = float(np.median(walls)) / _integral_minutes(10, 500, 5) if walls else 1.0
    return FACTOR, walls


def turbo_minutes(d, N):
    """Measured median wall time (min) of the TuRBO pilot, scaled linearly in N if unmeasured."""
    runs = {}
    for p in (ROOT / "results" / "raw" / "pilot_turbo").glob("*.json"):
        r = json.loads(p.read_text())
        runs.setdefault((r["spec"]["problem"]["d"], r["spec"]["budget"]), []).append(r["wall_s"] / 60)
    if (d, N) in runs:
        return float(np.median(runs[(d, N)])) * LOAD7
    dm = d if any(k[0] == d for k in runs) else min({k[0] for k in runs}, key=lambda x: abs(x - d))
    Nm = max(k[1] for k in runs if k[0] == dm)
    return float(np.median(runs[(dm, Nm)])) * N / Nm * LOAD7


def phases():
    """Ordered list of (key, name, note, group, [(n_runs, min_per_run)], fixed_cpu_h)."""
    b = lambda d, N: bo_minutes(d, N, EVERY)
    t = lambda d, N: turbo_minutes(d, N)
    P = [
        ("sens_bo", "Sensitivity grids", "BO: 6 cfgs x 2 problems x 10 seeds (CMA-ES grid takes seconds)", "core",
         [(60, b(10, 500)), (60, b(5, 400))], 0),
        ("e2_primary", "E2 primary (N=500)",
         "k=1e4: aligned, rotated, aligned replicate x 30 seeds x {ARD, iso, TuRBO} (+ CMA, sep-CMA, Sobol)", "core",
         [(180, b(10, 500)), (90, t(10, 500))], 0),
        ("e2_lowdim", "E2 low-eff-dim (d=20, d_eff=2, N=500)",
         "aligned+rotated x 20 seeds x {ARD (+1.5 min diagnostics), iso, TuRBO} (+ CMA, Sobol)", "core",
         [(40, b(20, 500) + 1.5), (40, b(20, 500)), (40, t(20, 500))], 0),
        ("h5", "H5 CMA-ES runs", "11 dims x 20 seeds, cap 100 d^2", "core", [], 0.5),
    ]
    for d, N in [(2, 200), (3, 200), (5, 400), (8, 640), (10, 1000)]:
        P.append((f"e1_d{d}", f"E1 d={d}, N={N}", "BO ARD x {ellipsoid, Rosenbrock} x 20 seeds (+ CMA, Sobol)", "core",
                  [(40, b(d, N))], 0))
    P.append(("followup", "Follow-up (reserved)", "one experiment chosen after E1 (about 20 wall-h)", "core", [],
              FOLLOW_UP_WALL_H * WORKERS))
    P += [
        ("e3_d2", "E3 d=2 (N=200)", "5 widths x 30 seeds x {LogEI, UCB}", "optional", [(300, b(2, 200))], 0),
        ("e3_d5", "E3 d=5 (N=400)", "5 widths x 30 seeds x {LogEI, UCB}", "optional", [(300, b(5, 400))], 0),
        ("e2_other_kappa", "E2 other kappa (N=500)",
         "k=1 aligned; k=1e2 aligned+rotated x 20 seeds x {ARD, iso, TuRBO}", "optional",
         [(120, b(10, 500)), (60, t(10, 500))], 0),
        ("e1_d20", "E1 d=20, N=1000", "BO ARD x {ellipsoid, Rosenbrock} x 20 seeds", "optional", [(40, b(20, 1000))], 0),
        ("e2_secondary", "E2 secondary N=1000", "k=1e4 aligned+rotated x 10 seeds x {ARD, iso, TuRBO}", "extra",
         [(40, b(10, 1000)), (20, t(10, 1000))], 0),
        ("a8c", "A8c noise (E5)", "d=5 N=300, 2 noise levels x 20 seeds, BO ARD", "extra", [(40, b(5, 300))], 0),
        ("a8d", "A8d long-budget probe", "d=10 k=1e4 rotated, N=2000, 10 seeds (extrapolated beyond n=1000)", "extra",
         [(10, b(10, 2000))], 0),
        ("a8e", "A8e second family", "4 BBOB-style functions, d=10, N=500, 10 seeds", "extra", [(40, b(10, 500))], 0),
    ]
    return P


def cpu_h(parts, fixed):
    return sum(n * m for n, m in parts) / 60 + fixed


def table():
    lines = [f"| Key | Experiment | Runs | CPU-h | Wall-h ({WORKERS} workers) | Expected finish | Group |",
             "|---|---|---|---|---|---|---|"]
    cum, totals = 0.0, {}
    for key, name, note, group, parts, fixed in phases():
        c = cpu_h(parts, fixed)
        totals[group] = totals.get(group, 0.0) + c / WORKERS
        cum += c / WORKERS * (1 + CONTINGENCY)
        t = START + timedelta(hours=cum)
        flag = ""
        if t > HARD_STOP:
            flag = " **after the hard stop**"
        elif group == "optional" and t > GATE:
            flag = " (past the 6 Nov gate)"
        runs = sum(n for n, _ in parts)
        lines.append(f"| {key} | {name}: {note} | {runs or '-'} | {c:.1f} | {c / WORKERS:.1f} | "
                     f"{t:%d %b %H:%M}{flag} | {group} |")
    core = totals["core"]
    lines += ["", f"- Lean core: **{core:.0f} wall-h** ({core * (1 + CONTINGENCY):.0f} with 15% contingency; "
              f"range {LOW * core * (1 + CONTINGENCY):.0f} to {HIGH * core * (1 + CONTINGENCY):.0f})",
              f"- Optional (E3, E2 other kappa, E1 d=20): {totals['optional']:.0f} wall-h",
              f"- Extras (E2 secondary, A8 c-e): {totals['extra']:.0f} wall-h",
              f"- Expected finish times are cumulative from the grid launch ({START:%d %b %H:%M}) with 15% contingency, in "
              f"this order; hard stop {HARD_STOP:%d %b %H:%M}."]
    return "\n".join(lines)


if __name__ == "__main__":
    fac, walls = calibrate()
    print(f"FACTOR = {fac:.2f} (measured {', '.join(f'{w:.1f}' for w in walls)} min vs integral "
          f"{_integral_minutes(10, 500, 5):.1f} min); LOAD7 = {LOAD7}")
    text = table()
    (ROOT / "notes" / "compute_plan.md").write_text(
        "# Compute plan (planning estimate, not a result)\n\n"
        f"BO cost = timing-pilot integral x FACTOR ({fac:.2f}, measured/integral at d=10, N=500, 3 real runs) x LOAD7 "
        f"({LOAD7}, measured: A4 check on 7 workers, 47.0 min per refit-every-5 run vs 37.7 min in the calibration run). "
        f"BO refits cold every {EVERY} iterations after n = 100. TuRBO cost = measured pilot median x LOAD7. "
        "Lean core first; optional rows run only if the core is ahead of schedule on 6 Nov.\n\n" + text + "\n",
        encoding="utf-8")
    print(text)
