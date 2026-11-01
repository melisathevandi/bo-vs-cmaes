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

Prints markdown tables and writes notes/compute_plan.md.
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


def phases(every):
    """Ordered list of (name, note, group, [(n_runs, min_per_run)], fixed_cpu_h)."""
    b = lambda d, N, e=None: bo_minutes(d, N, every if e is None else e)
    t = lambda d, N: turbo_minutes(d, N)
    P = []
    P.append(("A4 check d=10, N=500", "15 seeds x {cold every 5, cold every 10}, rotated k=1e4", "core",
              [(15, b(10, 500, 5)), (15, b(10, 500, 10))], 0))
    P.append(("Sensitivity grids", "BO: 6 cfgs x 2 problems x 10 seeds (CMA-ES grid negligible)", "core",
              [(60, b(10, 500)), (60, b(5, 400))], 0))
    P.append(("E2 primary (N=500)", "k=1e4: aligned, rotated, replicate x 30 seeds x {ARD, iso, TuRBO}", "core",
              [(180, b(10, 500)), (90, t(10, 500))], 0))
    P.append(("E2 low-eff-dim (d=20, N=500)", "aligned+rotated x 20 seeds x {ARD, iso, TuRBO}; ARD +1.5 min diagnostics",
              "core", [(80, b(20, 500) + 1.5), (40, t(20, 500))], 0))
    P.append(("H5 CMA-ES runs", "11 dims x 20 seeds, cap 100 d^2 (alongside)", "core", [], 0.5))
    for d, N in [(2, 200), (3, 200), (5, 400), (8, 640), (10, 1000)]:
        P.append((f"E1 d={d}, N={N}", "BO ARD x {ellipsoid, Rosenbrock} x 20 seeds", "core", [(40, b(d, N))], 0))
    P.append(("E3 d=2 (N=200)", "5 widths x 30 seeds x {LogEI, UCB}", "core", [(300, b(2, 200))], 0))
    P.append(("E3 d=5 (N=400)", "5 widths x 30 seeds x {LogEI, UCB}", "core", [(300, b(5, 400))], 0))
    P.append(("E2 other kappa (N=500)", "k=1 aligned; k=1e2 aligned+rotated x 20 seeds x {ARD, iso, TuRBO}", "core",
              [(120, b(10, 500)), (60, t(10, 500))], 0))
    P.append(("Follow-up (reserved)", "one experiment chosen after E1-E3 (given: about 20 wall-h)", "core", [],
              FOLLOW_UP_WALL_H * WORKERS))
    P.append(("E1 d=20, N=1000", "BO ARD x {ellipsoid, Rosenbrock} x 20 seeds (last core)", "core_d20",
              [(40, b(20, 1000))], 0))
    P.append(("E2 secondary N=1000", "k=1e4 aligned+rotated x 10 seeds x {ARD, iso, TuRBO}", "extra",
              [(40, b(10, 1000)), (20, t(10, 1000))], 0))
    P.append(("A8c noise (E5)", "d=5 N=300, 2 noise levels x 20 seeds, BO ARD", "extra", [(40, b(5, 300))], 0))
    P.append(("A8d long-budget probe", "d=10 k=1e4 rotated, N=2000, 10 seeds (extrapolated beyond n=1000)", "extra",
              [(10, b(10, 2000))], 0))
    P.append(("A8e second family", "4 BBOB-style functions, d=10, N=500, 10 seeds", "extra", [(40, b(10, 500))], 0))
    return P


def cpu_h(parts, fixed):
    return sum(n * m for n, m in parts) / 60 + fixed


def table(every, start):
    ph = phases(every)
    lines = [f"| # | Experiment | Runs | CPU-h | Wall-h ({WORKERS} workers) | Group |", "|---|---|---|---|---|---|"]
    cum, sched = 0.0, []
    sums = {"core": 0.0, "core_d20": 0.0, "extra": 0.0}
    for i, (name, note, group, parts, fixed) in enumerate(ph, 1):
        c = cpu_h(parts, fixed)
        runs = sum(n for n, _ in parts)
        sums[group] += c
        if group != "extra":
            cum += c / WORKERS * (1 + CONTINGENCY)
            sched.append((name, start + timedelta(hours=cum)))
        lines.append(f"| {i} | {name}: {note} | {runs or '-'} | {c:.1f} | {c / WORKERS:.1f} | {group} |")
    w = {k: v / WORKERS for k, v in sums.items()}
    core_wo, core_w = w["core"], w["core"] + w["core_d20"]
    lines += ["",
              f"- Core without E1 d=20: **{core_wo:.0f} wall-h** ({core_wo * (1 + CONTINGENCY):.0f} with 15% contingency; "
              f"range {LOW * core_wo * (1 + CONTINGENCY):.0f} to {HIGH * core_wo * (1 + CONTINGENCY):.0f})",
              f"- Core with E1 d=20: **{core_w:.0f} wall-h** ({core_w * (1 + CONTINGENCY):.0f} with 15% contingency; "
              f"range {LOW * core_w * (1 + CONTINGENCY):.0f} to {HIGH * core_w * (1 + CONTINGENCY):.0f})",
              f"- Extras (E2 secondary, A8 c-e), only if the core finishes early: {w['extra']:.0f} wall-h",
              "", "Cumulative finish times of the core phases in the agreed order (central estimate with "
              f"15% contingency, starting {start:%d %b %H:%M}; hard stop {HARD_STOP:%d %b %H:%M}):", ""]
    lines += [f"- {n}: {t:%d %b %H:%M}" + ("  **after the hard stop**" if t > HARD_STOP else "") for n, t in sched]
    return "\n".join(lines), core_wo, core_w


if __name__ == "__main__":
    fac, walls = calibrate()
    start = datetime.now()
    print(f"FACTOR = measured / integral = {fac:.2f} (measured {', '.join(f'{w:.1f}' for w in walls)} min vs integral "
          f"{_integral_minutes(10, 500, 5):.1f} min); LOAD7 assumption = {LOAD7}")
    out, totals = [], {}
    for every in (5, 10):
        tab, wo, w = table(every, start)
        totals[every] = (wo, w)
        out.append(f"### BO refit every {every} iterations (n > 100)\n\n{tab}\n")
    text = "\n".join(out)
    (ROOT / "notes" / "compute_plan.md").write_text(
        "# Compute plan (planning estimate, not a result)\n\n"
        f"BO cost = timing-pilot integral x FACTOR ({fac:.2f}, measured/integral at d=10, N=500, 3 real runs, "
        f"folding in real-trajectory fit cost and the load of the ~6 busy processes during that run) x LOAD7 "
        f"({LOAD7}, measured: A4 check at d=10 on 7 workers, 47.0 min per every-5 run vs 37.7 min in the calibration run). "
        f"TuRBO cost = measured pilot median x LOAD7. The A4-check row is the same in both tables.\n\n" + text,
        encoding="utf-8")
    print(text)
