"""H5: how many evaluations does CMA-ES need to learn the geometry, as a function of d?

    uv run python code/h5_learning_time.py --workers 7 [--quick]

Problem: rotated ellipsoid, kappa = 1e4, one instance per seed. CMA-ES (pycma) runs unbounded
in the unit-cube coordinates of the problem, x0 ~ U[0,1]^d, sigma0 = 0.3, default population,
with all termination criteria switched off (the loop below decides when to stop).

After every generation the learned geometry is scored by
    kappa_t = cond(C_t^(1/2) H C_t^(1/2)) = eigenvalue ratio of C_t H,
which starts at 1e4 (C_0 = I) and is 1 when C is proportional to H^-1. The learning time T is
the number of evaluations when kappa_t first drops below tau (primary tau = 10; 3 and 100 for
robustness). A run stops when kappa_t < 3, when it has used 100 d^2 evaluations (censored), or
when its best value reaches 1e-12 before kappa_t < tau (flagged "precision"). alpha is the slope
of an OLS fit of log T on log d; its 95% CI is a bootstrap over seeds within each d.

Writes results/raw/h5/*.json, results/summary/h5_learning_time.csv and h5_alpha.csv.
"""

import os

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw" / "h5"
SUMMARY = ROOT / "results" / "summary"
DIMS = [2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 80]
SEEDS = list(range(20))
KAPPA, SIGMA0, TAUS, F_PRECISION, CAP_FACTOR = 1e4, 0.3, (100, 10, 3), 1e-12, 100


def kappa_t(C, H):
    """Eigenvalue ratio of C H, computed through the symmetric matrix L^T H L with C = L L^T."""
    ev = np.linalg.eigvalsh(np.linalg.cholesky(C).T @ H @ np.linalg.cholesky(C))
    return float(ev[-1] / ev[0])


def one_run(d, seed):
    import cma
    from methods.base import method_seed
    from problems import ellipsoid

    f = ellipsoid(d, kappa=KAPPA, rotated=True, seed=seed)
    H = f.rotation.T @ np.diag(f.weights) @ f.rotation
    ms = method_seed(seed)
    x0 = np.random.default_rng(ms).uniform(size=d)
    opts = {"verbose": -9, "seed": ms % (2**31 - 1) + 1, "tolfun": 0, "tolx": 0, "tolfunhist": 0,
            "tolstagnation": 10**9, "tolflatfitness": 10**9, "tolconditioncov": 1e30,
            "tolupsigma": 1e30, "tolfacupx": 1e30, "maxiter": 10**9, "maxfevals": 10**12}
    es = cma.CMAEvolutionStrategy(x0, SIGMA0, opts)
    cap = CAP_FACTOR * d * d
    reached, kappas, evals, best = {}, [], 0, np.inf
    state = "cap"
    while evals < cap:
        X = es.ask()
        F = [f(np.asarray(x)) for x in X]
        es.tell(X, F)
        evals += len(F)
        best = min(best, min(F))
        k = kappa_t(es.sm.covariance_matrix, H)
        kappas.append((evals, k))
        for tau in TAUS:
            if k < tau and tau not in reached:
                reached[tau] = evals
        if 3 in reached:
            state = "done"
            break
        if best < F_PRECISION:
            state = "precision"
            break
    return {"d": d, "seed": seed, "cap": cap, "evals": evals, "state": state,
            "T": {str(t): reached.get(t) for t in TAUS}, "best_f": best,
            "popsize": es.sp.popsize, "kappa_trace": kappas[:: max(1, len(kappas) // 200)],
            "device": "cpu"}


def _job(args):
    d, seed, raw = args
    raw = Path(raw)
    path = raw / f"d{d}_s{seed}.json"
    if path.exists():
        return d, seed, 0.0
    t = time.perf_counter()
    rec = one_run(d, seed)
    rec["wall_s"] = time.perf_counter() - t
    raw.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.replace(path)
    return d, seed, rec["wall_s"]


def fit_alpha(table, tau, dmin, n_boot=10_000, seed=0):
    """OLS slope of log T on log d, bootstrap CI over seeds within each d."""
    t = table[(table.tau == tau) & table["T"].notna() & (table.d >= dmin)]
    groups = [g["T"].to_numpy() for _, g in t.groupby("d")]
    logd = np.log(np.array(sorted(t.d.unique()), float))
    def slope(samples):
        x = np.concatenate([np.full(len(s), ld) for s, ld in zip(samples, logd)])
        y = np.log(np.concatenate(samples))
        return float(np.polyfit(x, y, 1)[0])
    rng = np.random.default_rng(seed)
    boots = [slope([g[rng.integers(0, len(g), len(g))] for g in groups]) for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return slope(groups), float(lo), float(hi), len(t)


def summarise():
    rows = []
    for p in sorted(RAW.glob("*.json")):
        r = json.loads(p.read_text())
        for tau, T in r["T"].items():
            rows.append({"d": r["d"], "seed": r["seed"], "tau": int(tau), "T": T, "state": r["state"],
                         "evals": r["evals"], "cap": r["cap"], "popsize": r["popsize"]})
    table = pd.DataFrame(rows)
    table["censored"] = table["T"].isna() & (table["state"] == "cap")
    table["precision_before_tau"] = table["T"].isna() & (table["state"] == "precision")
    table.to_csv(SUMMARY / "h5_learning_time.csv", index=False)
    fits = []
    for tau in (10, 3, 100):
        for name, dmin in (("all d", 0), ("d >= 10", 10)):
            a, lo, hi, n = fit_alpha(table, tau, dmin)
            fits.append({"tau": tau, "fit": name, "alpha": a, "ci_lo": lo, "ci_hi": hi, "n_runs": n,
                         "contains_1.87": lo <= 1.87 <= hi, "contains_2": lo <= 2 <= hi})
    fits = pd.DataFrame(fits)
    fits.to_csv(SUMMARY / "h5_alpha.csv", index=False)
    pd.set_option("display.width", 200)
    print(fits.round(3).to_string(index=False))
    print("\nruns per d:", table[table.tau == 10].groupby("d").size().to_dict())
    print("censored (tau=10):", int(table[table.tau == 10].censored.sum()),
          "| precision before tau (tau=10):", int(table[table.tau == 10].precision_before_tau.sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--quick", action="store_true", help="d <= 10, 3 seeds, results/raw/h5_quick")
    args = ap.parse_args()
    if args.quick:
        RAW = ROOT / "results" / "raw" / "h5_quick"
        dims, seeds = [d for d in DIMS if d <= 10], SEEDS[:3]
    else:
        dims, seeds = DIMS, SEEDS
    jobs = [(d, s, str(RAW)) for d in sorted(dims, reverse=True) for s in seeds]     # longest first
    start = time.perf_counter()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, (d, s, secs) in enumerate(pool.imap_unordered(_job, jobs), 1):
            print(f"[{i}/{len(jobs)}] d={d} seed={s} {secs:.1f}s", flush=True)
    print(f"done in {time.perf_counter() - start:.0f}s")
    if not args.quick:
        summarise()
