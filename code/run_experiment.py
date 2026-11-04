"""Run every job of an experiment config in parallel.

    uv run python code/run_experiment.py --config configs/smoke.toml --workers 4
    uv run python code/run_experiment.py --config configs/e2_geometry.toml --quick

Config (TOML): `experiment`, `budget`, `seeds`, optional `instance_seed_offset`, a list of
[[problems]] (family + arguments, optional `budget`, `instance_seed_offset`) and a list of
[[methods]] (name, module, [methods.cfg]). Jobs are the product problems x methods x seeds.
Finished jobs are skipped, so an interrupted run can simply be started again.
"""

import os

# One thread per worker, set before numpy/torch are imported (also in spawned children).
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import multiprocessing as mp
import time
import tomllib
from pathlib import Path

QUICK_SEEDS, QUICK_BUDGET = 2, 60


def build_jobs(cfg, quick=False):
    seeds = cfg["seeds"][:QUICK_SEEDS] if quick else cfg["seeds"]
    jobs = []
    for prob in cfg["problems"]:
        for method in cfg["methods"]:
            for seed in seeds:
                budget = prob.get("budget", cfg["budget"])
                if quick:
                    budget = min(budget, QUICK_BUDGET)
                spec_prob = {k: v for k, v in prob.items()
                             if k not in ("budget", "instance_seed_offset")}
                offset = prob.get("instance_seed_offset", cfg.get("instance_seed_offset", 0))
                jobs.append({
                    "experiment": cfg["experiment"] + ("_quick" if quick else ""),
                    "budget": budget, "seed": seed, "instance_seed": seed + offset,
                    "problem": spec_prob,
                    "method": {"name": method["name"], "module": method["module"],
                               "cfg": method.get("cfg", {})},
                })
    return jobs


def _worker(args):
    spec, workers = args
    import json
    import torch
    torch.set_num_threads(1)
    from runner import RAW, job_id, run_job
    existed = (RAW / spec["experiment"] / f"{job_id(spec)}.json").exists()
    t = time.perf_counter()
    path = run_job(spec)
    if not existed:
        # record how many workers shared the machine; kept out of the spec so the job hash is unchanged
        record = json.loads(path.read_text())
        record["workers"] = workers
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record))
        tmp.replace(path)
    return spec, path, time.perf_counter() - t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--quick", action="store_true", help="2 seeds, budget <= 60")
    args = ap.parse_args()

    cfg = tomllib.loads(Path(args.config).read_text())
    jobs = build_jobs(cfg, args.quick)
    print(f"{len(jobs)} jobs, {args.workers} workers", flush=True)
    start = time.perf_counter()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, (spec, path, secs) in enumerate(pool.imap_unordered(_worker, [(j, args.workers) for j in jobs]), 1):
            print(f"[{i}/{len(jobs)}] {spec['problem'].get('name', spec['problem']['family'])} "
                  f"{spec['method']['name']} seed={spec['seed']} {secs:.1f}s", flush=True)
    print(f"done in {time.perf_counter() - start:.0f}s")
