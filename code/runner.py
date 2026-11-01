"""Run one (problem, method, seed) job and save its trajectory as a JSON file.

A job spec fully determines the run, so the file name is a hash of the spec: running the
same job twice reproduces the same file, and finished jobs are skipped (resume).

    spec = {"experiment": str, "budget": int, "seed": int, "instance_seed": int,
            "problem": {"family": ..., "d": ..., ...},
            "method": {"name": str, "module": str, "cfg": {...}}}

`seed` drives the method's randomness (through methods.base.method_seed, so its streams are
independent of the instance); `instance_seed` drives the problem instance (location of the
optimum, rotation). Paired comparisons give every method the same pair.
"""

import hashlib
import json
import subprocess
import time
from importlib import import_module
from pathlib import Path

import numpy as np

import problems
from methods.base import method_seed

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"


def make_problem(problem_spec, instance_seed):
    args = {k: v for k, v in problem_spec.items() if k not in ("name", "family")}
    return getattr(problems, problem_spec["family"])(seed=instance_seed, **args)


def job_id(spec):
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]


def _git_state():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "code"],
                                             cwd=ROOT, text=True).strip())
        return commit, dirty
    except Exception:
        return None, None


def _versions():
    out = {}
    for name in ("numpy", "scipy", "torch", "botorch", "gpytorch", "cma"):
        try:
            out[name] = import_module(name).__version__
        except Exception:
            out[name] = None
    return out


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"cannot serialise {type(obj)}")


def run_job(spec, out_dir=None):
    """Run a job unless its result file exists. Returns the path of the result file."""
    out_dir = Path(out_dir) if out_dir else RAW / spec["experiment"]
    path = out_dir / f"{job_id(spec)}.json"
    if path.exists():
        return path

    problem = make_problem(spec["problem"], spec["instance_seed"])
    method = import_module(f"methods.{spec['method']['module']}")
    start = time.perf_counter()
    result = method.run(problem, spec["budget"], method_seed(spec["seed"]),
                        **spec["method"]["cfg"])
    wall = time.perf_counter() - start

    commit, dirty = _git_state()
    record = {
        "spec": spec, "job_id": job_id(spec), "f_star": problem.f_star,
        "F": result["F"], "X": np.round(result["X"], 6), "t_method": result["t_method"],
        "t_eval": result["t_eval"], "wall_s": wall, "device": result["device"],
        "meta": result["meta"], "versions": _versions(),
        "git_commit": commit, "code_dirty": dirty,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, default=_jsonable))
    tmp.replace(path)                       # atomic: a crash never leaves a half-written file
    return path
