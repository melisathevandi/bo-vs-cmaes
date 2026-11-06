"""Clean-clone check: re-derive every table and figure from results/ in a fresh clone and compare with what is committed.

    uv run python code/check_reproduction.py --raw results/raw [--latex] [--keep]

Steps
  1. Required files exist in the committed tree (readme.md, authors.md, report/report.tex, code/, results/summary, ...).
  2. `git clone` of the committed state (HEAD) into a temporary directory, so anything that is only in the working
     tree (uncommitted code, untracked files) cannot be used. results/raw is not tracked (size), so it is linked into
     the clone from --raw.
  3. The derived files (everything in results/summary except the outputs of the scripts listed at the end, and all of
     report/generated and report/figures) are deleted in the clone, then every analysis script that reads results/raw
     is run, then make_tables.py and make_figures.py.
  4. Every file under results/summary, report/generated and report/figures in the clone must be identical to the
     committed one (`git status` in the clone shows no change). Figures are written without a creation date, so
     they compare byte for byte.
  5. Every macro of report/generated/numbers.tex appears in exactly one `% macros:` line of report/report.tex, and
     every listed prefix matches at least one macro.
  6. With --latex, report/report.tex compiles in the clone.
Not re-derived here (they run new computations rather than read results/raw): e0_invariance.py, anchor.py,
pilot_timing.py, diagnose_warm_start.py, e3_exploration_check.py, compute_plan.py, and the pilot run-time table. Their outputs are
tracked as they were run and kept in the clone.
Exit status 0 only if every check passes.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = ["readme.md", "authors.md", "predictions.md", "report/report.tex", "report/references.bib",
            "report/generated/numbers.tex", "code/make_figures.py", "code/make_tables.py", "code/run_experiment.py",
            "results/summary/e2_primary.csv", "uv.lock", "pyproject.toml"]
# (script, args, exit codes accepted). Order matters: cost_table before analyze_e1, tables and figures last.
PIPELINE = [
    ("analyze_e2.py", [], (0,)), ("descriptive_e2.py", [], (0,)), ("analyze_cma_long.py", [], (0,)),
    ("analyze_lowdim.py", [], (0,)), ("h5_expected_alpha.py", [], (0,)),
    ("h5_learning_time.py", ["--workers", "1"], (0,)),
    ("analyze_sensitivity.py", [], (0, 3)),
    ("analyze_a4.py", [], (0,)),
    ("analyze_a4.py", ["--experiment", "a4_d10_check", "--reference", "bo_cold_every5", "--variants", "bo_cold_every10",
                       "--budgets", "100", "300", "500", "--rule-budget", "500",
                       "--out", "results/summary/a4_d10_check.csv"], (0, 3)),
    ("cost_table.py", [], (0,)), ("analyze_e1.py", [], (0,)), ("analyze_e1_kappa1e4.py", [], (0,)),
    ("analyze_e3.py", [], (0,)), ("e3_sanity.py", [], (0,)), ("analyze_ablation.py", [], (0,)),
    ("make_tables.py", [], (0,)), ("make_figures.py", [], (0,)),
]


def run(cmd, cwd, env=None, check=True):
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True, check=check)


def link(target, link_path):
    """Directory junction on Windows (no admin rights needed), symlink elsewhere."""
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link_path), str(target)], check=True, capture_output=True)
    else:
        os.symlink(target, link_path)


def unlink(link_path):
    if os.name == "nt":
        os.rmdir(link_path)           # removes the junction only, never the target
    else:
        os.unlink(link_path)


def macros_ok(clone):
    nums = (clone / "report/generated/numbers.tex").read_text(encoding="utf-8")
    names = re.findall(r"\\newcommand\{\\num([A-Za-z]+)\}", nums)
    tex = (clone / "report/report.tex").read_text(encoding="utf-8")
    prefixes = []
    for line in tex.splitlines():
        m = re.match(r"%\s*macros:\s*(.*)", line)
        if m and m.group(1).strip() != "none":
            prefixes += [p.strip() for p in m.group(1).split(",") if p.strip()]
    problems = []
    cover = {n: [p for p in prefixes if n.startswith(p.rstrip("*"))] for n in names}
    problems += [f"macro {n} is in no section" for n, ps in cover.items() if not ps]
    problems += [f"macro {n} is in more than one section: {ps}" for n, ps in cover.items() if len(ps) > 1]
    problems += [f"prefix {p} matches no macro" for p in prefixes if not any(n.startswith(p.rstrip('*')) for n in names)]
    return len(names), problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "results" / "raw"), help="directory with the raw run JSONs")
    ap.add_argument("--latex", action="store_true", help="also compile report/report.tex in the clone")
    ap.add_argument("--keep", action="store_true", help="keep the temporary clone")
    args = ap.parse_args()
    failures = []

    tracked = set(run(["git", "ls-files"], ROOT).stdout.split())
    for f in REQUIRED:
        if f not in tracked:
            failures.append(f"not tracked: {f}")
    for d in ("code/", "results/summary/", "configs/", "report/figures/"):
        if not any(t.startswith(d) for t in tracked):
            failures.append(f"nothing tracked under {d}")
    print(f"[1] required files: {'ok' if not failures else failures}")

    raw = Path(args.raw).resolve()
    if not raw.is_dir():
        print(f"results/raw not found at {raw}: cannot re-derive anything")
        return 1
    tmp = Path(tempfile.mkdtemp(prefix="bo_vs_cmaes_clone_"))
    clone = tmp / "clone"
    run(["git", "clone", "--quiet", str(ROOT), str(clone)], ROOT)
    link(raw, clone / "results" / "raw")
    print(f"[2] cloned HEAD to {clone}; results/raw linked from {raw}")

    # Delete every derived file first, so that a script that silently writes nothing shows up as a missing file.
    kept = {"e0_rotation.csv", "anchor.csv", "pilot_timing.csv", "pilot_run_timing.csv", "a4_warm_start_diagnostic.csv",
            "e3_exploration_check.txt"}
    for f in (clone / "results" / "summary").iterdir():
        if f.name not in kept:
            f.unlink()
    for d in ("report/generated", "report/figures"):
        for f in (clone / d).iterdir():
            f.unlink()

    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="0")
    for script, extra, ok in PIPELINE:
        t0 = time.time()
        r = run([sys.executable, f"code/{script}", *extra], clone, env, check=False)
        status = "ok" if r.returncode in ok else f"FAILED (exit {r.returncode})"
        print(f"[3] {script} {' '.join(extra[:2])}: {status} ({time.time() - t0:.0f} s)")
        if r.returncode not in ok:
            failures.append(f"{script} exit {r.returncode}: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ''}")

    changed = run(["git", "status", "--porcelain", "--", "results/summary", "report/generated", "report/figures"],
                  clone).stdout.strip().splitlines()
    changed = [c for c in changed if "results/raw" not in c]
    print(f"[4] files that differ from the committed ones: {changed if changed else 'none'}")
    failures += [f"differs from committed: {c}" for c in changed]

    n, problems = macros_ok(clone)
    print(f"[5] numbers.tex macros: {n}; section coverage problems: {problems if problems else 'none'}")
    failures += problems

    if args.latex:
        r = run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "report.tex"], clone / "report", check=False)
        print(f"[6] LaTeX compile: {'ok' if r.returncode == 0 else 'FAILED'}")
        if r.returncode:
            failures.append("LaTeX compile failed")

    unlink(clone / "results" / "raw")
    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS" if not failures else "FAIL:\n  " + "\n  ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
