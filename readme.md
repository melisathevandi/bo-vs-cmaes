# Bayesian optimisation vs CMA-ES: when does each win?

DSA4212 final assignment. Report source: `report/report.tex` (figures in `report/figures/`, generated tables and numbers in
`report/generated/`). Authors: see [authors.md](authors.md). Repository: https://github.com/melisathevandi/bo-vs-cmaes

The study compares Gaussian-process Bayesian optimisation (ARD and isotropic Matérn kernels, LogEI and UCB), TuRBO-1,
CMA-ES (full, separable, IPOP) and Sobol sampling on synthetic test functions, with the log10 gap of the best true objective
value as the metric and all comparisons paired by seed. Hypotheses were written down before the runs (`predictions.md`, with dated
amendments); each analysis has an addendum fixed before the corresponding results were read (`analysis_addendum_*.md`).

## Environment

- [uv](https://docs.astral.sh/uv/); Python 3.12 (pinned in `.python-version`). Dependencies are pinned in `uv.lock`
  (pycma 4.5.0, BoTorch 0.18.1, GPyTorch 1.15.2, PyTorch 2.15 CPU, NumPy 2.5, SciPy 1.18).
- `uv sync` installs everything. All experiments run on CPU (`device: cpu` is stored in every result file).
- **One thread per worker.** `run_experiment.py` sets one torch/OpenMP thread per worker. This is not cosmetic: a GP-based
  trajectory (BO, TuRBO) is not identical when torch is multithreaded, so scripts that re-run a method directly must also
  call `torch.set_num_threads(1)` and set `OMP_NUM_THREADS=1`.
- Windows 11 was used; the code has no platform-specific parts except the clean-clone check (below), which uses a directory
  junction on Windows and a symlink elsewhere.

## Data

There is no external data. Test functions are generated from a seed (`code/problems.py`): ill-conditioned ellipsoid (aligned or
randomly rotated, condition number kappa), shifted Rosenbrock, low-effective-dimension quadratic, and a decoy-basin function.
Each result file in `results/raw/<experiment>/<job id>.json` holds the full trajectory, the timing split between method and
objective, the device, package versions, the git commit and a `code_dirty` flag (true if `code/` had uncommitted changes
when the run finished), and, for jobs run after commit 7e62b17, the worker count. The job id is the hash of the job
specification, so re-running a config skips finished jobs.

## Tests

```
uv run pytest
```

## Run the experiments

`code/run_experiment.py --config configs/<name>.toml [--workers N]` runs one config (`--quick`: 2 seeds, budget at most 60).
Wall times below are measured, with 7 workers unless stated (E2 low-effective dimension: 4 workers for most of it).

| Experiment | Config(s) | Runs | Wall time |
|---|---|---|---|
| Sensitivity grids (decision rule) | `sens_cmaes.toml`, `sens_bo.toml`, then `code/analyze_sensitivity.py` | 240 | n/a |
| E2 primary (rotation, d=10, kappa=1e4) | `e2_primary.toml`, `e2_cma_sigma015.toml`, `e2_cma_long.toml` | 540 + 90 + 60 | 13.1 h |
| E2 low effective dimension | `e2_lowdim.toml` | 200 | about 8 h (mixed 7 and 4 workers) |
| H5 CMA-ES learning time | `python code/h5_learning_time.py --workers 7` | 220 | 3 min |
| E1 (d = 2, 3, 5, 8, 10, kappa=100 ellipsoid and Rosenbrock) | `e1_d{2,3,5,8,10}.toml` and `e1_d*_cma_long.toml` | 600 + 200 | 14.2 h |
| E1 kappa=1e4 supplement | `e1k4_d2.toml`, `e1k4_d5.toml` | 240 | 1.0 h |
| E3 decoy basin, d=2 | `e3_d2.toml`, `e3_d2_long.toml` | 750 + 450 | 1.0 h |
| TuRBO rotation-penalty ablation | `turbo_ablation.toml` | 180 | 5.7 h |

Example: `uv run python code/run_experiment.py --config configs/e1_d2.toml --workers 7`. The TuRBO ablation needs the E2 primary
runs (default TuRBO is read from there). The CMA-ES long runs record the full trajectory to convergence and are used for the
break-even cost. Pilots and checks (timing pilot, refit-schedule check A4, correctness anchor, E0 rotation audit) are
`code/pilot_timing.py`, `analyze_a4.py`, `anchor.py`, `e0_invariance.py`, `diagnose_warm_start.py`.

**Raw results are not in the repository** (3.7 GB, git-ignored); `results/summary/` holds every aggregated table and is tracked.
Re-running everything takes about 2.5 days on a 7-core machine. TODO before submission: say here where the raw results are
provided, or confirm that the summaries are enough.

## Reproduce the tables and figures from the saved runs

Each analysis script reads `results/raw` and writes `results/summary/*.csv`; bootstrap seeds are fixed, so a rerun gives
identical files.

| Script | Output (in `results/summary/`) |
|---|---|
| `analyze_e2.py`, `descriptive_e2.py`, `analyze_cma_long.py` | `e2_primary.csv`, `e2_primary_descriptive.csv`, `e2_cma_long_reach.csv` |
| `analyze_lowdim.py` | `e2_lowdim_rho.csv`, `e2_lowdim_checkpoints.csv` |
| `h5_expected_alpha.py`, `h5_learning_time.py` | `h5_expected_alpha.csv`, `h5_learning_time.csv`, `h5_alpha.csv` |
| `cost_table.py`, `analyze_e1.py`, `analyze_e1_kappa1e4.py` | `cost_table.csv`, `cost_censored.csv`, `e1_*.csv` |
| `analyze_e3.py`, `e3_sanity.py` | `e3_success.csv`, `e3_tests.csv`, `e3_sanity.csv` |
| `analyze_ablation.py` | `turbo_ablation.csv`, `turbo_ablation_verdict.csv` |
| `analyze_sensitivity.py`, `analyze_a4.py` | `sensitivity_*.csv`, `a4_*.csv` |
| `make_tables.py` | `report/generated/tab_*.tex` |
| `make_figures.py` | `report/figures/fig*.pdf` and `report/generated/numbers.tex` (every number in the report text is a macro) |

Run everything and compare against the committed files in a fresh clone:

```
uv run python code/check_reproduction.py --raw results/raw [--latex]
```

It clones the committed state (so uncommitted files cannot be used), links `results/raw`, runs the scripts above, and fails if any
file under `results/summary`, `report/generated` or `report/figures` differs from the committed one, or if a macro of
`numbers.tex` is not listed in a section of `report/report.tex`. Outputs of `e0_invariance.py`, `anchor.py`, `pilot_timing.py`,
`diagnose_warm_start.py`, `e3_exploration_check.py` and `compute_plan.py` come from new computation, not from the saved runs; they
are tracked as they were run and are not part of that check.

## Layout

| Path | Contents |
|---|---|
| `code/problems.py` | Test functions |
| `code/methods/` | Sobol, CMA-ES (full, separable, IPOP), BO (ARD or isotropic Matérn, LogEI or UCB), TuRBO-1 |
| `code/runner.py`, `run_experiment.py` | One job per (config, method, seed); process pool; resume by job hash |
| `code/analyze_*.py`, `cost_table.py`, `h5_*.py` | Analyses (one per experiment) |
| `code/make_tables.py`, `make_figures.py` | Report tables, figures and numbers |
| `code/tests/` | Unit tests |
| `configs/` | One TOML per experiment |
| `results/raw/` | Per-run JSON (not tracked) |
| `results/summary/` | Aggregated tables |
| `report/` | `report.tex`, `references.bib`, `figures/`, `generated/` |
| `predictions.md`, `analysis_addendum_*.md` | Hypotheses and analysis rules, fixed before the results they govern |
| `notes/` | Lab notebook (process log), report notes (numbers with sources), plan, compute plan |
| `docs/` | Assignment guidelines and topic brief |
