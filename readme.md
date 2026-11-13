# Bayesian optimisation vs CMA-ES: when does each win?

Code, configurations, aggregated results and report source for a DSA4212 study. Authors: see [authors.md](authors.md).
Repository: https://github.com/melisathevandi/bo-vs-cmaes

## Summary

The study asks under which conditions Gaussian-process Bayesian optimisation (BO) or CMA-ES finds better solutions for
a given number of function evaluations, and when BO's model overhead is worth paying. Methods compared: BO with ARD or
isotropic Matérn kernels and LogEI or UCB acquisition (BoTorch), TuRBO-1, CMA-ES in full, separable and IPOP variants
(pycma), and Sobol sampling as a model-free reference. The metric is the log10 gap of the best true objective value;
all comparisons are paired by seed. Experiments: E0, rotation invariance of CMA-ES with and without box bounds; E1,
dimension against budget on a rotated ellipsoid and Rosenbrock (d = 2 to 10, with a kappa = 1e4 supplement), with the
break-even evaluation cost c\*; E2, rotation and axis alignment on an ill-conditioned ellipsoid (d = 10, kappa = 1e4) and
a low-effective-dimension function (d = 20, two relevant directions); H5, the scaling of CMA-ES's covariance learning time
with d; E3, a decoy-basin multimodal function (d = 2); and a follow-up ablation of TuRBO's rotation penalty. Hypotheses
are fixed in `predictions.md` (with dated amendments), and the analysis of each experiment in an `analysis_addendum_*.md`
file, both committed before the corresponding results were analysed.

## Repository layout

| Path | Contents |
|---|---|
| `code/problems.py` | Test functions (ellipsoid, Rosenbrock, low-effective-dimension quadratic, decoy basin) |
| `code/methods/` | Sobol, CMA-ES (full, separable, IPOP), BO (ARD or isotropic Matérn, LogEI or UCB), TuRBO-1 |
| `code/runner.py`, `code/run_experiment.py` | One job per (config, problem, method, seed); process pool; resume by job hash |
| `code/run_core.sh` | Runs the sensitivity grids, E2 and E1 in sequence, with retries |
| `code/analyze_*.py`, `code/cost_table.py`, `code/descriptive_e2.py`, `code/h5_*.py`, `code/e3_sanity.py` | Analyses: `results/raw` -> `results/summary` |
| `code/e0_invariance.py`, `code/anchor.py`, `code/pilot_timing.py`, `code/diagnose_warm_start.py`, `code/e3_exploration_check.py` | Checks that run their own computations |
| `code/compute_plan.py` | Compute estimate per experiment, written to `notes/compute_plan.md` (planning only) |
| `code/make_tables.py`, `code/make_figures.py` | Report tables, figures and `numbers.tex` |
| `code/check_reproduction.py` | Re-derives all tables and figures in a clean clone and compares them with the committed files |
| `code/tests/` | Unit tests |
| `configs/` | One TOML file per experiment |
| `results/raw/` | Per-run JSON files (not tracked, see below) |
| `results/summary/` | Aggregated tables (tracked) |
| `report/` | Report source: `report.tex`, `references.bib`, `figures/`, `generated/` |
| `predictions.md`, `analysis_addendum_*.md` | Hypotheses and analysis rules, fixed before the results they govern |
| `notes/` | Lab notebook (dated process record), results summary with sources, compute estimate |
| `docs/` | Assignment guidelines and topic description |

## Installation

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.12 (pinned in `.python-version`; `pyproject.toml` requires
>=3.12,<3.13). Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`: botorch 0.18.1, cma (pycma) 4.5.0,
gpytorch 1.15.2, matplotlib 3.11.2, numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, torch 2.15.0; pytest in the `dev` group.

```
uv sync
```

All experiments run on CPU (`device: cpu` is stored in every result file). `run_experiment.py` uses one torch/OpenMP
thread per worker. GP-based trajectories (BO, TuRBO) are not identical under multithreaded torch, so a script that re-runs
a method directly must also call `torch.set_num_threads(1)` and set `OMP_NUM_THREADS=1`. Compiling the report with
`check_reproduction.py --latex` additionally requires a LaTeX installation with `latexmk`. The experiments were run on
Windows 11; the only platform-specific code is the directory link in `check_reproduction.py` (junction on Windows,
symlink elsewhere).

## Data

There is no external data. Test functions are generated from a seed (`code/problems.py`). Each run writes
`results/raw/<experiment>/<job id>.json` with the full trajectory, the time split between method and objective, the
device, package versions, the git commit and a `code_dirty` flag (true if `code/` had uncommitted changes when the run
finished), and, for jobs run after commit 7e62b17, the worker count. The job id is the hash of the job specification, so
re-running a config skips finished jobs.

### Raw results are not included

`results/raw/` (about 3.7 GB, one JSON file per run, all experiments below) is excluded by `.gitignore` and is not part
of the repository. The aggregated tables in `results/summary/` are included, as are the report tables
(`report/generated/`) and figures (`report/figures/`).

The following scripts re-derive `results/summary/` from `results/raw/` (the raw files are needed; they are regenerated
by re-running the configs):

| Script | Output in `results/summary/` |
|---|---|
| `analyze_e2.py`, `descriptive_e2.py`, `analyze_cma_long.py` | `e2_primary.csv`, `e2_primary_descriptive.csv`, `e2_cma_long_reach.csv` |
| `analyze_lowdim.py` | `e2_lowdim_rho.csv`, `e2_lowdim_checkpoints.csv` |
| `h5_expected_alpha.py`, `h5_learning_time.py` | `h5_expected_alpha.csv` (no raw files needed), `h5_learning_time.csv`, `h5_alpha.csv` |
| `analyze_sensitivity.py` | `sensitivity_cmaes.csv`, `sensitivity_bo.csv` |
| `analyze_a4.py` (two calls, see `check_reproduction.py`) | `a4_refit_check.csv`, `a4_d10_check.csv` |
| `cost_table.py`, `analyze_e1.py`, `analyze_e1_kappa1e4.py` | `cost_table.csv`, `cost_censored.csv`, `e1_regime_map.csv`, `e1_h1.csv`, `e1_h4.csv`, `e1_cost_median_target.csv`, `e1_overhead.csv`, `e1_kappa1e4_h1.csv` |
| `analyze_e3.py`, `e3_sanity.py` | `e3_success.csv`, `e3_tests.csv`, `e3_sanity.csv` |
| `analyze_ablation.py` | `turbo_ablation.csv`, `turbo_ablation_verdict.csv` |

Files in `results/summary/` that are not derived from `results/raw/`:
- `e0_rotation.csv` (`e0_invariance.py`), `anchor.csv` (`anchor.py`), `pilot_timing.csv` (`pilot_timing.py`) and
  `a4_warm_start_diagnostic.csv` (`diagnose_warm_start.py`) come from scripts that run their own computations.
- `e3_exploration_check.txt` is the console output of `e3_exploration_check.py` (about 17 minutes on one core).
- `pilot_run_timing.csv` summarises the wall times of the `pilot_bo` and `pilot_turbo` runs; no script in `code/`
  writes it.

Without `results/raw/`, `make_tables.py` reproduces every table in `report/generated/` from `results/summary/`, and
`make_figures.py` reproduces Figures 4 and 5. Figures 1 and 2, the regime-map panels of Figure 3 and 30 of the 122
macros in `numbers.tex` need `results/raw/`.

## Quick checks (minutes)

```
uv run pytest                                                      # 57 unit tests, under 1 min
uv run python code/run_experiment.py --config configs/smoke.toml   # tiny budgets: runner, resume and JSON output
uv run python code/run_experiment.py --config configs/e1_d2.toml --quick   # 2 seeds, budget <= 60
uv run python code/h5_learning_time.py --quick                     # d <= 10, 3 seeds
uv run python code/make_tables.py                                  # report tables from results/summary
```

With `results/raw/` present, each `analyze_*.py` script takes seconds to minutes, and

```
uv run python code/check_reproduction.py --raw results/raw [--latex]
```

clones the committed state, links `results/raw`, deletes every derived file in the clone, runs the analysis scripts,
`make_tables.py` and `make_figures.py`, and fails if any file under `results/summary`, `report/generated` or
`report/figures` differs from the committed one, or if a macro of `numbers.tex` is not listed in a section of
`report/report.tex`. Bootstrap seeds are fixed and figures are written without a creation date, so the comparison is
byte for byte.

## Full runs (hours)

`code/run_experiment.py --config configs/<name>.toml [--workers N]` runs one config. Measured wall times, 7 workers unless
stated; `notes/compute_plan.md` gives the planning estimates per experiment.

| Experiment | Config(s) | Runs | Wall time |
|---|---|---|---|
| Sensitivity grids | `sens_cmaes.toml`, `sens_bo.toml`, then `code/analyze_sensitivity.py` | 240 | about 3.6 h (BO grid; CMA-ES grid 18 s) |
| A4 refit-schedule checks | `a4_refit_check.toml`, `a4_d10_check.toml` | 80 + 30 | 29-47 min per d = 10 run |
| E2 primary (d=10, kappa=1e4) | `e2_primary.toml`, `e2_cma_sigma015.toml`, `e2_cma_long.toml` | 540 + 90 + 60 | 13.1 h |
| E2 low effective dimension | `e2_lowdim.toml` | 200 | about 8 h (mixed 7 and 4 workers) |
| H5 CMA-ES learning time | `uv run python code/h5_learning_time.py --workers 7` | 220 | 3 min |
| E1 (d = 2, 3, 5, 8, 10) | `e1_d{2,3,5,8,10}.toml`, `e1_d{2,3,5,8,10}_cma_long.toml` | 600 + 200 | 14.2 h |
| E1 kappa=1e4 supplement | `e1k4_d2.toml`, `e1k4_d5.toml` | 240 | 1.0 h |
| E3 decoy basin, d=2 | `e3_d2.toml`, `e3_d2_long.toml` | 750 + 450 | 1.0 h |
| TuRBO rotation-penalty ablation | `turbo_ablation.toml` | 180 | 5.7 h |

Example: `uv run python code/run_experiment.py --config configs/e1_d2.toml --workers 7`. The TuRBO ablation reuses the
default TuRBO runs of E2 primary. Re-running everything takes about 2.5 days on a 7-core machine. Timing pilots and
checks: `code/pilot_timing.py`, `code/anchor.py`, `code/e0_invariance.py`, `code/diagnose_warm_start.py`
(configs `pilot_bo.toml`, `pilot_turbo.toml`).

## Report

The report source is in `report/` (`report.tex`, `references.bib`, `figures/`, `generated/`). The figures and tables it
is expected to include, and the code and data that produce them:

| Item | File | Produced by | Inputs |
|---|---|---|---|
| Figure 1: E2 primary, final gap and rotation penalty | `report/figures/fig1_e2_primary.pdf` | `make_figures.py` | `results/raw/e2_primary` |
| Figure 2: E2 low effective dimension, convergence, relevance ratio, GP rank correlation | `report/figures/fig2_e2_lowdim.pdf` | `make_figures.py` | `results/raw/e2_lowdim` |
| Figure 3: E1 regime maps, H1 statistic, H5 learning time | `report/figures/fig3_e1_h5.pdf` | `make_figures.py` | `results/raw/e1_d*`, `e1_h1.csv`, `e1_kappa1e4_h1.csv`, `h5_learning_time.csv`, `h5_alpha.csv` |
| Figure 4: E3 success probability against basin width | `report/figures/fig4_e3.pdf` | `make_figures.py` | `e3_success.csv` |
| Figure 5: TuRBO ablation | `report/figures/fig5_ablation.pdf` | `make_figures.py` | `turbo_ablation.csv` |
| Prediction vs outcome | `report/generated/tab_predictions.tex` | `make_tables.py` | `e1_h1.csv`, `e1_kappa1e4_h1.csv`, `e2_primary.csv`, `e2_lowdim_rho.csv`, `e3_tests.csv`, `e1_h4.csv`, `h5_alpha.csv`, `turbo_ablation*.csv` |
| E2 primary statistics | `report/generated/tab_e2.tex` | `make_tables.py` | `e2_primary.csv` |
| Relevance ratio and rank correlation | `report/generated/tab_lowdim.tex` | `make_tables.py` | `e2_lowdim_rho.csv` |
| Break-even cost c\* | `report/generated/tab_cstar.tex` | `make_tables.py` | `cost_table.csv` |
| H5 exponent | `report/generated/tab_h5.tex` | `make_tables.py` | `h5_alpha.csv` |
| E3 success | `report/generated/tab_e3.tex` | `make_tables.py` | `e3_success.csv` |
| TuRBO ablation | `report/generated/tab_ablation.tex` | `make_tables.py` | `turbo_ablation.csv` |
| Numbers in the text (122 macros) | `report/generated/numbers.tex` | `make_figures.py` | `results/summary/*.csv`, `results/raw` |

CSV files without a path are in `results/summary/`. The section of the report that uses each figure, table and macro
is listed in comments in `report/report.tex`.

## Tests

```
uv run pytest
```
