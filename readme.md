# Bayesian optimisation vs CMA-ES: when does each win?

Author: Melisa Abigail Thevandi (see [authors.md](authors.md)). Repository: https://github.com/melisathevandi/bo-vs-cmaes

Code, configurations, aggregated and raw results, and the report source for a DSA4212 study.

## Summary

The study asks under which conditions Gaussian-process Bayesian optimisation (BO) or CMA-ES finds better solutions for
a given number of function evaluations, and when BO's model overhead is worth paying. Methods compared: BO with ARD or
isotropic Matern kernels and LogEI or UCB acquisition (BoTorch), TuRBO-1, CMA-ES in full, separable and IPOP variants
(pycma), and Sobol sampling as a model-free reference. The metric is the log10 gap of the best true objective value;
all comparisons are paired by seed. Experiments: E0, rotation invariance of CMA-ES with and without box bounds; E1,
dimension against budget (d = 2 to 10) on a rotated ellipsoid (kappa = 100 main run, kappa = 1e4 supplement) and on
Rosenbrock, with the break-even evaluation cost c*; E2, rotation and axis alignment on an ill-conditioned ellipsoid
(d = 10, kappa = 1e4) and a low-effective-dimension function (d = 20, two relevant directions); H5, the scaling of
CMA-ES's covariance learning time with d; E3, a decoy-basin multimodal function (d = 2); and a follow-up ablation of
TuRBO's rotation penalty. Hypotheses are fixed in `predictions.md` (with dated amendments) and the analysis rules of
E1, E2, E3 and the supplementary runs in `analysis_addendum_*.md` (see Pre-registration).

## Repository layout

| Path | Contents |
|---|---|
| `readme.md`, `authors.md` | This file; author details |
| `pyproject.toml`, `uv.lock`, `.python-version` | Environment: dependencies, exact locked versions, Python 3.12 |
| `code/problems.py` | Test functions (ellipsoid, Rosenbrock, low-effective-dimension quadratic, decoy basin) |
| `code/methods/` | Sobol, CMA-ES (full, separable, IPOP), BO (ARD or isotropic Matern, LogEI or UCB), TuRBO-1 |
| `code/runner.py`, `code/run_experiment.py` | One job per (config, problem, method, seed); process pool; resume by job hash |
| `code/run_core.sh` | Chain: sensitivity grids, E2, H5, E1, with retries (see the warning under Reproduction) |
| `code/analyze_*.py`, `code/cost_table.py`, `code/descriptive_e2.py`, `code/h5_*.py`, `code/e3_sanity.py` | Analyses: `results/raw` -> `results/summary` |
| `code/e0_invariance.py`, `code/anchor.py`, `code/pilot_timing.py`, `code/diagnose_warm_start.py`, `code/e3_exploration_check.py` | Checks that run their own computations |
| `code/make_tables.py`, `code/make_figures.py` | Report tables, figures and `numbers.tex` |
| `code/check_reproduction.py` | Re-derives all tables and figures in a clean clone and compares them with the committed files |
| `code/tests/` | Unit tests (pytest) |
| `configs/` | One TOML file per experiment (problems, methods, budget, seeds) |
| `results/raw/` | Per-run JSON files, one per job: 3,769 files, about 266 MB |
| `results/summary/` | Aggregated CSV tables derived from `results/raw` (and a few from scripts that run their own computations) |
| `report/` | Report source: `report.tex`, `references.bib`, `figures/` (PDF), `generated/` (tables and `numbers.tex`) |
| `predictions.md`, `analysis_addendum_*.md` | Hypotheses and analysis rules, fixed before the results they govern |
| `notes/` | Lab notebook (dated process record) |

## Setup

Python 3.12 (`.python-version`; `pyproject.toml` requires >=3.12,<3.13). The environment is defined by `pyproject.toml`
and pinned in `uv.lock`; no `requirements.txt` or conda file is provided. With [uv](https://docs.astral.sh/uv/):

```
git clone https://github.com/melisathevandi/bo-vs-cmaes.git
cd bo-vs-cmaes
uv sync
```

Pinned versions (from `uv.lock`): botorch 0.18.1, cma (pycma) 4.5.0, gpytorch 1.15.2, torch 2.15.0, numpy 2.5.4,
pandas 3.1.0, scipy 1.18.1, matplotlib 3.11.2, pytest 9.2.0 (dev group). Without uv, in a Python 3.12 virtual
environment (`python3.12 -m venv .venv`, then activate it), the direct dependencies can be installed with pip:

```
pip install botorch==0.18.1 cma==4.5.0 gpytorch==1.15.2 torch==2.15.0 numpy==2.5.4 pandas==3.1.0 scipy==1.18.1 matplotlib==3.11.2 pytest==9.2.0
```

and every `uv run python code/...` command below becomes `python code/...`. Transitive dependencies are then resolved
by pip rather than by the lock file.

All experiments ran on CPU (`device: cpu` is stored in every result file). `run_experiment.py` uses one torch/OpenMP
thread per worker. A script that re-runs a BO or TuRBO job directly must call `torch.set_num_threads(1)` and set
`OMP_NUM_THREADS=1`, because GP trajectories are not identical under multithreaded torch. The experiments ran on
Windows 11 (AMD Ryzen 9 5900HS laptop, about 15 GB RAM, no GPU) with 7 worker processes.

There is no external data: test functions are generated from a seed (`code/problems.py`).

## Reproduction

Checks that take minutes:

```
uv run pytest                                                              # 57 unit tests, under 1 min
uv run python code/run_experiment.py --config configs/smoke.toml           # tiny budgets: runner, resume, JSON output
uv run python code/run_experiment.py --config configs/e1_d2.toml --quick   # 2 seeds, budget <= 60
uv run python code/h5_learning_time.py --quick                             # d <= 10, 3 seeds
uv run python code/check_reproduction.py                                   # see below
```

`check_reproduction.py [--latex]` clones the committed state (including `results/raw`), deletes every derived file in the
clone, runs the analysis scripts, `make_tables.py` and `make_figures.py`, and fails if any file under `results/summary`,
`report/generated` or `report/figures` differs from the committed one, or if a macro of `numbers.tex` is not listed in a
section of `report/report.tex`. With `--latex` it also compiles the report. It reads the tracked `results/raw`; it does
not re-run any experiment.

### Experiments

`uv run python code/run_experiment.py --config configs/<name>.toml [--workers N]` runs one config (default 6 workers);
each job writes `results/raw/<experiment>/<job id>.json` and finished jobs are skipped on re-runs. The analysis scripts
take no arguments unless stated. Wall times are measured on the machine above with 7 workers.

| Experiment | Run | Analysis -> output in `results/summary/` | Wall time |
|---|---|---|---|
| Correctness anchor | `code/anchor.py` | `anchor.csv` | not recorded |
| E0 (CMA-ES rotation, box) | `code/e0_invariance.py` (no config; own computations) | `e0_rotation.csv` | not recorded |
| Sensitivity grids | `sens_cmaes.toml`, `sens_bo.toml` | `analyze_sensitivity.py` -> `sensitivity_cmaes.csv`, `sensitivity_bo.csv` | BO grid about 3.6 h; CMA-ES grid seconds |
| A4 refit-schedule checks | `a4_refit_check.toml`, `a4_d10_check.toml` | `analyze_a4.py` -> `a4_refit_check.csv`; the second call (arguments in `code/check_reproduction.py`) -> `a4_d10_check.csv` | about 4 h together |
| E2 primary (d = 10, kappa = 1e4) | `e2_primary.toml` | `analyze_e2.py`, `descriptive_e2.py` -> `e2_primary.csv`, `e2_primary_descriptive.csv` | 13.1 h |
| E2 secondary (CMA-ES sigma0 = 0.15) | `e2_cma_sigma015.toml` | `analyze_e2.py` (secondary rows of `e2_primary.csv`) | seconds |
| E2 CMA-ES long runs | `e2_cma_long.toml` | `analyze_cma_long.py` -> `e2_cma_long_reach.csv` | 2 min |
| E2 low effective dimension | `e2_lowdim.toml` | `analyze_lowdim.py` -> `e2_lowdim_rho.csv`, `e2_lowdim_checkpoints.csv` | about 8 h |
| H5 (CMA-ES learning time) | `code/h5_learning_time.py --workers 7` (no config) | `h5_learning_time.csv`, `h5_alpha.csv`; `h5_expected_alpha.py` -> `h5_expected_alpha.csv` | 3 min |
| E1 (d = 2, 3, 5, 8, 10) | `e1_d{2,3,5,8,10}.toml` | `cost_table.py` then `analyze_e1.py` -> `e1_regime_map.csv`, `e1_h1.csv`, `e1_h4.csv`, `e1_cost_median_target.csv`, `e1_overhead.csv`, `cost_table.csv`, `cost_censored.csv` | d = 10 alone 14.2 h; all five about 19 h |
| E1 long CMA-ES runs | `e1_d{2,3,5,8,10}_cma_long.toml` | read by `cost_table.py` (above) | under 1 min |
| E1 kappa = 1e4 supplement (d = 2, 5) | `e1k4_d2.toml`, `e1k4_d5.toml` | `analyze_e1_kappa1e4.py` -> `e1_kappa1e4_h1.csv` | 1.0 h |
| E3 (decoy basin, d = 2) | `e3_d2.toml`, `e3_d2_long.toml` | `analyze_e3.py`, `e3_sanity.py` -> `e3_success.csv`, `e3_tests.csv`, `e3_sanity.csv` | 1.0 h |
| TuRBO ablation | `turbo_ablation.toml` (reuses the default TuRBO runs of E2 primary) | `analyze_ablation.py` -> `turbo_ablation.csv`, `turbo_ablation_verdict.csv` | 5.7 h |
| Timing pilots | `pilot_bo.toml`, `pilot_turbo.toml`; `code/pilot_timing.py` | `pilot_timing.csv` | not recorded |
| Warm-start diagnostic | `code/diagnose_warm_start.py` | `a4_warm_start_diagnostic.csv` | not recorded |

Notes on these outputs:
- `e3_exploration_check.txt` is saved console output of `code/e3_exploration_check.py` (UCB with three exploration
  weights, 6 seeds, about 17 minutes on one core); the script prints and does not write the file.
- `pilot_run_timing.csv` has no generating script. It lists the wall time (`wall_s / 60`) of the 9 runs in
  `results/raw/pilot_bo` and `results/raw/pilot_turbo` and the TuRBO restart counts from each run's `meta`.
- `bash code/run_core.sh [workers]` runs sensitivity grids, E2 primary, E2 low-dimension, H5 and E1 in sequence (retrying
  failed steps). It also runs `git add` and `git commit` after the sensitivity analysis, so run the configs individually
  unless that is intended.
- Re-running every row takes about 2.3 days (about 55 h summed over the rows). CPU time depends on the number of parallel
  workers and on other load; one BO run at d = 10 with 1000 evaluations took 1.9 to 2.8 h.

### Figures, tables and numbers

```
uv run python code/make_tables.py     # report/generated/tab_*.tex from results/summary
uv run python code/make_figures.py    # report/figures/fig1..fig5 (PDF) and report/generated/numbers.tex (122 macros)
```

Figures 1 to 5 are E2 primary, E2 low dimension, E1 and H5, E3, and the TuRBO ablation. `make_figures.py` reads
`results/raw` directly for Figures 1 and 2, the regime-map panels of Figure 3 and part of `numbers.tex`, and the CSV
files in `results/summary` for the rest. The section of the report that uses each figure, table and macro is listed in
comments in `report/report.tex`.

## Seeds and determinism

- Main runs use seeds 0-29 (E2 primary, E2 secondary and long runs, E3, TuRBO ablation) or 0-19 (E1, E1 long runs,
  E1 kappa = 1e4 supplement, E2 low dimension, H5). Tuning and checks use held-out seeds 1000 and above (sensitivity grids
  1000-1009, A4 checks 1000-1009 and 1000-1014), which do not overlap the main runs.
- Designs are paired by seed: for a given seed, all methods see the same problem instance, and aligned and rotated
  instances share the optimum. The aligned replicate used as a noise floor is an independent instance drawn with
  `instance_seed_offset = 5000`.
- Instance randomness and method randomness use independent streams (`method_seed` in `code/methods/base.py`;
  `code/tests/test_seeds.py`).
- Bootstrap analyses use fixed seeds, so analysis outputs are reproducible byte for byte (`check_reproduction.py`).
  Re-running a BO or TuRBO job with one thread reproduced stored trajectories exactly in the check that was done (9 TuRBO
  jobs); equality across machines or library versions was not tested.

## Pre-registration

Hypotheses were committed before the main runs and analysis rules before the results they govern (commit dates from
`git log`):

- `predictions.md`: frozen in ac4c90f (2026-11-01). Dated amendments: 0363ee5 (H1 tested at kappa = 100; the d = 20 part
  not run), f65181c (TuRBO ablation prediction, before its runs) and b78a3f4 (ablation noise floor).
- `analysis_addendum_e2.md`: e7a117d (2026-11-04), before any E2 result was read. `analysis_addendum_e2_cma_long.md`:
  10d1332. `analysis_addendum_e1_cma_long.md`: 4d901fa.
- `analysis_addendum_e1.md`: 4680bb8 (2026-11-06), before any E1 gap value was computed. `analysis_addendum_e3.md`:
  bf2ae7c, before the E3 runs.
- `analysis_addendum_e1_kappa1e4.md`: 290b815. This supplement was added after the E1 analysis and before its own runs,
  and is labelled as a supplement.

Later commits to these files (696c39b, a323c61) changed wording and references only. The lab notebook (`notes/`) records
deviations and unexpected observations with dates.

## Building the report

The report source is in `report/`; figures and tables are committed, so no experiment needs to be re-run. In `report/`:

```
pdflatex report
bibtex report
pdflatex report
pdflatex report
```

(or `latexmk -pdf report.tex`). A TeX Live installation with `natbib`, `hyperref` and the `libertine` package is required.

## AI assistance

AI tools (Claude) were used for coding and report assistance; the author reviewed and understands all content.
