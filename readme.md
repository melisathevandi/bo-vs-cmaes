# Bayesian optimisation vs CMA-ES: when does each win?

DSA4212 final assignment. Report: `report/report.tex` (Overleaf source, ≤ 8 pages).
Authors: see [authors.md](authors.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`.

```
uv sync
```

This installs the pinned versions from `uv.lock`. Everything runs on CPU.

## Run the tests

```
uv run pytest
```

## Reproduce results

Not yet implemented. The plan is to provide:

- `uv run python code/run_experiment.py --config configs/<name>.yaml --quick` (under 10 minutes)
- `uv run python code/run_experiment.py --config configs/<name>.yaml` (full)
- `uv run python code/make_figures.py` (regenerates every figure, and `report/generated/numbers.tex`)

## Layout

| Path | Contents |
|---|---|
| `code/problems.py` | Test functions (ellipsoid, rotations, Rosenbrock) |
| `code/tests/` | Unit tests for the test functions |
| `configs/` | One config per experiment |
| `results/raw/` | Per-run JSON output (not tracked by git) |
| `results/summary/` | Aggregated tables used by the report |
| `report/` | Report source, bibliography, generated figures and numbers |
| `docs/` | Assignment guidelines and topic brief |
| `notes/plan.md` | Critical review and agreed plan |
| `predictions.md` | Predictions, committed before the main runs (to be written) |
