# Addendum: supplementary CMA-ES long runs on the E1 problems (for the E4 break-even table)

Written before these runs were started and before any E1 BO run has finished. Nothing pre-registered changes: E1 keeps
its own budgets (`configs/e1_d{2,3,5,8,10}.toml`, not edited), and these runs are descriptive.

## What is run
CMA-ES only, default settings, on the same problems and seeds as E1 (ellipsoid kappa = 100 rotated, and shifted
Rosenbrock; 20 seeds), for d = 2, 3, 5, 8, 10 (experiments `e1_d{d}_cma_long`, `configs/e1_d{d}_cma_long.toml`). The budget cap
is 50,000 evaluations; pycma's own termination normally ends a run earlier. The runs record the full trajectory to
convergence and do not depend on any target. The method seed depends only on the seed, so the first evaluations of each run
are expected to equal the CMA-ES run of E1 (a technical check, not a test).

## Targets (a rule, fixed now; the values are computed afterwards)
1. **BO-relative target.** For each E1 problem (d, function): the median final gap of ARD-BO at its E1 budget (200, 200, 400,
   640, 1000 evaluations for d = 2, 3, 5, 8, 10), computed over seeds after the E1 BO runs finish. Per seed (for c*), the
   target is that BO run's own final gap.
2. **Absolute gap levels:** 1e-1, 1e-2 and 1e-4 (the gap being the best-so-far true objective minus the optimum).

For each target, per problem: the fraction of CMA-ES long runs that reach it, the median and IQR of the evaluations needed
(reported as censored when a run ends before reaching it, with the cause: budget cap or pycma termination), and the
break-even cost c* of `code/cost_table.py` (paired by seed, only where CMA-ES needs more evaluations than BO).

## Timings
The wall-clock times of these runs are taken at 1 worker while other jobs run on the machine (E2 low-effective dimension at
4 workers), so the CMA-ES side of c* carries that load. CMA-ES overhead is of the order of 0.1 ms per evaluation, so
this changes c* very little; it is stated here for completeness.
