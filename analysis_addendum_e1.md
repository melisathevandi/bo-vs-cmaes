# Addendum: E1 analysis (regime map, H1, break-even cost, censored seeds)

Written and committed before any E1 gap value (F), final gap or cost number was computed. What had been looked at before this
file: the number of result files per E1 experiment, the `code_dirty` and `workers` fields, per-job wall-clock lines in the run
logs, and the chain durations. No pre-registered definition in `predictions.md` is changed. Where this file adds a detail the
prediction left open (grid, statistic for "smaller lead"), it is fixed here, before the numbers.

## Design as run
E1 = d in {2, 3, 5, 8, 10}, budgets 200, 200, 400, 640, 1000, 20 seeds (0-19), ARD-BO (LogEI, cold refit every 10 iterations
after n = 100), CMA-ES (default, with box) and Sobol, on two problems: the rotated ellipsoid with kappa = 100 and the shifted
Rosenbrock function. `predictions.md` H1 mentions kappa = 1e4 for the ellipsoid; E1 was configured with kappa = 100
(`configs/e1_d*.toml`, committed before the runs). This mismatch is stated in the report, and H1 is evaluated on the problems
as run. E1 d = 20 is not run (decision of 6 Nov), so the "d = 20" part of H1 is not tested.

## Comparison A: regime map and H1
- Quantity: per seed, Delta(n) = log10 gap_BO(n) - log10 gap_CMA(n), best-so-far true gap (floor 1e-12), n evaluations;
  negative = BO better. Same seed = same instance, so the difference is paired.
- Grid, fixed now: n/d in {10, 20, 40, 60} (all four are within every E1 budget) and the final budget of each d.
- For each (problem, d, grid point): median Delta, 95% percentile bootstrap CI over seeds (10,000 resamples, seed 0) and the
  paired win rate P(BO < CMA) with its bootstrap CI. BO - Sobol is reported the same way as a reference.
- **H1 statistic (fixed here):** D(d) = median Delta at n/d = 20. S = D(10) - D(2), computed separately for the ellipsoid
  and for Rosenbrock; the CI resamples seeds independently at the two d (different instances). The prediction says the BO lead
  is at least 0.5 decades smaller at d = 10 than at d = 2, so S >= 0.5 on both problems supports the quantitative guess.
  S <= 0 on either problem refutes H1 for that problem, as in `predictions.md` ("same or larger"); 0 < S < 0.5 is reported as
  direction supported, size not. All five D(d) are reported, not only the endpoints.

## Comparison B: break-even cost c* (H4) and censored seeds
Computed exactly as in `code/cost_table.py` and `analysis_addendum_e1_cma_long.md`; the CMA-ES side is the `e1_d{d}_cma_long`
runs. Targets: (i) each BO run's own final gap (for c*, per seed), (ii) the median BO final gap over seeds at the E1 budget,
(iii) absolute levels 1e-1, 1e-2, 1e-4. Per problem and d: fraction of CMA-ES long runs that reach each target; median and IQR
of evaluations needed; for runs that never reach it, the cause (budget cap or pycma termination), listed per seed next to c*
together with whether BO also did not reach the absolute level on that seed (`results/summary/cost_censored.csv`). c* is
defined only for seeds where CMA-ES needs more evaluations than BO; the number of defined seeds is always shown.
- **H4 statistic (fixed here):** for each problem, the median c* (own target, defined seeds) at the five E1 budgets, and the
  Spearman correlation between this median and the budget N. H4 ("c* increases with the budget") is supported for a problem
  when the correlation is positive and refuted when it is negative (a decreasing median c* from d = 2 to d = 10 would be the
  refuting pattern). With five points this is descriptive, and the sign pattern is stated rather than a p-value.
- Method overhead o = t_method / n evaluations (seconds per evaluation) of the E1 runs is reported by method and d.
  E1 BO runs ran at 7 workers and the long CMA-ES runs at 1 worker; c* carries that load difference (CMA-ES overhead is of the
  order of 0.1 ms per evaluation, so the effect on c* is small).

## Sanity checks (allowed before the analysis)
Job counts (120 per d), no NaN, budgets reached, `code_dirty` false, equal first evaluations of the long CMA-ES run and the E1
CMA-ES run of the same seed. A failed check is reported before any result.

## Not done
No test of the d = 20 prediction, no TuRBO in E1, no new definition beyond the above. Anything else computed from E1 is
labelled descriptive in `notes/report_notes.md`.
