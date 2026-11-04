# Analysis addendum for E2 primary (written before any E2 result was read)

> Status at writing (04 Nov 2026 11:37): 7 of 540 E2 primary jobs had finished. No E2 primary result file has been
> opened or analysed; the analysis script (`code/analyze_e2.py`) has been run only on synthetic data in its unit tests.
> The commit timestamp of this file is the evidence. It supplements `predictions.md` (commit ac4c90f); where the two
> differ on the checks below, this addendum applies.

## 1. Noise-floor check (replaces "stop if the aligned-replicate CI excludes 0")
- **Quantity.** For each of the six methods (ARD-BO, iso-BO, TuRBO, CMA-ES, sep-CMA-ES, Sobol), at 500 evaluations:
  Delta_rep = log10 gap(aligned replicate) - log10 gap(aligned), paired by seed, 30 seeds.
- **Interval.** 99% percentile bootstrap CI (10,000 resamples, seed 0) for each method. 99% instead of 95% because six
  methods are checked at once: at 99% the chance that at least one method is flagged by chance alone is about 6%.
- **Tolerance.** A method **passes** if its 99% CI contains 0, or if its median |Delta_rep| is at most **0.15 decades**.
  The tolerance is about one third of the smallest rotation effect found so far (E0: +0.37 decades, bounded, kappa = 1e2) and
  well below the primary-endpoint effects the study is meant to resolve.
- **What a flag means.** A flagged method (CI excludes 0 and |median| > 0.15) is **not** a reason to stop. All results
  are still reported. For that method the rotation penalty is reported next to its Delta_rep, the text states that the
  noise floor is not zero for that method, and no claim about that method's Delta_rot is made beyond what exceeds
  Delta_rep. The cause is investigated and recorded in the lab notebook (candidates: seeding, instance difference).
- **Reporting.** All six rows (median, 99% CI, verdict) go in the results file `results/summary/e2_primary.csv`
  (quantity `noise_floor`), whether they pass or not.

## 2. Primary statistics (unchanged from predictions.md, restated so they are fixed here too)
- Endpoint: d = 10, kappa = 1e4, 500 evaluations, 30 seeds (0-29), all differences paired by seed.
- Delta_rot per method = log10 gap(rotated) - log10 gap(aligned), median with a 95% percentile bootstrap CI.
- Interactions: Delta_rot(ARD-BO) - Delta_rot(iso-BO) and Delta_rot(sep-CMA-ES) - Delta_rot(CMA-ES), 95% CI each. H2 is
  supported if both medians are > 0 and both CIs exclude 0.
- The six-method noise-floor check in section 1 is the only place a 99% interval is used.

## 3. Declared secondary variant: CMA-ES with sigma0 = 0.15
- **Why.** In the sensitivity grid, sigma0 = 0.15 beat the default on the rotated-ellipsoid pilot (-0.74 decades, CI
  [-1.04, -0.16]) but not on Rosenbrock (-0.04, CI [-0.12, 0.19]); the pre-set rule kept the default (sigma0 = 0.3).
  Because the ellipsoid result could matter for the rotation comparison, it is run as a declared **secondary** variant.
- **Scope.** CMA-ES only, same three problems, same seeds 0-29, 500 evaluations (`configs/e2_cma_sigma015.toml`,
  experiment `e2_cma_sigma015`, a separate results directory). It does not touch any BO or TuRBO run.
- **Use.** Reported as secondary: its Delta_rot, its noise-floor row (same rule as section 1) and the interaction
  Delta_rot(sep-CMA-ES) - Delta_rot(CMA-ES sigma0 = 0.15). It does not replace the default CMA-ES in the main
  interaction, in figures of the main comparison, or in the H2 verdict.

## 4. Not changed
No other analysis choice (metrics, budgets, seeds, methods in the main table) is altered by this addendum.
