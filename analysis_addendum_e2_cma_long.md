# Addendum: supplementary CMA-ES long runs on the E2 primary instances

Written before these runs were started. It adds a supplementary experiment and changes nothing that was
pre-registered: the E2 primary endpoint stays at n = 500 (`analysis_addendum_e2.md`), and these runs never
enter the E2 primary results, the noise-floor check or the H2 interactions.

## What is run
- Experiment `e2_cma_long` (`configs/e2_cma_long.toml`): CMA-ES only, default settings of `configs/e2_primary.toml`
  (sigma0 = 0.3, box bounds [0, 1]^d, pycma defaults including its termination criteria), on the aligned and rotated
  instances of E2 primary (d = 10, kappa = 1e4), the same 30 seeds and instance seeds, budget 10,000 evaluations
  (60 jobs). The aligned replicate is not run.
- Because the method seed depends only on the seed, the first 500 evaluations of each run are expected to equal the
  corresponding `cma` run of E2 primary. This is checked as a technical consistency check, not a statistical test.

## Purpose and quantities
1. **Evaluations CMA-ES needs to reach the gap BO reaches at n = 500.** For each instance and each BO variant (ARD-BO,
   iso-BO), the target is the median final log10 gap of that variant at n = 500 in E2 primary. For each CMA-ES run,
   n_reach is the first evaluation whose best-so-far gap is at or below the target. Reported per instance and variant:
   the fraction of the 30 runs that reach the target, and the median n_reach over runs when more than half reach it
   (otherwise reported as "above" the largest n observed). A run that has not reached the target is censored; the
   report states how many were censored because the 10,000 budget ran out and how many because pycma terminated first.
2. **Data for the E4 break-even table.** The same runs supply the CMA-ES side of the break-even cost c* of
   `code/cost_table.py` on this problem, with the target being each BO run's own final gap at n = 500 (paired by seed).
   c* is reported only for seeds where CMA-ES reaches the target, with the number of such seeds.

## What it does not do
No hypothesis test, no change to the E2 primary CIs, and no use of these runs for H2. The 10,000-evaluation endpoint
is supplementary and descriptive.
