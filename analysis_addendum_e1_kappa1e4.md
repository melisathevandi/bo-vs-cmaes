# Addendum: kappa = 1e4 supplement for H1 (added after the E1 analysis)

**Order of events.** E1 was run at kappa = 100 on the ellipsoid and analysed (`analysis_addendum_e1.md`, results in
`results/summary/e1_h1.csv`, 1fd7721; ellipsoid S = +0.34 [-0.55, 1.14]). H1 in `predictions.md` names kappa = 1e4, and the
amendment of 2026-11-06 (0363ee5) records that E1 did not test it. This supplement was **added after the E1 analysis, to also
test H1 at kappa = 1e4, the value named in `predictions.md`**. It is a supplement, labelled as such in the report, and does not replace the kappa = 100
result. Nothing in the E1 addendum is changed. The statistic and the thresholds are the ones of that addendum, applied to a new
set of runs. What has been looked at that bears on it: the E2 primary results at n = 500 (and its descriptive table); no kappa =
1e4 gap at n = 200 and no run at d = 2 or d = 5 with kappa = 1e4 exists yet.

## Runs
- **New:** ellipsoid only, **kappa = 1e4, rotated**, d = 2 and d = 5, budgets 200 and 400, seeds 0-19, ARD-BO (LogEI, same
  configuration as E1: cold refit every 10 iterations after n = 100), CMA-ES (default, box) and Sobol: the methods, budgets and
  seeds of E1 d = 2 and d = 5 (`configs/e1k4_d2.toml`, `configs/e1k4_d5.toml`). No new Rosenbrock runs.
- **d = 10 point, from existing runs:** the rotated ellipsoid of E2 primary (`ell_d10_k1e4_rotated`, 30 seeds 0-29, 500
  evaluations, ARD-BO, CMA-ES and Sobol), read at n = 200, which is 20 evaluations per dimension. This point uses 30 seeds and
  a budget of 500, not 20 seeds and 1000 as in E1; the BO runs are the same configuration as in E1.

## Statistic (as in `analysis_addendum_e1.md`, applied at kappa = 1e4)
D(d) = median over seeds of log10 gap_BO(n) - log10 gap_CMA(n) at n = 20 d (n = 40, 100, 200 for d = 2, 5, 10), paired by seed,
with a 95% percentile bootstrap CI (10,000 resamples, seed 0). **S = D(10) - D(2)**, the CI resampling the seeds of the two
dimensions independently. S >= 0.5 supports the quantitative guess of H1; S <= 0 refutes it; 0 < S < 0.5 is direction only.
D(5) is reported as well. The evaluations-per-dimension grid {10, 20, 40} is reported for the three dimensions as
descriptive (60 per dimension is not available at d = 10, whose runs have 500 evaluations). d = 3 and d = 8 are not run at
kappa = 1e4, so the trend rests on three dimensions.
