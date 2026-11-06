# Addendum: E3 at d = 2 (decoy-basin function), design and analysis fixed before the runs

H3 in `predictions.md` is unchanged. This file fixes the function, the grid and the statistics before any E3 run exists.
The only E3 runs that exist when this is written are a 2-seed smoke test (`results/raw/e3_d2_quick`, not used).

## Function (`code/problems.py: decoy_basin`, tests in `code/tests/test_problems.py`)
On the unit square, f(u) = min(B(u), D(u)) with B(u) = 1 + 4 ||u - c_b||^2 (broad decoy basin, minimum 1) and
D(u) = ||u - c_n||^2 / w^2 (narrow basin, minimum 0, global). B >= 1 everywhere, so **f(u) < 1 if and only if ||u - c_n|| < w**
(unit-tested). The centres c_b and c_n are drawn from [0.2, 0.8]^2, with ||c_n - c_b|| >= 0.4, from the instance seed; instances
with the same seed share the centres across widths. Widths w in {0.2, 0.1, 0.05, 0.025, 0.0125}, i.e. a success region of
area about 0.126, 0.031, 0.0079, 0.0020, 0.0005 of the square. d = 5 is not run.

## Runs
30 seeds (0-29, instance seed = seed), 5 widths. `configs/e3_d2.toml`, 200 evaluations: ARD-BO with LogEI, ARD-BO with UCB
(beta = 4, the code default; no tuning), default CMA-ES, IPOP-CMA-ES (population doubled at each restart) and Sobol.
`configs/e3_d2_long.toml`, 2000 evaluations: CMA-ES, IPOP-CMA-ES and Sobol only (BO is not run at this budget).
Default CMA-ES stops at its own termination criteria, so its run can end before the budget; its best value is then final.

## Success and statistics
- **Success at n:** the best true value among the first n evaluations is < 1 (strict). For the recommended point, "best
  observed" is the recommendation, as in the other experiments.
- P(success) per method and width with 95% Wilson interval, at n = 200 for all methods and at n = 2000 for CMA-ES, IPOP-CMA-ES
  and Sobol. Sobol is the reference line. A sanity check: Sobol's success rate should be close to 1 - (1 - pi w^2)^n
  (box truncation lowers it slightly).
- **Dependence on width (H3, first sentence):** per method, d_w = P(w = 0.2) - P(w = 0.0125), paired by seed, 95% percentile
  bootstrap over seeds (10,000 resamples, seed 0). "Falls with width" is supported for a method when the CI excludes 0 on
  the positive side. H3 is refuted on this point if the CI contains 0 for every method.
- **UCB against LogEI:** paired difference of success indicators (UCB - LogEI) at each width, same bootstrap. The H3
  prediction (UCB better on the narrowest basins) is supported if the difference at w = 0.0125 is > 0 with CI excluding 0,
  and refuted ("LogEI beats UCB on the narrowest basin") if it is < 0 with CI excluding 0. Anything else: not resolved.
- **IPOP against CMA-ES at the larger budget:** paired difference (IPOP - CMA-ES) at n = 2000, per width and pooled over
  the five widths (resampling seeds, widths together, because the widths share the centres). Supported if the pooled
  difference is > 0 with CI excluding 0. The same at n = 200 is reported as a descriptive comparison.
- **Not tested:** the statement that BO fails when the basin is smaller than the spacing of the initial design. With
  d = 2 the design has 2d = 4 points, so every tested width is far below the spacing; the statement cannot be separated from
  "narrow basins are hard", and BO against Sobol at each width is reported descriptively only.
- Evaluation overhead per method is reported from `t_method`. No other analysis.
