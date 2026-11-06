# Report notes (working summary, mirrors the report structure)

**Rules.** `results/` and `report/generated/numbers.tex` are the source of truth. Every number below cites its source
file and the commit that last changed it as `[file, commit]`; file names are in `results/summary/` unless a path is
given. If this file and the results disagree, the results win and this file is corrected. Process history is in
`notes/lab_notebook.md` and is linked by date, not repeated. Updated after each experiment's analysis finishes and
committed with the other notes.

**Status (4 Nov 2026).** Done: setup, E0, correctness anchor, A4 refit-schedule checks, sensitivity grids, H5 expected
exponent. Running: E2 primary (started 4 Nov about 10:52). Not yet run: E2 low-effective-dimension, H5 runs, E1, follow-up,
and the optional E3, E2 other kappa, E1 d=20, A8 c-e. `numbers.tex` is empty until the first figure script run on real
data, so no number below comes from it yet.

---

## 1. Setup and conventions
- **Problems** (`code/problems.py`): ellipsoid f(x) = sum_i kappa^((i-1)/(d-1)) y_i^2 with y = R(x - x*) (Hessian
  eigenvalues 1 to kappa), aligned (R = I) or rotated (Haar R); shifted Rosenbrock; low-effective-dimension quadratic
  (d_eff relevant directions). Domain [-5, 5]^d mapped from the unit cube; optimum x* drawn in [0.2, 0.8]^d, shared by the
  aligned and rotated instance of the same seed.
- **Metric:** gap = f(best point so far) - f*, on the true noiseless objective; reported as log10 gap, floored at 1e-12.
  Smaller is better.
- **Comparisons are paired by seed** (same instance for every method): median paired difference with a 95% percentile
  bootstrap CI (10,000 resamples) and the paired win rate. Held-out seeds 1000+ for tuning and checks; main runs use
  seeds 0-29 (primary) or 0-19.
- **Seeding:** method randomness is derived from the seed through `SeedSequence([seed, 7919])`, independent of the
  instance (see 2026-11-01 E0 entry for why).
- **Compute:** CPU only, one thread per worker, 7 workers; `device: cpu` in every result JSON. Results are named by the
  sha1 of the job spec and resumed by hash; each JSON stores package versions and the git commit.
- **Libraries:** pycma 4.5.0, BoTorch/GPyTorch (versions in each result JSON), Python 3.12.
- **BO refit schedule:** cold-start GP refit every iteration while n <= 100, then every 10 iterations (A4; section 3,
  "A4").

## 2. Methods (what the report must define)
- **CMA-ES** (pycma): full, sep (`CMA_diagonal`), IPOP (population doubling); bounds [0, 1], x0 uniform, sigma0 = 0.3,
  default population unless stated.
- **BO** (BoTorch SingleTaskGP): Matern-5/2 (nu = 2.5) ARD or isotropic kernel, dimension-scaled LogNormal lengthscale
  prior, standardised outputs on y = -f, LogEI (UCB with beta = 4 in E3), 2d Sobol initial points counted in the budget.
- **TuRBO-1:** trust region, Thompson sampling, restarts; shares the GP code. Plus a Sobol baseline in every comparison.
- **Operational definitions** (also in `notes/plan.md`, section 2c):
  - **Relevance ratio rho.** ARD kernel distance r^2 = (x - x')^T M (x - x') with M = diag(1/l_i^2). For the
    low-effective-dimension function, P is the orthogonal projector onto the d_eff relevant directions. rho = tr(PM)/tr(M),
    in [0, 1]; an isotropic kernel gives exactly d_eff/d. Stability: rho of the main fit and of 5 multi-start fits at each
    checkpoint, spread over all fits and over fits within 1 nat of the best likelihood (`code/diagnostics.py`).
  - **GP rank correlation.** At a checkpoint n, Spearman correlation between the GP posterior mean and the true f on 200
    fresh points from N(x_inc, 0.1^2 I) clipped to the box (x_inc = current best). Configured checkpoints are {25, 50, 100, 150, 200}; because the initial design already has 40 points, the first one is taken at n = 40, so the recorded checkpoints are n = 40, 50, 100, 150, 200 (`results/raw/e2_lowdim`, all 25 files finished by 5 Nov 01:30). Per ARD-BO checkpoint the record stores the lengthscales and marginal likelihood of a cold fit, the rank correlation and 5 multi-start fits (each with its lengthscales and marginal likelihood); iso-BO stores only the rank correlation. For the low-effective-dimension ARD-BO runs `t_method` and `wall_s` include the diagnostics time (cold fit, multi-start fits, rank correlation), so their cost is overstated relative to iso-BO.
  - **kappa_t (CMA-ES learning).** After each generation, kappa_t = cond(C_t^(1/2) H C_t^(1/2)) = eigenvalue ratio of
    C_t H, H the ellipsoid's Hessian. It starts at kappa and equals 1 when C is proportional to H^-1. Learning time T =
    evaluations when kappa_t first drops below tau = 10 (tau = 3 and 100 for robustness); unbounded, termination switched
    off, cap 100 d^2 evaluations, runs reaching f < 1e-12 before tau flagged (`code/h5_learning_time.py`).
  - **alpha fit.** OLS slope of log T on log d over d in {2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 80}, 20 seeds each; 95% CI
    by bootstrap over seeds within each d (10,000); primary fit d >= 10, also all d.
  - **E4 break-even cost.** c* = (T_BO - T_CMA)/(n_CMA - n_BO) per seed, T = cumulative method overhead at the evaluation
    that reaches the target gap; median over seeds where BO reaches the target first (`code/cost_table.py`).
- **Frozen predictions:** `predictions.md` (commit ac4c90f).

## 3. Experiments

### E0: does rotation invariance of CMA-ES survive the box? (done)
- **Question.** Full CMA-ES is rotation invariant in distribution. Does that hold with box bounds, as used in every main
  experiment?
- **Run.** CMA-ES, d = 10, 1000 evaluations, 50 paired seeds, kappa in {1e2, 1e4}, with and without the box
  (`code/e0_invariance.py`).
- **Results** (Delta_rot = log10 gap rotated - aligned, median [95% CI]; `e0_rotation.csv`, 06b6a98):

  | kappa | box | Delta_rot | aligned replicate (noise floor) |
  |---|---|---|---|
  | 1e2 | bounded | 0.37 [0.08, 0.76] | 0.03 [-0.32, 0.44] |
  | 1e2 | unbounded | -0.18 [-0.37, 0.07] | 0.05 [-0.31, 0.28] |
  | 1e4 | bounded | 0.54 [0.25, 0.66] | -0.02 [-0.16, 0.09] |
  | 1e4 | unbounded | -0.01 [-0.26, 0.15] | -0.16 [-0.31, -0.02] |

  Median log10 gap at kappa = 1e4, bounded: 1.50 aligned, 1.99 rotated (same file).
- **Interpretation.** Without the box the rotation effect is consistent with 0; with the box rotation costs about
  0.4-0.5 decades. The invariance is a property of the algorithm, not of the practical implementation with bounds. The
  mechanism is not tested (candidate: bound handling distorts the search distribution for an elongated ellipsoid that is
  diagonal to the box walls). The one noise-floor cell with CI below 0 (kappa = 1e4, unbounded, -0.16 [-0.31, -0.02]) is
  small next to the bounded effect. Monotone-transform invariance is exact (unit tests). Notebook: 2026-11-01
  (invariance, correction to the plan); 2026-11-01 (E0 rotation check; seeding bug).

### Correctness anchor (A8a, done)
- **Run.** `code/anchor.py` -> `anchor.csv` (22a3eab): the CMA-ES wrapper equals `cma.fmin2` on 10/10 seeds; on Branin
  (d = 2, 40 evaluations, 10 seeds) the pre-set criterion is median gap < 0.01.
- **Results.** BO (ARD, LogEI) median gap 2.0e-3 (max 1.8e-2); TuRBO 1.2e-3 (max 4.8e-3); Sobol 1.07 at 40 and 0.11 at 400
  evaluations (`anchor.csv`, 22a3eab). Both methods pass; the BO maximum comes from one slow seed under cold fits.
  Notebook: 2026-11-01 (correctness anchor); the later cold-fit rerun is noted in the A4 entry.

### A4: GP refit schedule (done)
- **Question.** Does warm-starting the GP refits, or refitting every 10 instead of every 5 iterations after n = 100,
  change BO's results?
- **d = 5 check** (rotated ellipsoid kappa = 1e2 and Rosenbrock, n <= 300, seeds 1000-1009; `a4_refit_check.csv`, 810e5df).
  Warm minus cold (log10 gap, positive = warm worse), rotated ellipsoid: n = 100 +0.84 [0.49, 0.94]; n = 300 +0.50
  [0.24, 0.71] (every 5) and +0.46 [0.16, 0.69] (every 10). Rosenbrock CIs contain 0. Cold every 10 minus cold every 5 on
  the ellipsoid at n = 300: +0.01 [-0.30, 0.34]; Rosenbrock +0.03 [-0.13, 0.24].
- **Diagnostic** (`a4_warm_start_diagnostic.csv`, cdc261f): warm and cold fits reach the same marginal likelihood with
  different lengthscales, so the likelihood is flat along a ridge.
- **d = 10 check** (rotated ellipsoid kappa = 1e4, n = 500, 15 seeds 1000-1014, rule fixed before the run:
  adopt if CI contains 0 and half-width <= 0.35; `a4_d10_check.csv`, 810e5df): every 10 minus every 5 = -0.03 [-0.20,
  0.20] at n = 500 (half-width 0.198), -0.01 [-0.13, 0.22] at n = 300. Median wall time per run 47.0 min (every 5) vs 29.1
  min (every 10) on 7 workers.
- **Decision.** Cold refits every 10 iterations after n = 100 in all main runs; warm start rejected.
- **Limits.** 10-15 seeds; two problems; warm-start cause only partly explained. Notebook: 2026-11-01 (A4 entry);
  2 Nov (d = 10 outcome).

### Sensitivity grids (done)
- **Question.** Does tuning one knob of each method change the comparison? Same effort for both methods: 6 configurations.
- **Run.** CMA-ES: sigma0 in {0.15, 0.3, 0.6} x population {default, 2x}. BO: n_init in {d, 2d, 4d} x nu in {1.5, 2.5}. Two
  pilot problems (rotated ellipsoid d = 10 kappa = 1e2, 500 evaluations; Rosenbrock d = 5, 400 evaluations), seeds
  1000-1009. Rule fixed in advance: keep the default unless an alternative is better on both problems with the CI
  excluding 0.
- **Results** (alternative minus default, log10 gap; `sensitivity_cmaes.csv`, `sensitivity_bo.csv`, eca7c30). CMA-ES
  sigma0 = 0.15 (default population): ellipsoid -0.74 [-1.04, -0.16], Rosenbrock -0.04 [-0.12, 0.19]; the doubled
  population is worse in all six comparisons (+0.27 to +1.03 depending on sigma0 and problem); the CI excludes 0
  for all three Rosenbrock cases and for sigma0 = 0.6 on the ellipsoid. BO: no alternative is better on either problem with a CI excluding 0; n_init = d with nu = 1.5 is worse on
  the ellipsoid (+0.62 [0.19, 0.90]), n_init = 4d with nu = 1.5 is worse on both (+0.45 [0.08, 0.68] and +0.25 [0.11, 0.71]).
- **Decision.** All defaults kept (sigma0 = 0.3, default population; n_init = 2d, nu = 2.5). Note for the report:
  sigma0 = 0.15 helps CMA-ES on the ellipsoid pilot only; this is a tuning-effort-symmetry result, reported in the appendix.
  Notebook: 4 Nov.

### E2 primary: rotation and axis alignment
- **Question (H2).** Do axis-aligned methods (ARD-BO, sep-CMA-ES) lose their advantage under rotation, relative to their
  rotation-insensitive counterparts (iso-BO, full CMA-ES)?
- **Run.** d = 10, kappa = 1e4, 500 evaluations, 30 seeds; aligned, rotated and an independent aligned replicate (noise
  floor, instance seed offset 5000); ARD-BO, iso-BO, TuRBO, CMA-ES, sep-CMA-ES, Sobol (`configs/e2_primary.toml`).
- **Results** (source `results/summary/e2_primary.csv`, from `code/analyze_e2.py`; log10 gap at n = 500, paired by seed,
  30 pairs). Noise floor (aligned replicate minus aligned, 99% CI): all six CIs contain 0 (check passes, no flags).
  Rotation penalty Delta_rot, median [95% CI]: ARD-BO 0.85 [0.71, 0.98]; iso-BO -0.09 [-0.25, 0.18]; TuRBO 1.56 [1.27, 1.63];
  CMA-ES 0.16 [0.00, 0.41]; sep-CMA-ES 0.87 [0.72, 1.07]; Sobol -0.01 [-0.17, 0.10]. Interactions: ARD minus iso 0.89
  [0.54, 1.20]; sep-CMA minus CMA 0.77 [0.62, 0.93]; both CIs exclude 0. Secondary CMA-ES sigma0 = 0.15: Delta_rot 0.32
  [-0.06, 0.61], interaction with sep-CMA 0.62 [0.45, 0.83].
- **Descriptive table (not pre-registered).** Median log10 gap at n = 500 with the IQR over 30 seeds, from
  `results/summary/e2_primary_descriptive.csv` (`code/descriptive_e2.py`; no test, adds nothing to the analysis above):

  | Method | Aligned median [Q1, Q3] | Rotated median [Q1, Q3] |
  |---|---|---|
  | ARD-BO | 0.95 [0.79, 1.11] | 1.75 [1.52, 2.05] |
  | iso-BO | 1.60 [1.46, 1.76] | 1.60 [1.46, 1.72] |
  | TuRBO | 0.87 [0.65, 0.99] | 2.29 [2.07, 2.48] |
  | CMA-ES | 2.73 [2.58, 2.93] | 2.97 [2.78, 3.24] |
  | sep-CMA-ES | 2.08 [1.91, 2.25] | 3.04 [2.84, 3.24] |
  | Sobol | 3.76 [3.62, 3.94] | 3.72 [3.63, 3.80] |
- **Supplementary CMA-ES long runs (descriptive; `analysis_addendum_e2_cma_long.md`, results
  `results/summary/e2_cma_long_reach.csv`, `code/analyze_cma_long.py`).** 60 CMA-ES runs of up to 10,000 evaluations on
  the aligned and rotated instances; all 60 ended by pycma's own termination (3,420 to 5,320 evaluations) at the 1e-12 floor.
  Evaluations needed to reach the median final gap that BO has at n = 500, median [Q1, Q3] over 30 runs (all 30 reach it, no
  censoring): aligned, ARD-BO target 9.0: 1,254 [1,100, 1,330]; aligned, iso-BO target 39.9: 954 [842, 1,070]; rotated,
  ARD-BO target 56.8: 1,156 [973, 1,330]; rotated, iso-BO target 39.6: 1,216 [1,082, 1,522]. Break-even cost c* (paired by
  seed, target = each BO run's own final gap, 30 of 30 seeds defined), median [Q1, Q3] in seconds per evaluation: aligned
  ARD-BO 1.84 [1.22, 2.30], aligned iso-BO 0.90 [0.61, 1.08], rotated ARD-BO 1.91 [1.26, 2.68], rotated iso-BO 0.78
  [0.52, 1.33]. Timings were taken while other jobs were running.
- **Interpretation.** Both pre-registered H2 interactions are positive with CIs excluding 0 in this bounded setting.
  Rotation costs the axis-aligned methods (ARD-BO, sep-CMA-ES, TuRBO) about 0.8 to 1.6 decades and the rotation-insensitive
  ones little or nothing. Unbounded behaviour is not tested here (E0 only).
- **Declared secondary variant.** CMA-ES with sigma0 = 0.15, run on the same instances and seeds in a separate experiment
  (`configs/e2_cma_sigma015.toml`, results in `results/raw/e2_cma_sigma015/`). It is reported as secondary and never
  replaces the default CMA-ES or enters the interactions of the main analysis.
- **Pre-registered analysis.** `analysis_addendum_e2.md` (written before any E2 primary result was read) fixes the
  noise-floor check (99% CIs for the six methods and a tolerance in decades), the primary statistics and the secondary
  variant. The script is `code/analyze_e2.py`.

### E2 low-effective dimension
- d = 20, d_eff = 2, aligned vs rotated, 500 evaluations, 20 seeds; rho and rank correlation at checkpoints
  (`configs/e2_lowdim.toml`). Prediction: rho >= 0.5 aligned, <= 0.2 rotated at n = 200 (guess).
- **Results at n = 200** (`results/summary/e2_lowdim_rho.csv`, `code/analyze_lowdim.py`; descriptive, 20 runs per instance,
  median [Q1, Q3]). rho of ARD-BO: aligned 0.998 [0.997, 0.999] (20 of 20 runs >= 0.5); rotated 0.168 [0.143, 0.183] (17 of 20
  runs <= 0.2; isotropic baseline 0.1). GP rank correlation: ARD-BO aligned 0.9999 [0.9996, 0.9999], rotated 0.44
  [0.35, 0.47]; iso-BO aligned 0.34 [0.31, 0.40], rotated 0.38 [0.29, 0.40]. Spread of rho across the 5 multi-start fits
  (median over runs): 0.021 aligned, 0.013 rotated.
- **Lengthscales.** The model has no upper bound on the lengthscale (Positive constraint, bound inf), so "at the bound" is
  reported as above 10 and above 100 box widths. Aligned: all 18 irrelevant lengthscales exceed 10 in 20 of 20 runs and
  exceed 100 in 14 of 20 runs (84% of irrelevant lengthscales on average exceed 100; largest 527); the median relevant
  lengthscale is 2.0. Rotated: 8.5% of lengthscales exceed 10, none exceed 100 (largest 21.8).
- Both thresholds hold at the median. The checkpoints are n = 40, 50, 100, 150, 200 (see the lab notebook, 5 Nov).

### H5: CMA-ES learning time
- **Expected exponent from the learning rates alone** (`h5_expected_alpha.csv`, 8a44b8c; evaluations ~ lambda/(c1 + c_mu),
  pycma 4.5.0 rates): 25.0 evaluations at d = 2, 257 at d = 10, 12,544 at d = 80 (file column `evals_pycma`, in units where
  the proportionality constant is 1). Fitted exponent over d in {2, ..., 80}: alpha = 1.71 (all d) and 1.87 (d >= 10), as
  frozen in `predictions.md` (ac4c90f). That number is a computed prediction from the rates; the measured alpha will come
  from `h5_alpha.csv` once `code/h5_learning_time.py` has run on the full grid.
- **Results** (`results/summary/h5_learning_time.csv`, `h5_alpha.csv`, `code/h5_learning_time.py`; 220 runs, 11 dimensions
  x 20 seeds, no run censored by the cap). Primary endpoint tau = 10, alpha [95% CI]: all d 1.82 [1.80, 1.84] (170 runs);
  d >= 10 2.02 [1.99, 2.06] (90 runs). Other thresholds: tau = 3 all d 1.74 [1.71, 1.78], d >= 10 1.72 [1.57, 1.85];
  tau = 100 all d 1.94 [1.92, 1.96], d >= 10 2.17 [2.16, 2.19]. The d >= 10 interval at tau = 10 excludes the predicted 1.87
  and contains 2; the all-d interval excludes the predicted 1.71 and 2.
- **Precision before the threshold.** At tau = 10, 50 of 220 runs reached machine precision in f before kappa_t fell below
  the threshold: 10 of 20 at d = 40 and 20 of 20 at d = 60 and 80. Those runs have no T and are left out of the fit, so the
  tau = 10 fits use d <= 40 with d = 40 half-observed (median T at d = 40 is from the 10 runs that reached it). T at d = 60 and
  80 is not measured at tau = 10 (tau = 100 has 18 to 20 runs there).
- **Results** (source `results/summary/e2_primary.csv`, from `code/analyze_e2.py`; log10 gap at n = 500, paired by seed,
  30 pairs). Noise floor (aligned replicate minus aligned, 99% CI): all six CIs contain 0 (check passes, no flags).
  Rotation penalty Delta_rot, median [95% CI]: ARD-BO 0.85 [0.71, 0.98]; iso-BO -0.09 [-0.25, 0.18]; TuRBO 1.56 [1.27, 1.63];
  CMA-ES 0.16 [0.00, 0.41]; sep-CMA-ES 0.87 [0.72, 1.07]; Sobol -0.01 [-0.17, 0.10]. Interactions: ARD minus iso 0.89
  [0.54, 1.20]; sep-CMA minus CMA 0.77 [0.62, 0.93]; both CIs exclude 0. Secondary CMA-ES sigma0 = 0.15: Delta_rot 0.32
  [-0.06, 0.61], interaction with sep-CMA 0.62 [0.45, 0.83].
- **Descriptive table (not pre-registered).** Median log10 gap at n = 500 with the IQR over 30 seeds, from
  `results/summary/e2_primary_descriptive.csv` (`code/descriptive_e2.py`; no test, adds nothing to the analysis above):

  | Method | Aligned median [Q1, Q3] | Rotated median [Q1, Q3] |
  |---|---|---|
  | ARD-BO | 0.95 [0.79, 1.11] | 1.75 [1.52, 2.05] |
  | iso-BO | 1.60 [1.46, 1.76] | 1.60 [1.46, 1.72] |
  | TuRBO | 0.87 [0.65, 0.99] | 2.29 [2.07, 2.48] |
  | CMA-ES | 2.73 [2.58, 2.93] | 2.97 [2.78, 3.24] |
  | sep-CMA-ES | 2.08 [1.91, 2.25] | 3.04 [2.84, 3.24] |
  | Sobol | 3.76 [3.62, 3.94] | 3.72 [3.63, 3.80] |
- **Supplementary CMA-ES long runs (descriptive; `analysis_addendum_e2_cma_long.md`, results
  `results/summary/e2_cma_long_reach.csv`, `code/analyze_cma_long.py`).** 60 CMA-ES runs of up to 10,000 evaluations on
  the aligned and rotated instances; all 60 ended by pycma's own termination (3,420 to 5,320 evaluations) at the 1e-12 floor.
  Evaluations needed to reach the median final gap that BO has at n = 500, median [Q1, Q3] over 30 runs (all 30 reach it, no
  censoring): aligned, ARD-BO target 9.0: 1,254 [1,100, 1,330]; aligned, iso-BO target 39.9: 954 [842, 1,070]; rotated,
  ARD-BO target 56.8: 1,156 [973, 1,330]; rotated, iso-BO target 39.6: 1,216 [1,082, 1,522]. Break-even cost c* (paired by
  seed, target = each BO run's own final gap, 30 of 30 seeds defined), median [Q1, Q3] in seconds per evaluation: aligned
  ARD-BO 1.84 [1.22, 2.30], aligned iso-BO 0.90 [0.61, 1.08], rotated ARD-BO 1.91 [1.26, 2.68], rotated iso-BO 0.78
  [0.52, 1.33]. Timings were taken while other jobs were running.
- **Interpretation.** Both pre-registered H2 interactions are positive with CIs excluding 0 in this bounded setting.
  Rotation costs the axis-aligned methods (ARD-BO, sep-CMA-ES, TuRBO) about 0.8 to 1.6 decades and the rotation-insensitive
  ones little or nothing. Unbounded behaviour is not tested here (E0 only).

### E1: dimension x budget (run 5-6 Nov; analysis as fixed in `analysis_addendum_e1.md`, 4680bb8)
- BO (ARD) vs CMA-ES vs Sobol on the rotated ellipsoid (kappa = 1e2, not 1e4 as H1 says; see the amendment in
  `predictions.md`, 0363ee5) and Rosenbrock, d in {2, 3, 5, 8, 10}, budgets 200, 200, 400, 640, 1000, 20 seeds
  (`configs/e1_d*.toml`). 600 runs, 7 workers, all `code_dirty = false`; 5 Nov 11:07 to 6 Nov 05:19 (14.2 h; d = 10 alone 8.9 h).
- Sources: `results/summary/e1_regime_map.csv`, `e1_h1.csv`, `e1_h4.csv`, `e1_cost_median_target.csv`, `e1_overhead.csv`,
  `cost_table.csv`, `cost_censored.csv` (code `analyze_e1.py`, `cost_table.py`; 1fd7721; a rerun gives identical files).
- **H1 statistic D(d)** = median paired log10 gap, ARD-BO minus CMA-ES, at 20 evaluations per dimension (negative = BO ahead):
  ellipsoid -2.35, -2.89, -2.98, -2.37, -2.01 for d = 2, 3, 5, 8, 10; Rosenbrock -1.48, -1.50, -0.98, -0.66, -0.82.
  S = D(10) - D(2): **ellipsoid +0.34 [-0.55, 1.14]** (direction only, below the 0.5 guess, CI contains 0);
  **Rosenbrock +0.65 [0.03, 1.22]** (meets the 0.5 guess). BO leads at 20 evaluations per dimension at every d, and the lead
  shrinks from d = 2 to d = 10 on both problems. The d = 20 part is not tested.
- **Final budgets** (median BO - CMA-ES, 95% CI): ellipsoid d = 2 +0.94 [0.44, 1.52], d = 3 -0.72 [-1.30, 0.01], d = 5 -0.24
  [-1.13, 0.31], d = 8 -0.66 [-1.28, -0.36], d = 10 -0.13 [-0.51, 0.10]; Rosenbrock d = 2 -1.06 [-1.44, 0.07], d = 3 -0.76
  [-0.92, -0.31], d = 5 +0.03 [-0.19, 0.23], d = 8 +0.56 [0.26, 0.83], d = 10 +0.72 [0.58, 0.84]. On Rosenbrock CMA-ES is ahead at
  the final budget from d = 8; on the ellipsoid the two are level at d = 10, and CMA-ES is ahead at d = 2 (200 evaluations).
- **c\* (own target, per seed; CMA-ES side from the long runs), median [IQR] in s per evaluation, seeds with c\* defined:**
  ellipsoid 0.99 [0.60, 3.05] (16), 0.80 [0.58, 1.38] (19), 1.39 [0.52, 3.56] (19), 5.77 [2.68, 11.25] (19), 59.85 [28.85, 103.71]
  (17) for d = 2, 3, 5, 8, 10; Rosenbrock 0.48 (19), 0.33 (19), 3.39 (15), 7.81 (7), 48.54 (**1 seed only**). On Rosenbrock at
  d = 8 and 10 CMA-ES usually needs fewer evaluations than BO needs for its own final gap (median 501 vs 884 at d = 10), so c\*
  is mostly undefined there: BO has no break-even to report. **H4:** the median c\* rises with the budget under both targets
  (Spearman 0.97 own target on both problems; median-target version 0.67 ellipsoid, 0.97 Rosenbrock); never decreasing.
- **Censored seeds:** 1 CMA-ES long run never reached BO's own final gap (d = 5 Rosenbrock); at the absolute levels 1e-1, 1e-2,
  1e-4, 1 run at d = 5 and d = 10 Rosenbrock and 2 at d = 8. They are listed per seed in `cost_censored.csv` with the cause
  and whether BO reached the level.
- **Overhead** of ARD-BO (mean method seconds per evaluation over the run, ellipsoid): 0.79, 1.10, 1.87, 4.10, 10.65 s for
  d = 2, 3, 5, 8, 10 (`e1_overhead.csv`; 7 workers). CMA-ES is far below 1 ms.
- Caveats: S for the ellipsoid has a CI containing 0. The E1 budgets differ per d, so the "final budget" rows mix d and budget.

### E3 at d = 2 (run 6 Nov; design and statistics fixed in `analysis_addendum_e3.md`, bf2ae7c)
- Decoy-basin function (`problems.decoy_basin`): success (f < 1) iff the point is within w of the narrow optimum. 5 widths x 30
  seeds x {LogEI-BO, UCB-BO, CMA-ES, IPOP-CMA-ES, Sobol} at 200 evaluations (750 runs, 11:39 to 12:37, 7 workers, 58 min) and
  CMA-ES, IPOP-CMA-ES, Sobol at 2000 evaluations (450 runs, 29 s). All `code_dirty = false`. Sources: `e3_success.csv`,
  `e3_tests.csv` (`analyze_e3.py`, 2a5baba).
- P(success), n = 200, w = 0.2, 0.1, 0.05, 0.025, 0.0125: LogEI-BO 0.80, 0.10, 0.03, 0, 0; UCB-BO 0.67, 0.10, 0.03, 0, 0; CMA-ES
  0.80, 0.40, 0.07, 0, 0 (IPOP identical at 200); Sobol 1.00, 1.00, 0.97, 0.30, 0.07 (i.i.d. expectation 1.00, 1.00, 0.79, 0.33,
  0.09; Sobol is a little above it at some widths, as expected for a low-discrepancy set).
- n = 2000: CMA-ES 0.80, 0.40, 0.07, 0, 0; IPOP 1.00, 0.93, 0.57, 0.27, 0.10; Sobol 1.00, 1.00, 1.00, 1.00, 0.87.
- **H3 statistics (addendum rules):** (i) P(w = 0.2) - P(w = 0.0125) is positive with CI excluding 0 for every method (BO-LogEI
  0.80 [0.67, 0.93], BO-UCB 0.67 [0.50, 0.83], CMA-ES 0.80, IPOP 0.80 at 200 and 0.90 at 2000, Sobol 0.93 at 200): **success
  falls with narrower basins: supported**. (ii) UCB - LogEI at w = 0.0125: 0.00 [0.00, 0.00], both at 0 of 30: **not resolved**
  (neither supported nor refuted; at w = 0.2 LogEI is better, -0.13 [-0.27, -0.03]). (iii) IPOP - CMA-ES pooled over widths at
  n = 2000: **+0.32 [0.25, 0.39]**: supported. At n = 200 the two are identical (no restart fits in the budget).
- **Descriptive:** at 200 evaluations BO is no better than Sobol at any width and significantly worse for w >= 0.025 (e.g.
  w = 0.05: LogEI minus Sobol -0.93 [-1.00, -0.83]). Sobol with 2000 points beats IPOP at every width (0.87 vs 0.10 at
  w = 0.0125). The claim about the initial-design spacing is not separable at d = 2 (stated in the addendum).

### Follow-up: TuRBO rotation-penalty ablation (run 6 Nov; prediction in `predictions.md`, f65181c and b78a3f4)
- Variants: ARD kernel with a cubic region, isotropic kernel (cubic region); default TuRBO = the E2 runs. 180 runs (30 seeds x
  {aligned, rotated, aligned replicate} x 2 variants), 13:06 to 19:09 (5.7 h, 7 workers), `code_dirty = false`, `turbo.py`
  change in 72cca55 (default reproduces 9 stored E2 trajectories exactly for 200 evaluations; the check must run
  single-threaded, torch multithreading changes the TuRBO trajectory). Sources: `turbo_ablation.csv`,
  `turbo_ablation_verdict.csv` (`analyze_ablation.py`, 2a5baba).
- Median log10 gap at n = 500 [aligned / rotated / replicate]: default 0.87 / 2.29 / 0.87; cubic ARD 1.82 / 2.39 / 1.69; isotropic
  2.42 / 2.45 / 2.42; ARD-BO 0.95 / 1.75; iso-BO 1.60 / 1.60; Sobol 3.76 / 3.72.
- Delta_rot: default 1.56 [1.27, 1.63]; **cubic ARD 0.60 [0.49, 0.69]** (floor -0.08 [-0.24, 0.11]); **isotropic 0.01 [-0.09, 0.15]**
  (floor 0.01 [-0.11, 0.14]). Contrast cubic ARD - default -0.96 [-1.08, -0.68]; isotropic - cubic ARD -0.59 [-0.72, -0.41].
- **Verdict by the pre-registered rules:** cubic ARD: Delta_rot in [0.45, 1.25] with CI excluding 0 and a negative contrast with the
  default: **prediction met**. Isotropic: |Delta_rot| <= 0.3 with CI containing 0 and the same after subtracting the noise floor:
  **prediction met**.
- **Reading that the rules do not show:** the reduction in penalty comes from the aligned side, not the rotated side. Cubic ARD
  loses 0.95 decades when aligned (0.87 to 1.82) and 0.10 when rotated (2.29 to 2.39); the isotropic variant is equally poor in
  both (2.42, 2.45), better than Sobol by 1.3 decades but worse than iso-BO (1.60). So the ARD-scaled region is what lets TuRBO
  exploit axis alignment, and removing it removes the aligned advantage rather than improving the rotated case. This was
  flagged as the main confound before the runs.

### E4: cost analysis (E1 table done, see E1 above)
- Break-even evaluation cost c* from the E1 logs (`code/cost_table.py` -> `cost_table.csv`).
- Measured run times so far, for context: BO at d = 10, 500 evaluations, 7 workers: 47.0 min per run with refit every 5
  and 29.1 min with refit every 10 (`a4_d10_check.csv`, 810e5df). TuRBO pilot, d = 10: 13.1-17.3 min at 500 evaluations,
  25.5-32.1 min at 1000; d = 20, 1000 evaluations: 53.1-54.5 min (`pilot_run_timing.csv`, bf39d63).

## 4. Prediction versus outcome (H1-H5)
| Hypothesis | Prediction (frozen in `predictions.md`, ac4c90f) | Status |
|---|---|---|
| H1 | BO lead over CMA-ES shrinks with d at equal evaluations per dimension; at least 0.5 decades smaller at d = 10 than d = 2 (guess); no lead at d = 20 | ellipsoid S = +0.34 [-0.55, 1.14] (direction only), Rosenbrock S = +0.65 [0.03, 1.22] (meets 0.5); d = 20 not tested; tested at kappa = 100 (amendment 0363ee5); `e1_h1.csv`, 1fd7721 |
| H2 | Both interactions (ARD vs iso BO; sep vs full CMA-ES) > 0 with CIs excluding 0; full CMA-ES Delta_rot > 0 with the box, about 0 without; rho >= 0.5 aligned and <= 0.2 rotated at n = 200 | CMA-ES part consistent with E0 (0.37 and 0.54 bounded; about 0 unbounded; `e0_rotation.csv`). The prediction was informed by E0, so this is not an independent test. E2 runs bounded only, so the unbounded part rests on E0 alone. rho at n = 200: 0.998 aligned (threshold 0.5), 0.168 rotated (threshold 0.2), medians over 20 runs (`e2_lowdim_rho.csv`); the two interactions are positive with CIs excluding 0 (`e2_primary.csv`) |
| H3 | P(success) falls with basin width; IPOP beats default CMA-ES at larger budgets; UCB beats LogEI on the narrowest basin | E3 at d = 2: success falls with width for every method (supported); IPOP beats CMA-ES at n = 2000, pooled +0.32 [0.25, 0.39] (supported); UCB vs LogEI at the narrowest width 0 vs 0 of 30 (not resolved); `e3_tests.csv`, 2a5baba |
| H4 | c* increases with budget and with d | median c* rises with the budget on both problems (Spearman 0.67 to 0.97); never decreasing; c* defined for 1 of 20 seeds at d = 10 Rosenbrock; `e1_h4.csv`, 1fd7721 |
| H5 | CI of alpha-hat (d >= 10) contains 1.87; alpha-hat (all d) about 1.7 | d >= 10 (tau = 10): 2.02 [1.99, 2.06], excludes 1.87, so the pre-registered refutation criterion is met; all d: 1.82 [1.80, 1.84] against about 1.7 (`h5_alpha.csv`). 50 of 220 runs reached machine precision first (d = 40, 60, 80), left out of the fit |

## 5. Surprises and negative results
- **Seeding bug** (found in E0): CMA-ES's start was a function of the optimum because instance and method randomness shared
  a seed; the aligned replicate "noise floor" came out significantly nonzero (+0.52 [0.32, 0.95] in the first E0 run,
  since deleted and regenerated). Fixed with an independent method seed and a test. 2026-11-01 (E0 entry).
- **Wrong test in the plan:** path-by-path rotation invariance of CMA-ES does not hold even in principle; invariance is in
  distribution, tested statistically. 2026-11-01 (invariance, correction to the plan).
- **Warm-start failure** (A4): warm-started refits reached the same likelihood but different lengthscales and gave worse BO
  on the rotated ellipsoid (+0.84 at n = 100). Cold refits are the default. 2026-11-01 (A4 entry).
- **E0 box effect:** rotation costs bounded CMA-ES about 0.4-0.5 decades while unbounded CMA-ES is rotation-insensitive
  (section 3, E0).

## 6. Limitations (so far)
- Noise is not studied unless the optional A8c runs; the report states this as a limitation.
- CPU only; timings come from a 7-worker Windows machine with a 1.25 load factor over single-thread pilots.
- A4 was checked at d = 5 and d = 10 with 10-15 seeds, so "refit every 10 does not matter" is evidence at those settings only.
- Sensitivity grids use two pilot problems and 10 seeds per configuration; a knob that matters elsewhere could be missed.
- Tuning is symmetric in effort (6 configurations each) but covers different knobs; sigma0 is the only CMA-ES knob that
  moved results on one pilot problem.
- Box constraints break CMA-ES rotation invariance (E0); every main experiment uses the box, so rotation penalties include
  this effect.
- Compute estimates for large N and d are extrapolated from a single calibration cell; the schedule has a wide range.

## 7. Open questions
- Why does the box break CMA-ES rotation invariance? (Bound-handling mechanism untested; candidate follow-up.)
- Does the BO flat-likelihood ridge explain the warm-start loss, or is something else at work? Only partially tested.
- Will iso-BO show a rotation penalty comparable to ARD-BO's (E2)? The box/design effect on BO is unmeasured.
- Which surprise from E1-E2 becomes the follow-up? Candidates are in `notes/plan.md`, section 3.
- Does the d = 20, N = 1000 E1 run get scheduled at all, or does it go in the "not run" list with its measured cost?
