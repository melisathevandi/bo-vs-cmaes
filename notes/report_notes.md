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

### E2 low-effective dimension (not yet run)
- d = 20, d_eff = 2, aligned vs rotated, 500 evaluations, 20 seeds; rho and rank correlation at checkpoints
  (`configs/e2_lowdim.toml`). Prediction: rho >= 0.5 aligned, <= 0.2 rotated at n = 200 (guess).

### H5: CMA-ES learning time (expected exponent done; runs not yet run)
- **Expected exponent from the learning rates alone** (`h5_expected_alpha.csv`, 8a44b8c; evaluations ~ lambda/(c1 + c_mu),
  pycma 4.5.0 rates): 25.0 evaluations at d = 2, 257 at d = 10, 12,544 at d = 80 (file column `evals_pycma`, in units where
  the proportionality constant is 1). Fitted exponent over d in {2, ..., 80}: alpha = 1.71 (all d) and 1.87 (d >= 10), as
  frozen in `predictions.md` (ac4c90f). That number is a computed prediction from the rates; the measured alpha will come
  from `h5_alpha.csv` once `code/h5_learning_time.py` has run on the full grid.
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

### E1: dimension x budget (not yet run)
- BO (ARD) vs CMA-ES vs Sobol on the rotated ellipsoid (kappa = 1e2) and Rosenbrock, d in {2, 3, 5, 8, 10},
  budgets 200, 200, 400, 640, 1000, 20 seeds (`configs/e1_d*.toml`). Feeds H1, the regime map and the E4 table.

### E3, E2 other kappa, E1 d = 20, A8 c-e (optional, not yet run)
- Run only if the core is ahead of schedule on 6 Nov (current estimate: it is not; `notes/compute_plan.md`). If not run, the
  report says so as a cut from the experiment list, with the BO computational cost behind it.

### Follow-up (reserved, not yet chosen)
- Chosen after E1; its prediction is committed to `predictions.md` before it runs.

### E4: cost analysis (not yet run)
- Break-even evaluation cost c* from the E1 logs (`code/cost_table.py` -> `cost_table.csv`).
- Measured run times so far, for context: BO at d = 10, 500 evaluations, 7 workers: 47.0 min per run with refit every 5
  and 29.1 min with refit every 10 (`a4_d10_check.csv`, 810e5df). TuRBO pilot, d = 10: 13.1-17.3 min at 500 evaluations,
  25.5-32.1 min at 1000; d = 20, 1000 evaluations: 53.1-54.5 min (`pilot_run_timing.csv`, bf39d63).

## 4. Prediction versus outcome (H1-H5)
| Hypothesis | Prediction (frozen in `predictions.md`, ac4c90f) | Status |
|---|---|---|
| H1 | BO lead over CMA-ES shrinks with d at equal evaluations per dimension; at least 0.5 decades smaller at d = 10 than d = 2 (guess); no lead at d = 20 | pending (E1) |
| H2 | Both interactions (ARD vs iso BO; sep vs full CMA-ES) > 0 with CIs excluding 0; full CMA-ES Delta_rot > 0 with the box, about 0 without; rho >= 0.5 aligned and <= 0.2 rotated at n = 200 | CMA-ES part consistent with E0 (0.37 and 0.54 bounded; about 0 unbounded; `e0_rotation.csv`). The prediction was informed by E0, so this is not an independent test. E2 runs bounded only, so the unbounded part rests on E0 alone. The rest is pending (E2) |
| H3 | P(success) falls with basin width; IPOP beats default CMA-ES at larger budgets; UCB beats LogEI on the narrowest basin | not run (E3 optional) |
| H4 | c* increases with budget and with d | pending (E1 logs) |
| H5 | CI of alpha-hat (d >= 10) contains 1.87; alpha-hat (all d) about 1.7 | expected values computed (1.87, 1.71; `h5_expected_alpha.csv`); measurement pending |

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
