# Lab notebook

Dated decisions, surprises, failed attempts. Newest at the bottom.

## 2026-11-01
- Plan revision 2 approved (see `plan.md`, amendments A1-A8 at the top win over the body).
- Decision: **CPU only.** Amendment A2 asked for CPU-vs-GPU timing. The machine has no CUDA
  (AMD Ryzen 9 5900HS, integrated Radeon graphics) and torch 2.15.0 is the CPU build.
  Pilot timings are CPU only, and every result JSON records `"device": "cpu"`.

## 2026-11-01 (invariance, correction to the plan)
- The plan's E0 test said "rotated problem with rotated x0 -> rotated trajectory, path by path".
  **Wrong.** With the same seed, pycma draws the same noise vector z in both runs, but the
  rotated run would need R z. Checked empirically (d=5, kappa=100, unbounded): the paths differ
  from the first generation on. Rotation invariance holds in distribution only.
- Consequence: monotone-transform invariance is tested exactly (`tests/test_invariance.py`);
  rotation invariance (and the effect of the box) is tested statistically in the E0 script,
  with paired differences over many seeds.
- First BO timing: about 1 s per iteration at d=5, n<=60 (single thread). The A2 budgets
  (1000 evaluations) are costly for BO in compute terms; budgets are fixed from the experiment requirements
  and the measured cost in the timing pilot (`results/summary/pilot_timing.csv`).
- Observed (n=40, aligned kappa=100, d=5): ARD lengthscales 13.4, 6.9, 4.0, 2.1, 1.0 follow the axis
  scales; GP rank correlation 0.97-1.0 (ARD) vs 0.6-0.95 (isotropic) at n=20-40.

## 2026-11-01 (timing pilot, CPU, 1 thread, 3 instance seeds per cell)
- Fit cost dominates BO's per-iteration cost. Warm-starting each refit from the previous
  hyperparameters cuts a refit by about 5-10x (e.g. d=10, n=1000: 42 s cold vs 4.9 s warm), and
  the average cost per iteration by about 3x (10.3 s -> 2.8 s). Acquisition optimisation is
  0.2-2.2 s per iteration over n=100-1000.
- Estimated BO cost per run with warm starts: d=10, 500 evaluations about 6 min; d=10 or d=20,
  1000 evaluations about 22 min (integrated over n; single-thread timings, so parallel runs may
  be somewhat slower). On these numbers the A2 budgets are kept as approved.
- TuRBO's cost is not covered by this pilot. In a first smoke run (d=10, 500 candidates) it took
  about 1 s per iteration at n<=80, because Thompson sampling needs a joint posterior sample over
  the candidates. It needs its own timing before the E2 runs.
- A4 check (warm vs cold refit, refit every 5 vs 10) is run in `configs/a4_refit_check.toml`
  on held-out seeds 1000-1009.

## 2026-11-01 (E0 rotation check for CMA-ES; seeding bug found and fixed)
- **Bug.** First E0 run: the aligned-vs-aligned replicate (noise floor) was significantly non-zero in one
  cell (+0.52, CI [0.32, 0.95]), which a noise floor must not be. Cause: instance randomness and method
  randomness both started from `default_rng(seed)`, so CMA-ES's start x0 was a function of the optimum x*
  (x* = 0.2 + 0.6 u, x0 = u). Aligned/rotated runs started close to x*, the replicate did not.
  Fix: methods now get `method_seed(seed)` (SeedSequence-mixed), applied in `runner.py` and `e0_invariance.py`;
  `tests/test_seeds.py` checks that raw seeding correlates x0 with x* (> 0.99) and the derived seed does not (< 0.2).
  All earlier outputs from the old scheme (smoke run, first A4 attempt, first E0 CSV) were deleted and re-generated.
- **E0 result** (CMA-ES, d=10, 1000 evaluations, 50 paired seeds, median difference of log10 gap [95% bootstrap CI];
  `results/summary/e0_rotation.csv`):
  - Unbounded: Delta_rot = -0.18 [-0.37, 0.07] (kappa=1e2), -0.02 [-0.26, 0.15] (kappa=1e4). Consistent with rotation invariance.
  - Box [0,1]^d (as in every main experiment): Delta_rot = +0.37 [0.09, 0.76] (kappa=1e2), +0.54 [0.25, 0.66] (kappa=1e4).
    Rotation costs CMA-ES about 0.4-0.5 decades once the box is present.
  - Noise floor (aligned replicate): +0.03 [-0.32, 0.44], +0.05 [-0.31, 0.28], -0.02 [-0.16, 0.09] and -0.16 [-0.31, -0.02];
    the last cell (kappa=1e4, unbounded) is marginally below 0, and is small compared with the bounded rotation effect.
- Reading: the invariance of full CMA-ES is a property of the algorithm without bounds; with box constraints it does not
  fully survive. This is the brief's "which theoretical invariances survive the practical implementation" question.
  The mechanism is not yet tested. Candidate explanation: the bound-handling transformation distorts the search
  distribution differently for an elongated ellipsoid that is diagonal to the box walls. Candidate follow-up: vary the
  bound handling or the box size relative to sigma0.
- At kappa=1e4 the median gap after 1000 evaluations is about 10^1.5-10^2, i.e. CMA-ES is still in its
  geometry-learning phase at this budget (relevant for H5 and the E2 budget).

## 2026-11-01 (correctness anchor, A8a: passed)
- `code/anchor.py` -> `results/summary/anchor.csv`.
- CMA-ES wrapper evaluates exactly the same points as `cma.fmin2` on the same bounded problem, start and seed
  (10/10 seeds, d=8 rotated ellipsoid, 200 evaluations).
- Branin (d=2), 40 evaluations, 10 seeds, threshold median gap < 0.01 fixed beforehand:
  BO (ARD, LogEI) median 1.0e-3 (max 6.4e-3); TuRBO median 2.3e-3 (max 5.1e-3); Sobol with 40 evaluations 1.07 and
  with 400 evaluations 0.11.

## 2026-11-01 (H5 expected exponent; predictions review)
- Review of the draft `predictions.md` found an error in the H5 derivation: the numerator of c1 + c_mu is
  2 mu_eff - 2 + 2/mu_eff, not 2 + 2 mu_eff (the -4 + 2/mu_eff term from c_mu was dropped).
- Checked numerically with the learning rates pycma actually uses (`code/h5_expected_alpha.py`):
  the lambda / numerator factor is almost constant in d (log-log slope about -0.005), so it does not change alpha.
  Rate-based expected alpha: 1.71 (d in [2, 80]) and 1.87 (d >= 10) with pycma 4.5.0; 1.69 / 1.86 with the
  tutorial formulas. pycma's c1 matches the tutorial; its c_mu is 7-48% larger (48% at d=2, 17% at d=10, 7% at d=80).
- Consequence: the draft prediction "CI for d >= 10 contains 2" was not what the rates predict at finite d.
  The prediction now compares the fitted alpha with the rate-based value 1.87.
- Decision: UCB stays in E3 (H3 makes a UCB-vs-LogEI prediction; E3 is at d <= 5 and cheap), so it is no longer
  on the cut list for E3.

## 2026-11-01 (A4 refit-schedule check: warm start rejected, cold refit every 10 adopted)
- Setup: BO (ARD, LogEI), budget 300, rotated ellipsoid (d=5, kappa=1e2) and Rosenbrock (d=5), held-out seeds
  1000-1009, paired by seed, reference = cold-start refit every 5 iterations
  (`configs/a4_refit_check.toml`, `results/summary/a4_refit_check.csv`).
- **Warm-started refits changed results.** Rotated ellipsoid, median difference in log10 gap, warm minus cold
  (positive = warm worse): n=100 +0.84 [0.49, 0.94]; n=300 +0.50 [0.24, 0.71] (refit every 5) and +0.46 [0.16, 0.69]
  (refit every 10). Rosenbrock: CIs contain 0 at every budget. At n=100 the refit schedule is identical, so the
  difference there comes from warm vs cold starting alone.
- **Diagnostic** (`code/diagnose_warm_start.py`, replaying 4 cold-run trajectories, n = 12 to 100): the warm chain's
  maximised marginal likelihood equals the cold fit's (median gap about 0, never below -0.5), but the fitted
  lengthscales differ at equal likelihood (e.g. seed 1003, n=100: cold [1.7, 4.3], warm [3.2, 7.9]). So the
  likelihood is flat along a ridge, and where a fit lands on it changes BO's behaviour. The direction of the effect on
  lengthscales is not consistent across the 4 seeds, and this replay uses the cold run's data, so it does not
  establish the cause of the end-to-end difference. Not pursued further for now.
- **Decision:** warm start is off (`warm_start=False`, now the default). Main runs use cold-start fits.
- **Cold refit every 10 vs cold every 5** (same A4 data): median differences at n=200 and n=300 are 0.00 to +0.03 with
  CIs containing 0 on both problems (widths about +-0.3 decades), and wall time is 25% (ellipsoid) and 10% (Rosenbrock)
  lower at n=300. **Decision:** main runs use cold fits and `refit_every = 10` once n > 100 (set explicitly in each
  config). Limits: tested at d=5 and n <= 300 with 10 seeds, so it is not evidence for d=10-20 at n=1000.
- Correctness anchor re-run under the cold default: BO (Branin, 40 evaluations) median gap 2.0e-3, worst seed
  1.8e-2 (above 0.01; the check is defined on the median); TuRBO median 1.2e-3.
- Compute estimate (planning only, from the timing pilot, cold fits, refit every 10, x1.3 for parallel load; not report
  numbers): BO run of d=10 with 500 evaluations about 15 min; d=10 with 1000 evaluations about 58 min; d=20 with 1000
  evaluations about 74 min. Rough totals for BO-type methods: E2 about 220 CPU-hours before TuRBO, E1 about 120,
  sensitivity grids about 30, E3 and the follow-up less; of the order of 80 hours of wall-clock time on 7 workers,
  with an uncertainty of +-50% (TuRBO is not yet timed).
- Anchor note: under cold-start fits one of the ten Branin seeds reached a gap of 1.8e-2 for BO (40 evaluations); the check passes on its pre-set criterion, a median gap below 0.01 (observed 2.0e-3).

## 2026-11-01 (compute plan approved with changes; A4 check at d=10 launched)
- Timing pilots: BO d=10, N=500, cold refit every 5: 36.5, 37.0, 38.9 min (3 real runs, ~6 busy processes); TuRBO d=10:
  13-17 min (N=500), 26-32 min (N=1000); d=20, N=1000: 53-55 min (`results/summary/pilot_run_timing.csv`).
  The raw pilot-based BO estimator was too optimistic (21 min vs 37 measured; 3.6 vs 10.0 min for the A4 d=5 runs).
- The calibration factor 2.28 = measured / pilot integral already contains the load of the ~6 processes that were
  running; the old 1.3 multiplier cancelled out and is gone. An assumed 1.10 covers the 7th worker; it is checked
  against the wall times of the A4 check below (7 workers).
- Approved scope and order (plan changes): A4 check at d=10 first, then sensitivity grids, E2 primary, E2
  low-effective-dimension, H5 runs (alongside), E1 d<=10, E3, E2 other kappa, the reserved follow-up (about 20 wall-h),
  E1 d=20 last of the core, then extras (A8 c-e, E2 secondary) only if the core finishes early. Runs that are not
  finished by 8 Nov are cut in the plan's cut order and reported as not run.
- Compute estimate (`notes/compute_plan.md`, refit every 5): core without E1 d=20 about 109 wall-h (126 with a 15%
  contingency; range 88-201); core with E1 d=20 about 133 wall-h (153; range 107-245).
- **A4 check at d=10, decision rule fixed before the runs** (`configs/a4_d10_check.toml`): adopt "cold refit every 10"
  only if, at n=500, the paired CI of (every 10 - every 5) contains 0 and its half-width is at most 0.35 decades;
  otherwise the main runs use every 5. 15 held-out seeds (1000-1014), rotated ellipsoid, d=10, kappa=1e4.

## 2 Nov: A4 check at d=10 (outcome) and load factor

- 30/30 runs finished (`results/summary/a4_d10_check.csv`). Paired difference in log10 gap (every 10 - every 5):
  n=100: 0.000 (identical path, as expected); n=300: -0.010, CI [-0.132, 0.218]; n=500: -0.026, CI [-0.197, 0.200],
  half-width 0.198.
- **Rule applied as pre-set: CI contains 0 and half-width 0.198 <= 0.35, so the main runs use cold refits every 10
  iterations (n > 100).** Documented as a schedule choice, backed by this paired check at d=5 (n<=300) and d=10 (n=500).
- Measured wall time per run on 7 workers: 47.0 min (every 5), 29.1 min (every 10). The calibration run took 37.7 min
  (~6 busy processes), so the real 7-worker load factor is about 1.25, not the assumed 1.10. `LOAD7` is set to 1.25 and
  `notes/compute_plan.md` regenerated. With every 10 and LOAD7 = 1.25: core without E1 d=20 about 96 wall-h (111 with
  contingency; range 78-177), with E1 d=20 about 112 (129; range 90-206); central estimate finishes E1 d=20 on 7 Nov,
  before the 8 Nov hard stop. The high end (x1.6) would not fit; the cut order then applies.

## 4 Nov: lean core adopted, grids launched, chain running

- Scope change (instruction of 4 Nov): the core is now the sensitivity grids, E2 primary, E2 low-effective-dimension, H5,
  E1 for d <= 10 and the reserved follow-up. E3, E2 other kappa, E1 d=20 and A8 c-e are optional and run only if the core is
  ahead of schedule on 6 Nov. Hard stop for runs: 8 Nov. The 4 local commits were pushed to origin/main.
- Sensitivity grids launched 4 Nov 00:52 (CMA-ES grid: 120 runs in 18 s; BO grid: 120 runs, 7 workers). Chain
  `code/run_core.sh` waits for the BO grid, applies the decision rule (`code/analyze_sensitivity.py`), commits the frozen
  configs, then runs E2 primary -> E2 low-effective-dimension -> H5 -> E1 d = 2, 3, 5, 8, 10, retrying a failed step up
  to 3 times. Step times go to `logs/chain_times.csv`; stop markers `logs/CHAIN_STOPPED` / `logs/CHAIN_DONE`.
- CMA-ES grid result (rule applied by hand before the BO grid finished): sigma0 = 0.15 beats the default on the
  rotated ellipsoid (median -0.74 decades, CI [-1.04, -0.16]) but not on Rosenbrock (-0.04, CI [-0.12, 0.19]), so the
  default is kept under the rule (needs both problems).
- Slip: my first chain script waited for 60 BO result files, but the grid has 120 jobs; I stopped it while it was still in
  its wait loop (nothing ran twice) and relaunched with 120. The compute-plan row for the grid had the same count mix-up
  (60 vs 120 runs) and was corrected: 6.3 wall-h, not 3.1.
- Choices made for configs the plan left open (to be stated in the report): E1 ellipsoid is rotated with kappa = 1e2;
  low-effective-dimension function uses d = 20, d_eff = 2, equal weights (kappa = 1); E2 primary also runs CMA-ES,
  sep-CMA-ES and Sobol (negligible cost); BO and TuRBO refit cold every 10 iterations after n = 100.
- Expected finish times per row are in `notes/compute_plan.md` (lean core about 78 wall-h with contingency: follow-up
  ends about 7 Nov 07:00, i.e. after the 6 Nov gate for the optional rows).

- **Slip (4 Nov): the chain did not start after the first launch.** I launched it with `setsid`, which does not exist in
  this shell, so it never ran. The BO grid finished at 04:28 and the 7 workers were idle until I noticed at 10:50 (about
  6.4 h lost). Relaunched with `nohup`; the decision rule was applied at 10:52: all defaults kept (BO: no alternative
  beats the default on both problems; CMA-ES: sigma0 = 0.15 wins on the rotated ellipsoid only). Frozen configs committed
  (eca7c30). E2 primary started 4 Nov about 10:52, so all expected finish times in `notes/compute_plan.md` move later by
  about 6.4 h.

## 4 Nov: E2 primary analysis fixed in advance (commit e7a117d)
- `analysis_addendum_e2.md` replaces the "stop if the aligned-replicate CI excludes 0" rule with 99% CIs for the
  six-method noise-floor check and a tolerance of 0.15 decades (a flag does not stop the analysis; it is reported and
  investigated). Written when 7 of 540 E2 primary jobs had finished and none had been opened.
- Declared secondary variant: CMA-ES with sigma0 = 0.15 on the E2 primary instances and seeds, in its own experiment
  (`configs/e2_cma_sigma015.toml`, 90 CMA-ES jobs, seconds). It does not touch the BO runs or the main results.

## 5 Nov: E2 primary finished and analysed
- E2 primary: 540 jobs finished 5 Nov about 00:58 (`logs/chain_times.csv`: 47,263 s = 13.1 wall-h) against an estimate of
  about 21 h with contingency; the estimate was high for this row (aligned-BO runs took 30 to 35 min each, TuRBO and
  non-BO runs were faster than assumed). E2 low-effective dimension started right after.
- Pre-registered analysis run once with `code/analyze_e2.py` (commit e7a117d rules): no noise-floor flag, both H2
  interactions positive with CIs excluding 0. Results in `results/summary/e2_primary.csv`.
- The BO runs emit many linear_operator NumericalWarnings (Cholesky not positive definite at jitter 1e-3, fallback to
  symeig; 4,097 lines in `logs/e2_primary.log`). All runs completed; effect on results not examined.

## 5 Nov: E2 audit and plan regeneration
- Audit of E2 primary (540/540, 18 cells x 30, no duplicates, `code_dirty` false throughout; 533 jobs at a00172f, 7 at
  f53f5f6, no method code differs between them) and of the secondary (90/90). No timing outlier beyond 1.5x the cell median.
- Low-effective-dimension checkpoints: configured at 25, 50, 100, 150, 200, recorded at 40, 50, 100, 150, 200 because the
  first checkpoint fires at the first iteration with n >= 25 and the initial design has 40 points. Unchanged for all runs.
- `t_method` is elapsed time minus time inside the objective, so for the ARD-BO low-effective-dimension runs it includes the
  diagnostics time. The fit/acquisition split, CPU time and per-run timestamps are not stored.
- `notes/compute_plan.md` regenerated: finished rows marked done, finish dates recomputed from the end of E2 primary
  (5 Nov 00:58) with the original per-run estimates; the estimates for the remaining BO rows were not rescaled.

## 5 Nov: supplementary CMA-ES long runs (e2_cma_long)
- Addendum `analysis_addendum_e2_cma_long.md` and `configs/e2_cma_long.toml` committed (10d1332) before running. 60 jobs
  (aligned and rotated, 30 seeds, budget 10,000), 1 worker, 113 s, commit 10d1332, no failures.
- Integrity: the first 500 evaluations of all 60 runs equal the matching `cma` runs of E2 primary. All 60 runs ended by
  pycma's own termination before 10,000 evaluations (3,420 to 5,320 evaluations, median 3,895), each at the 1e-12 gap
  floor, so no run is censored by the 10,000 budget. The reach analysis of the addendum has not been run yet.
- E1 configs: each E1 config has a single `budget` shared by every method (e1_d2/3: 200, d5: 400, d8: 640, d10: 1000), so
  CMA-ES in E1 stops at the BO budget. Queued configs not changed.

## 5 Nov: chain stopped at E2 low-effective dimension (out of memory)
- `logs/CHAIN_STOPPED`: `e2_lowdim failed 3 times` (about 02:00). 38 of 200 jobs finished (20 aligned ARD-BO, 18 aligned
  iso-BO); no TuRBO job has finished. All three attempts end in `RuntimeError: DefaultCPUAllocator: not enough memory`
  (256 MB, 256 MB, then 33 MB) raised in `code/methods/turbo.py:85` (`model.posterior(candidates).rsample()`, d = 20, 4,000
  Sobol candidates, so a 4,000 x 4,000 covariance whose eigendecomposition fallback is used after the Cholesky warnings).
  With 7 TuRBO workers at once this exhausts memory; the machine has 15.4 GB RAM and 6.0 of 31.0 GB commit free right
  after the stop. H5 and E1 have not started. No code or config was changed.
