# Lab notebook

Dated record of design choices, unexpected observations, failed attempts and their fixes. Oldest entry first.

**Check labels used in code and configs.** A1: aligned-replicate noise floor for the rotation penalty. A2: timing pilot
and budget choice. A4: GP refit-schedule check. A6: tuning on held-out seeds of the same problem families (sensitivity
grids). A7: H5 implementation (no termination, censoring flag, evaluation cap). A8a: correctness anchor. A8c-e (noise
experiment, long-budget probe, second problem family) were not run.

## 2026-11-01
- **CPU only.** The A2 timing pilot was specified for CPU and GPU. The machine has no CUDA device (AMD Ryzen 9 5900HS,
  integrated Radeon graphics) and torch 2.15.0 is the CPU build. Pilot timings are CPU only, and every result JSON records
  `"device": "cpu"`.

## 2026-11-01 (invariance test design)
- The originally specified E0 test ("rotated problem with rotated x0 -> rotated trajectory, path by path") is incorrect.
  With the same seed, pycma draws the same noise vector z in both runs, but the rotated run would need R z. Checked
  empirically (d=5, kappa=100, unbounded): the paths differ from the first generation on. Rotation invariance holds in
  distribution only.
- Consequence: monotone-transform invariance is tested exactly (`tests/test_invariance.py`); rotation invariance (and the
  effect of the box) is tested statistically in the E0 script, with paired differences over many seeds.
- First BO timing: about 1 s per iteration at d=5, n<=60 (single thread). Budgets of 1000 evaluations are costly for BO;
  budgets are set from the experiment requirements and the measured cost in the timing pilot
  (`results/summary/pilot_timing.csv`).
- Observed (n=40, aligned kappa=100, d=5): ARD lengthscales 13.4, 6.9, 4.0, 2.1, 1.0 follow the axis scales; GP rank
  correlation 0.97-1.0 (ARD) vs 0.6-0.95 (isotropic) at n=20-40.

## 2026-11-01 (timing pilot, CPU, 1 thread, 3 instance seeds per cell)
- Fit cost dominates BO's per-iteration cost. Warm-starting each refit from the previous hyperparameters cuts a refit by
  about 5-10x (e.g. d=10, n=1000: 42 s cold vs 4.9 s warm), and the average cost per iteration by about 3x
  (10.3 s -> 2.8 s). Acquisition optimisation is 0.2-2.2 s per iteration over n=100-1000.
- Estimated BO cost per run with warm starts: d=10, 500 evaluations about 6 min; d=10 or d=20, 1000 evaluations about
  22 min (integrated over n; single-thread timings, so parallel runs may be somewhat slower). On these numbers the A2
  budgets are kept.
- TuRBO's cost is not covered by this pilot. In a first smoke run (d=10, 500 candidates) it took about 1 s per iteration at
  n<=80, because Thompson sampling needs a joint posterior sample over the candidates. It is timed separately before E2.
- The A4 check (warm vs cold refit, refit every 5 vs 10) is run in `configs/a4_refit_check.toml` on held-out seeds
  1000-1009.

## 2026-11-01 (E0 rotation check for CMA-ES; seeding bug found and fixed)
- **Bug.** First E0 run: the aligned-vs-aligned replicate (noise floor) was significantly non-zero in one cell
  (+0.52, CI [0.32, 0.95]), which a noise floor must not be. Cause: instance randomness and method randomness both started
  from `default_rng(seed)`, so CMA-ES's start x0 was a function of the optimum x* (x* = 0.2 + 0.6 u, x0 = u). Aligned and
  rotated runs started close to x*, the replicate did not. Fix: methods now get `method_seed(seed)` (SeedSequence-mixed),
  applied in `runner.py` and `e0_invariance.py`; `tests/test_seeds.py` checks that raw seeding correlates x0 with x*
  (> 0.99) and the derived seed does not (< 0.2). All earlier outputs from the old scheme (smoke run, first A4 attempt,
  first E0 CSV) were deleted and regenerated.
- **E0 result** (CMA-ES, d=10, 1000 evaluations, 50 paired seeds, median difference of log10 gap [95% bootstrap CI];
  `results/summary/e0_rotation.csv`):
  - Unbounded: Delta_rot = -0.18 [-0.37, 0.07] (kappa=1e2), -0.02 [-0.26, 0.15] (kappa=1e4). Consistent with rotation
    invariance.
  - Box [0,1]^d (as in every main experiment): Delta_rot = +0.37 [0.09, 0.76] (kappa=1e2), +0.54 [0.25, 0.66] (kappa=1e4).
    Rotation costs CMA-ES about 0.4-0.5 decades once the box is present.
  - Noise floor (aligned replicate): +0.03 [-0.32, 0.44], +0.05 [-0.31, 0.28], -0.02 [-0.16, 0.09] and -0.16 [-0.31, -0.02];
    the last cell (kappa=1e4, unbounded) is marginally below 0, and is small compared with the bounded rotation effect.
- Interpretation: the invariance of full CMA-ES is a property of the algorithm without bounds; with box constraints it does
  not fully survive, which bears on which theoretical invariances survive a practical implementation. The mechanism is not
  tested. Candidate explanation: the bound-handling transformation distorts the search distribution differently for an
  elongated ellipsoid that is diagonal to the box walls. Candidate follow-up: vary the bound handling or the box size
  relative to sigma0.
- At kappa=1e4 the median gap after 1000 evaluations is about 10^1.5-10^2, i.e. CMA-ES is still in its geometry-learning
  phase at this budget (relevant for H5 and the E2 budget).

## 2026-11-01 (correctness anchor, A8a: passed)
- `code/anchor.py` -> `results/summary/anchor.csv`.
- The CMA-ES wrapper evaluates exactly the same points as `cma.fmin2` on the same bounded problem, start and seed
  (10/10 seeds, d=8 rotated ellipsoid, 200 evaluations).
- Branin (d=2), 40 evaluations, 10 seeds, threshold median gap < 0.01 fixed beforehand: BO (ARD, LogEI) median 1.0e-3
  (max 6.4e-3); TuRBO median 2.3e-3 (max 5.1e-3); Sobol with 40 evaluations 1.07 and with 400 evaluations 0.11.

## 2026-11-01 (H5 expected exponent; check of the draft predictions)
- A check of the draft `predictions.md` found an error in the H5 derivation: the numerator of c1 + c_mu is
  2 mu_eff - 2 + 2/mu_eff, not 2 + 2 mu_eff (the -4 + 2/mu_eff term from c_mu was dropped).
- Checked numerically with the learning rates pycma actually uses (`code/h5_expected_alpha.py`): the lambda / numerator
  factor is almost constant in d (log-log slope about -0.005), so it does not change alpha. Rate-based expected alpha:
  1.71 (d in [2, 80]) and 1.87 (d >= 10) with pycma 4.5.0; 1.69 / 1.86 with the tutorial formulas. pycma's c1 matches the
  tutorial; its c_mu is 7-48% larger (48% at d=2, 17% at d=10, 7% at d=80).
- Consequence: the draft prediction "CI for d >= 10 contains 2" is not what the rates predict at finite d. The prediction
  compares the fitted alpha with the rate-based value 1.87.
- UCB is kept in E3 (H3 makes a UCB-vs-LogEI prediction; E3 is at d <= 5, where BO is cheap).

## 2026-11-01 (A4 refit-schedule check: warm start rejected, cold refit every 10 adopted)
- Setup: BO (ARD, LogEI), budget 300, rotated ellipsoid (d=5, kappa=1e2) and Rosenbrock (d=5), held-out seeds
  1000-1009, paired by seed, reference = cold-start refit every 5 iterations (`configs/a4_refit_check.toml`,
  `results/summary/a4_refit_check.csv`).
- **Warm-started refits changed results.** Rotated ellipsoid, median difference in log10 gap, warm minus cold
  (positive = warm worse): n=100 +0.84 [0.49, 0.94]; n=300 +0.50 [0.24, 0.71] (refit every 5) and +0.46 [0.16, 0.69]
  (refit every 10). Rosenbrock: CIs contain 0 at every budget. At n=100 the refit schedule is identical, so the difference
  there comes from warm vs cold starting alone.
- **Diagnostic** (`code/diagnose_warm_start.py`, replaying 4 cold-run trajectories, n = 12 to 100): the warm chain's
  maximised marginal likelihood equals the cold fit's (median gap about 0, never below -0.5), but the fitted lengthscales
  differ at equal likelihood (e.g. seed 1003, n=100: cold [1.7, 4.3], warm [3.2, 7.9]). The likelihood is flat along a
  ridge, and where a fit lands on it changes BO's behaviour. The direction of the effect on lengthscales is not consistent
  across the 4 seeds, and this replay uses the cold run's data, so it does not establish the cause of the end-to-end
  difference. Not investigated further.
- **Outcome:** warm start is off (`warm_start=False`, the default). Main runs use cold-start fits.
- **Cold refit every 10 vs cold every 5** (same A4 data): median differences at n=200 and n=300 are 0.00 to +0.03 with CIs
  containing 0 on both problems (widths about +-0.3 decades), and wall time is 25% (ellipsoid) and 10% (Rosenbrock) lower
  at n=300. **Outcome:** main runs use cold fits and `refit_every = 10` once n > 100 (set explicitly in each config).
  Limits: tested at d=5 and n <= 300 with 10 seeds, so it is not evidence for d=10-20 at n=1000.
- Correctness anchor re-run under the cold default: BO (Branin, 40 evaluations) median gap 2.0e-3, worst seed 1.8e-2
  (above 0.01; the check is defined on the median, so it passes); TuRBO median 1.2e-3.
- Compute estimate (planning only, from the timing pilot, cold fits, refit every 10, x1.3 for parallel load; not results):
  BO run of d=10 with 500 evaluations about 15 min; d=10 with 1000 evaluations about 58 min; d=20 with 1000 evaluations
  about 74 min.

## 2026-11-01 (timing calibration; A4 check at d=10 set up)
- Timing pilots: BO d=10, N=500, cold refit every 5: 36.5, 37.0, 38.9 min (3 real runs, ~6 busy processes); TuRBO d=10:
  13-17 min (N=500), 26-32 min (N=1000); d=20, N=1000: 53-55 min (`results/summary/pilot_run_timing.csv`). The raw
  pilot-based BO estimator was too optimistic (21 min vs 37 measured; 3.6 vs 10.0 min for the A4 d=5 runs).
- The calibration factor 2.28 = measured / pilot integral already contains the load of the ~6 processes that were running;
  an assumed factor 1.10 covers the 7th worker and is checked against the wall times of the A4 check at d=10 (7 workers).
- **A4 check at d=10, decision rule fixed before the runs** (`configs/a4_d10_check.toml`): adopt "cold refit every 10"
  only if, at n=500, the paired CI of (every 10 - every 5) contains 0 and its half-width is at most 0.35 decades; otherwise
  the main runs use every 5. 15 held-out seeds (1000-1014), rotated ellipsoid, d=10, kappa=1e4.

## 2026-11-02: A4 check at d=10 (outcome) and load factor
- 30/30 runs finished (`results/summary/a4_d10_check.csv`). Paired difference in log10 gap (every 10 - every 5):
  n=100: 0.000 (identical path, as expected); n=300: -0.010, CI [-0.132, 0.218]; n=500: -0.026, CI [-0.197, 0.200],
  half-width 0.198.
- **Rule applied as pre-set: CI contains 0 and half-width 0.198 <= 0.35, so the main runs use cold refits every 10
  iterations (n > 100).** Backed by this paired check at d=5 (n<=300) and d=10 (n=500).
- Measured wall time per run on 7 workers: 47.0 min (every 5), 29.1 min (every 10). The calibration run took 37.7 min
  (~6 busy processes), so the 7-worker load factor is about 1.25, not the assumed 1.10; the compute estimate uses 1.25.

## 2026-11-04: sensitivity grids and frozen configs
- Scope of the main runs: sensitivity grids, E2 primary, E2 low-effective-dimension, H5, E1 for d <= 10 and one follow-up
  experiment. E3 at d=2 and a kappa = 1e4 supplement for H1 were added later; E2 at other kappa, E1 at d=20, E3 at d=5 and
  A8c-e were not run.
- Sensitivity grids: CMA-ES grid 120 runs (18 s); BO grid 120 runs, 7 workers. `code/run_core.sh` applies the decision
  rule (`code/analyze_sensitivity.py`), commits the frozen configs, then runs E2 primary -> E2 low-effective-dimension -> H5
  -> E1 d = 2, 3, 5, 8, 10, retrying a failed step up to 3 times. Step times go to `logs/chain_times.csv`; stop markers
  `logs/CHAIN_STOPPED` / `logs/CHAIN_DONE`.
- CMA-ES grid result: sigma0 = 0.15 beats the default on the rotated ellipsoid (median -0.74 decades, CI [-1.04, -0.16])
  but not on Rosenbrock (-0.04, CI [-0.12, 0.19]), so the default is kept under the rule (which requires both problems).
- A first version of the chain script waited for 60 BO result files although the grid has 120 jobs; it was stopped in its
  wait loop (no job ran twice) and restarted with 120. The compute estimate for the grid had the same count error
  (60 vs 120 runs) and was corrected (6.3 wall-h, not 3.1).
- A first launch of the chain used `setsid`, which is not available in the shell used, so the chain did not start; it was
  restarted with `nohup`. The decision rule was then applied: all defaults kept (BO: no alternative beats the default on
  both problems; CMA-ES: sigma0 = 0.15 wins on the rotated ellipsoid only). Frozen configs committed (eca7c30).
- Config choices not fixed by `predictions.md`: the E1 ellipsoid is rotated with kappa = 1e2; the low-effective-dimension
  function uses d = 20, d_eff = 2, equal weights (kappa = 1); E2 primary also runs CMA-ES, sep-CMA-ES and Sobol
  (negligible cost); BO and TuRBO refit cold every 10 iterations after n = 100.

## 2026-11-04: E2 primary analysis fixed in advance (commit e7a117d)
- `analysis_addendum_e2.md` replaces the "stop if the aligned-replicate CI excludes 0" rule with 99% CIs for the six-method
  noise-floor check and a tolerance of 0.15 decades (a flag does not stop the analysis; it is reported and investigated).
  Written when 7 of 540 E2 primary jobs had finished and none had been opened.
- Declared secondary variant: CMA-ES with sigma0 = 0.15 on the E2 primary instances and seeds, in its own experiment
  (`configs/e2_cma_sigma015.toml`, 90 CMA-ES jobs, seconds). It does not touch the BO runs or the main results.

## 2026-11-05: E2 primary finished and analysed
- E2 primary: 540 jobs, 47,263 s = 13.1 wall-h at 7 workers (`logs/chain_times.csv`), against an estimate of about 21 h
  with contingency (aligned-BO runs took 30 to 35 min each; TuRBO and non-BO runs were faster than estimated).
- Pre-registered analysis run once with `code/analyze_e2.py` (rules of commit e7a117d): no noise-floor flag, both H2
  interactions positive with CIs excluding 0. Results in `results/summary/e2_primary.csv`.
- The BO runs emit many linear_operator NumericalWarnings (Cholesky not positive definite at jitter 1e-3, fallback to
  symeig; 4,097 lines in `logs/e2_primary.log`). All runs completed; the effect on results was not examined.

## 2026-11-05: E2 audit
- Audit of E2 primary (540/540, 18 cells x 30, no duplicates, `code_dirty` false throughout; 533 jobs at a00172f, 7 at
  f53f5f6, no method code differs between them) and of the secondary (90/90). No timing outlier beyond 1.5x the cell median.
- Low-effective-dimension checkpoints: configured at 25, 50, 100, 150, 200, recorded at 40, 50, 100, 150, 200 because the
  first checkpoint fires at the first iteration with n >= 25 and the initial design has 40 points. The same for all runs.
- `t_method` is elapsed time minus time inside the objective, so for the ARD-BO low-effective-dimension runs it includes the
  diagnostics time. The fit/acquisition split, CPU time and per-run timestamps are not stored.

## 2026-11-05: supplementary CMA-ES long runs (e2_cma_long)
- Addendum `analysis_addendum_e2_cma_long.md` and `configs/e2_cma_long.toml` committed (10d1332) before the runs. 60 jobs
  (aligned and rotated, 30 seeds, budget 10,000), 1 worker, 113 s, no failures.
- Integrity: the first 500 evaluations of all 60 runs equal the matching `cma` runs of E2 primary. All 60 runs ended by
  pycma's own termination before 10,000 evaluations (3,420 to 5,320 evaluations, median 3,895), each at the 1e-12 gap
  floor, so no run is censored by the 10,000 budget.
- E1 configs have a single `budget` shared by every method (e1_d2/3: 200, d5: 400, d8: 640, d10: 1000), so CMA-ES in E1
  stops at the BO budget; the long runs supply the CMA-ES side of the break-even cost.

## 2026-11-05: E2 low-effective dimension stopped (out of memory) and resumed with 4 workers
- `logs/CHAIN_STOPPED`: `e2_lowdim failed 3 times`. 38 of 200 jobs had finished (20 aligned ARD-BO, 18 aligned iso-BO); no
  TuRBO job. All three attempts ended in `RuntimeError: DefaultCPUAllocator: not enough memory` (256 MB, 256 MB, then 33 MB)
  raised in `code/methods/turbo.py:85` (`model.posterior(candidates).rsample()`, d = 20, 4,000 Sobol candidates, so a
  4,000 x 4,000 covariance whose eigendecomposition fallback is used after the Cholesky warnings). With 7 TuRBO workers at
  once this exhausts memory (15.4 GB RAM).
- The remaining 162 jobs were resumed with 4 workers through `code/run_experiment.py`; no change to `turbo.py`,
  `runner.py`, `run_core.sh` or any config. H5 and E1 were started separately afterwards.
- `code/run_experiment.py` adds `"workers"` to each newly written result JSON (commit 7e62b17; not in the spec, so job
  hashes are unchanged). Jobs written before this have no such field: 7 workers for E2 primary, the 38 earlier e2_lowdim
  jobs and the sensitivity grids (log header `N jobs, 7 workers`). The 60 `e2_cma_long` jobs used 1 worker.

## 2026-11-05: reach analysis of the CMA-ES long runs
- `code/analyze_cma_long.py` run once as specified in the addendum; output `results/summary/e2_cma_long_reach.csv`. All 60
  runs reach every target (no censoring); c* is defined for 30 of 30 seeds in each cell. The long runs' wall-clock times were
  taken at 1 worker while the E2 low-effective-dimension jobs ran (4 workers), so the c* values carry that load.

## 2026-11-05: CMA-ES long runs on the E1 problems (e1_d{2,3,5,8,10}_cma_long)
- Addendum `analysis_addendum_e1_cma_long.md` and five configs committed (4d901fa) before the runs; no E1 config edited.
  200 jobs (5 dimensions x 2 problems x 20 seeds), 1 worker, run while E2 low-effective-dimension ran at 4 workers (timings
  carry that load). No failures.
- All 200 runs ended by pycma termination before the 50,000 cap (lengths: d=2 474 to 1,182; d=10 ellipsoid 2,750 to 3,270,
  Rosenbrock 5,270 to 7,190). The ellipsoid runs all end at the 1e-12 floor. Rosenbrock runs ending above gap 1e-6
  (a local minimum, gap about 4): d=5 1 of 20, d=8 2 of 20, d=10 1 of 20, none at d=2 and 3. The target rule of the addendum
  counts these as censored (terminated) where they never reach a target.

## 2026-11-05: cost_table.py reads the CMA-ES long runs
- `code/cost_table.py` takes the CMA-ES side from `<experiment>_cma_long` when it exists (column `cma_source`), uses the
  addenda's targets unchanged (each BO run's own final gap, and the absolute levels 1e-1, 1e-2, 1e-4), and reports per
  problem and target the CMA-ES seeds that never reach it (`cost_table.csv`) and, per such seed, whether CMA-ES was cut by the
  budget or by termination and whether BO reached each absolute level (`cost_censored.csv`).
- Check: `uv run python code/cost_table.py --check-e2` reproduces the E2 c* values of `e2_cma_long_reach.csv` exactly
  (largest absolute difference 0). A synthetic test covers the censored-seed reporting.

## 2026-11-05: E2 low-effective dimension finished
- 200 of 200 jobs, no duplicates, no error after the 4-worker restart (`done in 28,522 s` = 7.9 h for the 162 resumed jobs;
  the first 38 ran at 7 workers).
- Per-run wall time, median: aligned ARD-BO 1,541 s (7 workers), rotated ARD-BO 2,637 s (4 workers), iso-BO 724 to 806 s,
  TuRBO 1,116 s aligned and 1,208 s rotated (4 workers), CMA-ES and Sobol under 1 s. Rotated ARD-BO runs are 71% slower than
  aligned ones although they ran with fewer workers, so the per-instance cost is a property of the fit, not of the load;
  ARD-BO times are not comparable across the two instances or the two worker counts.
- 7 records have `code_dirty = true` (6 aligned iso-BO written at 977d3a3 and 244998e, 1 aligned TuRBO at b00b49d): analysis
  scripts under `code/` were uncommitted at the time. `code/methods`, `runner.py`, `problems.py` and `diagnostics.py` are
  identical to f53f5f6.

## 2026-11-05: H5 run, and a bug fix in the alpha fit
- H5: 220 jobs, 7 workers, about 3 min. The final summary step failed with `need at least one array to concatenate`:
  `fit_alpha` filtered with `table.T.notna()`, and `DataFrame.T` is the transpose, not the column `T`. Fixed to
  `table["T"].notna()` in `code/h5_learning_time.py`; the raw runs were untouched and the summary was rerun twice, with
  identical `h5_alpha.csv` and `h5_learning_time.csv` (bootstrap seed 0). No definition changed.
- Result against the frozen prediction (`predictions.md`): the d >= 10 CI excludes 1.87, so the refutation criterion is met
  (see `notes/report_notes.md`, H5). The H5 records have no worker field; the log header says 7 workers.
- rho and rank-correlation analysis of E2 low-effective dimension: `code/analyze_lowdim.py`.

## 2026-11-06: E1 finished, E1 long CMA-ES runs, E3 d=2, TuRBO ablation
- E1: 600 runs, 14.2 h at 7 workers (`done in 51270 s` for d = 10), no failures, all `code_dirty = false`. BO runs at d = 10
  took 1.9 to 2.8 h each.
- E1 long CMA-ES runs (200 runs, 1 worker, 20 s in total): the first 20 evaluations equal those of the E1 CMA-ES runs for all
  200 pairs.
- `analysis_addendum_e1.md` was committed (4680bb8) before any E1 gap value was computed; before it, only file counts,
  `code_dirty`/`workers` fields, per-job wall times and run durations had been inspected. `predictions.md` H1 names kappa =
  1e4 but the E1 configs use kappa = 100 (no recorded reason for the difference): dated amendment 0363ee5.
- E3 d = 2 and TuRBO ablation: design choices and results in `notes/report_notes.md`. A 2-seed smoke test of E3 continued
  to write records after it was stopped (orphaned workers); `results/raw/e3_d2_quick` was deleted, and no real records were
  affected. A test file was briefly placed under `code/` while E3 was running (removed within about a minute; the 633 records
  written by then have `code_dirty = false`).
- The first default-reproduction check of the TuRBO change failed on all 9 jobs because the check script used multithreaded
  torch while the E2 runs were single-threaded; with one thread the 9 trajectories match exactly. One new test (volume of the
  trust region) ignored clipping at the box edge and was corrected in the test, not in the code.
- E1 analysis scripts were developed outside `code/` while the ablation ran, then moved in (1fd7721); the rerun gave identical
  md5 for all 7 output files.

## 2026-11-06: kappa = 1e4 supplement, E3 sanity checks, figures
- kappa = 1e4 supplement for H1: addendum 290b815 (states that it was added after the E1 analysis), 240 runs at d = 2 and 5,
  7 workers, `code_dirty = false`. Before it, the E2 primary results at n = 500 had been inspected, but no n = 200 value.
- E3 sanity checks and the exploration check ran on one core alongside the supplement (no `code/` edits during the runs: the
  scripts were developed in an untracked folder and moved in after the runs, a0fffe1). The exploration check's console output
  is saved in `results/summary/e3_exploration_check.txt`.
- `code/make_figures.py` builds Figures 1-5 (E2 primary, E2 low dimension, E1 regime map + H1 statistic + H5, E3, TuRBO
  ablation) and `report/generated/numbers.tex` from `results/summary` and `results/raw`; two consecutive runs give identical
  `numbers.tex`. First renders had a legend over the data (Fig 2) and crowded log-axis tick labels (Figs 3c, 4b); fixed.

## 2026-11-06: report scaffolding and the clean-clone check
- Added `code/make_tables.py` (7 tables from `results/summary`, outcome column evaluated from the pre-registered rules),
  a placeholder `report/report.tex` (headings and, per section, figures, tables and macro prefixes), `report/references.bib`,
  and `code/check_reproduction.py`.
- The first version of the check could pass vacuously (a script that wrote nothing left the committed file in place). It
  deletes all derived files in the clone first. Result: all 17 steps run, no file differs from the committed one, all 122
  macros are listed in exactly one section, the placeholder compiles. Figures are saved without a creation date, so they
  compare byte for byte.
- `.gitignore` does not exclude `report/figures/`; LaTeX build files are ignored.
- `results/raw` (3.7 GB) is not tracked (see `readme.md`). Appendix tables (sensitivity, A4, anchor, E0, CMA-ES long runs,
  E3 sanity) are not generated; their source CSVs are listed in `report/report.tex`.

## 2026-11-07: figure revisions
- Figs 2, 3, 5 (58f8465, 8c94df3): legends, IQR bands, separate regime-map panels, fitted and predicted slopes, TuRBO variant
  names, 6.5 inch width.
- Fig 2 has one shared legend above the three panels (colour = method, line style = instance, dotted = isotropic value,
  band = IQR). Fig 3(d): the 1.71 line is anchored at the all-d centroid, the 1.87 and 2 lines at the d >= 10 centroid (each
  drawn over the d range of its fit); the legend states the centroid, and the caption should state it as well. Fig 5 uses the
  method colours of Figs 1 and 2 (the three TuRBO variants share the TuRBO colour and are distinguished by the axis labels);
  filled = aligned or Delta_rot, open = rotated or noise floor.
- Figure PDFs only; no number changed (122 macros).
