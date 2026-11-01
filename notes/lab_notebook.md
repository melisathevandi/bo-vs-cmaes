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
