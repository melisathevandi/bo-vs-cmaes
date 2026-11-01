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
