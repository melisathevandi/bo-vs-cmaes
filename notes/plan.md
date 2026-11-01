# Plan revision 2 (Nov 1): response to review, revised design, next steps

> **APPROVED by Melisa on Nov 1, with the amendments below. Where an amendment conflicts with the body of this plan, the amendment wins. No further plan revisions: problems are logged in `notes/lab_notebook.md`.**
>
> **A1. κ=1 is a sanity check only.** A rotated sphere is the same function, so Δ_rot = 0 at κ=1 by construction. It does *not* detect boundary artefacts. **The real null:** a second, independent draw of the aligned instance (different instance seeds), with Δ computed between the two aligned replicates for each method. That Δ is the noise floor for Δ_rot. Verification: *the aligned-vs-aligned replicate Δ has a CI containing 0.*
>
> **A2. Budgets** (final values set by the pilot):
> - d=2: 200, d=3: 200, d=5: 400, d=8: 640, d=10: 1000, d=20: 1000 (2000 only if the pilot shows that is cheap).
> - E2 runs at about 500 evaluations: it tests rotation, not scaling.
> - The Fig 3(b) collapse comparison uses d ∈ {2, 3, 5, 8, 10} only.
> - d=20 is labelled "BO's hard regime"; CMA-ES does not reach its learned-geometry regime there.
> - Pilot: time one BO iteration at n = 100, 300, 600, 1000, on CPU and on GPU (float64). Use the GPU only where it is faster, and record the device in every result JSON.
>
> **A3. H2 primary endpoint, fixed now:** final budget, κ=10⁴, d=10, 30 seeds. All other κ values and budgets are secondary and labelled as such.
>
> **A4. BO refit schedule:** refit the GP hyperparameters every 5–10 iterations once n > 100. Document this, and check once on a pilot problem that it doesn't change results (paired CI).
>
> **A5. H4:** c* is a quantity derived from measured overhead, not a blind prediction. The "0.1 to 10 s" range is removed.
>
> **A6. Tuning wording:** "held-out seeds on the same problem families", not "held-out problems".
>
> **A7. H5 implementation:**
> - Disable CMA-ES's termination criteria.
> - Flag runs that reach machine-precision f before κ_t < τ.
> - Cap evaluations at 100·d², raising it only if runs are censored.
> - Derive the expected α by hand from c_µ, λ and µ_eff before committing it. This replaces "may come out somewhat below 2".
>
> **A8. Additions in priority order, once the core plan is on track** (slip rule unchanged: cut c–e first):
> - a. **Correctness anchor (required):** check the wrappers against a standard reference curve (e.g. COCO/BBOB sphere, or a library baseline) before E1–E3.
> - b. 30 seeds on the primary H2 endpoint (A3).
> - c. Compact noise experiment (E5): noisy ellipsoid; compare best-observed, GP posterior mean and CMA-ES mean as the recommendation; score by the true f.
> - d. Long-budget probe: d=10, κ=10⁴ rotated, 2000 evaluations, 10 seeds. A candidate for the follow-up slot.
> - e. A second problem family (a few BBOB functions), appendix only.

## Context
The first review and plan (Oct 29) are saved in the repo at `notes/plan.md`, commit `361ce1e`. You then sent six critique points and asked me to check them critically rather than accept them as given. You also want a slot for a **follow-up experiment** after E1–E3, for the rubric's "follow-up on findings". This file is the revised plan. Once approved, it replaces `notes/plan.md`.

**Status:**
- **Done:** repo, uv + Python 3.12 environment, `code/problems.py` (ellipsoid with conditioning and rotation, shifted Rosenbrock), 20 passing unit tests.
- **Not started:** method wrappers, runner, pilot, all experiments.
- **Timing:** today is day 4 of 16. The deadline is 13 Nov 2026, 23:59.

---

## 1. Response to each critique point

**1. O(d²) is stated as fact. Agree, with a refinement.** I'll treat it as a hypothesis (H5 below) and check the source myself. Hansen's tutorial sets the covariance learning rates c₁ ≈ 2/d² and c_µ ≈ µ_eff/d². What that implies for the number of evaluations has to be worked out, not quoted.

The proposed test was to see whether the regime map collapses better on evals/d or evals/d². I'll do it, but as a *secondary* view. The BO-vs-CMA crossover mixes two methods' scaling, and BO has no reason to follow d². The cleaner primary test uses CMA-ES alone, which is cheap:
- Measure CMA-ES's learning time (defined in 2c) on the κ=10⁴ rotated ellipsoid, at 11 dimensions from 2 to 80 with 20 seeds each.
- Fit the log-log slope α with a bootstrap CI. If the heuristic holds, α ≈ 2. With default λ ≈ 4 + 3 ln d, the slope may come out somewhat below 2, which would be a finding in itself.

**2. "Isotropic BO is unaffected by rotation" needs a qualifier. Agree.** The isotropic kernel is invariant, but the box, the Sobol design and box-constrained acquisition optimisation are not. H2 will now say "approximately invariant, with the size of the deviation measured in E0 and E2".

On "keep the rotated optimum inside the box": for the ellipsoid this is **already guaranteed**. The rotation is about x*, x* is drawn in [0.2, 0.8]^d, and it's the same point for the aligned and rotated instances (unit-tested in `test_aligned_and_rotated_share_optimum`). What remains is subtler: rotation changes how the long axis of the ellipsoid meets the box walls. I'll handle that with two controls:
- **κ = 1 null check.** A rotated sphere is *the same function*, so I'll run κ=1 once (aligned only). Any aligned-vs-rotated gap at κ>1 then comes from the conditioning interacting with the box, not from rotation as such.
- **Low-effective-dimension function.** The optimum x* is inside the box by construction. Before E2 I'll also check numerically that uniform samples give similar f-value distributions in the aligned and rotated cases.

**3. Defaults plus sensitivity departs from the original tuning plan. Agree it must be symmetric, and this goes back to the held-out design you started with.**
- **Same tuning effort:** a 3 × 2 grid of 6 configurations per method.
  - CMA-ES: σ₀ ∈ {0.15, 0.3, 0.6} × λ ∈ {default, 2×default}.
  - BO: n_init ∈ {d, 2d, 4d} × Matérn ν ∈ {1.5, 2.5}.
- **Why ν instead of the kernel:** the brief's "kernel" knob, ARD vs isotropic, is the very factor E2 studies, so tuning it would bias the study.
- **Held out:** run on two pilot problems (κ=10² rotated ellipsoid at d=10; Rosenbrock at d=5) with **seeds 1000+**, which don't overlap the main-run seeds. 10 seeds per configuration.
- **Decision rule, fixed now:** keep the default unless another configuration is better on *both* pilot problems and its paired bootstrap CI excludes 0. If it is, adopt that configuration and report it.
- **Reporting:** an appendix table shows how far each knob moves the result.
- **Cost:** about 120 BO runs. Roughly half a day or one night with 7 workers.

**4. The scope is large. Agree.** Every experiment now feeds a specific figure or table (section 2b), and methods are limited per experiment (section 2a). Changes:
- sep-CMA-ES and isotropic BO only in E2.
- TuRBO only in E2.
- UCB is cut first, IPOP second.
- Rotated κ=1 is not run (it's the same function, as above).
- E4 becomes one table, built from logs.

**5. Put noise in the limitations. Agree.** It's named in the brief, so the limitations section will say explicitly that it wasn't studied and why: time, and the protocol for choosing a recommendation under noise.

**6. Timeline. Agree.** The next gate is the pilot and frozen configs, then E2 runs. One correction on ordering: E0 needs the same CMA-ES wrapper and takes minutes, so it runs during the pilot rather than as a separate step. The revised timeline is in section 4.

---

## 2. Revised design

**Hypotheses** (go into `predictions.md`, committed before the main runs):
- **H1:** BO's advantage over CMA-ES at a fixed evals/d shrinks as d grows.
- **H2:** Axis-aligned methods (ARD-BO, sep-CMA-ES) gain on aligned problems and lose that gain when rotated. Isotropic BO and full CMA-ES are *approximately* invariant. The deviations caused by the box and the initial design are quantified.
- **H3:** P(finding the narrow basin) falls with basin width. CMA-ES restarts recover it at larger budgets, and BO's fitted lengthscale sets the width below which it is missed.
- **H4:** The break-even evaluation cost c* rises with budget and d, because GP overhead grows with n. Prediction: c* is between 0.1 and 10 s per evaluation at the budgets used.
- **H5 (new):** CMA-ES's learning time (defined in 2c) scales as d^α. Hansen's heuristic suggests α ≈ 2. We report α̂ with a bootstrap 95% CI and claim "consistent with d²" only if the CI contains 2. To be tested, not assumed.

### 2a. Methods per experiment
| Method | Sens. | E0 | E1 | E2 | E3 |
|---|---|---|---|---|---|
| Sobol | | | ✓ | ✓ | ✓ |
| CMA-ES (full) | ✓ | ✓ | ✓ | ✓ | ✓ |
| sep-CMA-ES | | | | ✓ | |
| IPOP-CMA-ES | | | | | ✓ (cut 2nd) |
| BO: ARD, LogEI | ✓ | ✓ (monotone transform) | ✓ | ✓ | ✓ |
| BO: isotropic, LogEI | | | | ✓ | |
| BO: ARD, UCB | | | | | ✓ (cut 1st) |
| TuRBO-1 | | | (if time) | ✓ | |

### 2b. Experiment → figure or table (anything without one is dropped)
**Every figure shows Sobol as a reference** (a line or marker), so we can see whether either method beats sampling with no model at all.

| Output | Source | Content |
|---|---|---|
| **Fig 1** | E2, ellipsoid | **2×2 methods only** (ARD-BO, iso-BO, CMA, sep-CMA) plus the Sobol line. x-axis κ ∈ {1, 10², 10⁴}. (a) Median final log₁₀ gap, aligned and rotated (IQR bars). (b) **Rotation penalty Δ_rot with a paired bootstrap 95% CI** (2d). κ=1 is the null: Δ_rot should be ≈ 0, and anything else means a bug or a boundary artefact. **TuRBO is not in this figure**: its Δ_rot goes in the text and Table 3. |
| **Fig 2** | E2, low-eff-dim | (a) log₁₀ gap vs evaluations, aligned vs rotated: ARD-BO, iso-BO, CMA, Sobol. (b) Relevance ratio ρ vs n. (c) GP rank correlation vs n. Definitions are in 2c. |
| **Fig 3** | E1 + H5 | (a) Regime map: median paired difference (BO − CMA) in log₁₀ gap vs evals/d, one line per d. (b) The same against evals/d² (secondary view of collapse). (c) **CMA-ES-only** learning time vs d on log-log axes, d ∈ {2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 80}, with the fitted α̂ and its bootstrap CI. |
| **Fig 4** | E3 | P(success) vs basin width with Wilson CIs, d ∈ {2, 5}, Sobol included. Success is defined in 2c. |
| **Fig 5** | Follow-up | **Reserved.** Shares a row with Fig 4 (two half-width panels). |
| Table 3 | all | Prediction vs outcome, H1–H5 plus the follow-up. Each row gives the **test statistic and its paired CI** (2d). |
| Table 2 | E4 | Break-even c* by d and target accuracy (about 4 rows). |
| Table 1 → text | E0 | Invariance audit as about 3 sentences in the text. It becomes a table only if there's room. |
| Appendix | sensitivity | 6-configuration grids for CMA-ES and BO. |

### 2c. Operational definitions (for the methods section; you should be able to explain each)
- **Relevance ratio ρ (BO's ARD kernel).**
  - Inputs live in the unit cube. An ARD kernel measures distance as r² = (x−x′)ᵀ M (x−x′), with M = diag(1/ℓ₁², …, 1/ℓ_d²). Large 1/ℓᵢ² means "the GP thinks f changes quickly along coordinate i".
  - The low-effective-dimension function depends on x only through A x, where A is d_eff × d with orthonormal rows. P = AᵀA projects onto the directions that matter.
  - **ρ = tr(PM) / tr(M)** is the share of the kernel's total sensitivity that falls in the relevant directions. It lies in [0, 1].
  - An isotropic kernel gives exactly ρ = d_eff/d, which is the baseline. A perfect ARD fit on the aligned problem gives ρ → 1.
  - On the rotated problem, a diagonal M can only reach ρ > d_eff/d where some coordinate axis happens to line up with the relevant subspace, so we expect ρ ≈ d_eff/d.
  - Computed from the fitted GP at checkpoints n ∈ {25, 50, 100, 150, 200} (adjusted to the final budget).
- **GP rank correlation.**
  - At each checkpoint n, take the GP fitted to the n points. Draw m = 200 fresh points from N(x_inc, 0.1² I), clipped to the box, where x_inc is the current best point; this is the region the acquisition function searches.
  - Compute the **Spearman correlation between the GP posterior mean and the true f** at those points. 1 means the surrogate ranks candidates perfectly; 0 means it is no better than random.
  - Ranking is what matters for picking the next point, which is why rank correlation is used rather than RMSE.
- **CMA-ES learning time T(d).**
  - Use the rotated ellipsoid with κ = 10⁴ and Hessian H. CMA-ES runs **without box bounds**, so the bounds can't interfere with the adaptation being measured, with x₀ ~ U[0,1]^d and σ₀ = 0.3.
  - After each generation t, compute κ_t = cond(C_t^½ H C_t^½), which equals the eigenvalue ratio of C_t H. It starts at 10⁴ (C₀ = I) and reaches 1 when C ∝ H⁻¹, i.e. when the search distribution has fully learned the geometry.
  - **T = evaluations until κ_t first drops below τ = 10.** Runs are capped at 10⁴·d² evaluations, and runs that hit the cap are recorded as censored. Robustness: τ ∈ {3, 100} reported in the appendix.
- **α fit.**
  - Fit an OLS line to log T against log d (all seeds, 20 per d).
  - 95% CI by bootstrap: resample seeds within each d, 10,000 times.
  - Fixed now: the primary fit uses all d, and a secondary fit uses d ≥ 10, to check for small-d effects.
  - Caveat to state: the default λ = 4 + ⌊3 ln d⌋ grows with d, so α combines learning rate and population-size effects.
- **Success in E3.**
  - The decoy-basin function is built as f(x) = min(broad basin, narrow deep basin), with the narrow basin's minimum strictly lower than the decoy's minimum f_decoy.
  - **Success: f(x_rec) < f_decoy.** That is, the recommended point is better than anything the decoy basin can offer, which is only possible inside the narrow basin.
  - The definition uses only function values, it is unambiguous, and it doesn't depend on a distance threshold.

### 2d. Uncertainty on differences (the brief asks for this explicitly)
- **Pairing:** all comparisons pair by seed. Methods share the same instance for a given seed, and the aligned and rotated instances share x*.
- **Method difference:** for methods A and B at budget b, take each seed's difference Δ = log₁₀ gap_A − log₁₀ gap_B. Report the **median Δ with a 95% percentile bootstrap CI** (10,000 resamples of seeds) and the **paired win rate** P(A < B) with its CI.
- **H2 statistic:**
  - Rotation penalty for a method: Δ_rot = log₁₀ gap(rotated) − log₁₀ gap(aligned), paired by seed.
  - Axis-bias interaction: **Δ_rot(ARD-BO) − Δ_rot(iso-BO)**, and likewise **Δ_rot(sep-CMA) − Δ_rot(CMA)**, each with a bootstrap CI.
  - H2 is supported if both interactions are > 0 and their CIs exclude 0.
- Table 3 gives, for each hypothesis, the statistic, its point estimate, its CI, and the verdict.

### 2e. Page budget (8 pages target, hard cap 10)
| Section | Pages |
|---|---|
| Title, intro, hypotheses | 1.0 |
| Methods + definitions | 1.5 |
| Results: E2 (Figs 1–2) | 1.5 |
| Results: E1 + H5 (Fig 3) | 0.9 |
| Results: E3 + follow-up (Figs 4/5, one row) | 1.0 |
| Results: E0/E4 text + Table 2 | 0.4 |
| Discussion and limitations (incl. noise) | 0.7 |
| References | 0.5 |
| Appendix (sensitivity, τ robustness) | 0.5 |
| **Total** | **≈ 8.0** |

If it runs over, cut in this order: Table 2 → text, then shorter E0 text, then move Fig 2(c) to the appendix.

**Unchanged from revision 1:**
- Shared unit box, with 2d Sobol initial points counted in the budget.
- Paired seeds: main runs use seeds 0–19 (0–29 for E3); tuning uses 1000+.
- Metrics: log₁₀ gap of the true objective; median and IQR; paired win rate with bootstrap CIs.
- Diagnostics; auto-generated `numbers.tex`; Overleaf report of at most 8 pages; noise listed under limitations.

---

## 3. Follow-up experiment slot (new)
- **Reserved:** about 1.5 days (Nov 7–8), one overnight of compute, Fig 5 (a half-width panel next to Fig 4), and about 0.5 pages.
- **Process:**
  1. After E1–E3, list surprises in `notes/lab_notebook.md`.
  2. Pick the one whose explanation is most testable.
  3. Write its prediction into `predictions.md` and commit it, timestamped, **before** running.
  4. Report it whether or not the prediction holds.
- **Examples only, not committed to:**
  - If ARD-BO's rotation penalty is small, test whether it grows with κ or shrinks with budget. That would show whether the GP is "rescued" by data.
  - If α in H5 is well below 2, vary λ to separate generations from evaluations.
  - If TuRBO matches CMA-ES on rotated problems, swap TuRBO's axis-aligned trust region for an isotropic one.
  - If E3 shows BO missing basins, vary the lengthscale prior to test the mechanism.

---

## 4. Revised timeline (today Nov 1 = day 4)
| Date | Work | Gate |
|---|---|---|
| Nov 1 | Method wrappers (Sobol, CMA/sep/IPOP, BO ARD/iso LogEI/UCB, TuRBO-1); single-run runner with timing split; low-eff-dim function + tests | `pytest` passes; quick smoke run |
| Nov 2 | Timing pilot (BO s/iteration at d=2/10/20); E0 invariance; sensitivity grid overnight | Budgets fixed |
| Nov 3 | Apply decision rule; freeze configs; **commit `predictions.md`**; start **E2 overnight** | Predictions committed |
| Nov 4 | E2 analysis + diagnostics; H5 learning-time runs (CMA only, d up to 80, 20 seeds); **E1 overnight** | Figs 1–2 drafts |
| Nov 5 | E1 analysis + α fit; decoy-basin function + tests; **E3 run** | Fig 3 draft |
| Nov 6 | E3 analysis; E4 table from logs; choose follow-up, commit its prediction | Fig 4, Table 2 |
| Nov 7–8 | **Follow-up experiment** | Fig 5 |
| Nov 9–12 | Final figures and `numbers.tex`; you write the report in Overleaf | Draft by Nov 11 |
| Nov 13 | Clean-clone check; download Overleaf source into `report/`; submit (23:59) | Submitted |

**Slip rule:** if a gate is missed by more than half a day, apply the cut order: UCB → IPOP → TuRBO in E1 → d=20 in E1 → seeds down to 15. E2, the diagnostics, the sensitivity grid and the follow-up slot are never cut.

---

## 5. Implementation for the next session (Nov 1–2)
**Files to create:**
- `code/methods/sobol.py`, `cmaes.py` (full/sep/IPOP via pycma), `bo.py` (SingleTaskGP, ARD or isotropic Matérn, LogEI/UCB), `turbo.py` (adapted from BoTorch's TuRBO-1 tutorial, with a citation)
  - Every method exposes the same interface: `run(problem, budget, seed, **cfg) → trajectory`. Each record holds x, f, the incumbent recommendation, and overhead time.
- `code/runner.py`: one (config, method, seed) → one JSON file, recording package versions and the git commit.
- `code/run_experiment.py`: process pool with `OMP_NUM_THREADS=1` and `torch.set_num_threads(1)`, a `__main__` guard (Windows spawn), resume by config hash, and `--quick`.
- `code/problems.py`: add `low_effective_dim(d, d_eff, rotated, seed)`, reusing `random_rotation` and `_sample_optimum`.
- `code/tests/test_invariance.py`: same seed → identical CMA-ES trajectory for f vs g∘f; rotated problem with rotated x₀ and no bounds → rotated trajectory.
- `configs/pilot.yaml`, `configs/sensitivity.yaml`.
- Replace `notes/plan.md` with this revision.

**Other steps:**
- **Before writing the wrappers:** check the current pycma 4.5 and BoTorch 0.18 APIs (option names for sep-CMA/`CMA_diagonal`, IPOP via `cma.fmin2` restarts, `qLogExpectedImprovement`, isotropic kernel setup) against the installed packages and docs, not from memory.
- **Commits:** small, under `Melisa <melisathevandi@gmail.com>`, with no co-author line (your standing instruction).

## Verification
- `uv run pytest`: problem tests plus the invariance tests pass (CMA-ES exactly invariant to monotone transforms; invariant to rotation when unbounded).
- `run_experiment.py --quick` finishes in under 10 minutes and writes JSONs that resume correctly when rerun.
- Timing pilot output is saved in `results/summary/pilot_timing.csv` and used to fix the budgets in the configs.
- Unit tests for the diagnostics:
  - ρ = d_eff/d exactly for an isotropic M, and ρ = 1 for an M supported only on the relevant aligned coordinates.
  - κ_t = 1 when C = H⁻¹.
  - The decoy function: f_decoy is attained in the broad basin, and the narrow basin's minimum is below it.
- κ=1 null check: in E2, Δ_rot at κ=1 has a CI containing 0 for every method. If not, stop and investigate before reading any other result.
