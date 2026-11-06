# Predictions (written before the main runs)

> Committed on: 2026-11-01. This file was fixed before the E1, E2 and E3 runs; its commit timestamp is the evidence. Numbers marked (guess) are first guesses, not derived.

## What was already run before this file (disclosure)
- E0 (CMA-ES only, d=10, 1000 evaluations, `results/summary/e0_rotation.csv`): with the box
  [0,1]^d, rotation costs CMA-ES about +0.4 to +0.5 decades (log10 gap); without a box the
  rotation effect is consistent with 0. Predictions about CMA-ES's rotation sensitivity
  below are therefore informed by E0, and are labelled as such.
- A timing pilot and a correctness anchor. No BO, TuRBO or sep-CMA-ES comparison has been run.

## Conventions
Gap = f(best point so far) - f*, on the true objective. "Better" = smaller log10 gap.
All differences are paired by seed (same instance), with 95% bootstrap CIs.

## H1: BO's advantage shrinks as dimension grows relative to the budget
At the same evaluations per dimension, the median paired difference
(log10 gap of ARD-BO minus log10 gap of CMA-ES, negative = BO better) is more negative at
d=2,3,5 than at d=8,10, on both the ellipsoid (kappa=1e4) and Rosenbrock.
(guess) The BO lead at 20 evaluations per dimension is at least 0.5 decades smaller at d=10
than at d=2. At d=20 with 1000 evaluations (50 per dimension) BO is no longer ahead of CMA-ES.
*Would refute H1:* the BO lead is the same or larger at d=10 than at d=2.

## H2: axis-aligned methods gain on aligned problems and lose that gain when rotated
Primary endpoint, fixed now: d=10, kappa=1e4, 500 evaluations, 30 seeds. 1000 evaluations is reported as a secondary endpoint, and all other kappa values are secondary.
Rotation penalty Delta_rot = gap(rotated) - gap(aligned), paired by seed.
- Delta_rot(ARD-BO) - Delta_rot(isotropic BO) > 0, CI excluding 0.
- Delta_rot(sep-CMA-ES) - Delta_rot(full CMA-ES) > 0, CI excluding 0.
- Full CMA-ES: Delta_rot > 0 with the box (informed by E0), and consistent with 0 without it.
- Isotropic BO: approximately invariant; the deviation from 0 is reported, with the aligned
  replicate (independent aligned instance draw) as the noise floor.
- Low-effective-dimension function (d=20, 2 relevant directions): the relevance ratio
  rho = tr(PM)/tr(M) of the fitted ARD kernel is checked at n=200: rho is at least 0.5 when aligned and at most 0.2 when rotated (guess; the isotropic baseline is d_eff/d = 0.1) (a diagonal metric cannot align with an off-axis subspace).
*Would refute H2:* either interaction has a CI containing 0, or rho (aligned) < 0.5, or rho (rotated) > 0.2.

## H3: multimodality (exploratory)
Decoy-basin function: P(success) falls as the narrow basin gets narrower, for every method.
Success = best value below the decoy basin's minimum f_decoy.
- BO fails on narrow basins when the basin is smaller than the spacing of the initial design
  (2d points); UCB (more exploration) beats LogEI on the narrowest basins.
  (UCB is run in E3 because E3 is at d <= 5, where BO runs take minutes. The cost-based cuts in
  the plan apply to the expensive d=10 and d=20 experiments.)
- CMA-ES with IPOP restarts beats default CMA-ES at larger budgets; default CMA-ES is
  initialisation-dependent and rarely leaves the decoy basin.
*Would refute H3:* P(success) does not depend on basin width, or LogEI beats UCB on the narrowest basin.

## H4: break-even evaluation cost
For a target gap g, BO needs n_BO evaluations and CMA-ES n_CMA > n_BO. With measured method
overheads o_BO, o_CMA (seconds per evaluation, from the logs), BO is faster in wall-clock time
iff c > c* = (n_BO o_BO - n_CMA o_CMA) / (n_CMA - n_BO), where c is the cost of one evaluation.
Prediction (qualitative): c* increases with the budget and with d, because GP fitting cost
grows with n. c* is computed from the measured overheads, not predicted as a number.
*Would refute H4:* c* decreases with the budget.

## H5: CMA-ES learning time scales like d^alpha, with alpha below 2 at finite d
- **The idea.** Each generation closes a fraction c1 + c_mu of the remaining gap between the
  covariance matrix and its target, so learning takes about 1 / (c1 + c_mu) generations. Each
  generation costs lambda evaluations, so evaluations to learn ~ lambda / (c1 + c_mu).
- **Large d.** c1 ~ 2 / d^2 and c_mu ~ 2 (mu_eff - 2 + 1/mu_eff) / d^2, so
  c1 + c_mu ~ (2 mu_eff - 2 + 2/mu_eff) / d^2. The factor lambda / (2 mu_eff - 2 + 2/mu_eff) is
  almost constant in d (log-log slope about -0.005 over d = 2 to 80), so evaluations ~ d^2.
- **Finite d.** The exact rates have (d + 1.3)^2 and (d + 2)^2 in their denominators. The local
  log-log slope of (d + a)^2 is 2d / (d + a), which is below 2, so the exponent fitted over a
  finite range of d comes out below 2.
- **Numbers.** From `code/h5_expected_alpha.py` with pycma 4.5.0, fitted over
  d in {2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 80}: alpha = 1.71 (all d), 1.87 (d >= 10). The
  tutorial formulas give 1.69 and 1.86. pycma's c_mu is 7-48% larger (48% at d=2, 17% at d=10, 7% at d=80) than the tutorial value;
  the pycma defaults are what was measured.
- **Primary prediction.** The bootstrap 95% CI for alpha_hat (d >= 10) contains 1.87, and
  alpha_hat (all d) is lower, about 1.7. Also reported (secondary): whether the CI contains 2.
- **Would refute H5:** the CI for d >= 10 excludes 1.87. We would then report that the learning
  rates alone do not explain the scaling (the evolution path and step-size adaptation also affect
  learning), and say whether alpha is still compatible with 2.

## Follow-up experiment: what drives TuRBO's rotation penalty (committed 2026-11-06, before the runs)
**What was already seen.** The E2 primary analysis (d=10, kappa=1e4, 500 evaluations, 30 seeds) gave the rotation penalty
Delta_rot (median paired log10 gap, rotated minus aligned): TuRBO 1.56, ARD-BO 0.85, isotropic BO -0.09. This follow-up is
chosen because TuRBO's penalty is the largest; the numbers above are known, so the predictions below are informed by them.

**Design.** Same E2 primary instances (aligned and rotated ellipsoid, d=10, kappa=1e4, instance seeds 0-29), 500 evaluations,
seeds 0-29, same TuRBO tolerances and refit schedule. Three variants, a 2 x 2 design with one cell empty by construction:
1. *default* TuRBO (ARD kernel, trust region box with sides scaled by the ARD lengthscales): the E2 runs, not repeated;
2. *ARD kernel, cubic region*: the kernel is ARD, but the trust-region sides are all equal (weights = 1);
3. *isotropic kernel, cubic region*: one shared lengthscale (the region is then a cube anyway, so "isotropic kernel with
   ARD-scaled box" does not exist).
New runs: 2 variants x 2 problems x 30 seeds = 120. Default behaviour of the code is unchanged (checked before the runs).

**Statistics (paired by seed, 95% percentile bootstrap, 10,000 resamples, seed 0).** For each variant, Delta_rot as defined in
H2, and the rotated and aligned median gaps separately (a smaller Delta_rot could come from a worse aligned result and
must be visible). Contrasts: Delta_rot(cubic ARD) - Delta_rot(default) and Delta_rot(isotropic) - Delta_rot(cubic ARD),
resampling seeds independently across the problem pair of each variant (different runs, same instances).

**Predictions.**
- *Cubic ARD:* the rotation penalty is close to ARD-BO's, 0.85: Delta_rot in [0.45, 1.25] (ARD-BO value +/- 0.4), with the CI
  excluding 0, and Delta_rot(cubic ARD) - Delta_rot(default) < 0 with a CI excluding 0 (the ARD-scaled region adds penalty).
- *Isotropic:* the penalty is near 0: |Delta_rot| <= 0.3 with the CI containing 0.
*Would refute (cubic ARD):* Delta_rot outside [0.45, 1.25], or the contrast with the default CI containing 0 (then the region
shape does not explain TuRBO's excess penalty over ARD-BO).
*Would refute (isotropic):* |Delta_rot| > 0.3 or a CI excluding 0.
Both outcomes are reported whichever way they go. No aligned-replicate noise floor is run for the variants; the E2 replicate
for default TuRBO is the only noise reference and this is stated as a limitation.

## Amendments (dated; the text above is unchanged)

### 2026-11-06: H1 is tested at kappa = 100, and its d = 20 part was not run
- **kappa.** H1 above says kappa = 1e4 for the ellipsoid. The E1 configs (`configs/e1_d{2,3,5,8,10}.toml`, committed
  2026-11-04 and run 2026-11-05/06) use the rotated ellipsoid with **kappa = 100**, and Rosenbrock. H1 is therefore tested
  at kappa = 100 on the ellipsoid, not at kappa = 1e4. The 0.5-decade guess is unchanged and is applied to the kappa = 100
  results. No E1 run at kappa = 1e4 exists.
- **d = 20.** The sentence "At d=20 with 1000 evaluations (50 per dimension) BO is no longer ahead of CMA-ES" was **not
  tested**: E1 d = 20 was not run (decision of 2026-11-06). The tested range is d = 2, 3, 5, 8, 10.
- **When.** Written and committed before any E1 gap value or E1 result was analysed (see `analysis_addendum_e1.md`). The
  statistic for the 0.5-decade guess is fixed there.

### 2026-11-06 (second amendment, before any ablation run): aligned-replicate noise floor for both variants
The last sentence of the follow-up section above ("No aligned-replicate noise floor is run for the variants") is
**superseded**. Both variants are also run on the aligned replicate used in E2 (`ell_d10_k1e4_aligned_rep`, instance seed
offset 5000, seeds 0-29), 500 evaluations: 30 more runs per variant. The follow-up is then 3 problems x 30 seeds x 2 variants =
180 new runs. For each variant, the noise floor is the paired difference (aligned replicate minus aligned) in median log10
gap with its bootstrap CI, computed as for default TuRBO in the E2 analysis. Reading rule, fixed now: a variant's Delta_rot is
interpreted against its own noise floor. If the floor's CI excludes 0 for a variant, that is reported next to its Delta_rot and
the "near 0" criterion for the isotropic variant is judged on Delta_rot minus the floor as well as on Delta_rot. The thresholds
and the predictions above are unchanged.

