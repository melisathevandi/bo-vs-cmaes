# Critical review: DSA4212 BO vs CMA-ES plan

## Context
You handed over a plan for the DSA4212 final assignment, "BO vs evolutionary strategies", and asked for a critical review before any code is written. This file is that review. Nothing gets implemented until you approve.

### Environment check (read-only, run 2026-10-29)
| Item | Finding | Implication |
|---|---|---|
| CPU | AMD Ryzen 9 5900HS, 8 cores / 16 threads | Use about 6–7 worker processes, each with one BLAS/torch thread. |
| RAM | 15.4 GB | Each torch process uses about 0.3–0.5 GB, so 7 workers is safe. |
| OS | Windows 11 Home | Multiprocessing uses *spawn*: you need a `__main__` guard, and thread env vars must be set before torch is imported. Keep the laptop plugged in with sleep off for overnight runs. |
| Python | 3.13.1 (system, `C:\Python313`) | Decided: uv-managed Python 3.12 project environment. The system Python is left untouched. |
| Packages | numpy, scipy, torch, botorch, gpytorch, cma, matplotlib, pandas: **none installed** | Fresh `requirements.txt` with pinned versions. |
| uv | 0.13.6 installed (winget) | Ready to use. |
| git | Installed. The folder is **not** a git repo yet. | `git init` on day 1. The commit timestamp on `predictions.md` is your evidence that you predicted before you ran. |
| Location | `H:\Backup Laptop Mel\...` | If H: is an external or backup drive, it will be slower and riskier. Run from a local SSD and push to GitHub. (Question below.) |
| Docs | `assignment_guidelines.md` and `topic_bo_vs_es.md` sit at the root, not in `docs/` | Move them to `docs/`. |

---

## a) Problem and rubric in my words
The task is to explain *when and why* surrogate-based BO beats or loses to ranking-based CMA-ES on black-box problems. The explanation has to be stated up front and then tested with controlled, repeated, fairly budgeted experiments. Marks: 25% understanding (correct mechanisms, sources, justified choices), 35% experimental design (controlled comparisons and **follow-up on surprises**), 20% analysis (evidence-based claims and honest limits), 20% reproducibility (a repo that regenerates every reported number). Depth and mechanism are worth more than coverage. A well-investigated negative result is explicitly acceptable.

---

## b) Verdict on each part of section 3

### 3.1 Central claim: **agree, but sharpen it**
"It depends on the inductive bias" is true but hard to falsify. I would make it mechanistic and quantitative: *BO's advantage comes from a surrogate whose assumptions (smoothness, axis-aligned relevance through ARD) pay off from the first few evaluations. CMA-ES's geometric advantage (learning the covariance) only appears after roughly O(d²) evaluations. So the winner is set by the budget-to-dimension ratio and by whether the structure is axis-aligned.* The second sentence is the real insight. With default pycma at d=10, the rank-µ learning rate is about 2/d², so covariance learning needs hundreds to thousands of evaluations. **Within the budgets BO can afford, CMA-ES has mostly not learned the geometry yet.** That changes how H2 should be read (see below).

### Hypotheses
- **H1 (agree, improve).** This is the known result. Two changes make it testable:
  - Express budget as **evaluations / d**, COCO-style, so dimensions are comparable.
  - Commit to a direction *and a magnitude*, e.g. "the budget/d at which CMA-ES's median log-gap matches BO's falls as d grows".

  One caveat to check against the docs: BoTorch ≥0.12 changed the default lengthscale priors to dimension-scaled ones (Hvarfner et al., ICML 2024, "Vanilla BO performs great in high dimensions"). Modern vanilla BO may degrade with d much less than older papers suggest. H1 could partly fail, and that would be interesting.
- **H2 (agree on BO's side, misframed on CMA-ES's side).** The CMA-ES half, "CMA-ES does not degrade under rotation", is true by construction: an isotropic start is rotation-invariant. That makes it a sanity check, not a finding. The informative comparison is a **2×2 axis-bias ablation in both families**:

  |  | Exploits axis alignment | Rotation-invariant |
  |---|---|---|
  | BO | ARD kernel | isotropic kernel |
  | ES | sep-CMA-ES (diagonal covariance; pycma option) | full CMA-ES |

  Prediction: the members that exploit axis alignment win when the structure is aligned and lose that advantage when it is rotated. The rotation-invariant members are unaffected. That gives a symmetric, mechanistic test instead of "BO bad, CMA good".
- **H3 (keep as exploratory, but make it falsifiable).** Turn it into a curve: P(find narrow basin) as a function of **basin width** (one knob), at a fixed budget, for each method. Commit to the sign, e.g. "a GP with fitted lengthscales misses basins narrower than its lengthscale; IPOP restarts recover them at a higher budget".
- **H4 (as written it is close to a tautology).** "BO wins on wall-clock above the break-even cost" is true by definition. What you can predict is the break-even cost c*, its **magnitude**, and how it **scales** with d, budget and target accuracy. GP overhead grows with n, so c* rises with budget. Also say clearly that evaluation cost is *simulated*: your test functions take microseconds, so wall-clock = measured optimiser overhead + n·c.

### 3.3 Methods: **improve**
- Use **LogEI** (`qLogExpectedImprovement`, Ament et al., NeurIPS 2023) instead of plain EI. BoTorch now recommends it because EI has vanishing-gradient problems. Plain EI would be a reviewer's easy criticism.
- Add **sep-CMA-ES**. It costs nothing and completes the 2×2.
- **Promote TuRBO-1 from stretch to core**, at least in E1 and E2. It is in the brief's own reading list, and the marker expects it. It is the natural "local BO" bridge between the two families. Its trust region is also axis-aligned and scaled by ARD lengthscales, so it *also* belongs in the H2 analysis. BoTorch has an official TuRBO-1 tutorial implementation we can adapt and cite.
- Drop the hybrid: **agree**.
- Keep the Sobol baseline: **agree**.

### 3.4 Fairness protocol: **mostly agree. Replace full tuning with defaults plus a sensitivity analysis.**
- A held-out tuning protocol for 4–5 methods is a lot of extra compute and writing. It mostly re-derives library defaults that are already tuned. Cheaper and more informative: use **documented defaults** (CMA-ES σ₀ = 0.3 × range per Hansen's tutorial; BO n_init = 2d Sobol points, counted in the budget), then run a **sensitivity check** on one pilot problem (σ₀ ×{0.5, 1, 2}; n_init ∈ {d, 2d, 4d}). The rubric rewards "sensitivity to assumptions" directly, and this addresses it.
- Initial information: CMA-ES x₀ is a seeded uniform point in the box, *not* the centre. Optima are placed randomly inside the box, away from its centre, to avoid centre bias. All methods work in the same normalised box [0,1]^d.
- Paired instances across methods: **agree**.

### 3.5 Metrics and statistics: **agree, a few additions**
- Primary metric: **log10 gap** of the true objective at the recommended point. Read the whole trajectory from one long run: **agree**.
- Summaries: median and IQR, plus a **paired win rate** P(BO better than CMA on the same instance) with a bootstrap CI. Win rate is easy to read and handles heavy tails.
- Report effect sizes with CIs rather than many p-values. There are many comparisons.
- A COCO-style ECDF of runtime-to-target: use it for **one** aggregate figure only.
- Timing: separate method overhead (fit and acquisition, or ask/tell) from function evaluation time. **Agree.**

### 3.6 Experiments: **too many for 8 pages. Restructure.**
- **E2 (centrepiece, agree)**, plus two cheap but strong additions:
  - **Exact invariance tests.** For the *same seed*, CMA-ES on f and on g∘f (g strictly increasing) should give *identical* trajectories. Without bounds, CMA-ES on a rotated problem with a rotated x₀ should give the rotated trajectory, path by path. Then turn on box bounds and *measure* how far invariance breaks. Do the same for BO: it is not invariant to monotone transforms, so compare f vs log f on the ellipsoid. This is exactly the brief's "which invariances survive practical implementation" question, it costs minutes of compute, and it makes a clean table.
  - **A design fix for the low-effective-dimension function.** Use f(x) = g(A(x − x*)) with orthonormal A (2×20). The confound to watch: a random rotation changes the shape of the projected box, so the reachable region of g differs between the aligned and rotated cases. Keep x* inside the box with margin, and choose g, e.g. a quadratic or a shifted Branin-like function, so that its optimum is reachable in both cases. Check numerically that the f-value distribution of uniform samples is similar across the two cases.
- **E1 (agree, reshape)**: d ∈ {2, 5, 10, 20}, budget cap set by a **timing pilot**. Run CMA-ES (which is cheap) far beyond BO's cap, e.g. to 50·d², and show where it overtakes. The regime map's axes are d × evaluations/d.
- **E3 (shrink)**: one hand-made function with a broad decoy basin and a narrow deep basin, a basin-width knob, and d ∈ {2, 5}. CMA-ES default / IPOP restarts (pycma `cma.fmin` with `restarts` and `incpopsize`); BO LogEI vs UCB. Use Wilson CIs: **agree**. Rastrigin is optional; the knob gives a curve, which is more useful than another benchmark.
- **E4 (agree as analysis-only)**: half a figure built from logged timings. Model parallelism analytically and label it as a model. Skip batch BO.
- **E5 noise: drop it from the plan now**, rather than cutting it later. Mention it in limitations.
- **BBOB/COCO**: don't use the `cocoex` package (it is harder to control budgets and instances). Implement BBOB-*style* functions (ellipsoid f10, rotated Rosenbrock f9) yourself, with unit tests, and cite COCO for the definitions.

### 3.7 Mechanism diagnostics: **agree. This is where the marks are. Make them concrete.**
- **BO, relevance alignment.** With the ARD metric M = diag(1/ℓ²) and true projector P = AᵀA, compute **tr(PM)/tr(M)**. Expect about 1 when aligned and at most about 2/20 = 0.1 when rotated: a diagonal metric *cannot* concentrate on an off-axis subspace. Raw lengthscale plots don't show this as directly.
- **BO, surrogate quality.** Spearman rank correlation of the GP posterior mean on fresh points near the incumbent, as n grows, aligned vs rotated and across d. This measures directly whether the surrogate is useful.
- **CMA-ES, geometry learned.** On quadratics, track cond(C^{1/2} H C^{1/2}) over evaluations. It should fall towards 1, and the plot shows *when* CMA-ES learns the geometry compared with BO's budget. This one figure could carry the whole story.
- σ trajectory: optional.

### 3.8 Report plan: **agree.** Planned figures: (1) geometry 2×2 plus the relevance diagnostic, (2) regime map, (3) CMA-ES learning-time curve, (4) multimodality curve, plus half a figure for cost. Prediction-vs-outcome table and invariance table. Write the **numbers into LaTeX automatically** (scripts emit `report/generated/numbers.tex` with `\newcommand` macros and table `.tex` files). Then no number in the PDF is typed by hand, which is the strongest defence if the report is checked against the repo by AI.

### 3.9 Reproducibility: **agree.** Additions: `OMP_NUM_THREADS=1` and `torch.set_num_threads(1)` per worker; one JSON result file per (config hash, method, seed) so runs resume; record package versions and git commit in each result; a quick mode under 10 minutes; a final check from a clean clone.

### 3.11 Cut order: **agree with modifications.** New order: UCB in E3, then d=20 in E1, then TuRBO in E1 (keep it in E2), then reduce seeds (never below 15). Never cut E2, the diagnostics, or the invariance tests.

### CPU feasibility (estimate; the **pilot must measure this**)
BoTorch on CPU at n ≤ 300 and d ≤ 20 takes roughly 0.5–3 s per iteration (GP fit plus optimising the acquisition function with about 10 restarts and 512 raw samples). That is roughly 3–10 min per 300-evaluation run. A rough count with 7 workers:
- E2: about 8 problems × 3 BO-type methods × 20 seeds ≈ 480 runs at budget about 200 → **about 4–6 h**
- E1: about 8 settings × 2 BO-type methods × 20 seeds ≈ 320 runs → **about 4–6 h**
- E3: small d, budget about 100 → **about 1 h**
- Sensitivity checks plus invariance tests → **about 1 h**
- CMA-ES/Sobol, all runs → **minutes**

That is **about 10–15 h in total**: two or three overnight runs. It is feasible, with headroom for one round of follow-up. If the pilot shows over 3 s per iteration at d=20, cap d=20 at 200 evaluations or reduce `raw_samples`.

### Biggest risks against the rubric
1. **Breadth over depth** (35% exploration). Fix: fewer experiments, more diagnostics, and a planned slot for following up surprises.
2. **A known result with no mechanism** ("BO wins early"). Fix: the CMA learning-time curve and the relevance diagnostic.
3. **Straw-man BO or CMA-ES.** Fix: LogEI, TuRBO, defaults from the libraries' own guidance, sensitivity checks.
4. **Mismatch between report and repo** (20%). Fix: auto-generated numbers and tables, and a clean-clone check.
5. **Integrity, "genuine engagement".** Fix: *you* write the report prose. Keep a dated `notes/lab_notebook.md` of decisions and surprises, commit `predictions.md` before the main runs, and read the four starter references yourself. I will explain each design choice as we go.
6. **Compute surprises on Windows.** Fix: the timing pilot on day 2 decides the budgets.

---

## c) Recommended final design (★ = my change)
1. **Methods:** Sobol; CMA-ES (pycma defaults, σ₀ = 0.3); ★sep-CMA-ES; ★IPOP-CMA-ES (E3); BO = SingleTaskGP, Matérn-5/2 ARD, ★LogEI, BoTorch default priors, normalise/standardise; BO-isotropic (E2); BO-UCB (E3); ★TuRBO-1 (E1, E2).
2. **Protocol:** shared normalised box, Sobol initial design of size 2d counted in the budget, random x₀ and optima, paired seeds (20; 30 for E3), ★defaults plus a sensitivity check instead of held-out tuning.
3. **Experiments:** ★E0 invariance tests (exact, cheap) → E2 geometry with the ★2×2 axis-bias ablation and the ★corrected low-effective-dimension design → E1 regime map with ★budget/d axes and ★long CMA-ES runs → E3 ★basin-width curve → E4 cost analysis (★H4 reframed as the magnitude and scaling of c*). ★E5 removed.
4. **Diagnostics:** ★tr(PM)/tr(M), ★GP rank correlation, ★cond(C^{1/2}HC^{1/2}) over time.
5. **Statistics:** log10 gap, median/IQR, ★paired win rate with bootstrap CI, Wilson CIs, one ECDF.
6. **Report:** ≤8 pages, ★auto-generated `numbers.tex` and tables, prediction-vs-outcome table.

### Suggested timeline (Oct 29 → about Nov 13)
- **Oct 29–30:** venv, git, test functions plus unit tests, method wrappers, invariance tests.
- **Oct 31:** timing pilot and sensitivity checks; set budgets; **commit `predictions.md`**.
- **Nov 1–5:** E2 and E1 overnight runs; diagnostics.
- **Nov 6–7:** E3 and E4.
- **Nov 8:** reserved for follow-up on surprises.
- **Nov 9–12:** figures and report (you write).
- **Nov 13:** clean-clone reproduction check and submission.

---

## d) Decisions (answered 2026-10-29)
1. **Solo:** Melisa Abigail Thevandi, A0288030J. Because you are working alone, scope stays tight: TuRBO is core in **E2**, and it goes into E1 only if the pilot shows spare compute. UCB in E3 is first on the cut list.
2. **Deadline:** 13 Nov 2026, 23:59. The timeline above stands, with Nov 13 kept as a buffer only.
3. **Location:** keep working in the H: external SSD folder. Push to your **private** GitHub repo and **invite `alexxthiery`** as a collaborator before submitting.
4. **Accepted:** drop E5, defaults plus a sensitivity check, TuRBO core, sep-CMA-ES, invariance tests.
5. **Overleaf:** scripts write `report/figures/` and `report/generated/*.tex`, and you upload those to Overleaf. At the end, download the Overleaf source zip into `report/` so the repo holds the exact submitted source. The clean-clone check then compares `generated/numbers.tex` against it.
6. **Environment:** **uv** with **Python 3.12**, pinned via `.python-version`. 3.12 has the safest torch/botorch wheels on Windows, and uv installs it without touching your system 3.13. Use `pyproject.toml` plus `uv.lock`, and the readme says to run `uv sync` then `uv run ...`. `requirements.txt` is replaced by `uv.lock`.
7. **References not read yet.** Suggested reading, done alongside days 1–3: Hansen's CMA-ES tutorial (sections on invariance, learning rates and step-size control: the basis of the "O(d²) learning time" argument); Frazier's BO tutorial (GP, EI, noise); the TuRBO paper (motivation and trust-region rule); COCO (budgets/d, ECDFs, instances). I will point to the relevant section whenever a design choice depends on it.

---

## e) Repo skeleton (proposed)
```
bo-vs-cmaes/
├── readme.md                 # setup, quick/full reproduction, figure ↔ script map
├── authors.md
├── pyproject.toml            # uv project, deps
├── uv.lock                   # pinned versions
├── .python-version           # 3.12
├── predictions.md            # committed before main runs
├── docs/
│   ├── assignment_guidelines.md
│   └── topic_bo_vs_es.md
├── notes/
│   └── lab_notebook.md       # dated decisions, surprises, failed attempts
├── configs/
│   ├── pilot.yaml
│   ├── e0_invariance.yaml
│   ├── e1_dimension.yaml
│   ├── e2_geometry.yaml
│   └── e3_multimodal.yaml
├── code/
│   ├── problems.py           # ellipsoid, rosenbrock, low-eff-dim, decoy-basin; rotations
│   ├── methods/
│   │   ├── sobol.py
│   │   ├── cmaes.py          # full / sep / IPOP via pycma
│   │   ├── bo.py             # GP + LogEI/UCB, ARD/iso
│   │   └── turbo.py
│   ├── runner.py             # one (config, method, seed) → results JSON, timing split
│   ├── run_experiment.py     # parallel pool, resume by hash, --quick
│   ├── diagnostics.py        # relevance ratio, GP rank corr, CMA cond(C^½HC^½)
│   ├── analysis.py           # stats, bootstrap, Wilson
│   ├── make_figures.py       # every figure + numbers.tex + tables
│   └── tests/
│       ├── test_problems.py  # f(x*)=f*, rotation correctness
│       └── test_invariance.py
├── results/
│   ├── raw/<experiment>/<hash>_<method>_<seed>.json
│   └── summary/              # aggregated CSVs
└── report/
    ├── report.tex
    ├── references.bib
    ├── figures/              # generated
    └── generated/            # numbers.tex, tables
```

## Verification (once implementation starts)
- Unit tests: every test function attains f* at x*, and rotations are orthogonal.
- Exact invariance tests pass for CMA-ES (identical trajectories) before any main run.
- Quick mode finishes in under 10 minutes and regenerates placeholder figures.
- Final check: from a clean clone, `run_experiment --quick` then `make_figures` rebuild `numbers.tex`. Diff it against the submitted version.

**References to verify yourself before citing** (I am fairly confident but have not checked them in this session): Frazier 2018 BO tutorial (arXiv:1807.02811); Hansen 2016 CMA-ES tutorial (arXiv:1604.00772); Eriksson et al. 2019 TuRBO (NeurIPS); Hansen et al. 2021 COCO (Optim. Methods Softw.); Ament et al. 2023 LogEI (NeurIPS); Hvarfner et al. 2024 (ICML); Auger & Hansen 2005 IPOP-CMA-ES (CEC); Ros & Hansen 2008 sep-CMA-ES (PPSN).
