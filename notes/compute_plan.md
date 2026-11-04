# Compute plan (planning estimate, not a result)

BO cost = timing-pilot integral x FACTOR (2.28, measured/integral at d=10, N=500, 3 real runs) x LOAD7 (1.25, measured: A4 check on 7 workers, 47.0 min per refit-every-5 run vs 37.7 min in the calibration run). BO refits cold every 10 iterations after n = 100. TuRBO cost = measured pilot median x LOAD7. Lean core first; optional rows run only if the core is ahead of schedule on 6 Nov.

| Key | Experiment | Runs | CPU-h | Wall-h (7 workers) | Expected finish | Group |
|---|---|---|---|---|---|---|
| sens_bo | Sensitivity grids: BO: 6 cfgs x 2 problems x 10 seeds (CMA-ES grid takes seconds) | 120 | - | - | done | core |
| e2_primary | E2 primary (N=500): k=1e4: aligned, rotated, aligned replicate x 30 seeds x {ARD, iso, TuRBO} (+ CMA, sep-CMA, Sobol) | 270 | - | - | done | core |
| e2_lowdim | E2 low-eff-dim (d=20, d_eff=2, N=500): aligned+rotated x 20 seeds x {ARD (+1.5 min diagnostics), iso, TuRBO} (+ CMA, Sobol) | 120 | 55.5 | 7.9 | 05 Nov 10:05 | core |
| h5 | H5 CMA-ES runs: 11 dims x 20 seeds, cap 100 d^2 | - | 0.5 | 0.1 | 05 Nov 10:10 | core |
| e1_d2 | E1 d=2, N=200: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds (+ CMA, Sobol) | 40 | 2.4 | 0.3 | 05 Nov 10:33 | core |
| e1_d3 | E1 d=3, N=200: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds (+ CMA, Sobol) | 40 | 2.4 | 0.3 | 05 Nov 10:57 | core |
| e1_d5 | E1 d=5, N=400: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds (+ CMA, Sobol) | 40 | 7.1 | 1.0 | 05 Nov 12:07 | core |
| e1_d8 | E1 d=8, N=640: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds (+ CMA, Sobol) | 40 | 33.3 | 4.8 | 05 Nov 17:35 | core |
| e1_d10 | E1 d=10, N=1000: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds (+ CMA, Sobol) | 40 | 84.1 | 12.0 | 06 Nov 07:24 | core |
| followup | Follow-up (reserved): one experiment chosen after E1 (about 20 wall-h) | - | 140.0 | 20.0 | 07 Nov 06:24 | core |
| e3_d2 | E3 d=2 (N=200): 5 widths x 30 seeds x {LogEI, UCB} | 300 | 18.0 | 2.6 | 07 Nov 09:21 (past the 6 Nov gate) | optional |
| e3_d5 | E3 d=5 (N=400): 5 widths x 30 seeds x {LogEI, UCB} | 300 | 52.9 | 7.6 | 07 Nov 18:03 (past the 6 Nov gate) | optional |
| e2_other_kappa | E2 other kappa (N=500): k=1 aligned; k=1e2 aligned+rotated x 20 seeds x {ARD, iso, TuRBO} | 180 | 85.7 | 12.2 | 08 Nov 08:08 (past the 6 Nov gate) | optional |
| e1_d20 | E1 d=20, N=1000: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 108.3 | 15.5 | 09 Nov 01:55 **after the hard stop** | optional |
| e2_secondary | E2 secondary N=1000: k=1e4 aligned+rotated x 10 seeds x {ARD, iso, TuRBO} | 60 | 96.1 | 13.7 | 09 Nov 17:42 **after the hard stop** | extra |
| a8c | A8c noise (E5): d=5 N=300, 2 noise levels x 20 seeds, BO ARD | 40 | 4.5 | 0.6 | 09 Nov 18:26 **after the hard stop** | extra |
| a8d | A8d long-budget probe: d=10 k=1e4 rotated, N=2000, 10 seeds (extrapolated beyond n=1000) | 10 | 178.7 | 25.5 | 10 Nov 23:48 **after the hard stop** | extra |
| a8e | A8e second family: 4 BBOB-style functions, d=10, N=500, 10 seeds | 40 | 22.2 | 3.2 | 11 Nov 03:27 **after the hard stop** | extra |

- Remaining lean core: **46 wall-h** (53 with 15% contingency; range 37 to 86)
- Optional (E3, E2 other kappa, E1 d=20): 38 wall-h
- Extras (E2 secondary, A8 c-e): 43 wall-h
- Expected finish times are cumulative from 05 Nov 00:58 (end of E2 primary) over the remaining rows, with 15% contingency, in this order; hard stop 08 Nov 23:59.
