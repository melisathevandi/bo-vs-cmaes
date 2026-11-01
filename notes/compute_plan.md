# Compute plan (planning estimate, not a result)

BO cost = timing-pilot integral x FACTOR (2.28, measured/integral at d=10, N=500, 3 real runs, folding in real-trajectory fit cost and the load of the ~6 busy processes during that run) x LOAD7 (1.25, measured: A4 check at d=10 on 7 workers, 47.0 min per every-5 run vs 37.7 min in the calibration run). TuRBO cost = measured pilot median x LOAD7. The A4-check row is the same in both tables.

### BO refit every 5 iterations (n > 100)

| # | Experiment | Runs | CPU-h | Wall-h (7 workers) | Group |
|---|---|---|---|---|---|
| 1 | A4 check d=10, N=500: 15 seeds x {cold every 5, cold every 10}, rotated k=1e4 | 30 | 19.9 | 2.8 | core |
| 2 | Sensitivity grids: BO: 6 cfgs x 2 problems x 10 seeds (CMA-ES grid negligible) | 120 | 59.0 | 8.4 | core |
| 3 | E2 primary (N=500): k=1e4: aligned, rotated, replicate x 30 seeds x {ARD, iso, TuRBO} | 270 | 167.2 | 23.9 | core |
| 4 | E2 low-eff-dim (d=20, N=500): aligned+rotated x 20 seeds x {ARD, iso, TuRBO}; ARD +1.5 min diagnostics | 120 | 73.6 | 10.5 | core |
| 5 | H5 CMA-ES runs: 11 dims x 20 seeds, cap 100 d^2 (alongside) | - | 0.5 | 0.1 | core |
| 6 | E1 d=2, N=200: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 2.6 | 0.4 | core |
| 7 | E1 d=3, N=200: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 2.6 | 0.4 | core |
| 8 | E1 d=5, N=400: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 8.5 | 1.2 | core |
| 9 | E1 d=8, N=640: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 48.2 | 6.9 | core |
| 10 | E1 d=10, N=1000: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 132.9 | 19.0 | core |
| 11 | E3 d=2 (N=200): 5 widths x 30 seeds x {LogEI, UCB} | 300 | 19.7 | 2.8 | core |
| 12 | E3 d=5 (N=400): 5 widths x 30 seeds x {LogEI, UCB} | 300 | 63.8 | 9.1 | core |
| 13 | E2 other kappa (N=500): k=1 aligned; k=1e2 aligned+rotated x 20 seeds x {ARD, iso, TuRBO} | 180 | 111.5 | 15.9 | core |
| 14 | Follow-up (reserved): one experiment chosen after E1-E3 (given: about 20 wall-h) | - | 140.0 | 20.0 | core |
| 15 | E1 d=20, N=1000: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds (last core) | 40 | 188.8 | 27.0 | core_d20 |
| 16 | E2 secondary N=1000: k=1e4 aligned+rotated x 10 seeds x {ARD, iso, TuRBO} | 60 | 144.9 | 20.7 | extra |
| 17 | A8c noise (E5): d=5 N=300, 2 noise levels x 20 seeds, BO ARD | 40 | 5.2 | 0.7 | extra |
| 18 | A8d long-budget probe: d=10 k=1e4 rotated, N=2000, 10 seeds (extrapolated beyond n=1000) | 10 | 316.2 | 45.2 | extra |
| 19 | A8e second family: 4 BBOB-style functions, d=10, N=500, 10 seeds | 40 | 30.8 | 4.4 | extra |

- Core without E1 d=20: **121 wall-h** (140 with 15% contingency; range 98 to 223)
- Core with E1 d=20: **148 wall-h** (171 with 15% contingency; range 119 to 273)
- Extras (E2 secondary, A8 c-e), only if the core finishes early: 71 wall-h

Cumulative finish times of the core phases in the agreed order (central estimate with 15% contingency, starting 02 Nov 07:29; hard stop 08 Nov 23:59):

- A4 check d=10, N=500: 02 Nov 10:45
- Sensitivity grids: 02 Nov 20:26
- E2 primary (N=500): 03 Nov 23:54
- E2 low-eff-dim (d=20, N=500): 04 Nov 12:00
- H5 CMA-ES runs: 04 Nov 12:05
- E1 d=2, N=200: 04 Nov 12:31
- E1 d=3, N=200: 04 Nov 12:57
- E1 d=5, N=400: 04 Nov 14:20
- E1 d=8, N=640: 04 Nov 22:15
- E1 d=10, N=1000: 05 Nov 20:05
- E3 d=2 (N=200): 05 Nov 23:19
- E3 d=5 (N=400): 06 Nov 09:48
- E2 other kappa (N=500): 07 Nov 04:07
- Follow-up (reserved): 08 Nov 03:07
- E1 d=20, N=1000: 09 Nov 10:08  **after the hard stop**

### BO refit every 10 iterations (n > 100)

| # | Experiment | Runs | CPU-h | Wall-h (7 workers) | Group |
|---|---|---|---|---|---|
| 1 | A4 check d=10, N=500: 15 seeds x {cold every 5, cold every 10}, rotated k=1e4 | 30 | 19.9 | 2.8 | core |
| 2 | Sensitivity grids: BO: 6 cfgs x 2 problems x 10 seeds (CMA-ES grid negligible) | 120 | 43.9 | 6.3 | core |
| 3 | E2 primary (N=500): k=1e4: aligned, rotated, replicate x 30 seeds x {ARD, iso, TuRBO} | 270 | 128.5 | 18.4 | core |
| 4 | E2 low-eff-dim (d=20, N=500): aligned+rotated x 20 seeds x {ARD, iso, TuRBO}; ARD +1.5 min diagnostics | 120 | 56.5 | 8.1 | core |
| 5 | H5 CMA-ES runs: 11 dims x 20 seeds, cap 100 d^2 (alongside) | - | 0.5 | 0.1 | core |
| 6 | E1 d=2, N=200: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 2.4 | 0.3 | core |
| 7 | E1 d=3, N=200: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 2.4 | 0.3 | core |
| 8 | E1 d=5, N=400: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 7.1 | 1.0 | core |
| 9 | E1 d=8, N=640: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 33.3 | 4.8 | core |
| 10 | E1 d=10, N=1000: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 84.1 | 12.0 | core |
| 11 | E3 d=2 (N=200): 5 widths x 30 seeds x {LogEI, UCB} | 300 | 18.0 | 2.6 | core |
| 12 | E3 d=5 (N=400): 5 widths x 30 seeds x {LogEI, UCB} | 300 | 52.9 | 7.6 | core |
| 13 | E2 other kappa (N=500): k=1 aligned; k=1e2 aligned+rotated x 20 seeds x {ARD, iso, TuRBO} | 180 | 85.7 | 12.2 | core |
| 14 | Follow-up (reserved): one experiment chosen after E1-E3 (given: about 20 wall-h) | - | 140.0 | 20.0 | core |
| 15 | E1 d=20, N=1000: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds (last core) | 40 | 108.3 | 15.5 | core_d20 |
| 16 | E2 secondary N=1000: k=1e4 aligned+rotated x 10 seeds x {ARD, iso, TuRBO} | 60 | 96.1 | 13.7 | extra |
| 17 | A8c noise (E5): d=5 N=300, 2 noise levels x 20 seeds, BO ARD | 40 | 4.5 | 0.6 | extra |
| 18 | A8d long-budget probe: d=10 k=1e4 rotated, N=2000, 10 seeds (extrapolated beyond n=1000) | 10 | 178.7 | 25.5 | extra |
| 19 | A8e second family: 4 BBOB-style functions, d=10, N=500, 10 seeds | 40 | 22.2 | 3.2 | extra |

- Core without E1 d=20: **96 wall-h** (111 with 15% contingency; range 78 to 177)
- Core with E1 d=20: **112 wall-h** (129 with 15% contingency; range 90 to 206)
- Extras (E2 secondary, A8 c-e), only if the core finishes early: 43 wall-h

Cumulative finish times of the core phases in the agreed order (central estimate with 15% contingency, starting 02 Nov 07:29; hard stop 08 Nov 23:59):

- A4 check d=10, N=500: 02 Nov 10:45
- Sensitivity grids: 02 Nov 17:58
- E2 primary (N=500): 03 Nov 15:04
- E2 low-eff-dim (d=20, N=500): 04 Nov 00:21
- H5 CMA-ES runs: 04 Nov 00:26
- E1 d=2, N=200: 04 Nov 00:50
- E1 d=3, N=200: 04 Nov 01:14
- E1 d=5, N=400: 04 Nov 02:23
- E1 d=8, N=640: 04 Nov 07:52
- E1 d=10, N=1000: 04 Nov 21:40
- E3 d=2 (N=200): 05 Nov 00:38
- E3 d=5 (N=400): 05 Nov 09:20
- E2 other kappa (N=500): 05 Nov 23:24
- Follow-up (reserved): 06 Nov 22:24
- E1 d=20, N=1000: 07 Nov 16:11
