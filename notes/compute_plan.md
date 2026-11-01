# Compute plan (planning estimate, not a result)

BO cost = timing-pilot integral x FACTOR (2.28, measured/integral at d=10, N=500, 3 real runs, folding in real-trajectory fit cost and the load of the ~6 busy processes during that run) x LOAD7 (1.1, assumed for a 7-worker pool; to be checked against the A4 check at d=10). TuRBO cost = measured pilot median x LOAD7. The A4-check row is the same in both tables.

### BO refit every 5 iterations (n > 100)

| # | Experiment | Runs | CPU-h | Wall-h (7 workers) | Group |
|---|---|---|---|---|---|
| 1 | A4 check d=10, N=500: 15 seeds x {cold every 5, cold every 10}, rotated k=1e4 | 30 | 17.5 | 2.5 | core |
| 2 | Sensitivity grids: BO: 6 cfgs x 2 problems x 10 seeds (CMA-ES grid negligible) | 120 | 51.9 | 7.4 | core |
| 3 | E2 primary (N=500): k=1e4: aligned, rotated, replicate x 30 seeds x {ARD, iso, TuRBO} | 270 | 147.1 | 21.0 | core |
| 4 | E2 low-eff-dim (d=20, N=500): aligned+rotated x 20 seeds x {ARD, iso, TuRBO}; ARD +1.5 min diagnostics | 120 | 65.0 | 9.3 | core |
| 5 | H5 CMA-ES runs: 11 dims x 20 seeds, cap 100 d^2 (alongside) | - | 0.5 | 0.1 | core |
| 6 | E1 d=2, N=200: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 2.3 | 0.3 | core |
| 7 | E1 d=3, N=200: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 2.3 | 0.3 | core |
| 8 | E1 d=5, N=400: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 7.5 | 1.1 | core |
| 9 | E1 d=8, N=640: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 42.4 | 6.1 | core |
| 10 | E1 d=10, N=1000: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 116.9 | 16.7 | core |
| 11 | E3 d=2 (N=200): 5 widths x 30 seeds x {LogEI, UCB} | 300 | 17.3 | 2.5 | core |
| 12 | E3 d=5 (N=400): 5 widths x 30 seeds x {LogEI, UCB} | 300 | 56.1 | 8.0 | core |
| 13 | E2 other kappa (N=500): k=1 aligned; k=1e2 aligned+rotated x 20 seeds x {ARD, iso, TuRBO} | 180 | 98.1 | 14.0 | core |
| 14 | Follow-up (reserved): one experiment chosen after E1-E3 (given: about 20 wall-h) | - | 140.0 | 20.0 | core |
| 15 | E1 d=20, N=1000: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds (last core) | 40 | 166.2 | 23.7 | core_d20 |
| 16 | E2 secondary N=1000: k=1e4 aligned+rotated x 10 seeds x {ARD, iso, TuRBO} | 60 | 127.5 | 18.2 | extra |
| 17 | A8c noise (E5): d=5 N=300, 2 noise levels x 20 seeds, BO ARD | 40 | 4.6 | 0.7 | extra |
| 18 | A8d long-budget probe: d=10 k=1e4 rotated, N=2000, 10 seeds (extrapolated beyond n=1000) | 10 | 278.3 | 39.8 | extra |
| 19 | A8e second family: 4 BBOB-style functions, d=10, N=500, 10 seeds | 40 | 27.1 | 3.9 | extra |

- Core without E1 d=20: **109 wall-h** (126 with 15% contingency; range 88 to 201)
- Core with E1 d=20: **133 wall-h** (153 with 15% contingency; range 107 to 245)
- Extras (E2 secondary, A8 c-e), only if the core finishes early: 62 wall-h

Cumulative finish times of the core phases in the agreed order (central estimate with 15% contingency, starting 01 Nov 23:46; hard stop 08 Nov 23:59):

- A4 check d=10, N=500: 02 Nov 02:38
- Sensitivity grids: 02 Nov 11:10
- E2 primary (N=500): 03 Nov 11:20
- E2 low-eff-dim (d=20, N=500): 03 Nov 22:01
- H5 CMA-ES runs: 03 Nov 22:06
- E1 d=2, N=200: 03 Nov 22:29
- E1 d=3, N=200: 03 Nov 22:52
- E1 d=5, N=400: 04 Nov 00:05
- E1 d=8, N=640: 04 Nov 07:03
- E1 d=10, N=1000: 05 Nov 02:16
- E3 d=2 (N=200): 05 Nov 05:07
- E3 d=5 (N=400): 05 Nov 14:20
- E2 other kappa (N=500): 06 Nov 06:27
- Follow-up (reserved): 07 Nov 05:27
- E1 d=20, N=1000: 08 Nov 08:45

### BO refit every 10 iterations (n > 100)

| # | Experiment | Runs | CPU-h | Wall-h (7 workers) | Group |
|---|---|---|---|---|---|
| 1 | A4 check d=10, N=500: 15 seeds x {cold every 5, cold every 10}, rotated k=1e4 | 30 | 17.5 | 2.5 | core |
| 2 | Sensitivity grids: BO: 6 cfgs x 2 problems x 10 seeds (CMA-ES grid negligible) | 120 | 38.6 | 5.5 | core |
| 3 | E2 primary (N=500): k=1e4: aligned, rotated, replicate x 30 seeds x {ARD, iso, TuRBO} | 270 | 113.1 | 16.2 | core |
| 4 | E2 low-eff-dim (d=20, N=500): aligned+rotated x 20 seeds x {ARD, iso, TuRBO}; ARD +1.5 min diagnostics | 120 | 50.0 | 7.1 | core |
| 5 | H5 CMA-ES runs: 11 dims x 20 seeds, cap 100 d^2 (alongside) | - | 0.5 | 0.1 | core |
| 6 | E1 d=2, N=200: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 2.1 | 0.3 | core |
| 7 | E1 d=3, N=200: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 2.1 | 0.3 | core |
| 8 | E1 d=5, N=400: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 6.2 | 0.9 | core |
| 9 | E1 d=8, N=640: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 29.3 | 4.2 | core |
| 10 | E1 d=10, N=1000: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 74.0 | 10.6 | core |
| 11 | E3 d=2 (N=200): 5 widths x 30 seeds x {LogEI, UCB} | 300 | 15.8 | 2.3 | core |
| 12 | E3 d=5 (N=400): 5 widths x 30 seeds x {LogEI, UCB} | 300 | 46.6 | 6.7 | core |
| 13 | E2 other kappa (N=500): k=1 aligned; k=1e2 aligned+rotated x 20 seeds x {ARD, iso, TuRBO} | 180 | 75.4 | 10.8 | core |
| 14 | Follow-up (reserved): one experiment chosen after E1-E3 (given: about 20 wall-h) | - | 140.0 | 20.0 | core |
| 15 | E1 d=20, N=1000: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds (last core) | 40 | 95.3 | 13.6 | core_d20 |
| 16 | E2 secondary N=1000: k=1e4 aligned+rotated x 10 seeds x {ARD, iso, TuRBO} | 60 | 84.5 | 12.1 | extra |
| 17 | A8c noise (E5): d=5 N=300, 2 noise levels x 20 seeds, BO ARD | 40 | 4.0 | 0.6 | extra |
| 18 | A8d long-budget probe: d=10 k=1e4 rotated, N=2000, 10 seeds (extrapolated beyond n=1000) | 10 | 157.2 | 22.5 | extra |
| 19 | A8e second family: 4 BBOB-style functions, d=10, N=500, 10 seeds | 40 | 19.5 | 2.8 | extra |

- Core without E1 d=20: **87 wall-h** (100 with 15% contingency; range 70 to 161)
- Core with E1 d=20: **101 wall-h** (116 with 15% contingency; range 81 to 186)
- Extras (E2 secondary, A8 c-e), only if the core finishes early: 38 wall-h

Cumulative finish times of the core phases in the agreed order (central estimate with 15% contingency, starting 01 Nov 23:46; hard stop 08 Nov 23:59):

- A4 check d=10, N=500: 02 Nov 02:38
- Sensitivity grids: 02 Nov 08:59
- E2 primary (N=500): 03 Nov 03:34
- E2 low-eff-dim (d=20, N=500): 03 Nov 11:46
- H5 CMA-ES runs: 03 Nov 11:51
- E1 d=2, N=200: 03 Nov 12:12
- E1 d=3, N=200: 03 Nov 12:33
- E1 d=5, N=400: 03 Nov 13:34
- E1 d=8, N=640: 03 Nov 18:23
- E1 d=10, N=1000: 04 Nov 06:33
- E3 d=2 (N=200): 04 Nov 09:09
- E3 d=5 (N=400): 04 Nov 16:48
- E2 other kappa (N=500): 05 Nov 05:11
- Follow-up (reserved): 06 Nov 04:11
- E1 d=20, N=1000: 06 Nov 19:50
