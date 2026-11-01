# Compute plan (planning estimate, not a result)

BO estimates are the timing-pilot integral times 1.3 (parallel load) times a calibration factor of 1.75 (measured / estimated at d=10, N=500).

### BO refit every 5 iterations (n > 100)

| Experiment | Runs (BO-type) | Min/run | CPU-h | Wall-h on 7 workers | Cut-order mark |
|---|---|---|---|---|---|
| Sensitivity grids: BO: 6 cfgs x 2 problems x 10 seeds; CMA-ES grid negligible | 120 | 23.6 | 47.2 | 6.7 | never cut |
| E0 invariance (done): CMA-ES only | 0 | 0.0 | 0.0 | 0.0 | done |
| E1 d=2, N=200: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 3.2 | 2.1 | 0.3 | core |
| E1 d=3, N=200: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 3.2 | 2.1 | 0.3 | core |
| E1 d=5, N=400: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 10.2 | 6.8 | 1.0 | core |
| E1 d=8, N=640: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 57.8 | 38.5 | 5.5 | core |
| E1 d=10, N=1000: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 159.5 | 106.3 | 15.2 | core |
| E1 d=20, N=1000: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 226.6 | 151.1 | 21.6 | cut 4th (d=20 in E1) |
| E2 primary (N=500): k=1e4: aligned, rotated, replicate x 30 seeds x {ARD, iso, TuRBO} | 270 | 29.7 | 133.8 | 19.1 | never cut |
| E2 secondary N=1000 (optional): k=1e4 aligned+rotated x 10 seeds x {ARD, iso, TuRBO} | 60 | 115.9 | 115.9 | 16.6 | optional |
| E2 other kappa (N=500): k=1 aligned; k=1e2 aligned+rotated x 20 seeds x {ARD, iso, TuRBO} | 180 | 29.7 | 89.2 | 12.7 | core |
| E2 low-eff-dim (d=20, N=500): aligned+rotated x 20 seeds x {ARD, iso, TuRBO}; ARD adds ~3 min diagnostics | 120 | 29.6 | 59.3 | 8.5 | never cut |
| E3 d=2 (N=200): 5 widths x 30 seeds x {LogEI, UCB} | 300 | 3.2 | 15.8 | 2.3 | UCB cut 1st (kept in E3) |
| E3 d=5 (N=400): 5 widths x 30 seeds x {LogEI, UCB} | 300 | 10.2 | 51.0 | 7.3 | UCB cut 1st (kept in E3) |
| H5 CMA-ES runs: 11 dims x 20 seeds, cap 100 d^2 | 220 | 0.0 | 0.0 | 0.0 | core |
| A8c noise (E5): d=5 N=300, 2 noise levels x 20 seeds, BO ARD | 40 | 6.3 | 4.2 | 0.6 | cut 1st (A8 c-e) |
| A8d long-budget probe: d=10 k=1e4 rotated, N=2000, 10 seeds, BO ARD (extrapolated beyond n=1000) | 10 | 1517.8 | 253.0 | 36.1 | cut 1st (A8 c-e) |
| A8e second family: 4 BBOB-style functions, d=10, N=500, 10 seeds, BO ARD | 40 | 37.0 | 24.7 | 3.5 | cut 1st (A8 c-e) |
| **Total** | | | **1101** | **157** (range 110 to 252) | |

### BO refit every 10 iterations (n > 100)

| Experiment | Runs (BO-type) | Min/run | CPU-h | Wall-h on 7 workers | Cut-order mark |
|---|---|---|---|---|---|
| Sensitivity grids: BO: 6 cfgs x 2 problems x 10 seeds; CMA-ES grid negligible | 120 | 17.6 | 35.1 | 5.0 | never cut |
| E0 invariance (done): CMA-ES only | 0 | 0.0 | 0.0 | 0.0 | done |
| E1 d=2, N=200: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 2.9 | 1.9 | 0.3 | core |
| E1 d=3, N=200: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 2.9 | 1.9 | 0.3 | core |
| E1 d=5, N=400: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 8.5 | 5.6 | 0.8 | core |
| E1 d=8, N=640: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 40.0 | 26.7 | 3.8 | core |
| E1 d=10, N=1000: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 100.9 | 67.3 | 9.6 | core |
| E1 d=20, N=1000: BO ARD x {ellipsoid, Rosenbrock} x 20 seeds | 40 | 129.9 | 86.6 | 12.4 | cut 4th (d=20 in E1) |
| E2 primary (N=500): k=1e4: aligned, rotated, replicate x 30 seeds x {ARD, iso, TuRBO} | 270 | 22.8 | 102.8 | 14.7 | never cut |
| E2 secondary N=1000 (optional): k=1e4 aligned+rotated x 10 seeds x {ARD, iso, TuRBO} | 60 | 76.9 | 76.9 | 11.0 | optional |
| E2 other kappa (N=500): k=1 aligned; k=1e2 aligned+rotated x 20 seeds x {ARD, iso, TuRBO} | 180 | 22.8 | 68.5 | 9.8 | core |
| E2 low-eff-dim (d=20, N=500): aligned+rotated x 20 seeds x {ARD, iso, TuRBO}; ARD adds ~3 min diagnostics | 120 | 22.8 | 45.6 | 6.5 | never cut |
| E3 d=2 (N=200): 5 widths x 30 seeds x {LogEI, UCB} | 300 | 2.9 | 14.4 | 2.1 | UCB cut 1st (kept in E3) |
| E3 d=5 (N=400): 5 widths x 30 seeds x {LogEI, UCB} | 300 | 8.5 | 42.3 | 6.0 | UCB cut 1st (kept in E3) |
| H5 CMA-ES runs: 11 dims x 20 seeds, cap 100 d^2 | 220 | 0.0 | 0.0 | 0.0 | core |
| A8c noise (E5): d=5 N=300, 2 noise levels x 20 seeds, BO ARD | 40 | 5.4 | 3.6 | 0.5 | cut 1st (A8 c-e) |
| A8d long-budget probe: d=10 k=1e4 rotated, N=2000, 10 seeds, BO ARD (extrapolated beyond n=1000) | 10 | 857.7 | 142.9 | 20.4 | cut 1st (A8 c-e) |
| A8e second family: 4 BBOB-style functions, d=10, N=500, 10 seeds, BO ARD | 40 | 26.7 | 17.8 | 2.5 | cut 1st (A8 c-e) |
| **Total** | | | **740** | **106** (range 74 to 169) | |

Refit every 5 costs 361 CPU-h more than every 10 (52 wall-h on 7 workers).
