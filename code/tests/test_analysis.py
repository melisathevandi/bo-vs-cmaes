import numpy as np
import pandas as pd

from analysis import GAP_FLOOR, bootstrap_ci, gap_table, log_gap, paired_difference


def test_log_gap_uses_best_so_far_and_floor():
    F = np.array([100.0, 10.0, 50.0, 1e-3])
    assert log_gap(F, 0.0, 3) == np.log10(10.0)            # best of the first 3
    assert log_gap(F, 0.0, 4) == np.log10(1e-3)
    assert log_gap(np.array([1e-20]), 0.0, 1) == np.log10(GAP_FLOOR)


def test_log_gap_forward_fills_short_runs():
    F = np.array([5.0, 1.0])                                # run stopped after 2 evaluations
    assert log_gap(F, 0.0, 500) == 0.0


def test_bootstrap_ci_contains_true_median_for_symmetric_data():
    x = np.random.default_rng(0).standard_normal(200)
    med, lo, hi = bootstrap_ci(x)
    assert lo < med < hi and lo < 0.0 < hi


def test_paired_difference_detects_a_clear_shift():
    rng = np.random.default_rng(1)
    base = rng.uniform(-4, -1, 30)
    rows = [{"problem": "p", "seed": s, "method": "a", "F": np.array([10 ** g]), "f_star": 0.0}
            for s, g in enumerate(base - 1.0)]               # a is 1 decade better
    rows += [{"problem": "p", "seed": s, "method": "b", "F": np.array([10 ** g]), "f_star": 0.0}
             for s, g in enumerate(base)]
    table = gap_table(pd.DataFrame(rows), n=1)
    out = paired_difference(table, "a", "b", "p")
    assert abs(out["median_diff"] + 1.0) < 1e-9 and out["ci_hi"] < 0 and out["win_rate"] == 1.0
