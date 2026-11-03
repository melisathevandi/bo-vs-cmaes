import numpy as np

from cost_table import break_even, reach


def test_reach_uses_best_so_far():
    assert reach([5.0, 1.0, 3.0, 0.05], 0.0, 0.1) == 3
    assert reach([5.0, 1.0], 0.0, 0.1) is None


def test_break_even_equalises_total_time():
    # BO reaches eps at evaluation 10 with 100 s overhead; CMA at evaluation 60 with 1 s overhead.
    bo = {"F": np.r_[np.ones(10), 0.0], "f_star": 0.0, "t_method": np.linspace(0, 100, 11)}
    cma = {"F": np.r_[np.ones(60), 0.0], "f_star": 0.0, "t_method": np.linspace(0, 1, 61)}
    c = break_even(bo, cma, 0.1)
    # nb = 10, nc = 60: (100 - 1) / 50
    assert np.isclose(c, 99 / 50)
    assert np.isclose(100 + 10 * c, 1 + 60 * c)


def test_break_even_undefined_when_bo_is_not_first():
    bo = {"F": np.r_[np.ones(60), 0.0], "f_star": 0.0, "t_method": np.zeros(61)}
    cma = {"F": np.r_[np.ones(10), 0.0], "f_star": 0.0, "t_method": np.zeros(11)}
    assert break_even(bo, cma, 0.1) is None
