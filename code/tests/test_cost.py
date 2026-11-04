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


def _runs_df(rows):
    import pandas as pd
    return pd.DataFrame(rows)


def test_table_reports_censored_cma_seeds_and_bo_status():
    from cost_table import table
    def run(method, seed, F, T, budget):
        return {"problem": "p", "method": method, "seed": seed, "F": np.array(F, float), "f_star": 0.0,
                "t_method": np.array(T, float), "budget": budget}
    bo = [run("bo_ard", 0, [1, 0.01, 1e-6, 1e-6], [0, 10, 20, 30], 4),      # reaches every level
          run("bo_ard", 1, [1, 0.5, 0.5, 0.5], [0, 10, 20, 30], 4)]        # stuck at 0.5
    cma = [run("cma", 0, [1, 1, 1, 1, 1, 1e-6], [0, 0, 0, 0, 0, 1], 100),   # reaches at 6 evaluations
           run("cma", 1, [1, 0.5, 0.5], [0, 0, 1], 100)]                    # terminated early at 0.5: censored
    t, c = table(_runs_df(bo), _runs_df(cma))
    row = t[t.target == "0.01"].iloc[0]
    assert row.cma_censored == 1 and row.cma_censored_seeds == "1"
    assert row.bo_reaches == 1 and row.cma_reaches == 1 and row.c_star_defined == 1
    assert np.isclose(row.median_c_star_s, (10 - 1) / (6 - 2))            # nb index 1, nc index 5, T 10 vs 1
    cs = c[c.target == "0.01"].iloc[0]
    assert cs.seed == 1 and cs.cma_cause == "terminated" and not cs["bo_reaches_0.1"]
