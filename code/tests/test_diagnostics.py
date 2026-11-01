import numpy as np
import pytest
import torch

from diagnostics import relevance_ratio, rho_stability
from methods import bo
from problems import ellipsoid, low_effective_dim


def projector(d, d_eff, seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    return q[:, :d_eff] @ q[:, :d_eff].T


def test_relevance_ratio_isotropic_equals_deff_over_d():
    P = projector(10, 2)
    assert relevance_ratio(np.full(10, 3.7), P) == pytest.approx(2 / 10)


def test_relevance_ratio_is_one_when_kernel_sees_only_relevant_axes():
    P = np.diag([1.0, 1.0, 0.0, 0.0, 0.0])
    ls = np.array([1.0, 1.0, 1e4, 1e4, 1e4])
    assert relevance_ratio(ls, P) == pytest.approx(1.0, abs=1e-6)
    ls_wrong = np.array([1e4, 1e4, 1.0, 1.0, 1.0])
    assert relevance_ratio(ls_wrong, P) == pytest.approx(0.0, abs=1e-6)


def test_relevance_ratio_stays_in_unit_interval():
    rng = np.random.default_rng(0)
    for _ in range(50):
        P = projector(6, 2, seed=int(rng.integers(1000)))
        assert 0.0 <= relevance_ratio(np.exp(rng.normal(size=6)), P) <= 1.0


def test_rho_stability_identical_fits_have_zero_spread():
    diag = {"checkpoint_n": [50], "lengthscales": [[1.0, 2.0, 3.0]], "mll": [-5.0],
            "multistart": [[{"mll": -5.0, "lengthscales": [1.0, 2.0, 3.0]}] * 3]}
    row = rho_stability(diag, np.diag([1.0, 0.0, 0.0])).iloc[0]
    assert row.spread_all == pytest.approx(0.0) and row.n_fits == 4


def test_rho_stability_near_optimal_excludes_worse_fits():
    P = np.diag([1.0, 0.0])
    diag = {"checkpoint_n": [50], "lengthscales": [[1.0, 1.0]], "mll": [-5.0],
            "multistart": [[{"mll": -5.5, "lengthscales": [1.0, 1.2]},
                            {"mll": -20.0, "lengthscales": [5.0, 0.5]}]]}   # much worse fit
    row = rho_stability(diag, P, tol=1.0).iloc[0]
    assert row.n_near_optimal == 2
    assert row.spread_near_optimal < 0.1 < row.spread_all


def test_multistart_fits_returns_finite_results_for_every_start():
    f = low_effective_dim(4, 2, rotated=False, seed=1)
    x = np.random.default_rng(0).uniform(size=(20, 4))
    y = -np.array([f(u) for u in x])
    out = bo.multistart_fits(torch.tensor(x), torch.tensor(y).unsqueeze(-1), True, 2.5, 3,
                             np.random.default_rng(1))
    assert len(out) == 3
    assert all(np.isfinite(o["mll"]) and len(o["lengthscales"]) == 4 for o in out)


def test_checkpoint_at_or_beyond_budget_is_rejected():
    f = ellipsoid(2, kappa=1e2, rotated=False, seed=0)
    with pytest.raises(ValueError):
        bo.run(f, 10, seed=0, checkpoints=(10,))


def test_diagnostics_do_not_change_the_optimisation_path():
    f = ellipsoid(2, kappa=1e2, rotated=True, seed=0)
    kw = dict(num_restarts=2, raw_samples=32)
    plain = bo.run(f, 12, seed=5, **kw)
    observed = bo.run(f, 12, seed=5, checkpoints=(8,), multistart=2, **kw)
    np.testing.assert_array_equal(plain["X"], observed["X"])
    diag = observed["meta"]["diagnostics"]
    assert diag["checkpoint_n"] == [8] and len(diag["multistart"][0]) == 2
