"""Unit tests for code/problems.py. These check the definitions, not the optimisers."""

import numpy as np
import pytest

from problems import ellipsoid, low_effective_dim, random_rotation, rosenbrock


@pytest.mark.parametrize("d", [2, 5, 10, 20])
@pytest.mark.parametrize("rotated", [False, True])
def test_ellipsoid_zero_at_optimum_and_positive_elsewhere(d, rotated):
    f = ellipsoid(d, kappa=1e4, rotated=rotated, seed=3)
    assert f(f.u_star) == pytest.approx(f.f_star, abs=1e-12)
    rng = np.random.default_rng(0)
    for u in rng.uniform(0, 1, size=(20, d)):
        assert f(u) > 0.0


@pytest.mark.parametrize("d", [2, 10])
def test_rotation_is_orthogonal(d):
    rot = random_rotation(d, np.random.default_rng(1))
    np.testing.assert_allclose(rot.T @ rot, np.eye(d), atol=1e-10)
    assert np.linalg.det(rot) == pytest.approx(1.0) or np.linalg.det(rot) == pytest.approx(-1.0)


@pytest.mark.parametrize("kappa", [1.0, 1e2, 1e4])
def test_ellipsoid_hessian_condition_number(kappa):
    """Finite-difference Hessian in unit-cube coordinates has condition number kappa."""
    d = 6
    f = ellipsoid(d, kappa=kappa, rotated=True, seed=7)
    h = 1e-4
    hess = np.zeros((d, d))
    e = np.eye(d)
    x0 = f.u_star + 0.01  # a point near the optimum, where the function is quadratic
    for i in range(d):
        for j in range(d):
            hess[i, j] = (
                f(x0 + h * e[i] + h * e[j]) - f(x0 + h * e[i] - h * e[j])
                - f(x0 - h * e[i] + h * e[j]) + f(x0 - h * e[i] - h * e[j])
            ) / (4 * h * h)
    eig = np.linalg.eigvalsh((hess + hess.T) / 2)
    assert eig.max() / eig.min() == pytest.approx(kappa, rel=1e-3)


def test_aligned_and_rotated_share_optimum():
    aligned = ellipsoid(5, kappa=10, rotated=False, seed=11)
    rotated = ellipsoid(5, kappa=10, rotated=True, seed=11)
    np.testing.assert_array_equal(aligned.u_star, rotated.u_star)
    np.testing.assert_allclose(aligned.rotation, np.eye(5))


def test_ellipsoid_is_deterministic_for_seed():
    a = ellipsoid(4, kappa=100, rotated=True, seed=5)
    b = ellipsoid(4, kappa=100, rotated=True, seed=5)
    u = np.full(4, 0.3)
    assert a(u) == b(u)


@pytest.mark.parametrize("d", [2, 5, 10, 20])
def test_rosenbrock_zero_at_optimum(d):
    f = rosenbrock(d, seed=2)
    assert f(f.u_star) == pytest.approx(0.0, abs=1e-12)
    rng = np.random.default_rng(1)
    for u in rng.uniform(0, 1, size=(20, d)):
        assert f(u) >= 0.0


@pytest.mark.parametrize("rotated", [False, True])
def test_low_eff_dim_structure(rotated):
    d, d_eff = 20, 2
    f = low_effective_dim(d, d_eff, kappa=10.0, rotated=rotated, seed=4)
    np.testing.assert_allclose(f.A @ f.A.T, np.eye(d_eff), atol=1e-10)   # orthonormal rows
    assert np.trace(f.P) == pytest.approx(d_eff)                         # projector rank
    assert f(f.u_star) == pytest.approx(0.0, abs=1e-12)
    # f is constant along directions orthogonal to the relevant subspace
    rng = np.random.default_rng(0)
    u = rng.uniform(0.3, 0.7, d)
    v = rng.standard_normal(d)
    v -= f.P @ v
    assert f(u + 0.05 * v / np.linalg.norm(v)) == pytest.approx(f(u), rel=1e-9)


def test_low_eff_dim_aligned_axes_and_shared_optimum():
    aligned = low_effective_dim(10, 2, rotated=False, seed=9)
    rotated = low_effective_dim(10, 2, rotated=True, seed=9)
    np.testing.assert_array_equal(aligned.u_star, rotated.u_star)
    # aligned: each row of A is a coordinate axis
    assert set(np.abs(aligned.A).sum(axis=1)) == {1.0}
    assert np.count_nonzero(aligned.A) == 2


def test_low_eff_dim_uniform_sample_scale_similar_aligned_vs_rotated():
    """Rotation should not change how large f is on average, or the comparison is biased."""
    ratios = []
    for seed in range(20):
        u = np.random.default_rng(100 + seed).uniform(0, 1, size=(300, 20))
        fa = low_effective_dim(20, 2, rotated=False, seed=seed)
        fr = low_effective_dim(20, 2, rotated=True, seed=seed)
        ratios.append(np.median([fr(x) for x in u]) / np.median([fa(x) for x in u]))
    assert 0.5 < np.median(ratios) < 2.0


def test_ellipsoid_rejects_dim_one():
    with pytest.raises(ValueError):
        ellipsoid(1)
