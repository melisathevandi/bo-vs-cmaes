"""TuRBO variants for the rotation-penalty ablation: cubic region and isotropic kernel."""

import numpy as np

from methods import turbo
from problems import ellipsoid


class _Model:
    class covar_module:
        class base_kernel:
            class lengthscale:
                @staticmethod
                def detach():
                    class T:
                        @staticmethod
                        def numpy():
                            return np.array([[0.5, 1.0, 4.0]])
                    return T
            lengthscale = lengthscale


def test_cubic_region_has_equal_sides_and_ard_region_does_not():
    c = np.full(3, 0.5)
    lb, ub = turbo.trust_region(_Model(), c, 0.1, 3, "cube")
    np.testing.assert_allclose(ub - lb, 0.1)
    lb, ub = turbo.trust_region(_Model(), c, 0.1, 3, "ard")
    assert np.ptp(ub - lb) > 0.1
    np.testing.assert_allclose(np.prod(ub - lb), 0.1 ** 3, rtol=1e-9)    # same volume as the cube


def test_variants_run_and_default_is_unchanged():
    f = ellipsoid(3, kappa=100.0, rotated=True, seed=1)
    base = turbo.run(f, 14, 5, refit_every=10)
    again = turbo.run(f, 14, 5, refit_every=10, ard=True, region="ard")
    np.testing.assert_array_equal(base["F"], again["F"])
    for cfg in ({"region": "cube"}, {"ard": False}):
        out = turbo.run(f, 14, 5, refit_every=10, **cfg)
        assert len(out["F"]) == 14 and np.isfinite(out["F"]).all()
    assert out["meta"]["region"] == "cube"
