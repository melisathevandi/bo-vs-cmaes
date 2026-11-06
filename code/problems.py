"""Test functions for the BO vs CMA-ES experiments.

Every function takes a point u in the unit cube [0, 1]^d. The point is mapped to the
search domain [LO, HI]^d before the function is evaluated, so every method searches
the same box. An instance is fixed by its family, its parameters and a seed:
the seed sets the location of the optimum and, for rotated instances, the rotation.

Each factory returns a callable f(u) -> float with these attributes:
    f.u_star  optimum location in the unit cube
    f.f_star  optimal value (0 for all families here)
    f.dim     dimension d
"""

import numpy as np

LO, HI = -5.0, 5.0      # search domain per coordinate
WIDTH = HI - LO


def random_rotation(d, rng):
    """Return a Haar-distributed random orthogonal d x d matrix.

    A QR decomposition of a Gaussian matrix gives an orthogonal Q. Multiplying each
    column by the sign of R's diagonal makes the distribution uniform over rotations.
    """
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))


def _sample_optimum(d, rng):
    """Draw the optimum away from the box edges, so it is not at a corner."""
    return rng.uniform(0.2, 0.8, size=d)


def ellipsoid(d, kappa=1.0, rotated=False, seed=0):
    """Ill-conditioned ellipsoid f(x) = sum_i kappa^(i/(d-1)) * y_i^2.

    y = R (x - x*) is measured in the original scale. The Hessian eigenvalues range
    from 1 to kappa, so kappa is the condition number. R is the identity when
    rotated=False ("aligned": the axes match the coordinates). The optimum is drawn
    first, so the aligned and rotated instances with the same seed share it.
    """
    if d < 2:
        raise ValueError("ellipsoid needs d >= 2 so that the weights are defined")
    rng = np.random.default_rng(seed)
    u_star = _sample_optimum(d, rng)
    rot = random_rotation(d, rng) if rotated else np.eye(d)
    weights = kappa ** (np.arange(d) / (d - 1))

    def f(u):
        y = rot @ ((np.asarray(u, dtype=float) - u_star) * WIDTH)
        return float(np.sum(weights * y * y))

    f.u_star = u_star
    f.f_star = 0.0
    f.dim = d
    f.rotation = rot
    f.weights = weights
    return f


def low_effective_dim(d, d_eff=2, kappa=1.0, rotated=False, seed=0):
    """f(x) = sum_k w_k * (a_k . (x - x*))^2, with only d_eff relevant directions.

    The rows a_k of A (d_eff x d) are orthonormal. Aligned: they are d_eff randomly
    chosen coordinate axes. Rotated: the first d_eff rows of a random rotation. The
    function is constant along the other d - d_eff directions, so the optimum is an
    affine subspace that contains x* (inside the box in both cases). P = A^T A projects
    onto the relevant subspace; the relevance diagnostic compares it to the GP's
    lengthscales. Aligned and rotated instances with the same seed share x*.
    """
    if not 1 <= d_eff <= d:
        raise ValueError("need 1 <= d_eff <= d")
    rng = np.random.default_rng(seed)
    u_star = _sample_optimum(d, rng)
    if rotated:
        a = random_rotation(d, rng)[:d_eff]
    else:
        a = np.eye(d)[rng.choice(d, size=d_eff, replace=False)]
    weights = kappa ** (np.arange(d_eff) / max(d_eff - 1, 1))

    def f(u):
        z = a @ ((np.asarray(u, dtype=float) - u_star) * WIDTH)
        return float(np.sum(weights * z * z))

    f.u_star = u_star
    f.f_star = 0.0
    f.dim = d
    f.A = a
    f.P = a.T @ a
    f.weights = weights
    return f


def rosenbrock(d, seed=0):
    """Shifted Rosenbrock function, with its minimum at the seeded optimum.

    z = x - x* + 1 moves the classic minimum at z = 1 onto x*, so f(x*) = 0.
    The curved valley makes this a harder, non-separable test problem.
    """
    if d < 2:
        raise ValueError("rosenbrock needs d >= 2")
    rng = np.random.default_rng(seed)
    u_star = _sample_optimum(d, rng)

    def f(u):
        x = np.asarray(u, dtype=float) * WIDTH + LO
        z = x - (u_star * WIDTH + LO) + 1.0
        return float(np.sum(100.0 * (z[1:] - z[:-1] ** 2) ** 2 + (1.0 - z[:-1]) ** 2))

    f.u_star = u_star
    f.f_star = 0.0
    f.dim = d
    return f


F_DECOY = 1.0       # minimum of the broad (decoy) basin; the narrow basin reaches 0


def decoy_basin(d=2, width=0.1, seed=0):
    """Decoy-basin function for E3: f(u) = min(B(u), D(u)) on the unit cube.

    B(u) = F_DECOY + 4 ||u - c_b||^2 is a broad basin whose minimum is F_DECOY = 1 at c_b.
    D(u) = ||u - c_n||^2 / width^2 is a narrow basin with minimum 0 at c_n (the global
    optimum). D < F_DECOY exactly when ||u - c_n|| < width, and B >= F_DECOY everywhere, so
    f(u) < F_DECOY if and only if u is inside the ball of radius `width` around c_n: the
    success event "better than anything the decoy basin offers" is a ball of that radius.
    c_b is drawn uniformly in [0.2, 0.8]^d, then c_n the same way, redrawn until
    ||c_n - c_b|| >= 0.4. The seed fixes both centres (and nothing depends on the width, so
    instances with the same seed share the centres across widths).
    """
    rng = np.random.default_rng(seed)
    c_b = _sample_optimum(d, rng)
    c_n = _sample_optimum(d, rng)
    while np.linalg.norm(c_n - c_b) < 0.4:
        c_n = _sample_optimum(d, rng)

    def f(u):
        u = np.asarray(u, dtype=float)
        broad = F_DECOY + 4.0 * float(np.sum((u - c_b) ** 2))
        narrow = float(np.sum((u - c_n) ** 2)) / width ** 2
        return min(broad, narrow)

    f.u_star = c_n
    f.f_star = 0.0
    f.dim = d
    f.centre_broad = c_b
    f.width = width
    f.f_decoy = F_DECOY
    return f
