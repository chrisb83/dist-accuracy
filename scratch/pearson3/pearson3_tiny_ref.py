"""Auxiliary reference for scipy.stats.pearson3 at small |skew| (alpha = 4/skew**2 >= 4e6).

Same formulas as refs/pearson3_ref.py; the cdf is computed by the quadrature
route there (mpmath's gammainc does not converge for alpha >= 1e7 and is very
slow at 4e6). 100 digits. Parameter sets straddle SciPy's normal-branch
threshold |skew| < 1.6e-5 (line 8490) and include alpha = 4e6 (skew 1e-3),
where scipy.special.gammainc's 2000-term cap already bites (gamma F1).

Tail integrals come from refs/pearson3_ref.py::_tail_quad, which integrates
the pdf rescaled by exp(-logpdf(x)) (fixed after verification: the unscaled
integrand lost ~5e-12 relative at 1e-198 with 100 digits). This note also
changes the file hash, so the harness cache built before the fix is not reused.
"""
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "..", "refs"))
import pearson3_ref as _R  # noqa: E402
import mpmath as mp  # noqa: E402

_R.ALPHA_QUAD = mp.mpf(1e5)

DIST = "pearson3"
DPS = 100
REPLACE_GRID = True
PARAMS = [(1e-3,), (-1e-3,), (2e-5,), (-2e-5,), (1.6e-5,), (1.59e-5,),
          (-1.59e-5,), (1e-8,)]


def ppf(q, k):
    """Newton on log F (q <= 1/2) or log(1 - F) (q > 1/2), seeded with the
    Cornish-Fisher quantile; much faster than accuracy.invert here because
    each cdf evaluation is a quadrature. Residual checked to 1e-60."""
    q = mp.mpf(q)
    upper = q > mp.mpf(0.5)
    Lt = mp.log(1 - q) if upper else mp.log(q)
    z = mp.sqrt(2) * mp.erfinv(2 * q - 1)
    x = z + k / 6 * (z ** 2 - 1)
    for _ in range(60):
        F = _R.cdf(x, k)
        G = 1 - F if upper else F
        r = mp.log(G) - Lt
        d = _R.pdf(x, k) / G * (-1 if upper else 1)   # d log G / dx
        dx = -r / d
        x += dx
        if abs(r) < mp.mpf(10) ** -70:
            return x
    raise mp.NoConvergence("ppf Newton")


REFS = dict(_R.REFS)
REFS["ppf"] = ppf
REFS["isf"] = lambda q, k: ppf(1 - q, k)


def x_points(k):
    z = np.array([0.0, 0.3, -0.3, 1, -1, 2, -2, 3, -3, 4, -4, 4.6, -4.6, 5,
                  -5, 6, -6, 7, -7, 8, -8, 10, -10, 12, -12, 15, -15, 20, -20,
                  30, -30, 37, -37])
    return z


def q_points(k):
    return np.array([1e-20, 1e-12, 1e-8, 1e-5, 0.01, 0.1, 0.5, 0.9, 0.99,
                     1 - 1e-5, 1 - 1e-8])


def sf(x, k):
    """Right-tail integral directly (1 - cdf loses everything below 1e-100 at
    100 digits)."""
    if x > _R._mode(k):
        return _R._tail_quad(x, k, +1)
    return 1 - _R.cdf(x, k)


REFS["sf"] = sf
REFS["logsf"] = lambda x, k: mp.log(sf(x, k))
