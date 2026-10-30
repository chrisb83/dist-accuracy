"""mpmath reference for scipy.stats.invgamma (standard form, loc=0, scale=1).

Parametrization from the docstring pdf:

    f(x; a) = x**(-a-1) * exp(-1/x) / Gamma(a),    x > 0, a > 0
    F(x; a) = Q(a, 1/x)    (upper regularized incomplete gamma)
    S(x; a) = P(a, 1/x)    (lower regularized incomplete gamma)

Speed: as in refs/gamma_ref.py, evaluate whichever mpmath incomplete gamma
is fast/robust on its side (upper for y >= a, lower for y < a, y = 1/x) and
get the other by subtracting from 1 (exact enough at 400 digits).
1/x is computed in mpmath, i.e. exactly at working precision.

ppf/isf are inverted from the cdf with ``accuracy.invert``; the bracket is
seeded from the SciPy value (speed-up only).
"""
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".claude", "skills",
                                "dist-accuracy-review", "scripts"))
from accuracy import invert  # noqa: E402

DIST = "invgamma"
DPS = 400

# inventory example; tiny/small shapes (gammaincinv with exponent/a), a = 1,
# moderate, a = 1e3 (gamma F6 regime) and a = 1e6 (igam_series cap, gamma F1;
# large cancellation in _logpdf).
PARAMS = [(4.066899613699307,), (1e-3,), (0.1,), (1.0,), (10.5,),
          (1e3,), (1e6,)]


def _P(a, y):
    """Lower regularized incomplete gamma P(a, y)."""
    if y >= a:
        return 1 - mp.gammainc(a, y, mp.inf, regularized=True)
    return mp.gammainc(a, 0, y, regularized=True)


def pdf(x, a):
    return mp.exp(-(a + 1) * mp.log(x) - 1 / x - mp.loggamma(a))


def logpdf(x, a):
    return -(a + 1) * mp.log(x) - 1 / x - mp.loggamma(a)


def cdf(x, a):
    return 1 - _P(a, 1 / x)


def sf(x, a):
    return _P(a, 1 / x)


def _seed(method, q, a):
    from scipy import stats
    with np.errstate(all="ignore"):
        v = float(getattr(stats.invgamma, method)(q, a))
    if not (np.isfinite(v) and v > 0):
        return None, None
    # tight bracket: skips invert's coarse bisection (the harness also calls
    # ppf/isf many times for condition numbers); invert widens it if the
    # SciPy value is off by more than 1e-4 relative.
    return v * (1 - 1e-4), v * (1 + 1e-4)


def ppf(q, a):
    lo, hi = _seed("ppf", float(q), float(a))
    return invert(lambda x: cdf(x, a), q, lo=lo, hi=hi, positive=True)


def isf(q, a):
    lo, hi = _seed("isf", float(q), float(a))
    return invert(lambda x: cdf(x, a), 1 - q, lo=lo, hi=hi, positive=True)


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": lambda x, a: mp.log(cdf(x, a)),
    "sf": sf,
    "logsf": lambda x, a: mp.log(sf(x, a)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(a):
    """x = 1/y for y around the igam/igamc regime switches (y = 1, y = a,
    a +- k*sqrt(a), y = a/2, 2a) and the mode 1/(a+1)."""
    s = np.sqrt(a)
    ys = np.array([
        a * 0.5, a * 0.9, a * 0.99, a, a * 1.01, a * 1.1, a * 2.0, a * 10.0,
        a + 1.0, a - s, a + s, a - 3 * s, a + 3 * s, a - 5 * s, a - 8 * s,
        a - 12 * s, a + 10 * s, 1.0 - 1e-8, 1.0, 1.0 + 1e-8, 0.5, 2.0,
        710.0, 745.0, 1e-8,
    ], dtype=float)
    ys = ys[np.isfinite(ys) & (ys > 0)]
    return 1.0 / ys


def q_points(a):
    return np.array([1e-17, 1e-16, 1e-8, 0.5 - 1e-16, 0.5, 0.5 + 1e-16,
                     1 - 1e-16, 1 - 2e-16, 1 - 1e-13, 1 - 1e-9])
