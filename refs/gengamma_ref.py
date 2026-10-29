"""mpmath reference for scipy.stats.gengamma (standard form, loc=0, scale=1).

Parametrization from the docstring pdf:

    f(x; a, c) = |c| x**(c*a - 1) exp(-x**c) / Gamma(a),   x > 0, a > 0, c != 0

With y = x**c and P, Q the regularized lower/upper incomplete gamma functions:

    c > 0:  F(x) = P(a, y),  S(x) = Q(a, y)
    c < 0:  F(x) = Q(a, y),  S(x) = P(a, y)

Speed: mp.gammainc(a, y, inf) is slow (~1 s) for moderate y, the lower
mp.gammainc(a, 0, y) is fast up to large y. So P is the lower function for
y < a + 300 and 1 - Q (upper function) beyond, where Q < ~1e-130 must keep
its relative accuracy; Q = 1 - P likewise.
Quantiles: y = P^{-1}(a, p) is found with accuracy.invert (log-y space), then
x = y**(1/c); p = q (ppf, c > 0), 1 - q (ppf, c < 0), and the reverse for isf.
"""
import mpmath as mp
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".claude", "skills",
                                "dist-accuracy-review", "scripts"))
from accuracy import invert  # noqa: E402

DIST = "gengamma"
DPS = 400

# inventory examples (both signs of c); a << 1 with large |c| (x**c and the
# incomplete-gamma inverse underflow while the result is normal); a = 1/3,
# c = 3 (c*a - 1 rounds to 0, halfgennorm(3)); large a (logpdf cancellation);
# tiny c.
PARAMS = [(4.4162385429431925, 3.119309167924276),
          (4.4162385429431925, -3.119309167924276),
          (1e-3, 10.0), (1e-3, -10.0), (1.0 / 3.0, 3.0), (1e3, 2.0),
          (2.0, 1e-3), (0.5, -0.5)]


def _P(a, y):
    if y == 0:
        return mp.mpf(0)
    if y == mp.inf:
        return mp.mpf(1)
    if y >= a + 300:
        return 1 - mp.gammainc(a, y, mp.inf, regularized=True)
    return mp.gammainc(a, 0, y, regularized=True)


def _Q(a, y):
    if y == 0:
        return mp.mpf(1)
    if y == mp.inf:
        return mp.mpf(0)
    if y >= a + 300:
        return mp.gammainc(a, y, mp.inf, regularized=True)
    return 1 - mp.gammainc(a, 0, y, regularized=True)


def logpdf(x, a, c):
    if x == 0:
        return mp.mpf("-inf")  # harness only measures normal results
    return (mp.log(abs(c)) + (c * a - 1) * mp.log(x) - x ** c
            - mp.loggamma(a))


def pdf(x, a, c):
    if x == 0:
        return mp.mpf(0)
    return mp.exp(logpdf(x, a, c))


def cdf(x, a, c):
    if x == 0:
        return mp.mpf(0)
    y = x ** c
    return _P(a, y) if c > 0 else _Q(a, y)


def sf(x, a, c):
    if x == 0:
        return mp.mpf(1)
    y = x ** c
    return _Q(a, y) if c > 0 else _P(a, y)


def _invP(a, p):
    from scipy import special
    with np.errstate(all="ignore"):
        v = float(special.gammaincinv(float(a), float(p)))
    lo = hi = None
    if np.isfinite(v) and v > 0:
        lo, hi = v / 4, v * 4
    return invert(lambda y: _P(a, y), p, lo=lo, hi=hi, positive=True)


def ppf(q, a, c):
    p = q if c > 0 else 1 - q
    return _invP(a, p) ** (1 / c)


def isf(q, a, c):
    p = 1 - q if c > 0 else q
    return _invP(a, p) ** (1 / c)


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": lambda x, a, c: mp.log(cdf(x, a, c)),
    "sf": sf,
    "logsf": lambda x, a, c: mp.log(sf(x, a, c)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(a, c):
    """Bulk of y = x**c around a, and where x**c under/overflows."""
    s = np.sqrt(a)
    ys = np.array([a * 0.5, a, a * 2.0, a - s, a + s, a - 3 * s, a + 3 * s,
                   a + 10 * s, a * 0.9, a * 1.1, 1e-3 * a, 1e-8])
    ys = ys[ys > 0]
    with np.errstate(all="ignore"):
        xs = list(ys ** (1.0 / c))
        for L in (-745.0, -708.0, -700.0, 700.0, 709.0, 710.0):
            xs.append(np.exp(L / c))
        xs += [1.0 - 1e-8, 1.0, 1.0 + 1e-8, 0.5, 2.0]
    xs = np.array(xs, float)
    return xs[np.isfinite(xs) & (xs > 0)]


def q_points(a, c):
    return np.array([1e-17, 1e-8, 1e-3, 0.1, 0.5, 0.9, 1 - 1e-3,
                     1 - 1e-9, 1 - 1e-16])
