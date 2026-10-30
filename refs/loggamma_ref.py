"""mpmath reference for scipy.stats.loggamma (standard form, loc=0, scale=1).

Parametrization from the docstring pdf:

    f(x; c) = exp(c*x - exp(x)) / Gamma(c),     x real, c > 0

i.e. X = log(Y) with Y ~ gamma(c).  Hence

    F(x; c) = P(c, exp(x)),   S(x; c) = Q(c, exp(x)) = 1 - F.

Speed (mpmath, dps=400): the lower regularized gammainc is fast for z <~ c,
the upper one for z >~ c (and the lower one raises NoConvergence for large c
with z > c), so whichever is on its fast side is evaluated and the other is
obtained by subtraction from 1 (exact enough at 400 digits).
ppf/isf by inversion of the cdf (accuracy.invert, x unrestricted in sign).
"""
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".claude", "skills",
                                "dist-accuracy-review", "scripts"))
from accuracy import invert  # noqa: E402

DIST = "loggamma"
DPS = 400

# inventory example; tiny shapes (heavy left tail, ppf branch g < XMIN);
# c = 1; moderate; large shapes (cancellation in logpdf, gammainc series).
PARAMS = [(0.4141193182605212,), (1e-10,), (1e-3,), (1.0,), (10.5,),
          (1e3,), (1e6,)]


def _upper(x, c):
    """True if the upper incomplete gamma is the fast/robust choice."""
    return mp.exp(x) >= c


_HUGE = mp.mpf(10) ** 5


def _far_right(x, c):
    """exp(x) - c*x > 1e5: sf, pdf < exp(-1e5 + ...), far below the double
    range. mpmath's exp recurses for exp(-exp(1e300)), so such points get
    the leading asymptotic term, capped (not measured: not a normal double)."""
    return x > 20 and mp.exp(x) - c * x > _HUGE + abs(mp.loggamma(c))


def pdf(x, c):
    if _far_right(x, c):
        return mp.exp(-_HUGE)
    return mp.exp(c * x - mp.exp(x) - mp.loggamma(c))


def logpdf(x, c):
    return c * x - mp.exp(x) - mp.loggamma(c)


def cdf(x, c):
    if _far_right(x, c):
        return 1 - mp.exp(-_HUGE)
    z = mp.exp(x)
    if z >= c:
        return 1 - mp.gammainc(c, z, mp.inf, regularized=True)
    return mp.gammainc(c, 0, z, regularized=True)


def sf(x, c):
    if _far_right(x, c):
        return mp.exp(-_HUGE)
    z = mp.exp(x)
    if z >= c:
        return mp.gammainc(c, z, mp.inf, regularized=True)
    return 1 - mp.gammainc(c, 0, z, regularized=True)


def _seed(method, q, c):
    from scipy import stats
    with np.errstate(all="ignore"):
        v = float(getattr(stats.loggamma, method)(q, c))
    if not np.isfinite(v):
        return None, None
    w = max(abs(v) * 1e-3, 1e-3 / np.sqrt(c) if c > 1 else 1e-3)
    return v - w, v + w


def ppf(q, c):
    lo, hi = _seed("ppf", float(q), float(c))
    return invert(lambda x: cdf(x, c), q, lo=lo, hi=hi, positive=False)


def isf(q, c):
    lo, hi = _seed("isf", float(q), float(c))
    return invert(lambda x: cdf(x, c), 1 - q, lo=lo, hi=hi, positive=False)


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": lambda x, c: mp.log(cdf(x, c)),
    "sf": sf,
    "logsf": lambda x, c: mp.log(sf(x, c)),
    "ppf": ppf,
    "isf": isf,
}

_LOGXMIN = float(np.log(np.finfo(float).tiny))


def x_points(c):
    """Mode region log(c) +- k/sqrt(c), both sides of the x < _LOGXMIN
    branch, the left tail where c*x ~ -700 (cdf still normal), x near 0
    and the right tail where sf approaches the underflow limit."""
    lc = np.log(c)
    s = 1 / np.sqrt(c)
    k = np.array([-10, -6, -4.6, -4, -3, -2, -1, -0.3, 0, 0.3, 1, 2, 3, 4,
                  6, 10, 20, 30])
    mode = lc + k * s
    edges = np.array([_LOGXMIN, np.nextafter(_LOGXMIN, -np.inf),
                      np.nextafter(_LOGXMIN, np.inf), _LOGXMIN - 1,
                      _LOGXMIN + 1, -745.0, -720.0, -700.0, -1e-8, 1e-8,
                      -1e-3, 1e-3, 0.5, 3.0, 5.0, 6.0, 6.4, 6.5, 6.55])
    tail = np.array([-1.0, -10.0, -100.0, -300.0, -700.0]) / c
    right = np.log(c + np.sqrt(c) * np.array([20.0, 30.0, 35.0]) + 700)
    pts = np.concatenate([mode, edges, tail, right])
    return pts[np.isfinite(pts)]


def q_points(c):
    return np.array([1e-300, 1e-200, 1e-17, 1e-8, 0.3, 0.5, 0.7,
                     1 - 1e-9, 1 - 1e-13, 1 - 2e-16, 1 - 1e-16])
