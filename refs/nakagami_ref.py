"""mpmath reference for scipy.stats.nakagami (standard form, loc=0, scale=1).

Parametrization from the docstring pdf (shape nu > 0):

    f(x; nu) = 2 nu**nu / Gamma(nu) * x**(2 nu - 1) * exp(-nu x**2),  x >= 0
    F(x; nu) = P(nu, nu x**2)   (regularized lower incomplete gamma)
    S(x; nu) = Q(nu, nu x**2) = 1 - F

nu*x**2 is formed in mpmath (exact for a double x), so the reference has no
overflow/underflow of x**2.

Speed (as refs/chi_ref.py): mp.gammainc(a, 0, y) is fast for y <~ a,
mp.gammainc(a, y, inf) for y >~ a; the other side is 1 - (...).
ppf/isf come from the cdf with accuracy.invert (seeded from SciPy only to
speed up the bracket search).
"""
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".claude", "skills",
                                "dist-accuracy-review", "scripts"))
from accuracy import invert  # noqa: E402

DIST = "nakagami"
DPS = 400

# 4.967...: inventory example; 0.5 (halfnorm), 1.5 (maxwell up to scale),
# small nu (nu*x*x underflow, quantile underflow), tiny nu (igami exp(./a)),
# large nu (logpdf cancellation) and nu = 1e6 (inside the cephes igam
# 2000-term window of reports/gamma.md F1).
PARAMS = [(4.967379486666624,), (0.5,), (1.5,), (1e-3,), (0.05,), (1e3,),
          (1e6,)]


def _upper_side(a, y):
    return y >= a and (a >= 50 or y > 800)


def _P(a, y):
    if y == 0:
        return mp.mpf(0)
    if _upper_side(a, y):
        return 1 - mp.gammainc(a, y, mp.inf, regularized=True)
    return mp.gammainc(a, 0, y, regularized=True)


def _Q(a, y):
    if y == 0:
        return mp.mpf(1)
    if _upper_side(a, y):
        return mp.gammainc(a, y, mp.inf, regularized=True)
    return 1 - _P(a, y)


def logpdf(x, nu):
    x = mp.mpf(x)
    nu = mp.mpf(nu)
    if x == 0:
        if nu == mp.mpf(1) / 2:
            return mp.log(2 * mp.sqrt(nu) / mp.gamma(nu))
        return mp.inf if nu < mp.mpf(1) / 2 else -mp.inf
    return (mp.log(2) + nu * mp.log(nu) - mp.loggamma(nu)
            + (2 * nu - 1) * mp.log(x) - nu * x * x)


def pdf(x, nu):
    return mp.exp(logpdf(x, nu))


def cdf(x, nu):
    x = mp.mpf(x)
    nu = mp.mpf(nu)
    return _P(nu, nu * x * x)


def sf(x, nu):
    x = mp.mpf(x)
    nu = mp.mpf(nu)
    return _Q(nu, nu * x * x)


def _seed(method, q, nu):
    from scipy import stats
    with np.errstate(all="ignore"):
        v = float(getattr(stats.nakagami, method)(q, nu))
    if not (np.isfinite(v) and v > 1e-300):
        return None, None
    return v / 4, v * 4


def ppf(q, nu):
    lo, hi = _seed("ppf", float(q), float(nu))
    return invert(lambda x: cdf(x, nu), q, lo=lo, hi=hi, positive=True)


def isf(q, nu):
    lo, hi = _seed("isf", float(q), float(nu))
    return invert(lambda x: cdf(x, nu), 1 - q, lo=lo, hi=hi, positive=True)


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": lambda x, nu: mp.log(cdf(x, nu)),
    "sf": sf,
    "logsf": lambda x, nu: mp.log(sf(x, nu)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(nu):
    """Mode/bulk, the cephes igam windows y = a + k*sqrt(a) (a = nu,
    y = nu*x**2 <=> x = sqrt(1 + k/sqrt(nu))), the x**2 overflow edges
    (x > 1.34e154; -nu*x**2 stays finite up to sqrt(1.8e308/nu)) and the
    nu*x*x underflow edges (subnormal below sqrt(2.2e-308/nu))."""
    pts = [np.sqrt(max(1 - 0.5 / nu, 0.0)), 1.0, 0.5, 2.0, 5.0]
    for k in (-30, -15, -10, -7, -5, -4.6, -4.4, -3, -1, 1, 3, 4.4, 4.6, 5,
              7, 10, 15, 30):
        v = 1 + k / np.sqrt(nu)
        if v > 0:
            pts.append(np.sqrt(v))
    pts += [1.3e154, 1.34e154, 1.35e154, 1.5e154, 1.8e154]
    with np.errstate(over="ignore"):
        big = np.sqrt(1.8e308 / nu)
    pts += [big * 0.5, big * 0.9, big * 0.99]
    small = np.sqrt(2.2e-308 / nu)
    pts += [small * f for f in (10, 1.01, 0.5, 1e-3, 1e-8, 1e-12)]
    a = np.array(pts, dtype=float)
    return a[np.isfinite(a) & (a > 0)]


def q_points(nu):
    return np.array([1e-17, 1e-16, 1e-8, 0.5 - 1e-16, 0.5, 0.5 + 1e-16,
                     1 - 1e-16, 1 - 2e-16, 1 - 1e-13, 1 - 1e-9])
