"""mpmath reference for scipy.stats.chi (standard form, loc=0, scale=1).

Parametrization from the docstring pdf (shape df = k > 0):

    f(x; k) = x**(k-1) exp(-x**2/2) / (2**(k/2-1) Gamma(k/2)),   x >= 0
    F(x; k) = P(k/2, x**2/2)        (regularized lower incomplete gamma)
    S(x; k) = Q(k/2, x**2/2) = 1 - F

x**2/2 is formed in mpmath (exact for a double x), so the reference has no
overflow/underflow of x**2.

Speed (as in refs/gamma_ref.py): mp.gammainc(a, 0, y) is fast for y <~ a,
mp.gammainc(a, y, inf) for y >~ a; the other side is obtained as 1 - (...).
For a = df/2 >= ~1e7 mpmath's gammainc does not converge; the lower function
is then evaluated by the Kummer series summed to full convergence (the series
reference of scratch/gamma/gamma_a1e8_ref.py).

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

DIST = "chi"
DPS = 400

# 78: inventory example; 1 (halfnorm), 3 (maxwell), small df (x**2 underflow
# matters, cdf ~ x**df), tiny df (igami exp(./a)), large df (logpdf
# cancellation) and df = 2e6 (a = 1e6, inside the cephes igam 2000-term
# window of reports/gamma.md F1).
PARAMS = [(78.0,), (1.0,), (3.0,), (1e-3,), (0.1,), (1e3,), (2e6,)]


def _kummer_lower(a, y):
    ax = mp.exp(a * mp.log(y) - y - mp.loggamma(a + 1))
    r, c, s = a, mp.mpf(1), mp.mpf(1)
    tol = mp.mpf(10) ** (-mp.mp.dps - 10)
    for _ in range(50_000_000):
        r += 1
        c *= y / r
        s += c
        if c < s * tol:
            break
    return s * ax


def _upper_side(a, y):
    # the upper function is slow for small a; for a < 50 the lower one is
    # fast up to y ~ 800, and 1 - P keeps >= 50 digits wherever Q is normal
    return y >= a and (a >= 50 or y > 800)


def _P(a, y):
    if y == 0:
        return mp.mpf(0)
    if _upper_side(a, y):
        return 1 - mp.gammainc(a, y, mp.inf, regularized=True)
    if a >= 1e7:
        return _kummer_lower(a, y)
    return mp.gammainc(a, 0, y, regularized=True)


def _Q(a, y):
    if y == 0:
        return mp.mpf(1)
    if _upper_side(a, y):
        return mp.gammainc(a, y, mp.inf, regularized=True)
    return 1 - _P(a, y)


def logpdf(x, df):
    x = mp.mpf(x)
    df = mp.mpf(df)
    if x == 0:
        if df == 1:
            return mp.log(mp.sqrt(2 / mp.pi))
        return mp.inf if df < 1 else -mp.inf
    return (mp.log(2) - df / 2 * mp.log(2) - mp.loggamma(df / 2)
            + (df - 1) * mp.log(x) - x * x / 2)


def pdf(x, df):
    return mp.exp(logpdf(x, df))


def cdf(x, df):
    x = mp.mpf(x)
    return _P(mp.mpf(df) / 2, x * x / 2)


def sf(x, df):
    x = mp.mpf(x)
    return _Q(mp.mpf(df) / 2, x * x / 2)


def _seed(method, q, df):
    from scipy import stats
    with np.errstate(all="ignore"):
        v = float(getattr(stats.chi, method)(q, df))
    if not (np.isfinite(v) and v > 0):
        return None, None
    return v / 4, v * 4


def ppf(q, df):
    lo, hi = _seed("ppf", float(q), float(df))
    return invert(lambda x: cdf(x, df), q, lo=lo, hi=hi, positive=True)


def isf(q, df):
    lo, hi = _seed("isf", float(q), float(df))
    return invert(lambda x: cdf(x, df), 1 - q, lo=lo, hi=hi, positive=True)


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": lambda x, df: mp.log(cdf(x, df)),
    "sf": sf,
    "logsf": lambda x, df: mp.log(sf(x, df)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(df):
    """Mode/bulk, the cephes igam windows in the gamma variable y = x**2/2
    (y = a + k*sqrt(a) <=> x = sqrt(df + k*sqrt(2*df))), and the x**2
    overflow (x > 1.34e154) / underflow (x < 1.5e-154 subnormal,
    x < 1e-162 zero) edges."""
    pts = [np.sqrt(max(df - 1, 0.0)), np.sqrt(df), 0.5, 1.0, 2.0, 5.0]
    for k in (-30, -15, -10, -7, -5, -4.6, -4.4, -3, -1, 1, 3, 4.4, 4.6, 5,
              7, 10, 15, 30):
        v = df + k * np.sqrt(2 * df)
        if v > 0:
            pts.append(np.sqrt(v))
    pts += [1.3e154, 1.34e154, 1.35e154, 1.5e154, 1.8e154, 1.9e154,
            2e-154, 1e-154, 1e-155, 1e-158, 1e-160, 1e-162, 1e-163, 1e-170]
    a = np.array(pts, dtype=float)
    return a[np.isfinite(a) & (a > 0)]


def q_points(df):
    return np.array([1e-17, 1e-16, 1e-8, 0.5 - 1e-16, 0.5, 0.5 + 1e-16,
                     1 - 1e-16, 1 - 2e-16, 1 - 1e-13, 1 - 1e-9])
