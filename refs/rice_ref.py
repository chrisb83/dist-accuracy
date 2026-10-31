"""mpmath reference for scipy.stats.rice (standard form, loc=0, scale=1).

Docstring pdf:  f(x, b) = x exp(-(x**2 + b**2)/2) I_0(x b),  x >= 0, b >= 0.
rice(b) is the square root of ncx2(df=2, nc=b**2), so with y = x**2/2,
mu = b**2/2 and Poisson weights w_j = exp(-mu) mu**j / j!:
    cdf(x) = sum_j w_j P(1 + j, y),   sf(x) = sum_j w_j Q(1 + j, y)
(P, Q regularized incomplete gamma).  Evaluated with positive terms only,
as in refs/ncx2_ref.py (a = df/2 = 1):
  * cdf: P_J once from mp.gammainc, then P_j = P_{j+1} + d_j backwards,
    d_j = y**(1+j) exp(-y) / Gamma(2+j); truncation bound P_{J+1} * tail(J).
  * sf: Q_0 = exp(-y), then Q_{j+1} = Q_j + d_j forwards.
b = 0 is the Rayleigh distribution (closed forms).
ppf/isf invert the cdf or the sf, whichever side has target <= 1/2.
"""
import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".claude", "skills",
                                "dist-accuracy-review", "scripts"))
from accuracy import invert  # noqa: E402

DIST = "rice"
DPS = 100

# inventory example; tiny b (nc = 1e-10: Boost Ding cdf path, ncx2 F7 band);
# b = 0 (Rayleigh limit, chndtr with nc = 0); b = 3 (nc = 9, Ding path);
# b = 15 and 40 (nc = 225 and 1600 >= 200: Boost cdf series from the
# Poisson mode, ncx2 F1/F6).
PARAMS = [(0.7749725210111873,), (1e-5,), (0.0,), (3.0,), (15.0,), (40.0,)]


def _eps():
    return mp.mpf(10) ** (-(mp.mp.dps + 3))


def _logw(j, mu):
    return -mu + j * mp.log(mu) - mp.loggamma(j + 1)


def _logd(j, y):
    return (1 + j) * mp.log(y) - y - mp.loggamma(j + 2)


def _ptail_bound(J, mu):
    return mp.exp(_logw(J + 1, mu)) / (1 - mu / (J + 2))


def cdf(x, b):
    x, b = mp.mpf(x), mp.mpf(b)
    if x <= 0:
        return mp.mpf(0)
    y, mu = x * x / 2, b * b / 2
    if mu == 0:
        return -mp.expm1(-y)
    J = int(mp.ceil(mu + 30 * mp.sqrt(mu) + 60))
    while True:
        P = mp.gammainc(1 + J, 0, y, regularized=True)
        d = mp.exp(_logd(J - 1, y))
        w = mp.exp(_logw(J, mu))
        total = w * P
        for j in range(J - 1, -1, -1):
            P = P + d
            w = w * (j + 1) / mu
            total += w * P
            if j > 0:
                d = d * (1 + j) / y
        PJ1 = mp.gammainc(2 + J, 0, y, regularized=True)
        if PJ1 * _ptail_bound(J, mu) <= _eps() * total:
            return total
        J = int(J * 1.5) + 10


def sf(x, b):
    x, b = mp.mpf(x), mp.mpf(b)
    if x <= 0:
        return mp.mpf(1)
    y, mu = x * x / 2, b * b / 2
    if mu == 0:
        return mp.exp(-y)
    if (x - b) > 60 and x * b > 2e5:
        # far right tail: sf < ~exp(-(x-b)**2/2) * sqrt(x/b) < exp(-1800)
        # (below the double range; the harness skips the point)
        return mp.exp(-(x - b) ** 2 / 2) * mp.sqrt(x / b)
    Q = mp.exp(-y)
    d = mp.exp(_logd(0, y))
    w = mp.exp(-mu)
    total = w * Q
    j = 0
    eps = _eps()
    while True:
        Q = Q + d
        d = d * y / (j + 2)
        w = w * mu / (j + 1)
        j += 1
        total += w * Q
        if j + 2 > mu and w * mu / (j + 1) / (1 - mu / (j + 2)) <= eps * total:
            return total
        if j > 3 * 10**5:
            raise mp.NoConvergence("sf series")


def pdf(x, b):
    x, b = mp.mpf(x), mp.mpf(b)
    return x * mp.exp(-(x * x + b * b) / 2) * mp.besseli(0, x * b)


def logpdf(x, b):
    return mp.log(pdf(x, b))


def _quantile(p, pc, b):
    if p <= mp.mpf(0.5):
        return invert(lambda x: cdf(x, b), p, positive=True, digits=30)
    return invert(lambda x: sf(x, b), pc, increasing=False, positive=True,
                  digits=30)


def ppf(q, b):
    q = mp.mpf(q)
    return _quantile(q, 1 - q, b)


def isf(q, b):
    q = mp.mpf(q)
    return _quantile(1 - q, q, b)


def logcdf(x, b):
    F = cdf(x, b)
    return mp.log(F) if F < mp.mpf(0.5) else mp.log1p(-sf(x, b))


def logsf(x, b):
    S = sf(x, b)
    return mp.log(S) if S < mp.mpf(0.5) else mp.log1p(-cdf(x, b))


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": logcdf,
    "sf": sf,
    "logsf": logsf,
    "ppf": ppf,
    "isf": isf,
}


def x_points(b):
    # ncx2 F7 band: (x**2/2)**2 subnormal (x ~ 1e-81 ... 2e-77); x**2
    # underflow edge (1.5e-154); Boost cdf switch x**2 = 2 + b**2; around
    # the mode x ~ b and right tail b + 10 ... b + 38.
    import numpy as np
    b = float(b)
    m = np.sqrt(2 + b * b)
    pts = [1e-81, 1e-80, 3e-80, 1e-79, 1e-78, 1e-77, 1e-76, 1e-155, 2e-154,
           m * (1 - 1e-6), m, m * (1 + 1e-6), b + 5, b + 10, b + 20, b + 30,
           b + 36, b + 38]
    if b > 12:
        pts += [b - 10, b - 5, b / 2, 1.0, 0.1]
    pts = np.array(pts)
    return pts[(pts > 0) & np.isfinite(pts)]
