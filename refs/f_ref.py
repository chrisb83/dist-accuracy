"""mpmath reference for scipy.stats.f (standard form, loc=0, scale=1).

Parametrization from the docstring pdf (n = dfn, m = dfd):

    f(x; n, m) = m**(m/2) n**(n/2) x**(n/2-1) / ((m + n x)**((n+m)/2) B(n/2, m/2))
    F(x; n, m) = I_y(n/2, m/2)     with y  = n x / (m + n x)
    S(x; n, m) = I_yc(m/2, n/2)    with yc = m / (m + n x)

i.e. F is betaprime(n/2, m/2) evaluated at t = n x / m.  The incomplete beta
is evaluated with the continued fraction (modified Lentz) on its convergent
side, as in refs/betaprime_ref.py (mp.betainc is far too slow for large
shapes).  F and S are each computed where they are the small quantity.
ppf/isf invert F or S, choosing the side where the target is <= 1/2.
"""
import mpmath as mp

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".claude", "skills",
                                "dist-accuracy-review", "scripts"))
from accuracy import invert  # noqa: E402

DIST = "f"
DPS = 120

# inventory example; small dfn (pole at 0) and small dfd (heavy right tail);
# dfn = 2 (x**0 factor); large dfd (logpdf cancellation of m/2 log m vs
# (n+m)/2 log(m+nx)); large balanced shapes; huge dfn (n*x overflow).
PARAMS = [(29.0, 18.0), (1.0, 1.0), (0.01, 3.0), (2.0, 0.01), (5.0, 1e4),
          (1e3, 1e3), (1e6, 2.0)]
# (1e5, 3e5) is measured separately for pdf, logpdf, cdf, sf only
# (quantile references take ~5-30 s per point there):
#   --params "1e5,3e5" --methods pdf,logpdf,cdf,sf


def _logbeta(a, b):
    return mp.loggamma(a) + mp.loggamma(b) - mp.loggamma(a + b)


def _betacf(a, b, x):
    """Continued fraction for I_x(a,b) (modified Lentz, Numerical Recipes)."""
    tiny = mp.mpf(10) ** (-(mp.mp.dps + 50))
    eps = mp.mpf(10) ** (-(mp.mp.dps + 5))
    qab, qap, qam = a + b, a + 1, a - 1
    c = mp.mpf(1)
    d = 1 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1 / d
    h = d
    m = 1
    while True:
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        de = d * c
        h *= de
        if abs(de - 1) < eps:
            return h
        m += 1
        if m > 10**7:
            raise mp.NoConvergence("betacf")


def _lower(a, b, y, logy, logyc):
    """I_y(a, b) on the convergent side; logy, logyc = log(y), log(1-y)."""
    lp = a * logy + b * logyc - _logbeta(a, b) - mp.log(a)
    return mp.exp(lp) * _betacf(a, b, y)


def _cdf_sf(x, n, m):
    x, n, m = mp.mpf(x), mp.mpf(n), mp.mpf(m)
    if x <= 0:
        return mp.mpf(0), mp.mpf(1)
    if mp.isinf(x):
        return mp.mpf(1), mp.mpf(0)
    a, b = n / 2, m / 2
    t = n * x / m
    l1p = mp.log1p(t)
    logy = mp.log(t) - l1p        # log(t/(1+t)) = log(n x/(m + n x))
    logyc = -l1p                  # log(m/(m + n x))
    y = t / (1 + t)
    yc = 1 / (1 + t)
    if y < (a + 1) / (a + b + 2):
        F = _lower(a, b, y, logy, logyc)
        return F, 1 - F
    S = _lower(b, a, yc, logyc, logy)
    return 1 - S, S


def logpdf(x, n, m):
    x, n, m = mp.mpf(x), mp.mpf(n), mp.mpf(m)
    return (m / 2 * mp.log(m) + n / 2 * mp.log(n) + (n / 2 - 1) * mp.log(x)
            - (n + m) / 2 * mp.log(m + n * x) - _logbeta(n / 2, m / 2))


def pdf(x, n, m):
    x = mp.mpf(x)
    if x == 0:
        if n == 2:
            return mp.mpf(1)
        return mp.inf if n < 2 else mp.mpf(0)
    return mp.exp(logpdf(x, n, m))


def cdf(x, n, m):
    return _cdf_sf(x, n, m)[0]


def sf(x, n, m):
    return _cdf_sf(x, n, m)[1]


def _quantile(p, pc, n, m):
    """x with F(x) = p, S(x) = pc (p + pc = 1); invert the smaller side."""
    if p <= mp.mpf(0.5):
        return invert(lambda x: cdf(x, n, m), p, positive=True)
    return invert(lambda x: sf(x, n, m), pc, increasing=False, positive=True)


def ppf(q, n, m):
    q = mp.mpf(q)
    return _quantile(q, 1 - q, n, m)


def isf(q, n, m):
    q = mp.mpf(q)
    return _quantile(1 - q, q, n, m)


def logcdf(x, n, m):
    F, S = _cdf_sf(x, n, m)
    return mp.log(F) if F < mp.mpf(0.5) else mp.log1p(-S)


def logsf(x, n, m):
    F, S = _cdf_sf(x, n, m)
    return mp.log(S) if S < mp.mpf(0.5) else mp.log1p(-F)


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


def x_points(n, m):
    # bulk around the mean/mode (~1, relative sd ~ sqrt(2/n + 2/m)),
    # n*x/m = 1 (incomplete-beta side switch), n*x ~ m*eps (log(m + n*x)
    # rounding), and n*x near overflow (x ~ 1.8e308/n).
    import numpy as np
    n, m = float(n), float(m)
    pts = [0.5, 1 - 1e-12, 1.0, 1 + 1e-12, 2.0, m / n, m / n * 1e-16,
           m / n * 1e-10, 1e300 / n, 1.7e308 / n, 1e308]
    s = np.sqrt(2 / n + 2 / m)
    if s < 0.5:
        pts += [np.exp(k * s) for k in (-10, -5, -2, -1, -0.3, 0.3, 1, 2, 5, 10)]
    pts = np.array(pts)
    return pts[(pts > 0) & np.isfinite(pts)]
