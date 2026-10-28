"""mpmath reference for scipy.stats.betaprime (standard form, loc=0, scale=1).

Parametrization from the docstring pdf:

    f(x; a, b) = x**(a-1) * (1+x)**(-a-b) / B(a, b),   x > 0, a, b > 0
    F(x; a, b) = I_y(a, b)     with y = x/(1+x)
    S(x; a, b) = I_yc(b, a)    with yc = 1 - y = 1/(1+x)

The incomplete beta is evaluated with the continued fraction (modified Lentz)
on its convergent side, as in refs/beta_ref.py (mp.betainc is far too slow for
large shapes).  y and yc are passed separately (both exact at working
precision), so x = 1e300 (yc = 1e-300) keeps full accuracy.  F and S are each
computed where they are the small quantity.  ppf/isf invert F or S in log x,
choosing the side where the target is <= 1/2, so no 1 - q is needed for the
small tail.
"""
import mpmath as mp

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".claude", "skills",
                                "dist-accuracy-review", "scripts"))
from accuracy import invert  # noqa: E402

DIST = "betaprime"
DPS = 120

# inventory example; a < 1 < b and (2, 10) where Boost ibeta_inv (beta F1)
# fails for small q; small b (fat right tail) and small a; large a with
# small b (ppf near r = 0.9999 in the bulk, logpdf cancellation at large x);
# large balanced shapes (betaln / logpdf cancellation near the mode).
PARAMS = [(5.0, 6.0), (0.5, 2.0), (2.0, 10.0), (2.0, 1e-3), (1e-3, 2.0),
          (1e3, 0.5), (300.0, 1e3)]
# (1e5, 3e5) is measured separately for pdf, logpdf, cdf, sf only
# (quantile references take ~10-25 s per point there):
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


def _cdf_sf(x, a, b):
    x, a, b = mp.mpf(x), mp.mpf(a), mp.mpf(b)
    if x <= 0:
        return mp.mpf(0), mp.mpf(1)
    if mp.isinf(x):
        return mp.mpf(1), mp.mpf(0)
    l1p = mp.log1p(x)
    logy = mp.log(x) - l1p        # log(x/(1+x))
    logyc = -l1p                  # log(1/(1+x))
    y = x / (1 + x)
    yc = 1 / (1 + x)
    if y < (a + 1) / (a + b + 2):
        F = _lower(a, b, y, logy, logyc)
        return F, 1 - F
    S = _lower(b, a, yc, logyc, logy)
    return 1 - S, S


def logpdf(x, a, b):
    x = mp.mpf(x)
    return (a - 1) * mp.log(x) - (a + b) * mp.log1p(x) - _logbeta(a, b)


def pdf(x, a, b):
    x = mp.mpf(x)
    if x == 0:
        if a == 1:
            return 1 / mp.beta(a, b)
        return mp.inf if a < 1 else mp.mpf(0)
    return mp.exp(logpdf(x, a, b))


def cdf(x, a, b):
    return _cdf_sf(x, a, b)[0]


def sf(x, a, b):
    return _cdf_sf(x, a, b)[1]


def _F(x, a, b):
    return _cdf_sf(x, a, b)[0]


def _S(x, a, b):
    return _cdf_sf(x, a, b)[1]


def _quantile(p, pc, a, b):
    """x with F(x) = p, S(x) = pc (p + pc = 1); invert the smaller side."""
    if p <= mp.mpf(0.5):
        return invert(lambda x: _F(x, a, b), p, positive=True)
    return invert(lambda x: _S(x, a, b), pc, increasing=False, positive=True)


def ppf(q, a, b):
    q = mp.mpf(q)
    return _quantile(q, 1 - q, a, b)


def isf(q, a, b):
    q = mp.mpf(q)
    return _quantile(1 - q, q, a, b)


def logcdf(x, a, b):
    F, S = _cdf_sf(x, a, b)
    return mp.log(F) if F < mp.mpf(0.5) else mp.log1p(-S)


def logsf(x, a, b):
    F, S = _cdf_sf(x, a, b)
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


def x_points(a, b):
    # branch at x = 1 (cdf/sf), x = 9999 (r = 0.9999, ppf switch), mode,
    # a/b (bulk for large a), and a few points around the bulk.
    import numpy as np
    a, b = float(a), float(b)
    pts = [0.5, 1 - 1e-12, 1.0, 1 + 1e-12, 2.0, 9998.0, 9999.0, 10000.0]
    m = a / max(b, 1e-300)
    for k in (0.01, 0.1, 0.3, 0.5, 0.9, 1.0, 1.1, 2.0, 3.0, 10.0, 100.0):
        pts.append(m * k)
    if a > 1:
        pts.append((a - 1) / (b + 1))
    if a > 100 and b > 100:
        # large balanced shapes: mean +- k sd (relative sd ~ sqrt(1/a + 1/b))
        s = np.sqrt(1 / a + 1 / b)
        pts += [m * np.exp(k * s) for k in (-10, -5, -2, -1, -0.3, 0.3, 1, 2, 5, 10)]
    pts = np.array(pts)
    return pts[(pts > 0) & np.isfinite(pts)]


def q_points(a, b):
    # q where the ppf switches branch (beta r = 0.9999, i.e. x = 9999)
    import numpy as np
    from scipy import stats
    qs = []
    for x in (9990.0, 9998.0, 9999.0, 10000.0, 10010.0):
        qs.append(float(stats.betaprime.cdf(x, a, b)))
    qs += [1e-9, 1e-12, 1e-14, 1e-16, 1e-50, 1e-101, 1e-120, 1e-200]
    qs = np.array(qs)
    return qs[(qs > 0) & (qs < 1)]
