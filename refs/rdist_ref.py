"""mpmath reference for scipy.stats.rdist (standard form, loc=0, scale=1).

Parametrization from the docstring pdf:

    f(x; c) = (1 - x**2)**(c/2 - 1) / B(1/2, c/2),   -1 <= x <= 1, c > 0

Equivalently X = 2*B - 1 with B ~ Beta(a, a), a = c/2, so with y = (1 + x)/2

    F(x; c) = I_y(a, a),      S(x; c) = F(-x; c)   (symmetry).

Speed: ``mp.betainc`` (hyp2f1) is far too slow for large a (minutes at
a = 5e5, 400 digits), so I_y(a, a) is computed with the classical continued
fraction (as in refs/beta_ref.py) on y <= 1/2, which is always the
convergent side for a = b, i.e. F(x) for x <= 0 and F(x) = 1 - F(-x) for
x > 0 (1 + x, 1 - x and -x are exact at 120 digits for every double x).
sf uses the symmetry S(x) = F(-x), so it never cancels.

Quantiles (Newton from a double guess, verified to |log F - log q| < 1e-100,
fallback ``invert``): for q <= 1/2 solve I_y(a, a) = q for y in (0, 1/2] (in log y, so
tiny y keeps full relative accuracy) and return 2*y - 1; for q > 1/2 use
ppf(q) = -ppf(1 - q) (1 - q is exact). Near q = 1/2 the root y is found to
~1e-60 relative, so x = 2*y - 1 keeps > 40 correct digits even for
|q - 1/2| = 5.5e-17. isf(q) = -ppf(q).

DPS = 120: the continued fraction needs O(sqrt(a)) terms for large a; 120
digits are plenty for every quantity measured (the harness checks
stability against higher precision).
"""
import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".claude", "skills",
                                "dist-accuracy-review", "scripts"))
from accuracy import invert  # noqa: E402

DIST = "rdist"
DPS = 120

# inventory example; c = 3 (semicircular); c < 2 and c < 1 (pdf -> inf at
# the ends); c = 10; large c (logpdf cancels terms of size ~c, x ~ 1/sqrt(c)
# so ppf = 2p - 1 cancels everywhere); tiny c (almost all mass at +-1).
# Tiny c is not among the first three sets (self-check round trips cannot
# resolve 1 - |x| ~ 1e-600).
PARAMS = [(1.6,), (3.0,), (10.0,), (0.5,), (1e3,), (1e6,), (1e-3,)]


def _logbeta(a, b):
    return mp.loggamma(a) + mp.loggamma(b) - mp.loggamma(a + b)


def _betacf(a, b, x):
    """Continued fraction for I_x(a,b) (modified Lentz)."""
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


def _I_low(y, a):
    """I_y(a, a) for mpf y in (0, 1/2] (convergent side of the CF)."""
    if y <= 0:
        return mp.mpf(0)
    if y >= mp.mpf(0.5):
        return mp.mpf(0.5)
    lp = a * mp.log(y) + a * mp.log1p(-y) - _logbeta(a, a) - mp.log(a)
    return mp.exp(lp) * _betacf(a, a, y)


def cdf(x, c):
    x, a = mp.mpf(x), mp.mpf(c) / 2
    if x <= -1:
        return mp.mpf(0)
    if x >= 1:
        return mp.mpf(1)
    if x <= 0:
        return _I_low((1 + x) / 2, a)
    return 1 - _I_low((1 - x) / 2, a)


def sf(x, c):
    return cdf(-mp.mpf(x), c)


def logpdf(x, c):
    x, c = mp.mpf(x), mp.mpf(c)
    return (c / 2 - 1) * mp.log((1 - x) * (1 + x)) - _logbeta(mp.mpf(0.5), c / 2)


def pdf(x, c):
    x, c = mp.mpf(x), mp.mpf(c)
    if abs(x) == 1:
        if c == 2:
            return mp.mpf(0.5)
        return mp.inf if c < 2 else mp.mpf(0)
    return mp.exp(logpdf(x, c))


def logcdf(x, c):
    x = mp.mpf(x)
    if x <= 0:
        return mp.log(cdf(x, c))
    return mp.log1p(-sf(x, c))


def logsf(x, c):
    return logcdf(-mp.mpf(x), c)


def _ppf_low(q, a):
    """x with F(x) = q for 0 < q <= 1/2."""
    if q == mp.mpf(0.5):
        return mp.mpf(0)
    y = _newton_low(q, a)
    if y is None:
        y = invert(lambda y: _I_low(y, a), q, lo=mp.mpf("1e-10"),
                   hi=mp.mpf("0.5"), positive=True)
    return 2 * y - 1


def _newton_low(q, a):
    """Newton on g(u) = log I_{e^u}(a, a) - log q, u = log y.

    Seeded with a double-precision guess (scipy.special.betaincinv, used only
    as a starting point); the result is accepted only if |g| < 1e-(dps-20)
    and the root lies in (0, 1/2]; otherwise None (the caller then falls back
    to the bracketing solver ``invert``)."""
    try:
        import numpy as np
        from scipy import special as sc
        with np.errstate(all="ignore"):
            y0 = float(sc.betaincinv(float(a), float(a), float(q)))
        if not np.isfinite(y0) or not (0 < y0 <= 0.5):
            return None
    except Exception:
        return None
    lq = mp.log(q)
    if abs(q - mp.mpf(0.5)) < mp.mpf("1e-6"):
        # the double seed rounds to 0.5 here; linearize at the median instead
        f_half = mp.exp((2 * a - 2) * mp.log(mp.mpf(0.5)) - _logbeta(a, a))
        y0 = mp.mpf(0.5) - (mp.mpf(0.5) - q) / f_half
    u = mp.log(mp.mpf(y0))
    tol = mp.mpf(10) ** (-(mp.mp.dps - 20))
    lB = _logbeta(a, a)
    for _ in range(40):
        y = mp.exp(u)
        if not (0 < y < mp.mpf(0.5)):
            return None
        F = _I_low(y, a)
        if F <= 0:
            return None
        g = mp.log(F) - lq
        if abs(g) < tol:
            return y
        # d/du log F = y f(y) / F, f(y) = y**(a-1) (1-y)**(a-1) / B(a, a)
        dlog = mp.exp(a * mp.log(y) + (a - 1) * mp.log1p(-y) - lB) / F
        u = u - g / dlog
    return None


def ppf(q, c):
    q, a = mp.mpf(q), mp.mpf(c) / 2
    if q <= mp.mpf(0.5):
        return _ppf_low(q, a)
    return -_ppf_low(1 - q, a)


def isf(q, c):
    return -ppf(q, c)


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


def x_points(c):
    # Both endpoints approached from inside (where (x+1)/2 rounds for x > 0.5),
    # tiny |x| (cdf near 1/2), and for large c the bulk at k standard
    # deviations (sd = 1/sqrt(c+1)), where logpdf cancels terms of size ~c.
    import numpy as np
    c = float(c)
    pts = [0.0, 1e-300, -1e-300, 1e-20, -1e-20, 1e-8, -1e-8, 0.5, -0.5,
           0.3, -0.3, 0.7, -0.7]
    for k in [1, 2, 3, 10, 1e4, 1e8]:
        pts += [1 - k * np.spacing(0.5), -1 + k * np.spacing(0.5)]
    for d in [1e-15, 1e-13, 1e-11, 1e-9, 1e-7, 1e-5, 1e-3, 1e-2, 0.1]:
        pts += [1 - d, -1 + d]
    s = 1 / np.sqrt(c + 1)
    for k in [0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 30]:
        if k * s < 1:
            pts += [k * s, -k * s]
    pts = np.array(pts)
    return pts[(pts > -1) & (pts < 1)]


def q_points(c):
    # near q = 1/2 (x = 2p - 1 cancels), tails, near 1.
    import numpy as np
    pts = [0.5, 0.25, 0.75, 0.4, 0.6]
    for e in [-300, -250, -200, -150, -100, -50, -30, -20, -16, -12, -8,
              -6, -4, -3, -2]:
        pts.append(10.0 ** e)
    for d in [1e-16, 1e-15, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
        pts += [1 - d, 0.5 + d, 0.5 - d]
    return np.array(pts)
