"""mpmath reference for scipy.stats.beta (standard form, loc=0, scale=1).

Parametrization from the docstring pdf:

    f(x; a, b) = x**(a-1) * (1-x)**(b-1) / B(a, b),   0 <= x <= 1, a, b > 0
    F(x; a, b) = I_x(a, b)                (regularized incomplete beta)
    S(x; a, b) = 1 - F = I_{1-x}(b, a)

Speed notes (mpmath): ``mp.betainc`` (via hyp2f1) takes seconds
to minutes for large shapes (a = b = 1e6), so the incomplete beta is computed
here with the classical continued fraction (Numerical Recipes ``betacf``,
modified Lentz) on its convergent side x < (a+1)/(a+b+2), and with the
symmetry I_x(a, b) = 1 - I_{1-x}(b, a) otherwise.  ``1 - x`` is exact at
400 digits for every double x.  ``_cdf_sf`` returns both F and S, each
computed on the side where it is the small quantity, so both have full
relative accuracy.  DPS = 120 (the continued fraction needs O(sqrt(a))
terms for large shapes near the mean; 400 digits made the ppf/isf runs for
a = 1e5 too slow).  isf inverts sf directly, so it needs no 1 - q.
Spot-checked against ``mp.betainc`` in
``scratch/beta/check_ref.py``.
"""
import mpmath as mp

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".claude", "skills",
                                "dist-accuracy-review", "scripts"))
from accuracy import invert  # noqa: E402

DIST = "beta"
DPS = 120

# inventory example; tiny shapes (pdf -> inf at both ends, cdf ~ x**a);
# a < 1 < b; a = 1 (xlogy(0, x)); moderate symmetric; large shapes where
# xlogy/xlog1py/betaln in _logpdf are huge and cancel; very unbalanced.
# (1e-3, 1e-3) is not among the first three sets (the harness's self-checks
# use those): its quantiles for q >~ 0.3 are 1 - 1e-222 and closer, which
# cannot be resolved at any practical precision; the reference returns 1,
# the correct double, but the cdf(ppf(q)) == q round trip cannot hold.
PARAMS = [(2.3098496451481823, 0.6268795430096368),
          (0.5, 5.0), (1.0, 3.0), (1e-3, 1e-3), (20.0, 50.0),
          (1e3, 1e3), (1e5, 3e5), (1e6, 2.0)]


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


def _lower(a, b, x):
    """I_x(a, b) for 0 < x < (a+1)/(a+b+2) (convergent side)."""
    lp = a * mp.log(x) + b * mp.log1p(-x) - _logbeta(a, b) - mp.log(a)
    return mp.exp(lp) * _betacf(a, b, x)


def _cdf_sf(x, a, b):
    x, a, b = mp.mpf(x), mp.mpf(a), mp.mpf(b)
    if x <= 0:
        return mp.mpf(0), mp.mpf(1)
    if x >= 1:
        return mp.mpf(1), mp.mpf(0)
    if x < (a + 1) / (a + b + 2):
        F = _lower(a, b, x)
        return F, 1 - F
    S = _lower(b, a, 1 - x)
    return 1 - S, S


def pdf(x, a, b):
    x = mp.mpf(x)
    if x == 0:
        if a == 1:
            return 1 / mp.beta(a, b)
        return mp.inf if a < 1 else mp.mpf(0)
    if x == 1:
        if b == 1:
            return 1 / mp.beta(a, b)
        return mp.inf if b < 1 else mp.mpf(0)
    return mp.exp(logpdf(x, a, b))


def logpdf(x, a, b):
    x = mp.mpf(x)
    if x == 0 or x == 1:
        return mp.log(pdf(x, a, b))
    return (a - 1) * mp.log(x) + (b - 1) * mp.log1p(-x) - _logbeta(a, b)


def cdf(x, a, b):
    return _cdf_sf(x, a, b)[0]


def sf(x, a, b):
    return _cdf_sf(x, a, b)[1]


def _I(y, a, b):
    """I_y(a, b) for an mpf y in (0, 1) (no rounding of 1 - y needed)."""
    return _cdf_sf_y(y, a, b)[0]


def _cdf_sf_y(y, a, b):
    if y <= 0:              # invert() may probe outside (0, 1) when
        return mp.mpf(0), mp.mpf(1)    # widening its bracket
    if y >= 1:
        return mp.mpf(1), mp.mpf(0)
    if y < (a + 1) / (a + b + 2):
        F = _lower(a, b, y)
        return F, 1 - F
    S = _lower(b, a, 1 - y)
    return 1 - S, S


def _quantile(p, a, b):
    """x with I_x(a, b) = p, where p = cdf target and pc = 1 - p.

    For p <= 1/2 solve I_x(a, b) = p for x.  Otherwise solve
    I_y(b, a) = 1 - p for y = 1 - x and return 1 - y: for tiny shapes
    1 - x can be far below 10**-dps (e.g. a = b = 1e-3, q = 0.7 gives
    1 - x ~ 1e-300), which x itself cannot resolve.
    """
    p = mp.mpf(p)
    if p <= mp.mpf(0.5):
        return invert(lambda x: _I(x, a, b), p, lo=mp.mpf("1e-10"),
                      hi=mp.mpf("0.5"), positive=True)
    y = invert(lambda y: _I(y, b, a), 1 - p, lo=mp.mpf("1e-10"),
               hi=mp.mpf("0.5"), positive=True)
    return 1 - y


def ppf(q, a, b):
    # q is a double <= 1 - 1.1e-16, so 1 - q is exact at 120 digits
    return _quantile(q, a, b)


def isf(q, a, b):
    # isf(q) = x with S(x) = q, i.e. I_y(b, a) = q for y = 1 - x
    q = mp.mpf(q)
    if q <= mp.mpf(0.5):
        y = invert(lambda y: _I(y, b, a), q, lo=mp.mpf("1e-10"),
                   hi=mp.mpf("0.5"), positive=True)
        return 1 - y
    return invert(lambda x: _I(x, a, b), 1 - q, lo=mp.mpf("1e-10"),
                  hi=mp.mpf("0.5"), positive=True)


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
    # mode, mean, the switch point of the continued fraction / of Boost's
    # ibeta (~ a/(a+b)), points a few standard deviations around the mean
    # (large shapes), tiny x where x**a nears the underflow limit, and the
    # smallest normal doubles (a <= 1 only).  Subnormal x is NOT included: scu._beta_pdf
    # raises OverflowError there, which aborts the harness (see report F1).
    import numpy as np
    a, b = float(a), float(b)
    m = a / (a + b)
    s = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    pts = [m, (a + 1) / (a + b + 2)]
    for k in (-20, -10, -5, -2, -1, -0.3, 0.3, 1, 2, 5, 10, 20):
        pts.append(m + k * s)
    if a > 1 and b > 1:
        pts.append((a - 1) / (a + b - 2))
    pts += [1e-300 ** (1 / max(a, 1)) * 10 ** k for k in (0, 1)]
    if a <= 1:
        # for a > 1 these also raise OverflowError in scu._beta_pdf (the
        # threshold grows with a: 1.7e-306 at a = 20, 1.7e-305 at a = 1e3)
        pts += [2.2250738585072014e-308, 3e-308, 1e-307]
    pts = np.array(pts)
    return pts[(pts > 0) & (pts < 1)]
