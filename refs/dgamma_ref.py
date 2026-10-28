"""mpmath reference for scipy.stats.dgamma (standard form, loc=0, scale=1).

Parametrization from the docstring pdf:

    f(x; a) = |x|**(a-1) * exp(-|x|) / (2 Gamma(a)),   x real, a > 0
    F(x; a) = 1/2 + P(a, x)/2   for x > 0
            = Q(a, -x)/2        for x <= 0
    S(x; a) = F(-x; a)          (symmetry)

P/Q are evaluated on whichever side mpmath is fast/robust (as in
refs/gamma_ref.py): lower integral for y < a, upper integral for y >= a,
the other one by 1 - (...). DPS = 120 for speed (the harness checks
stability against higher precision).

Quantiles: by symmetry ppf(q) = P^-1(a, 2q - 1) for q > 1/2 and
-P^-1(a, 1 - 2q) for q <= 1/2; isf(q) = -ppf(q). P^-1 is computed with
accuracy.invert on the gamma cdf in log x (positive=True), so roots of any
magnitude get full relative accuracy. 2q, 2q - 1 and 2 - 2q are exact for
double q; 1 - 2q is not needed in the tail (the small complement 2q is used).
"""
import mpmath as mp
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".claude", "skills",
                                "dist-accuracy-review", "scripts"))
from accuracy import invert  # noqa: E402

DIST = "dgamma"
DPS = 120   # mpmath upper gammainc takes seconds at 400 digits

# inventory example; tiny shapes (gammaincinv amplification), a = 1 (Laplace),
# moderate; a = 150 (|x|**(a-1) overflows for |x| > 117 although the pdf is
# normal); a = 200 and 1e3 (sc.gamma(a) overflows for a > 171.6); a = 1e6
# (large-a cancellation in logpdf and the gammainc issues of reports/gamma.md).
PARAMS = [(1.1023326088288166,), (1e-3,), (0.1,), (1.0,), (10.5,),
          (150.0,), (200.0,), (1e3,), (1e6,)]


def _Q(a, y):
    """Regularized upper incomplete gamma Q(a, y), y >= 0."""
    if y == 0:
        return mp.mpf(1)
    if y >= a:
        return mp.gammainc(a, y, mp.inf, regularized=True)
    return 1 - mp.gammainc(a, 0, y, regularized=True)


def _P(a, y):
    """Regularized lower incomplete gamma P(a, y), y >= 0."""
    if y == 0:
        return mp.mpf(0)
    if y >= a:
        return 1 - _Q(a, y)
    return mp.gammainc(a, 0, y, regularized=True)


def pdf(x, a):
    ax = abs(x)
    if ax == 0:
        if a == 1:
            return mp.mpf(1) / 2
        return mp.inf if a < 1 else mp.mpf(0)
    return mp.exp((a - 1) * mp.log(ax) - ax - mp.loggamma(a)) / 2


def cdf(x, a):
    if x > 0:
        return (1 + _P(a, x)) / 2
    return _Q(a, -x) / 2


def sf(x, a):
    return cdf(-x, a)


def _ginv(a, p, c):
    """Solve P(a, y) = p for y > 0 (0 < p < 1); c = 1 - p is passed
    separately (computed exactly by the caller: at 120 digits 1 - (1 - 2e-300)
    would not be exact)."""
    """

    Speed: Newton in u = log(y) on log P (p <= 1/2) or log Q (p > 1/2),
    started from SciPy's gammaincinv (only a starting point); each step costs
    one incomplete-gamma evaluation, against ~50 for accuracy.invert. Falls
    back to accuracy.invert if Newton does not converge. The harness checks
    the result against the cdf (quantile self-check)."""
    from scipy import special
    upper = p > mp.mpf(1) / 2
    with np.errstate(all="ignore"):
        if upper:
            v = float(special.gammainccinv(float(a), float(c)))
        else:
            v = float(special.gammaincinv(float(a), float(p)))
    lt = mp.log(c) if upper else mp.log(p)
    if np.isfinite(v) and v > 0:
        u = mp.log(mp.mpf(v))
        tol = mp.mpf(10) ** (-(mp.mp.dps - 20))
        for _ in range(40):
            y = mp.exp(u)
            T = _Q(a, y) if upper else _P(a, y)
            if not T > 0:
                break
            # d log P / du = y f(y) / P,  d log Q / du = -y f(y) / Q
            yf = mp.exp(a * u - y - mp.loggamma(a))
            g = mp.log(T) - lt
            dg = -yf / T if upper else yf / T
            step = g / dg
            u -= step
            if abs(step) < tol:
                return mp.exp(u)
    lo = hi = None
    if np.isfinite(v) and v > 0:
        lo, hi = v / 4, v * 4
    if upper:
        return invert(lambda y: _Q(a, y), c, lo=lo, hi=hi, positive=True,
                      increasing=False)
    return invert(lambda y: _P(a, y), p, lo=lo, hi=hi, positive=True)


def ppf(q, a):
    half = mp.mpf(1) / 2
    if q == half:
        return mp.mpf(0)
    if q > half:
        return _ginv(a, 2 * q - 1, 2 - 2 * q)
    return -_ginv(a, 1 - 2 * q, 2 * q)


def isf(q, a):
    return -ppf(q, a)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, a: mp.log(pdf(x, a)),
    "cdf": cdf,
    "logcdf": lambda x, a: mp.log(cdf(x, a)),
    "sf": sf,
    "logsf": lambda x, a: mp.log(sf(x, a)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(a):
    """Bulk of |x| ~ a, both sides of the x**(a-1) overflow threshold, and
    the transitions of igam/igamc (x = a, x = 1), mirrored to x < 0."""
    s = np.sqrt(a)
    pos = [a * 0.5, a * 0.9, a * 0.99, a, a * 1.01, a * 1.1, a * 2.0,
           a * 10.0, a - 1.0, a - s, a + s, a - 3 * s, a + 3 * s, a + 10 * s,
           a - 5 * s, a - 8 * s,
           1.0 - 1e-8, 1.0, 1.0 + 1e-8, 0.5, 2.0, 710.0, 745.0, 800.0, 1e-8]
    if a > 1:
        t = 709.78 / (a - 1)
        if t < 700:
            pos += [np.exp(t * 0.98), np.exp(t * 1.02)]
    pos = np.array(pos, dtype=float)
    pos = pos[np.isfinite(pos) & (pos > 0)]
    return np.concatenate([pos, -pos])


def q_points(a):
    return np.array([1e-17, 1e-16, 1e-8, 0.5 - 1e-8, 0.5 - 1e-12,
                     0.5 - 1e-16, 0.5 - 5.551115123125783e-17,
                     0.5 + 1.1102230246251565e-16, 0.5 + 1e-12, 0.5 + 1e-8,
                     0.5 + 1e-4, 0.5 - 1e-4,
                     1 - 1e-16, 1 - 2e-16, 1 - 1e-13, 1 - 1e-9])
