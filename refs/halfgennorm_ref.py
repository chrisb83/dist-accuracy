"""mpmath reference for scipy.stats.halfgennorm (standard form, loc=0, scale=1).

Parametrization from the docstring pdf:

    f(x; beta) = beta / Gamma(1/beta) * exp(-x**beta),   x > 0, beta > 0

With a = 1/beta and y = x**beta (P, Q regularized incomplete gamma):

    F(x) = P(a, y),   S(x) = Q(a, y)

P/Q are evaluated on whichever side mpmath is fast (as in refs/gennorm_ref.py):
lower integral for y < a, upper integral for y >= a, the other by 1 - (...).
y is formed in mpmath, so it never under/overflows.

Quantiles: ppf(q) = x with P(a, x**beta) = q, isf(q) = x with Q(a, x**beta) = q
(1 - q exact at the working precision). Solved in log x by Newton seeded from
SciPy when SciPy's value is usable, falling back to accuracy.invert; the
harness checks the result against the cdf.
"""
import mpmath as mp
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".claude", "skills",
                                "dist-accuracy-review", "scripts"))
from accuracy import invert  # noqa: E402

DIST = "halfgennorm"
DPS = 400

# inventory example; small beta (huge gammaln(1/beta), 1/beta amplification
# in ppf); exponential (1) and half-normal (2); large beta where x**beta
# underflows inside (0, 1) although the cdf is ~x there (20, 1e3, 1e6);
# huge beta where log(beta) and gammaln(1/beta) cancel (1e300).
PARAMS = [(0.6748054997000371,), (0.01,), (0.2,), (1.0,), (2.0,),
          (20.0,), (1e3,), (1e6,), (1e300,)]


def _exp(L):
    """exp(L); for L < -1e5 return 2**round(L/ln2) (mp.exp of huge negative
    arguments hangs); such values are far below double range anyway."""
    if L > -1e5:
        return mp.exp(L)
    if L < -1e15:
        return mp.mpf(0)
    return mp.ldexp(mp.mpf(1), int(L / mp.ln2))


def _Q_asym(a, y):
    """Q(a, y) for y >> a, asymptotic series (only where far below double range)."""
    s, t = mp.mpf(1), mp.mpf(1)
    for k in range(1, 400):
        t *= (a - k) / y
        s += t
        if abs(t) < mp.eps * abs(s):
            break
    return _exp((a - 1) * mp.log(y) - y - mp.loggamma(a)) * s


def _Q(a, y):
    if y == 0:
        return mp.mpf(1)
    if y > 2000 + 4 * a:
        return _Q_asym(a, y)
    if y >= a:
        return mp.gammainc(a, y, mp.inf, regularized=True)
    return 1 - mp.gammainc(a, 0, y, regularized=True)


def _P(a, y):
    if y == 0:
        return mp.mpf(0)
    if y >= a:
        return 1 - _Q(a, y)
    return mp.gammainc(a, 0, y, regularized=True)


def logpdf(x, beta):
    return mp.log(beta) - mp.loggamma(1 / beta) - x ** beta


def pdf(x, beta):
    return _exp(logpdf(x, beta))


def cdf(x, beta):
    return _P(1 / beta, x ** beta)


def sf(x, beta):
    return _Q(1 / beta, x ** beta)


def _solve(p, c, beta, seed):
    """x > 0 with P(1/beta, x**beta) = p, c = 1 - p passed exactly.
    For p <= 1/2 solve log P = log p, else log Q = log c."""
    a = 1 / beta
    upper = p > mp.mpf(1) / 2
    T = (lambda x: _Q(a, x ** beta)) if upper else (lambda x: _P(a, x ** beta))
    lt = mp.log(c) if upper else mp.log(p)
    if seed is not None:
        u = mp.log(mp.mpf(seed))
        tol = mp.mpf(10) ** (-(mp.mp.dps - 20))
        lga = mp.loggamma(a)
        for _ in range(60):
            x = mp.exp(u)
            y = x ** beta
            v = T(x)
            if not v > 0:
                break
            dv = _exp(a * mp.log(y) - y - lga) * beta / v
            step = (mp.log(v) - lt) / (-dv if upper else dv)
            if abs(step) > 5:
                break
            u -= step
            if abs(step) < tol:
                return mp.exp(u)
    lo = hi = None
    if seed is not None:
        lo, hi = seed / 4, seed * 4
    if upper:
        return invert(T, c, lo=lo, hi=hi, positive=True, increasing=False)
    return invert(T, p, lo=lo, hi=hi, positive=True)


def _seed(method, q, beta):
    from scipy import stats
    with np.errstate(all="ignore"):
        v = float(getattr(stats.halfgennorm, method)(float(q), float(beta)))
    return v if np.isfinite(v) and v > 0 else None


def ppf(q, beta):
    return _solve(q, 1 - q, beta, _seed("ppf", q, beta))


def isf(q, beta):
    return _solve(1 - q, q, beta, _seed("isf", q, beta))


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": lambda x, b: mp.log(cdf(x, b)),
    "sf": sf,
    "logsf": lambda x, b: mp.log(sf(x, b)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(beta):
    """x = 1 neighbourhood (large beta), both sides of where x**beta
    underflows (beta*log x < -708 / -745) and overflows (> 709.78), the
    zero of logpdf (x**beta = -gammaln(1 + 1/beta)) and bulk points."""
    pos = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.99, 1 - 1e-6, 1.0, 1 + 1e-6,
           1.01, 1.5, 3.0, 1e-3, 1e-8]
    with np.errstate(all="ignore"):
        for t in (-745.0, -708.0, 709.78, 5.0, 30.0, 100.0, 700.0):
            for f in (0.98, 1.02):
                pos.append(np.exp(f * t / beta))
    pos = np.array(pos, dtype=float)
    return pos[np.isfinite(pos) & (pos > 0)]


def q_points(beta):
    return np.array([1e-4, 1e-8, 1e-12, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                     0.7, 0.8, 0.9, 0.95, 1 - 1e-4, 1 - 1e-9, 1 - 1e-13,
                     1 - 1e-16, 1 - 2e-16])
