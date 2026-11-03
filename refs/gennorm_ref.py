"""mpmath reference for scipy.stats.gennorm (standard form, loc=0, scale=1).

Parametrization from the docstring pdf:

    f(x; beta) = beta / (2 Gamma(1/beta)) * exp(-|x|**beta),   x real, beta > 0

With a = 1/beta and y = |x|**beta (P, Q regularized incomplete gamma):

    F(x) = 1/2 + P(a, y)/2   for x > 0
         = Q(a, y)/2         for x <= 0
    S(x) = F(-x)             (symmetry)

P/Q are evaluated on whichever side mpmath is fast (as in refs/gamma_ref.py):
lower integral for y < a, upper integral for y >= a, the other by 1 - (...).
y is formed in mpmath, so it never under/overflows (beta = 1e300 gives
y = 0.5**1e300 without trouble).

Quantiles: ppf(q) = +x with P(a, x**beta) = 2q - 1 for q > 1/2 and
-x with P(a, x**beta) = 1 - 2q for q < 1/2; isf(q) = -ppf(q). The half-line
equation is solved directly in log x with accuracy.invert (seeded from SciPy
when SciPy's value is usable), so the quantile gets full relative accuracy
whatever beta is. 2q - 1, 1 - 2q and their complements are exact at the
working precision for double q.
"""
import mpmath as mp
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".claude", "skills",
                                "dist-accuracy-review", "scripts"))
from accuracy import invert  # noqa: E402

DIST = "gennorm"
DPS = 400  # see speed note in _half; harness checks stability

# inventory example; small beta (huge gammaln(1/beta) constant, 1/beta
# amplification in ppf; 0.006 keeps logpdf ~ -680 so the pdf stays normal);
# Laplace (1) and normal (2); large beta where |x|**beta underflows for |x|
# inside (-1, 1) although the cdf moves linearly there (20, 1e3, 1e6), and
# huge beta where log(0.5*beta) and gammaln(1/beta) cancel (1e100, 1e300).
PARAMS = [(1.2988442399460265,), (0.006,), (0.05,), (0.5,), (1.0,), (2.0,),
          (20.0,), (1e3,), (1e6,), (1e100,), (1e300,)]


def _exp(L):
    """exp(L); for L < -1e5 (far below double range) return 2**round(L/ln2),
    because mp.exp of arguments like -1e10000 needs enormous internal
    precision and hangs."""
    if L > -1e5:
        return mp.exp(L)
    if L < -1e15:
        return mp.mpf(0)   # 2**(L/ln2) would need an exponent with >1e15 digits
    # only the order of magnitude is kept (L - n*ln2 is not representable
    # for |L| ~ 1e10000); such values are never compared with doubles
    return mp.ldexp(mp.mpf(1), int(L / mp.ln2))


def _Q_asym(a, y):
    """Q(a, y) for y >> a: y**(a-1) e**-y / Gamma(a) * sum_k (a-1)...(a-k)/y**k.
    Only used where y > 2000 + 4a and the value is far below double range
    (mpmath's upper gammainc hangs for astronomically large y)."""
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
    return mp.log(beta / 2) - mp.loggamma(1 / beta) - abs(x) ** beta


def pdf(x, beta):
    return _exp(logpdf(x, beta))


def cdf(x, beta):
    a = 1 / beta
    y = abs(x) ** beta
    if x > 0:
        return (1 + _P(a, y)) / 2
    return _Q(a, y) / 2


def sf(x, beta):
    return cdf(-x, beta)


def _half(p, c, beta, seed):
    """x > 0 with P(1/beta, x**beta) = p, c = 1 - p passed exactly.

    For p <= 1/2 solve log P = log p, else log Q = log c, so no 1 - (...) of
    a quantity near 1 is ever formed. Newton in u = log x from SciPy's value
    (only a starting point; d log P/du = beta y**a e**-y / (Gamma(a) P)),
    falling back to accuracy.invert. The harness checks the result against
    the cdf."""
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


def _seed(q, beta):
    from scipy import stats
    with np.errstate(all="ignore"):
        v = abs(float(stats.gennorm.ppf(float(q), float(beta))))
    return v if np.isfinite(v) and v > 0 else None


def ppf(q, beta):
    half = mp.mpf(1) / 2
    if q == half:
        return mp.mpf(0)
    s = _seed(q, beta)
    if q > half:
        return _half(2 * q - 1, 2 - 2 * q, beta, s)
    return -_half(1 - 2 * q, 2 * q, beta, s)


def isf(q, beta):
    return -ppf(q, beta)


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
    """|x| = 1 neighbourhood (large beta), both sides of where |x|**beta
    underflows (beta*log|x| < -708 / -745) and overflows (> 709.78), the bulk
    in units of the scale Gamma(3/beta)/Gamma(1/beta) where representable;
    mirrored to x < 0."""
    pos = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.99, 1 - 1e-6, 1.0, 1 + 1e-6,
           1.01, 1.5, 3.0, 1e-3, 1e-8]
    with np.errstate(all="ignore"):
        for t in (-745.0, -708.0, 709.78, 5.0, 30.0, 100.0, 700.0):
            for f in (0.98, 1.02):
                pos.append(np.exp(f * t / beta))
    pos = np.array(pos, dtype=float)
    pos = pos[np.isfinite(pos) & (pos > 0)]
    return np.concatenate([pos, -pos])


def q_points(beta):
    return np.array([0.5 - 1e-4, 0.5 - 1e-8, 0.5 - 1e-12, 0.5 - 5.551115123125783e-17,
                     0.5 + 1.1102230246251565e-16, 0.5 + 1e-12, 0.5 + 1e-8,
                     0.5 + 1e-4, 0.55, 0.6, 0.7, 0.74, 0.8, 0.9, 0.95,
                     0.3, 0.26, 0.2, 0.1, 0.05,
                     1 - 1e-16, 1 - 2e-16, 1 - 1e-13, 1 - 1e-9])
