"""Reference module for scipy.stats.irwinhall.

Standard form, shape n (positive integer). Irwin-Hall = sum of n U(0,1):
  pdf(x) = 1/(n-1)! sum_{k=0}^{floor(x)} (-1)^k C(n,k) (x-k)^(n-1)
  cdf(x) = 1/n!     sum_{k=0}^{floor(x)} (-1)^k C(n,k) (x-k)^n
  sf(x)  = cdf(n - x)   (symmetry about n/2; n - x is exact in mpmath)
The alternating sum cancels by up to ~ n*log10(2e) digits, so each call
raises the working precision by that amount (exact rational arithmetic
in effect).  The CLT is not used anywhere.
"""
import mpmath as mp
from accuracy import invert

DIST = "irwinhall"

PARAMS = [(1.0,), (2.0,), (3.0,), (10.0,), (30.0,), (100.0,), (300.0,)]


def _extra(n):
    return int(0.75 * float(n)) + 20


def _sum(x, n, p):
    n = int(n)
    x = mp.mpf(x)
    if x <= 0:
        return mp.mpf(0)
    m = min(int(mp.floor(x)), n)
    s = mp.mpf(0)
    for k in range(m + 1):
        t = mp.binomial(n, k) * (x - k) ** p
        s += -t if k % 2 else t
    return s


def pdf(x, n):
    with mp.workdps(mp.mp.dps + _extra(n)):
        if x >= n:
            r = mp.mpf(0)
        else:
            r = _sum(x, n, int(n) - 1) / mp.factorial(int(n) - 1)
    return +r


def cdf(x, n):
    with mp.workdps(mp.mp.dps + _extra(n)):
        if x >= n:
            r = mp.mpf(1)
        else:
            r = _sum(x, n, int(n)) / mp.factorial(int(n))
    return +r


def sf(x, n):
    return cdf(mp.mpf(n) - mp.mpf(x), n)


def ppf(q, n):
    return invert(lambda x: cdf(x, n), q)


def isf(q, n):
    return invert(lambda x: cdf(x, n), 1 - q)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, n: mp.log(pdf(x, n)),
    "cdf": cdf,
    "logcdf": lambda x, n: mp.log(cdf(x, n)),
    "sf": sf,
    "logsf": lambda x, n: mp.log(sf(x, n)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(n):
    # Knots (interval boundaries of the piecewise polynomial) and both sides,
    # the centre, and left/right tails where the pdf/cdf reach ~1e-300
    # (x**(n-1)/(n-1)! ~ 1e-300) for large n.
    import numpy as np
    from scipy.special import gammaln
    n = int(n)
    ks = np.unique(np.r_[np.arange(0, min(n, 6) + 1), n - np.arange(0, min(n, 6) + 1),
                         n // 2, n // 4])
    pts = [ks, ks + 1e-9, ks - 1e-9, ks + 0.5, [n / 2 + 0.25, n / 2 - 1e-7]]
    if n > 1:
        # x where log pdf ~ -690 and -100 in the left tail and mirror
        for L in (-690.0, -300.0, -100.0, -30.0):
            x = np.exp((L + gammaln(n)) / (n - 1))
            if 0 < x < n:
                pts.append([x, n - x])
    p = np.concatenate([np.asarray(a, float) for a in pts])
    return p[(p >= 0) & (p <= n)]
