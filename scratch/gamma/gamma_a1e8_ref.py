"""Auxiliary reference for scipy.stats.gamma at very large a (a = 1e8).

mpmath's own ``mp.gammainc(a, 0, x)`` raises NoConvergence for a >= 1e7, so the
cdf is evaluated here with the Kummer series that cephes' ``igam_series`` uses

    P(a, x) = x**a e**-x / Gamma(a+1) * sum_{n>=0} prod_{i=1..n} x/(a+i),

summed to the working precision (validated against mp.gammainc at a = 1e6 to
20 digits, see scratch/gamma/probe5.py). Only cdf/sf are measured, on a small
grid around x = a, which is where scipy's 2000-iteration cap bites.
"""
import mpmath as mp
import numpy as np

DIST = "gamma"
DPS = 100
PARAMS = [(1e8,)]
METHODS = ["cdf", "sf"]
REPLACE_GRID = True


def _lower(a, x):
    ax = mp.exp(a * mp.log(x) - x - mp.loggamma(a + 1))
    r, c, ans = a, mp.mpf(1), mp.mpf(1)
    tol = mp.mpf(10) ** (-mp.mp.dps - 10)
    for _ in range(10_000_000):
        r += 1
        c *= x / r
        ans += c
        if c < ans * tol:
            break
    return ans * ax


def cdf(x, a):
    if x == 0:
        return mp.mpf(0)
    if x >= a:
        return 1 - mp.gammainc(a, x, mp.inf, regularized=True)
    return _lower(a, x)


def sf(x, a):
    if x >= a:
        return mp.gammainc(a, x, mp.inf, regularized=True)
    return 1 - cdf(x, a)


def pdf(x, a):
    return mp.exp((a - 1) * mp.log(x) - x - mp.loggamma(a))


REFS = {"cdf": cdf, "sf": sf, "pdf": pdf}


def x_points(a):
    s = np.sqrt(a)
    return np.array([a + k * s for k in
                     (-10.0, -7.0, -5.0, -4.6, -4.0, -3.0, 0.0, 3.0, 5.0, 10.0)])


def q_points(a):
    return np.array([0.5])
