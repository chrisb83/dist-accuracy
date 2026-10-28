"""Auxiliary reference for scipy.stats.chi2 at very large df (df = 2e7, 2e8).

mpmath's mp.gammainc(a, 0, y) does not converge for a = df/2 >~ 1e7, so the
cdf for y = x/2 < a is the Kummer series (the one cephes' igam_series uses),
summed to the working precision:

    P(a, y) = y**a e**-y / Gamma(a+1) * sum_{n>=0} prod_{i=1..n} y/(a+i).

Upper side (y >= a) uses mp.gammainc(a, y, inf), which converges there.
Same construction as scratch/gamma/gamma_a1e8_ref.py (validated there against
mp.gammainc at a = 1e6 and by an independent quadrature verifier).
"""
import mpmath as mp
import numpy as np

DIST = "chi2"
DPS = 100
PARAMS = [(2e7,), (2e8,)]
METHODS = ["cdf", "sf", "pdf", "logpdf"]
REPLACE_GRID = True


def _lower(a, y):
    ay = mp.exp(a * mp.log(y) - y - mp.loggamma(a + 1))
    r, c, ans = a, mp.mpf(1), mp.mpf(1)
    tol = mp.mpf(10) ** (-mp.mp.dps - 10)
    for _ in range(10_000_000):
        r += 1
        c *= y / r
        ans += c
        if c < ans * tol:
            break
    return ans * ay


def cdf(x, k):
    a, y = k / 2, x / 2
    if y >= a:
        return 1 - mp.gammainc(a, y, mp.inf, regularized=True)
    return _lower(a, y)


def sf(x, k):
    a, y = k / 2, x / 2
    if y >= a:
        return mp.gammainc(a, y, mp.inf, regularized=True)
    return 1 - _lower(a, y)


def pdf(x, k):
    a = k / 2
    return mp.exp((a - 1) * mp.log(x) - x / 2 - mp.loggamma(a) - a * mp.log(2))


REFS = {"cdf": cdf, "sf": sf, "pdf": pdf,
        "logpdf": lambda x, k: mp.log(pdf(x, k))}


def x_points(k):
    s = np.sqrt(2 * k)
    return np.array([k + j * s for j in
                     (-30.0, -20.0, -15.0, -10.0, -7.0, -5.0, -4.6, -4.4,
                      -4.0, -3.0, 0.0, 3.0, 5.0, 10.0)])


def q_points(k):
    return np.array([0.5])
