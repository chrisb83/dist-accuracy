"""Reference module for scipy.stats.wald.

Docstring pdf (standard form): f(x) = (2 pi x^3)^(-1/2) exp(-(x - 1)^2 / (2 x)),
i.e. invgauss with mu = 1.
cdf: F = Phi((x - 1)/sqrt(x)) + exp(2) Phi(-(x + 1)/sqrt(x)).
"""
import mpmath as mp
from accuracy import invert

DIST = "wald"

PARAMS = [()]


def _erfc(t):
    # mpmath's erfc overflows internally for huge t (x up to 1e300 gives
    # t ~ 1e150); use the asymptotic series
    # erfc(t) = exp(-t^2)/(t sqrt(pi)) * sum_n (-1)^n (2n-1)!! / (2 t^2)^n
    # (terms below 10^-(8n) for t > 1e4). Same helper as refs/invgauss_ref.py.
    if t > 10**4:
        u = 1 / (2 * t * t)
        term, tot = mp.mpf(1), mp.mpf(1)
        for n in range(1, 80):
            term *= -(2 * n - 1) * u
            tot += term
        return mp.exp(-t * t) / (t * mp.sqrt(mp.pi)) * tot
    if t < -10**4:
        return 2 - _erfc(-t)
    return mp.erfc(t)


def _Phi(z):
    return _erfc(-z / mp.sqrt(2)) / 2


def pdf(x):
    return 1 / mp.sqrt(2 * mp.pi * x**3) * mp.exp(-(x - 1) ** 2 / (2 * x))


def cdf(x):
    s = mp.sqrt(x)
    return _Phi((x - 1) / s) + mp.exp(2) * _Phi(-(x + 1) / s)


def sf(x):
    return 1 - cdf(x)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x: mp.log(pdf(x)),
    "cdf": cdf,
    "logcdf": lambda x: mp.log(cdf(x)),
    "sf": sf,
    "logsf": lambda x: mp.log(sf(x)),
    "ppf": lambda q: invert(lambda x: cdf(x), q),
    "isf": lambda q: invert(lambda x: cdf(x), 1 - q),
}


def x_points():
    import numpy as np
    # mode/mean region, right tail up to where sf underflows (~1416),
    # left tail down to where cdf underflows (~1.4e-3),
    # x**3 overflow (5.6e102) and (x - 1)**2 overflow (1.34e154) thresholds.
    pts = [1e-3, 1.4e-3, 1.5e-3, 2e-3, 0.5, 1 - 1e-8, 1.0, 1 + 1e-8, 2.0, 10.0,
           100.0, 300.0, 1000.0, 1400.0, 1410.0,
           5e102, 6e102, 1.3e154, 1.4e154, 1e200, 1e300]
    return np.array(pts, float)


def q_points():
    return [0.5 - 1e-12, 0.5, 0.5 + 1e-12, 1e-20, 1e-50, 1e-70, 1e-76, 1e-80, 1e-100]
