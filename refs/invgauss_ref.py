"""Reference module for scipy.stats.invgauss.

Docstring pdf (standard form): f(x; mu) = (2 pi x^3)^(-1/2) exp(-(x - mu)^2 / (2 mu^2 x)).
cdf (standard inverse Gaussian with lambda = 1, mean mu):
    F = Phi((x/mu - 1)/sqrt(x)) + exp(2/mu) Phi(-(x/mu + 1)/sqrt(x)).
"""
import mpmath as mp
from accuracy import invert

DIST = "invgauss"

# example, mu = 1, small mu (cancellation of 2/mu in the log_ndtr term),
# large mu (right tail x >> mu, Levy-like regime, x**3 overflow with normal pdf).
PARAMS = [(0.14546264555347513,), (1.0,), (1e-3,), (1e-8,), (1e-20,), (1e3,), (1e8,), (1e20,)]


def _erfc(t):
    # mpmath's erfc overflows internally for huge t; use the asymptotic series
    # erfc(t) = exp(-t^2)/(t sqrt(pi)) * sum_n (-1)^n (2n-1)!! / (2 t^2)^n,
    # whose terms fall below 10^-(8n) for t > 1e4 (60 terms >> 400 digits).
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


def pdf(x, mu):
    return 1 / mp.sqrt(2 * mp.pi * x**3) * mp.exp(-(x - mu) ** 2 / (2 * mu**2 * x))


def cdf(x, mu):
    s = mp.sqrt(x)
    return _Phi((x / mu - 1) / s) + mp.exp(2 / mu) * _Phi(-(x / mu + 1) / s)


def sf(x, mu):
    return 1 - cdf(x, mu)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, mu: mp.log(pdf(x, mu)),
    "cdf": cdf,
    "logcdf": lambda x, mu: mp.log(cdf(x, mu)),
    "sf": sf,
    "logsf": lambda x, mu: mp.log(sf(x, mu)),
    "ppf": lambda q, mu: invert(lambda x: cdf(x, mu), q),
    "isf": lambda q, mu: invert(lambda x: cdf(x, mu), 1 - q),
}


def x_points(mu):
    import numpy as np
    m = float(mu)
    pts = [m * f for f in (1e-3, 0.1, 0.5, 1 - 1e-8, 1.0, 1 + 1e-8, 2.0, 10.0, 100.0, 1e4, 1e8)]
    # x**3 overflow threshold; (x/mu)**2 overflow threshold
    pts += [5e102, 6e102, 1.3e154 * m, 1.4e154 * m]
    # right tail where sf is still normal: x up to ~1416 mu^2
    pts += [m * m * f for f in (1.0, 10.0, 100.0, 1000.0)]
    with np.errstate(over="ignore", under="ignore"):
        return np.array(pts, float)


def q_points(mu):
    return [0.5 - 1e-12, 0.5, 0.5 + 1e-12]
