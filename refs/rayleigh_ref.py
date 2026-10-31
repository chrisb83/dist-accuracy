"""mpmath reference for scipy.stats.rayleigh (standard form, loc=0, scale=1).

Docstring pdf (no shapes; chi with df = 2):

    f(x) = x exp(-x**2/2),  x >= 0
    F(x) = 1 - exp(-x**2/2)
    S(x) = exp(-x**2/2)
    ppf(q) = sqrt(-2 log(1 - q)),  isf(q) = sqrt(-2 log(q))

Closed forms evaluated at 400 digits; x**2 is formed in mpmath, so there is
no overflow/underflow of x**2 in the reference.
"""
import mpmath as mp
import numpy as np

DIST = "rayleigh"
DPS = 400

PARAMS = [()]


def logpdf(x):
    x = mp.mpf(x)
    if x == 0:
        return -mp.inf
    return mp.log(x) - x * x / 2


def pdf(x):
    x = mp.mpf(x)
    return x * mp.exp(-x * x / 2)


def cdf(x):
    x = mp.mpf(x)
    return 1 - mp.exp(-x * x / 2)


def sf(x):
    x = mp.mpf(x)
    return mp.exp(-x * x / 2)


def logsf(x):
    x = mp.mpf(x)
    return -x * x / 2


def ppf(q):
    q = mp.mpf(q)
    return mp.sqrt(-2 * mp.log(1 - q))


def isf(q):
    q = mp.mpf(q)
    return mp.sqrt(-2 * mp.log(q))


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": lambda x: mp.log(cdf(x)),
    "sf": sf,
    "logsf": logsf,
    "ppf": ppf,
    "isf": isf,
}


def x_points():
    """Mode 1, bulk, pdf/sf underflow edge (~37.5-38.6), the x*x overflow
    window in logpdf/logsf (1.34e154 ... 1.897e154), the x**2 underflow edge
    in cdf (~1.5e-154 ... 2.1e-154) and tiny x where log(x) is large."""
    pts = [1.0, 0.5, 2.0, 3.0, 5.0, 8.0, 10.0, 20.0, 30.0, 35.0,
           37.0, 37.5, 37.6, 37.7, 38.0, 38.3, 38.5, 38.6,
           1e10, 1e100, 1.3e154, 1.34e154, 1.35e154, 1.5e154, 1.8e154,
           1.89e154, 1.896e154,
           1.5e-154, 2e-154, 2.2e-154, 3e-154, 1e-150, 1e-100, 1e-50,
           1e-20, 1e-10, 1e-5, 1e-3, 1e-300, 1e-305, 1e-307]
    return np.array(pts, dtype=float)


def q_points():
    return np.array([1e-300, 1e-200, 1e-100, 1e-17, 1e-16, 1e-8,
                     0.5 - 1e-16, 0.5, 0.5 + 1e-16, 0.9,
                     1 - 1e-16, 1 - 2e-16, 1 - 1e-13, 1 - 1e-9])
