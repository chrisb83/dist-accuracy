"""Reference module for scipy.stats.foldcauchy.

Standard form, docstring pdf:
    f(x, c) = 1/(pi (1+(x-c)^2)) + 1/(pi (1+(x+c)^2)),  x >= 0, c >= 0.
cdf = (atan(x-c) + atan(x+c))/pi, evaluated as the exactly equivalent
arg((1 + i(x-c))(1 + i(x+c)))/pi = atan2(2x, 1 + c^2 - x^2)/pi, which avoids
the cancellation of +-atan(c) (up to ~400 digits for c = 1e100, x = 1e-300).
"""
import mpmath as mp
from accuracy import invert

DIST = "foldcauchy"

PARAMS = [(4.716467345583189,), (0.0,), (1e-3,), (1.0,), (1e3,), (1e6,), (1e100,)]


def pdf(x, c):
    return (1 / (1 + (x - c) ** 2) + 1 / (1 + (x + c) ** 2)) / mp.pi


def cdf(x, c):
    return mp.atan2(2 * x, 1 + c ** 2 - x ** 2) / mp.pi


def sf(x, c):
    return 1 - cdf(x, c)


def ppf(q, c):
    return invert(lambda x: cdf(x, c), q)


def isf(q, c):
    return invert(lambda x: cdf(x, c), 1 - q)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, c: mp.log(pdf(x, c)),
    "cdf": cdf,
    "logcdf": lambda x, c: mp.log(cdf(x, c)),
    "sf": sf,
    "logsf": lambda x, c: mp.log(sf(x, c)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(c):
    import numpy as np
    pts = [0.5, 1.0, 2.0, 1e-10, 1e-5, 0.1]
    if c > 0:
        pts += [c * (1 - 1e-6), c, c * (1 + 1e-6), c * 0.5, c * 2, np.sqrt(1 + c * c)]
    return np.array(pts)
