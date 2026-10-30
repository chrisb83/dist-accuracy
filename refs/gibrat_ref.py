"""Reference module for scipy.stats.gibrat.

Standard form, docstring pdf (no shapes)
    f(x) = 1/(x sqrt(2 pi)) exp(-log(x)**2 / 2),  x > 0
i.e. lognorm with s = 1.
cdf = Phi(log x), sf = Phi(-log x);
ppf = exp(Phi^-1(q)), isf = exp(Phi^-1(1 - q)), Phi^-1 by numerical inversion.
"""
import mpmath as mp
from accuracy import invert

DIST = "gibrat"

PARAMS = [()]


def pdf(x):
    return mp.exp(-mp.log(x) ** 2 / 2) / (x * mp.sqrt(2 * mp.pi))


def logpdf(x):
    return -mp.log(x) ** 2 / 2 - mp.log(x * mp.sqrt(2 * mp.pi))


def cdf(x):
    return mp.ncdf(mp.log(x))


def sf(x):
    return mp.ncdf(-mp.log(x))


def ppf(q):
    return mp.exp(invert(mp.ncdf, q, positive=False))


def isf(q):
    return mp.exp(invert(mp.ncdf, 1 - q, positive=False))


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": lambda x: mp.log(cdf(x)),
    "sf": sf,
    "logsf": lambda x: mp.log(sf(x)),
    "ppf": ppf,
    "isf": isf,
}


def x_points():
    # x = 1 and its neighbours (log x = 0), the mode exp(-1), x where
    # z = log x is +-8, +-30, +-37.5, +-38.4 (ndtr tails, ndtr underflow),
    # and both sides of x*sqrt(2 pi) = 1.8e308 (logpdf overflow, x ~ 7.2e307).
    import numpy as np
    pts = [1.0, 1 + 2**-52, 1 - 2**-53, 1 + 1e-12, 1 - 1e-12, 1 + 1e-8, 0.5, 2.0,
           np.exp(-1.0), 1e300, 1e305, 1e307, 7e307, 8e307, 1.7e308]
    for z in [-38.4, -37.5, -30.0, -20.0, -8.0, 8.0, 20.0, 30.0, 37.5, 38.4]:
        pts.append(np.exp(z))
    return np.array(pts, dtype=float)


def q_points():
    import numpy as np
    return np.array([1e-300, 1e-200, 1e-100, 1e-20, 1e-16, 1e-12, 1e-8, 1e-4,
                     1e-2, 0.3, 0.5, 0.7, 0.9, 1 - 1e-6, 1 - 1e-10, 1 - 1e-14,
                     1 - 2**-52, 1 - 2**-53])
