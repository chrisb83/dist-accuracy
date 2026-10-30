"""Reference module for scipy.stats.halfnorm.

Standard form, docstring pdf f(x) = sqrt(2/pi) exp(-x**2/2), x >= 0.
cdf = erf(x/sqrt(2)), sf = erfc(x/sqrt(2)); ppf/isf by inversion of cdf.
"""
import mpmath as mp
from accuracy import invert

DIST = "halfnorm"
PARAMS = [()]


def pdf(x):
    return mp.sqrt(2 / mp.pi) * mp.exp(-x * x / 2)


def cdf(x):
    return mp.erf(x / mp.sqrt(2))


def sf(x):
    return mp.erfc(x / mp.sqrt(2))


def ppf(q):
    return invert(lambda x: cdf(x), q)


def isf(q):
    return invert(lambda x: cdf(x), 1 - q)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x: mp.log(2 / mp.pi) / 2 - x * x / 2,
    "cdf": cdf,
    "logcdf": lambda x: mp.log(cdf(x)),
    "sf": sf,
    "logsf": lambda x: mp.log(sf(x)),
    "ppf": ppf,
    "isf": isf,
}


def x_points():
    import numpy as np
    # x*x overflow region for logpdf (1.34e154 .. 1.9e154); sf/pdf underflow
    # near 37-38.6; subnormal x/sqrt(2) near 3e-308.
    return np.array([2.5e-308, 3e-308, 1e-307, 1e-8, 0.5, 1.0, 5.0, 8.0, 20.0,
                     37.0, 37.5, 38.0, 38.5, 1e10, 1e100, 1.3e154, 1.35e154,
                     1.5e154, 1.8e154, 1.89e154])


def q_points():
    import numpy as np
    return np.array([1e-300, 1e-100, 1e-20, 1e-17, 1e-16, 2.2e-16, 1e-15,
                     1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.3, 0.5, 0.9,
                     1 - 1e-6, 1 - 1e-10, 1 - 1e-14, 1 - 2**-52, 1 - 2**-53,
                     1 - 3 * 2**-53])
