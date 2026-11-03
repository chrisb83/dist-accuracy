"""Reference module for scipy.stats.norm.

Standard form, docstring pdf f(x) = exp(-x**2/2)/sqrt(2*pi), x real.
cdf = erfc(-x/sqrt(2))/2, sf = erfc(x/sqrt(2))/2; ppf/isf by inversion of cdf.
"""
import mpmath as mp
from accuracy import invert

DIST = "norm"
PARAMS = [()]


def pdf(x):
    return mp.exp(-x * x / 2) / mp.sqrt(2 * mp.pi)


def logpdf(x):
    return -x * x / 2 - mp.log(2 * mp.pi) / 2


def cdf(x):
    return mp.erfc(-x / mp.sqrt(2)) / 2


def sf(x):
    return mp.erfc(x / mp.sqrt(2)) / 2


def ppf(q):
    if q == mp.mpf(1) / 2:
        return mp.mpf(0)  # exact; invert() returns ~1e-60 here
    return invert(lambda x: cdf(x), q, positive=False)


def isf(q):
    if q == mp.mpf(1) / 2:
        return mp.mpf(0)
    return invert(lambda x: cdf(x), 1 - q, positive=False)


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
    import numpy as np
    # log_ndtr branch at x = -1 (both signs for logsf); ndtr/pdf underflow
    # near |x| = 37.5 .. 38.6; x*x overflow 1.34e154 .. 1.896e154 (logpdf,
    # log_ndtr's t*t); tiny |x|.
    pos = np.array([1e-300, 1e-20, 1e-8, 1e-3, 0.1, 0.5, 1 - 1e-9, 1.0,
                    1 + 1e-9, 1.3, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 26.0,
                    30.0, 35.0, 37.0, 37.5, 38.0, 38.4, 38.6, 40.0, 100.0,
                    1e10, 1e100, 1e150, 1.3e154, 1.35e154, 1.5e154, 1.8e154,
                    1.89e154,
                    # non-integers: x*x is rounded before exp (pdf)
                    0.7071, 1.1, 2.3456789, 5.55, 9.87654321, 12.3456789,
                    19.9, 25.7, 29.123456789, 33.33, 36.987654321,
                    37.77, 38.123456789, 38.55])
    return np.concatenate([pos, -pos, [0.0]])


def q_points():
    import numpy as np
    return np.array([1e-300, 1e-200, 1e-100, 1e-50, 1e-20, 1e-17, 1e-16,
                     2.2e-16, 1e-15, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2,
                     0.02, 0.1, 0.13, 0.135, 0.14, 0.3, 0.49, 0.5 - 1e-9,
                     0.5, 0.5 + 1e-9, 0.51, 0.7, 0.85, 0.86, 0.865, 0.87,
                     0.9, 0.99, 1 - 1e-6, 1 - 1e-10, 1 - 1e-14, 1 - 2**-52,
                     1 - 2**-53, 1 - 3 * 2**-53])
