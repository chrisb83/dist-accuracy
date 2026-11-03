"""Reference module for scipy.stats.expon.

Standard form, docstring pdf f(x) = exp(-x), x >= 0. No shapes.
"""
import mpmath as mp

DIST = "expon"

PARAMS = [()]


def pdf(x):
    return mp.exp(-x)


def cdf(x):
    return 1 - mp.exp(-x)


def sf(x):
    return mp.exp(-x)


def ppf(q):
    return -mp.log(1 - q)


def isf(q):
    return -mp.log(q)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x: mp.log(pdf(x)),
    "cdf": cdf,
    "logcdf": lambda x: mp.log(cdf(x)),
    "sf": sf,
    "logsf": lambda x: mp.log(sf(x)),
    "ppf": ppf,
    "isf": isf,
}


def x_points():
    # Smallest normal x (sc.expm1 near subnormal arguments), the region
    # where -expm1(-x) rounds to 1 (x ~ 36.7..37.5), and where exp(-x)
    # approaches the subnormal range (x ~ 708).
    import numpy as np
    return np.array([2.2250738585072014e-308, 3e-308, 1e-307, 1e-305,
                     1e-300, 1e-200, 1e-17, 1.1e-16, 1e-8, 0.5, 0.6931471805599453,
                     1.0, 36.0, 36.7, 37.0, 37.5, 38.0, 700.0, 708.0, 708.39])


def q_points():
    import numpy as np
    return np.array([2.2250738585072014e-308, 1e-305, 1e-17, 0.5,
                     1 - 2**-53, 1 - 2**-52, 1 - 1e-10, 0.9999])
