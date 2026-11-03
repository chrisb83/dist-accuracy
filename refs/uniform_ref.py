"""Reference module for scipy.stats.uniform.

Standard form, docstring pdf f(x) = 1 for 0 <= x <= 1. No shapes.
"""
import mpmath as mp

DIST = "uniform"

PARAMS = [()]


def pdf(x):
    return mp.mpf(1) if 0 <= x <= 1 else mp.mpf(0)


def cdf(x):
    return min(max(mp.mpf(x), mp.mpf(0)), mp.mpf(1))


def sf(x):
    return 1 - cdf(x)


def ppf(q):
    return mp.mpf(q)


def isf(q):
    return 1 - mp.mpf(q)


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
    # Support edges, the smallest normal x, and x just below 1 where the
    # inherited sf = 1 - cdf is still exact (Sterbenz) - kept for opportunities.
    import numpy as np
    return np.array([2.2250738585072014e-308, 1e-300, 1e-17, 1e-8, 0.25, 0.5,
                     0.75, 1 - 1e-8, 1 - 2**-52, 1 - 2**-53, 1.0])


def q_points():
    import numpy as np
    return np.array([2.2250738585072014e-308, 1e-300, 1e-17, 0.5,
                     1 - 2**-53, 1 - 2**-52, 1 - 1e-10, 0.9999])
