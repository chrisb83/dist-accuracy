"""Reference module for scipy.stats.laplace.

Standard form, no shapes. Docstring pdf: f(x) = exp(-|x|)/2, real x.
cdf F(x) = exp(x)/2 for x <= 0, 1 - exp(-x)/2 for x > 0; sf = 1 - F;
ppf = log(2q) for q <= 1/2, -log(2(1-q)) otherwise; isf = ppf(1 - q).
"""
import mpmath as mp

DIST = "laplace"

PARAMS = [()]


def pdf(x):
    return mp.exp(-abs(x)) / 2


def cdf(x):
    if x <= 0:
        return mp.exp(x) / 2
    return 1 - mp.exp(-x) / 2


def sf(x):
    return 1 - cdf(x)


def ppf(q):
    if q <= mp.mpf(1) / 2:
        return mp.log(2 * q)
    return -mp.log(2 * (1 - q))


def isf(q):
    return ppf(1 - q)


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
    # cdf branch at x = 0 (both sides, tiny |x|), x near log 2 (cdf = 3/4),
    # where exp(-|x|)/2 underflows (|x| ~ 708 normal limit, 745 subnormal),
    # where exp(x) in the unused np.where branch overflows (|x| > 709.78).
    import numpy as np
    pts = [0.0, 1e-300, 1e-200, 1e-100, 1e-20, 1e-16, 1e-12, 1e-8, 1e-4, 0.1,
           0.5, np.log(2), 1.0, 2.0, 10.0, 36.0, 37.0, 38.0, 50.0, 100.0,
           700.0, 707.0, 708.0, 708.3, 709.0, 709.7, 709.8, 710.0, 745.0, 1e3]
    pts = pts + [-p for p in pts if p != 0]
    return np.array(pts)


def q_points():
    # Both sides of the q = 0.5 branch, q near 1, tiny q.
    import numpy as np
    d = [1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.25]
    pts = [0.5, np.nextafter(0.5, 0), np.nextafter(0.5, 1)]
    pts += [0.5 + v for v in d] + [0.5 - v for v in d]
    pts += [1 - 2.2e-16, 1 - 1e-15, 1 - 1e-12, 1 - 1e-8, 0.75, 0.25,
            1e-308, 1e-300, 1e-200, 1e-100, 1e-20, 1e-16]
    return np.array(pts)
