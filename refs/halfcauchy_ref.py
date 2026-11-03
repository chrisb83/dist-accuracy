"""Reference module for scipy.stats.halfcauchy.

Standard form, no shapes. Docstring pdf: f(x) = 2/(pi (1 + x**2)), x >= 0.
cdf F(x) = 2 atan(x)/pi, sf = 1 - F, ppf = tan(pi q/2), isf = ppf(1 - q).
"""
import mpmath as mp

DIST = "halfcauchy"

PARAMS = [()]


def pdf(x):
    return 2 / (mp.pi * (1 + x * x))


def cdf(x):
    return 2 * mp.atan(x) / mp.pi


def sf(x):
    return 1 - cdf(x)


def ppf(q):
    return mp.tan(mp.pi * q / 2)


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
    # x = 0, x near 1, both sides of x*x overflow (~1.34e154) and
    # underflow (~1.5e-154), and where the pdf becomes subnormal (~5.4e153).
    import numpy as np
    return np.array([0.0, 1e-300, 1e-160, 1e-154, 1e-20, 1e-8, 0.5,
                     1 - 1e-12, 1.0, 1 + 1e-12, 2.0, 1e8, 1e16, 1e100,
                     1e150, 5e153, 1.3e154, 1.4e154, 1e155, 1e200, 1e250,
                     1e300, 1e307, 1.7e308])


def q_points():
    import numpy as np
    return np.array([1e-308, 1e-300, 1e-200, 1e-100, 1e-20, 1e-8, 0.25, 0.5,
                     0.75, 1 - 1e-8, 1 - 1e-12, 1 - 1e-15, 1 - 2.2e-16,
                     1 - 1.1e-16])
