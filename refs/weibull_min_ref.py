"""Reference module for scipy.stats.weibull_min.

Standard form, docstring pdf: f(x; c) = c x**(c-1) exp(-x**c), x > 0, c > 0.
"""
import mpmath as mp

DIST = "weibull_min"

# Example parameter, exact c = 1 (expon) and 2 (rayleigh), small c (c - 1
# and 1/c inexact, huge/tiny quantiles), 1/c inexact (3), and large c where
# pow(x, c - 1) or exp(-x**c) become subnormal while the pdf is normal
# (1e14: the pdf stays normal up to x**c = 745).
PARAMS = [(1.7866166930421596,), (1.0,), (2.0,), (3.0,), (0.1,), (0.01,), (1e6,), (1e10,), (1e14,)]


def _expneg(t):
    # exp(-t); for t > 1e6 the value is far below any double (and mpmath's
    # exp of a huge argument is extremely slow), so return 0.
    return mp.mpf(0) if t > 1e6 else mp.exp(-t)


def pdf(x, c):
    return c * x ** (c - 1) * _expneg(x ** c)


def cdf(x, c):
    return 1 - _expneg(x ** c)


def sf(x, c):
    return _expneg(x ** c)


def logsf(x, c):
    return -x ** c


def ppf(q, c):
    return (-mp.log(1 - q)) ** (1 / c)


def isf(q, c):
    return (-mp.log(q)) ** (1 / c)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, c: mp.log(c) + (c - 1) * mp.log(x) - x ** c,
    "cdf": cdf,
    "logcdf": lambda x, c: mp.log(cdf(x, c)),
    "sf": sf,
    "logsf": logsf,
    "ppf": ppf,
    "isf": isf,
}


def x_points(c):
    # x**c in [700, 750] (exp(-x**c) subnormal, x > 1) and
    # (c-1)*log(x) in [-750, -700] (pow(x, c-1) subnormal, x < 1),
    # plus x around 1 where everything happens for large c.
    import numpy as np
    t = np.array([700.0, 708.0, 710.0, 720.0, 730.0, 740.0, 745.0])
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        right = np.exp(np.log(t) / c)
        left = np.exp(-t / (c - 1)) if c > 1 else np.array([])
        mid = np.exp(np.array([-30, -10, -3, -1, 0.0, 1, 2, 3]) / c)
    return np.concatenate([right, left, mid, [1 - 1e-6, 1.0, 1 + 1e-6]])
