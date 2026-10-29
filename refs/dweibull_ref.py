"""Reference module for scipy.stats.dweibull.

Standard form, docstring pdf: f(x; c) = c/2 |x|**(c-1) exp(-|x|**c), x real, c > 0.
cdf(x) = 0.5 exp(-|x|**c) for x <= 0, 1 - 0.5 exp(-x**c) for x > 0.
"""
import mpmath as mp

DIST = "dweibull"

# Example parameter, exact c = 1 (Laplace) and 2, inexact 1/c (3), small c
# (c - 1, 1/c inexact; huge/tiny quantiles) and large c (subnormal
# intermediates near |x| = 1, pdf nan from inf*0).
PARAMS = [(2.0685080649914673,), (1.0,), (2.0,), (3.0,), (0.3,), (0.1,), (0.01,), (1e6,), (1e10,)]


def _expneg(t):
    # exp(-t); for t > 1e6 far below any double (mpmath exp of huge args is slow).
    return mp.mpf(0) if t > 1e6 else mp.exp(-t)


def pdf(x, c):
    ax = abs(x)
    return c / 2 * ax ** (c - 1) * _expneg(ax ** c)


def logpdf(x, c):
    ax = abs(x)
    return mp.log(c / 2) + (c - 1) * mp.log(ax) - ax ** c


def cdf(x, c):
    h = _expneg(abs(x) ** c) / 2
    return h if x <= 0 else 1 - h


def sf(x, c):
    return 1 - cdf(x, c)


def ppf(q, c):
    if q <= mp.mpf(1) / 2:
        return -(-mp.log(2 * q)) ** (1 / c)
    return (-mp.log(2 * (1 - q))) ** (1 / c)


def isf(q, c):
    return ppf(1 - q, c)


def logcdf(x, c):
    t = abs(x) ** c
    if x <= 0:
        return -mp.log(2) - t
    return mp.log(cdf(x, c))


def logsf(x, c):
    t = abs(x) ** c
    if x >= 0:
        return -mp.log(2) - t
    return mp.log(sf(x, c))


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": logcdf,
    "sf": sf,
    "logsf": logsf,
    "ppf": ppf,
    "isf": isf,
}


def x_points(c):
    # |x|**c in [700, 745] (exp(-|x|**c) subnormal), (c-1)*log|x| in
    # [-750, -700] (pow subnormal), |x| around 1, both signs.
    import numpy as np
    t = np.array([20.0, 100.0, 700.0, 708.0, 710.0, 720.0, 730.0, 740.0, 745.0])
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        right = np.exp(np.log(t) / c)
        left = np.exp(-t[2:] / (c - 1)) if c > 1 else np.array([])
        mid = np.exp(np.array([-30, -10, -3, -1, 0.0, 1, 2, 3]) / c)
    pos = np.concatenate([right, left, mid, [1 - 1e-6, 1.0, 1 + 1e-6]])
    return np.concatenate([pos, -pos])


def q_points(c):
    # around the median (branch q <= 0.5) and symmetric tails
    import numpy as np
    d = 10.0 ** np.array([-15.0, -12, -8, -4, -2])
    return np.concatenate([0.5 - d, [0.5], 0.5 + d])
