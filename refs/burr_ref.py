"""Reference module for scipy.stats.burr.

Standard form, SciPy parametrization from the docstring pdf:
    f(x; c, d) = c d x**(-c-1) (1 + x**-c)**(-d-1),  x > 0, c, d > 0.
cdf = (1 + x**-c)**-d; ppf closed form; everything at harness precision.
"""
import mpmath as mp
import numpy as np

DIST = "burr"

# example params; c*d = 1 (logpdf cancellation); small d with small/large c
# (ppf/isf overflow); large d (power amplification, ppf cancellation);
# large c*d (subnormal x**(-c-1)); small c; fisk-like d = 1.
PARAMS = [(10.5, 4.3), (2.0, 3.0), (0.5, 1e-3), (100.0, 1e-3), (2.0, 0.5),
          (3.0, 1e4), (100.0, 1e6), (0.01, 10.0), (1.0, 1.0)]


def pdf(x, c, d):
    return c * d * x ** (-c - 1) * (1 + x ** (-c)) ** (-d - 1)


def cdf(x, c, d):
    return (1 + x ** (-c)) ** (-d)


def sf(x, c, d):
    return 1 - cdf(x, c, d)


def ppf(q, c, d):
    return (q ** (-1 / d) - 1) ** (-1 / c)


def isf(q, c, d):
    return ppf(1 - q, c, d)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, c, d: mp.log(pdf(x, c, d)),
    "cdf": cdf,
    "logcdf": lambda x, c, d: mp.log(cdf(x, c, d)),
    "sf": sf,
    "logsf": lambda x, c, d: mp.log(sf(x, c, d)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(c, d):
    # x == 1 neighbourhood, and both sides of where x**-c overflows
    # (c*log x < -709.8) / becomes subnormal or underflows (c*log x > 708..745),
    # and where x**(-c-1) becomes subnormal.
    with np.errstate(over="ignore", under="ignore"):
        logs = np.array([-709.8, -745.0, 708.0, 745.0]) / c
        logs1 = np.array([708.0, 745.0]) / (c + 1)
        edges = np.exp(np.concatenate([logs * 0.99, logs * 1.01,
                                       logs1 * 0.99, logs1 * 1.01]))
    return np.concatenate([[0.5, 1 - 1e-6, 1.0, 1 + 1e-6, 2.0, 1e-10], edges])
