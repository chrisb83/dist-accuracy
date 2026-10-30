"""Reference module for scipy.stats.loglaplace.

Standard form. Docstring pdf: f(x, c) = c/2 x**(c-1) for 0 < x < 1,
c/2 x**(-c-1) for x >= 1, c > 0.
cdf F = x**c/2 (x < 1), 1 - x**-c/2 (x >= 1); sf = 1 - F (closed form);
ppf = (2q)**(1/c) for q < 1/2, (2(1-q))**(-1/c) otherwise; isf = ppf(1 - q).
"""
import mpmath as mp

DIST = "loglaplace"

# example; non-dyadic c (0.3, 0.7: c +- 1 and 1/c inexact); c = 1 and 3
# (1/3 inexact, c +- 1 exact); tiny and huge c.
PARAMS = [(3.2505926592051435,), (0.3,), (0.7,), (1.0,), (3.0,), (1e-3,), (50.0,), (1e6,)]


def pdf(x, c):
    if x < 1:
        return c / 2 * x ** (c - 1)
    return c / 2 * x ** (-c - 1)


def cdf(x, c):
    if x < 1:
        return x ** c / 2
    return 1 - x ** (-c) / 2


def sf(x, c):
    # closed form: 1 - cdf would be 0 at 400 digits once sf < 1e-400
    if x < 1:
        return 1 - x ** c / 2
    return x ** (-c) / 2


def ppf(q, c):
    if q < mp.mpf(1) / 2:
        return (2 * q) ** (1 / c)
    return (2 * (1 - q)) ** (-1 / c)


def isf(q, c):
    return ppf(1 - q, c)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, c: mp.log(pdf(x, c)),
    "cdf": cdf,
    "logcdf": lambda x, c: mp.log(cdf(x, c)),
    "sf": sf,
    "logsf": lambda x, c: mp.log(sf(x, c)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(c):
    # branch at x = 1; where x**(+-c) and x**(c-1), x**(-c-1) leave the
    # normal range (|c log x| ~ 708).
    import numpy as np
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        logs = np.array([-708.0, 708.0]) / c
        edges = np.exp(np.concatenate([logs * 0.99, logs * 0.5, logs * 0.1]))
    return np.concatenate([[1 - 1e-12, 1 - 1e-6, 0.999, 1.0, 1.001, 1 + 1e-6, 1 + 1e-12], edges])


def q_points(c):
    import numpy as np
    return np.array([0.5 - 1e-12, 0.5 - 1e-6, 0.5, 0.5 + 1e-6, 0.5 + 1e-12, 0.49, 0.51])
