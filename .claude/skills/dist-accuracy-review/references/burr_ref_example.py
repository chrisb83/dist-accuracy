"""Reference module for scipy.stats.burr (worked example).

Copy this file to refs/<dist>_ref.py and adapt it. Rules it follows:

* Standard form only (loc=0, scale=1); shapes in SciPy's parametrization,
  taken from the docstring pdf:  f(x; c, d) = c d x**(-c-1) (1 + x**-c)**(-d-1).
* Defining formulas, evaluated at the harness precision (400 digits): any
  algebraically equivalent form is fine at that precision, so cancellation in
  1 - cdf or q**(-1/d) - 1 needs no special treatment here.
* Every function takes mpmath numbers and returns an mpmath number.
"""
import mpmath as mp

DIST = "burr"

# Typical values, small/large shapes (small d with large c keeps isf results
# representable), and c*d = 1 (exact cancellation in logpdf).
PARAMS = [(10.5, 4.3), (2.0, 3.0), (0.5, 1e-3), (100.0, 1e-3), (2.0, 0.5), (3.0, 1e4), (100.0, 1e6)]


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
    # Branch boundaries in the code: x == 0 (pdf/logpdf) and around x = 1,
    # plus both sides of where x**-c overflows (c*log(x) < -709.8) and
    # underflows (c*log(x) > 745). The harness drops non-finite points.
    import numpy as np
    with np.errstate(over="ignore", under="ignore"):
        logs = np.array([-709.8, -745.0, 709.8, 745.0]) / c
        edges = np.exp(np.concatenate([logs * 0.99, logs * 1.01]))
    return np.concatenate([[0.5, 1 - 1e-6, 1.0, 1 + 1e-6, 2.0], edges])
