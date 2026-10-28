"""mpmath reference for scipy.stats.burr12 (Burr type XII), standard form.

Docstring pdf:  f(x; c, d) = c d x**(c-1) / (1 + x**c)**(d+1),  x > 0, c, d > 0.
sf = (1 + x**c)**(-d), cdf = 1 - sf,
ppf(q) = ((1 - q)**(-1/d) - 1)**(1/c), isf(q) = (q**(-1/d) - 1)**(1/c).
Simplest defining forms; cancellation is harmless at 400 digits.
"""
import numpy as np
import mpmath as mp

DIST = "burr12"

# inventory example; d = 1 (fisk); small d with large c (x**c overflows while
# pdf/sf stay normal; expm1 overflows in ppf/isf); large d (rounding raised to
# power d, subnormal x**c lifted by d); small c; large c with tiny d (logpdf
# term cancellation); huge d (subnormal expm1 in ppf).
PARAMS = [
    (10.0, 4.0),
    (2.0, 1.0),
    (100.0, 1e-3),
    (3.0, 1e6),
    (1e-3, 2.0),
    (0.5, 1e-3),
    (1e4, 1e-6),
    (10.0, 1e12),
]


def sf(x, c, d):
    return (1 + x ** c) ** (-d)


def cdf(x, c, d):
    return 1 - sf(x, c, d)


def pdf(x, c, d):
    return c * d * x ** (c - 1) * (1 + x ** c) ** (-d - 1)


def ppf(q, c, d):
    return ((1 - q) ** (-1 / d) - 1) ** (1 / c)


def isf(q, c, d):
    return (q ** (-1 / d) - 1) ** (1 / c)


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
    """Both sides of the thresholds: x**c overflow/underflow/subnormal,
    1 + x**c rounding to 1 (x**c ~ eps), and x = 1."""
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        a = np.array([709.8, 745.0, -708.0, -709.8, -715.0, -720.0, -730.0,
                      -740.0, -745.0, -36.0, -36.7, -37.0, -40.0, -50.0]) / c
        logs = np.concatenate([a])
        edges = np.exp(np.concatenate([logs * 0.999, logs, logs * 1.001]))
        # d * x**c ~ t: bulk of the distribution for large d, where
        # (1 + x**c)**(-d) in logcdf amplifies rounding (pattern 6)
        t = np.array([1e-3, 0.1, 0.5, 1.0, 3.0, 10.0, 100.0])
        bulk = np.exp((np.log(t) - np.log(d)) / c)
    return np.concatenate([[0.5, 1 - 1e-8, 1.0, 1 + 1e-8, 2.0], edges, bulk])


def q_points(c, d):
    """q where expm1(-log1p(-q)/d) or expm1(-log(q)/d) passes 709.8."""
    pts = [0.5, 0.9, 0.99]
    for z in (700.0, 709.0, 710.0, 720.0):
        t = z * d
        if t < 745:
            pts.append(float(np.exp(-t)))        # isf: -log(q)/d = z
            pts.append(float(-np.expm1(-t)))     # ppf: -log1p(-q)/d = z
    return np.array([p for p in pts if 0 < p < 1])
