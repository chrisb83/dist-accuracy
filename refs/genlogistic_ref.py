"""Reference module for scipy.stats.genlogistic.

Standard form, SciPy's parametrization (docstring pdf):
    f(x; c) = c exp(-x) / (1 + exp(-x))**(c + 1),  x real, c > 0.
    F(x; c) = (1 + exp(-x))**(-c)
    F^-1(q; c) = -log(q**(-1/c) - 1)
"""
import mpmath as mp

DIST = "genlogistic"

# Inventory example; small c (rounding of c - 1 in logpdf, left-tail overflow of
# exp(-x) where the true cdf is still normal, ppf overflow of q**(-1/c));
# c = 1 (logistic); large c (power amplification in cdf, log(c) - x cancellation;
# c = 1e15 makes the cdf error ~10%).
PARAMS = [(0.41192440799679475,), (1e-3,), (1e-10,), (1e-20,), (1.0,), (50.0,), (1e6,), (1e15,)]


def pdf(x, c):
    return c * mp.exp(-x) / (1 + mp.exp(-x)) ** (c + 1)


def cdf(x, c):
    return (1 + mp.exp(-x)) ** (-c)


def sf(x, c):
    return 1 - cdf(x, c)


def ppf(q, c):
    return -mp.log(q ** (-1 / c) - 1)


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
    # Branch x < 0 in logpdf; exp(-x) overflow at x = -709.78; mode at log(c);
    # left-tail points where c*|x| is O(1..700) for small c.
    import numpy as np
    pts = [-1e-8, 0.0, 1e-8, 0.5, -0.5, -700.0, -709.0, -710.0, -720.0, -800.0,
           -1000.0, -1e4, -1e5, np.log(c) - 1, np.log(c), np.log(c) + 1,
           np.log(c) + 20, np.log(c) + 40]
    pts += [-t / c for t in (1e-3, 1.0, 10.0, 100.0, 700.0)]
    return np.array([p for p in pts if np.isfinite(p) and abs(p) < 1e300])


def q_points(c):
    import numpy as np
    return np.array([1e-20, 1e-17, 1e-16, 1e-12, 1e-8, 1e-4, 0.5, 1 - 1e-8])
