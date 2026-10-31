"""Reference module for scipy.stats.lomax.

Standard form, SciPy's parametrization from the docstring:
f(x; c) = c / (1 + x)**(c + 1), x >= 0, c > 0.
sf = (1 + x)**(-c), cdf = 1 - sf, ppf = (1 - q)**(-1/c) - 1, isf = q**(-1/c) - 1.
"""
import mpmath as mp

DIST = "lomax"

# Inventory example, c = 1, tiny c (c + 1.0 rounds to 1; isf/ppf overflow
# quickly), small c, large c (rounding of 1 + x raised to c + 1; isf
# cancellation in q**(-1/c) - 1).
PARAMS = [(1.8771398388773268,), (1.0,), (1e-17,), (1e-3,), (1e3,), (1e6,)]


def sf(x, c):
    return (1 + x) ** (-c)


def pdf(x, c):
    return c * (1 + x) ** (-c - 1)


def cdf(x, c):
    return 1 - sf(x, c)


def ppf(q, c):
    return (1 - q) ** (-1 / c) - 1


def isf(q, c):
    return q ** (-1 / c) - 1


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, c: mp.log(pdf(x, c)),
    "cdf": cdf,
    "logcdf": lambda x, c: mp.log(cdf(x, c)),
    "sf": sf,
    "logsf": lambda x, c: -c * mp.log(1 + x),
    "ppf": ppf,
    "isf": isf,
}


def x_points(c):
    # Where (1 + x)**(c + 1) overflows ((c+1)*log1p(x) > 709.8) and where the
    # pdf/sf underflow ((c+1)*log1p(x) ~ 708..745), the zero of logpdf
    # (x = c**(1/(c+1)) - 1), and small x where 1.0 + x rounds.
    import numpy as np
    with np.errstate(over="ignore"):
        edges = np.expm1(np.array([300.0, 600.0, 700.0, 709.0, 710.0, 720.0, 740.0]) / (c + 1))
        zero = np.expm1(np.log(c) / (c + 1))
    small = np.array([1e-17, 1e-16, 3e-16, 1e-15, 1e-12, 1e-9])
    return np.concatenate([edges, [zero, zero * (1 + 1e-6), zero * (1 - 1e-6)], small])


def q_points(c):
    import numpy as np
    return np.array([1e-17, 1e-16, 5e-16, 1e-14, 1e-12, 1e-10, 0.5, 0.9, 1 - 1e-6, 1 - 1e-10, 1 - 1e-14])
