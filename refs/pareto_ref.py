"""Reference module for scipy.stats.pareto.

Standard form, SciPy's parametrization from the docstring:
f(x; b) = b / x**(b+1), x >= 1, b > 0.
cdf = 1 - x**-b, sf = x**-b, ppf = (1 - q)**(-1/b), isf = q**(-1/b).
"""
import mpmath as mp

DIST = "pareto"

# Inventory example, typical b, tiny b (cdf/ppf cancellation), very tiny b
# (-b-1 rounds to -1 in pdf), large b (subnormal x**(-b-1) in pdf).
PARAMS = [(2.621716532144454,), (1.0,), (1e-3,), (1e-8,), (1e-17,), (1e3,), (1e6,)]


def pdf(x, b):
    return b * x ** (-b - 1)


def cdf(x, b):
    return 1 - x ** (-b)


def sf(x, b):
    return x ** (-b)


def ppf(q, b):
    return (1 - q) ** (-1 / b)


def isf(q, b):
    return q ** (-1 / b)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, b: mp.log(pdf(x, b)),
    "cdf": cdf,
    "logcdf": lambda x, b: mp.log(cdf(x, b)),
    "sf": sf,
    "logsf": lambda x, b: mp.log(sf(x, b)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(b):
    # Close to the support edge x = 1 (cdf cancellation), and where
    # x**(-b-1) becomes subnormal: (b+1)*log(x) in [708, 745].
    import numpy as np
    near1 = 1 + np.array([2.3e-16, 1e-15, 1e-13, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2])
    with np.errstate(over="ignore"):
        edges = np.exp(np.array([700.0, 708.0, 715.0, 730.0, 740.0]) / (b + 1))
    return np.concatenate([near1, edges])


def q_points(b):
    import numpy as np
    return np.array([1e-17, 1e-16, 5e-16, 1e-14, 1e-12, 1e-10])
