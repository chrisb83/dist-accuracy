"""Reference module for scipy.stats.kappa3.

Docstring pdf: f(x, a) = a (a + x^a)^{-(a + 1)/a}, x > 0, a > 0.
cdf F(x) = x (a + x^a)^{-1/a}; F^a = x^a/(a + x^a) gives
ppf(q) = (a / (q^-a - 1))^(1/a).
"""
import mpmath as mp

DIST = "kappa3"

# Example (1); small a (amplification by 1/a, cancellation in q**-a - 1);
# moderate and large a (x**a / q**-a / expm1 overflow).
PARAMS = [(1.0,), (1e-3,), (1e-6,), (1e-12,), (0.1,), (2.0,), (10.0,), (20.0,), (1e3,)]


def pdf(x, a):
    return a * (a + x ** a) ** (-1 / a - 1)


def cdf(x, a):
    return x * (a + x ** a) ** (-1 / a)


def sf(x, a):
    return 1 - cdf(x, a)


def ppf(q, a):
    return (a / (q ** (-a) - 1)) ** (1 / a)


def isf(q, a):
    return ppf(1 - q, a)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, a: mp.log(pdf(x, a)),
    "cdf": cdf,
    "logcdf": lambda x, a: mp.log(cdf(x, a)),
    "sf": sf,
    "logsf": lambda x, a: mp.log(sf(x, a)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(a):
    # x**a overflow (a*log x = 709.8) and underflow (-745), both sides;
    # x near 1 (sf cutoff region for large a).
    import numpy as np
    with np.errstate(over="ignore", under="ignore"):
        logs = np.array([709.8, -745.0, 600.0]) / a
        edges = np.exp(np.concatenate([logs * 0.99, logs * 1.01]))
    return np.concatenate([[0.5, 1 - 1e-6, 1.0, 1 + 1e-6, 1.01, 1.1, 2.0, 3.0], edges])


def q_points(a):
    import numpy as np
    # q**-a overflow (q = exp(-709.8/a)) on both sides
    with np.errstate(over="ignore", under="ignore"):
        e = np.exp(-709.8 / a * np.array([0.99, 1.01]))
    return np.concatenate([[0.4, 0.49, 0.5, 0.51, 0.6, 0.9, 0.99], e])
