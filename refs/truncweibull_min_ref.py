"""Reference module for scipy.stats.truncweibull_min.

Standard form, shapes (c, a, b) in SciPy's order, support a < x <= b:
    f(x) = c x**(c-1) exp(-x**c) / (exp(-a**c) - exp(-b**c))
    F(x) = (exp(-a**c) - exp(-x**c)) / (exp(-a**c) - exp(-b**c))
    ppf(q) = (-log((1-q) exp(-a**c) + q exp(-b**c)))**(1/c)

Differences of exponentials are written with expm1 (factoring out exp(-a**c))
so that they stay exact for tiny a**c, b**c in the lower-precision stability
check as well.
"""
import mpmath as mp

DIST = "truncweibull_min"

# Example; c=1, a=0 (truncexpon); small b (denominator 1 - exp(-b**c) cancels);
# small a and b with a > 0; large a (exp(-a**c) underflows); tiny c (a**c ~ b**c ~ 1);
# a**c ~ 530 (exp of a rounded large power); a = 0 with large c (1 - exp(-x**c));
PARAMS = [(2.5, 0.25, 1.75), (1.0, 0.0, 4.69), (2.0, 0.0, 1e-6),
          (0.5, 1e-10, 1e-8), (2.0, 30.0, 31.0), (1e-3, 0.5, 2.0),
          (2.3, 15.3, 16.0), (5.0, 0.0, 3.0),
          # narrow interval away from 0; a**c = 718 (exp(-a**c) subnormal);
          # moderately small b**c = 1e-6 with a = 0
          (2.0, 1.0, 1.00000001), (2.0, 26.8, 27.5), (3.0, 0.0, 0.01)]


def _denum(c, a, b):
    A, B = a ** c, b ** c
    return mp.exp(-A) * -mp.expm1(A - B)


def pdf(x, c, a, b):
    return c * x ** (c - 1) * mp.exp(-x ** c) / _denum(c, a, b)


def cdf(x, c, a, b):
    A, X = a ** c, x ** c
    return mp.exp(-A) * -mp.expm1(A - X) / _denum(c, a, b)


def sf(x, c, a, b):
    X, B = x ** c, b ** c
    return mp.exp(-X) * -mp.expm1(X - B) / _denum(c, a, b)


def ppf(q, c, a, b):
    A, B = a ** c, b ** c
    return (A - mp.log1p(q * mp.expm1(A - B))) ** (1 / c)


def isf(q, c, a, b):
    return ppf(1 - q, c, a, b)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, c, a, b: mp.log(pdf(x, c, a, b)),
    "cdf": cdf,
    "logcdf": lambda x, c, a, b: mp.log(cdf(x, c, a, b)),
    "sf": sf,
    "logsf": lambda x, c, a, b: mp.log(sf(x, c, a, b)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(c, a, b):
    import numpy as np
    w = b - a
    pts = [a + w * f for f in (1e-12, 1e-6, 1e-3, 0.1, 0.5, 0.9)]
    pts += [b - w * f for f in (1e-3, 1e-6, 1e-12)] + [b]
    if a == 0:
        pts += [b * f for f in (1e-300, 1e-100, 1e-20, 1e-8)]
    return np.array([p for p in pts if a < p <= b])
