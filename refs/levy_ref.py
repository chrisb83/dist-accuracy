"""Reference module for scipy.stats.levy.

Standard form, no shapes. Docstring pdf:
    f(x) = exp(-1/(2x)) / sqrt(2 pi x^3),  x > 0.
cdf F(x) = erfc(1/sqrt(2x)); sf = 1 - F = erf(1/sqrt(2x));
ppf(q) = 1/(2 erfinv(1 - q)^2) (erfcinv(q) = erfinv(1 - q), 1 - q exact at
400 digits); isf(q) = ppf(1 - q) = 1/(2 erfinv(q)^2).
"""
import mpmath as mp

DIST = "levy"

PARAMS = [()]


def pdf(x):
    return mp.exp(-1 / (2 * x)) / mp.sqrt(2 * mp.pi * x ** 3)


def cdf(x):
    return mp.erfc(1 / mp.sqrt(2 * x))


def sf(x):
    return 1 - cdf(x)


def ppf(q):
    return 1 / (2 * mp.erfinv(1 - q) ** 2)


def isf(q):
    return 1 / (2 * mp.erfinv(q) ** 2)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x: mp.log(pdf(x)),
    "cdf": cdf,
    "logcdf": lambda x: mp.log(cdf(x)),
    "sf": sf,
    "logsf": lambda x: mp.log(sf(x)),
    "ppf": ppf,
    "isf": isf,
}


def x_points():
    # pdf: exp(-1/(2x)) becomes subnormal for 1/(2x) > 708.4 (x < 7.06e-4)
    # and underflows for 1/(2x) > 745.1 (x < 6.71e-4) while the prefactor
    # x**-1.5 ~ 5e4 keeps the pdf normal a little longer.
    # cdf: erfc(z) underflows near z = 26.5 (x ~ 7.1e-4).
    # sf: 0.5/x is subnormal for x > 2.25e307.
    import numpy as np
    pts = [6.0e-4, 6.5e-4, 6.6e-4, 6.65e-4, 6.7e-4, 6.75e-4, 6.8e-4, 6.9e-4,
           7.0e-4, 7.06e-4, 7.1e-4, 7.2e-4, 7.5e-4, 8e-4, 1e-3, 2e-3, 5e-3,
           0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 1e10, 1e100, 1e200, 1e300,
           1e305, 1e306, 1e307, 2e307, 2.2e307, 2.3e307, 3e307, 5e307,
           1e308, 1.5e308, 1.7e308]
    return np.array(pts)


def q_points():
    # ppf: q/2 subnormal for q < 4.45e-308; isf: erfinv(q)**2 is subnormal
    # or zero for q < ~1.7e-154 (isf ~ 2/(pi q^2) normal for q > ~6e-155).
    import numpy as np
    pts = [2.3e-308, 3e-308, 4e-308, 4.4e-308, 4.5e-308, 1e-307, 1e-306,
           1e-200, 1e-160, 1e-155, 6e-155, 7e-155, 1e-154, 1.5e-154,
           2e-154, 1e-153, 1e-150, 1e-100, 1e-20, 1e-10, 0.3, 0.5, 0.9,
           1 - 1e-6, 1 - 1e-10, 1 - 1e-14, 1 - 2.2e-16]
    return np.array(pts)
