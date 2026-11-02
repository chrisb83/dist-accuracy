"""Reference module for scipy.stats.truncexpon.

Standard form, shape b > 0, support [0, b]:
    f(x, b) = exp(-x) / (1 - exp(-b))
    F(x, b) = (1 - exp(-x)) / (1 - exp(-b))
    ppf(q)  = -log(1 - q (1 - exp(-b)))

expm1/log1p are used so the formulas stay exact for b down to 1e-300 even in
the harness's lower-precision stability check.
"""
import mpmath as mp

DIST = "truncexpon"

# Example value, tiny b (exp(-b) ~ 1: cancellation in sf/isf), moderate and
# large b (log(-expm1(-b)) rounds to 0 in logpdf), b > 745 (exp(-b) underflows).
PARAMS = [(4.690772545681048,), (1e-300,), (1e-12,), (1e-6,), (0.5,),
          (20.0,), (50.0,), (800.0,)]


def pdf(x, b):
    return mp.exp(-x) / -mp.expm1(-b)


def cdf(x, b):
    return mp.expm1(-x) / mp.expm1(-b)


def sf(x, b):
    return 1 - cdf(x, b)


def ppf(q, b):
    return -mp.log1p(q * mp.expm1(-b))


def isf(q, b):
    return ppf(1 - q, b)


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
    import numpy as np
    pts = [0.0, 1e-300, 1e-20, 1e-12, 1e-6, 0.5 * b, b * (1 - 1e-6), b]
    pts += [v for v in (1.0, 10.0, 30.0, 700.0, 708.0, 709.0, 740.0, 744.0) if v < b]
    return np.array(pts)
