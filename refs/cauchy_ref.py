"""Reference module for scipy.stats.cauchy.

Standard form, no shapes. Docstring pdf: f(x) = 1/(pi (1 + x**2)), real x.
cdf F(x) = 1/2 + atan(x)/pi, sf = 1 - F, ppf = tan(pi (q - 1/2)),
isf = ppf(1 - q).
"""
import mpmath as mp

DIST = "cauchy"

PARAMS = [()]


def pdf(x):
    return 1 / (mp.pi * (1 + x * x))


def cdf(x):
    return mp.mpf(1) / 2 + mp.atan(x) / mp.pi


def sf(x):
    return 1 - cdf(x)


def ppf(q):
    return mp.tan(mp.pi * (q - mp.mpf(1) / 2))


def isf(q):
    return ppf(1 - q)


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
    # logpdf branch at |x| = 1 (both sides), where x*x overflows
    # (|x| ~ 1.34e154) and underflows, both signs, near 0.
    import numpy as np
    pts = [0.0, 1e-300, 1e-160, 1e-20, 1e-8, 1e-4, 0.5, 1 - 1e-12, 1 - 1e-16,
           1.0, 1 + 2.2e-16, 1 + 1e-12, 1.5, 2.0, 1e8, 1e16, 1e100, 1e150,
           1.3e154, 1.4e154, 1e200, 1e300, 1e307, 1.7e308]
    pts = pts + [-p for p in pts if p != 0]
    return np.array(pts)


def q_points():
    # Both sides of q = 0.5 (ppf/isf ~ 0 there), q near 1, tiny q.
    import numpy as np
    d = [1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.25]
    pts = [0.5] + [0.5 + v for v in d] + [0.5 - v for v in d]
    pts += [1 - 2.2e-16, 1 - 1e-15, 1 - 1e-12, 1 - 1e-8, 0.75, 0.25,
            1e-308, 1e-300, 1e-200, 1e-100, 1e-20]
    return np.array(pts)
