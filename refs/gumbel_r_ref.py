"""Reference module for scipy.stats.gumbel_r.

Standard form, no shapes. Docstring pdf: f(x) = exp(-(x + exp(-x))), real x.
cdf F(x) = exp(-exp(-x)); sf = -expm1(-exp(-x)) (mpmath expm1 is exact in
relative terms, so the far right tail is fine even where 1 - F would cancel
past 400 digits, x > ~920); logpdf/logcdf in direct form so huge |x| stay
cheap. ppf = -log(-log q), isf = ppf(1 - q).

Speed guard: for x < -800, exp(-exp(-x)) takes seconds to minutes in mpmath
(exp(-x) > 1e347). There pdf, cdf are below exp(-1e347) and 1 - sf, logsf
likewise, i.e. far outside the double range (never measured), so the
guarded functions return 0 / 1 directly.
"""
import mpmath as mp

DIST = "gumbel_r"

PARAMS = [()]


def logpdf(x):
    return -x - mp.exp(-x)


LEFT = -800


def pdf(x):
    if x < LEFT:
        return mp.mpf(0)
    return mp.exp(logpdf(x))


def cdf(x):
    if x < LEFT:
        return mp.mpf(0)
    return mp.exp(-mp.exp(-x))


def sf(x):
    if x < LEFT:
        return mp.mpf(1)
    return -mp.expm1(-mp.exp(-x))


def ppf(q):
    return -mp.log(-mp.log(q))


def isf(q):
    return ppf(1 - q)


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": lambda x: -mp.exp(-x),
    "sf": sf,
    "logsf": lambda x: mp.log(sf(x)),
    "ppf": ppf,
    "isf": isf,
}


def x_points():
    # Around 0 (mode), where exp(-x) < eps (x ~ 36.7) and where cdf ~ 1,
    # sf/pdf underflow (x ~ 708 .. 745); left tail where exp(-x) is large and
    # pdf/cdf underflow (x ~ -6.6) and logpdf/logcdf overflow (x ~ -709.78).
    import numpy as np
    pos = [1e-300, 1e-100, 1e-20, 1e-10, 1e-5, 0.01, 0.1, 0.5, 1.0, 2.0,
           5.0, 10.0, 20.0, 30.0, 36.0, 36.7, 37.5, 40.0, 50.0, 100.0,
           300.0, 600.0, 700.0, 708.0, 709.0, 709.78, 720.0, 740.0, 744.0,
           745.0, 746.0, 800.0, 1e4, 1e10, 1e100, 1e300, 1.7e308]
    neg = [-1e-300, -1e-20, -1e-10, -1e-5, -0.01, -0.1, -0.5, -1.0, -2.0,
           -3.0, -3.6, -4.0, -5.0, -6.0, -6.5, -6.6, -6.7, -7.0, -10.0,
           -20.0, -100.0, -300.0, -600.0, -700.0, -709.0, -709.7]
    return np.array([0.0] + pos + neg)


def q_points():
    # Tiny q, q near 1, and around q = 1/e (ppf = 0) and q = 1 - 1/e
    # (isf = 0), where -log(-log q) is a log of a number near 1.
    import numpy as np
    e1 = float(mp.exp(-1))
    d = [1e-15, 1e-12, 1e-8, 1e-4, 1e-2]
    pts = [e1, 1 - e1] + [e1 + v for v in d] + [e1 - v for v in d] \
        + [1 - e1 + v for v in d] + [1 - e1 - v for v in d]
    pts += [1e-308, 1e-300, 1e-200, 1e-100, 1e-20, 1e-8, 1e-3, 0.1, 0.5,
            0.9, 1 - 1e-3, 1 - 1e-8, 1 - 1e-12, 1 - 1e-15, 1 - 2.2e-16]
    return np.array(pts)
