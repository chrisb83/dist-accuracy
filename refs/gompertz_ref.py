"""Reference module for scipy.stats.gompertz.

Docstring pdf: f(x, c) = c exp(x) exp(-c (e^x - 1)), x >= 0, c > 0.
cdf = 1 - exp(-c (e^x - 1)), closed-form inverses.
"""
import mpmath as mp

DIST = "gompertz"

# Example, c = 1 exactly (log(c) = 0: logpdf cancellation at small x),
# c near 1, tiny and huge c; c = 1e-307 and 1e-310 make c*expm1(x) and
# -log(q)/c overflow while the result is finite.
PARAMS = [(0.9474371307510525,), (1.0,), (1.0 + 1e-8,), (1e-3,), (1e-300,),
          (1e-307,), (1e-310,), (1e3,), (1e6,)]


def _u(x, c):
    return c * mp.expm1(x)


# exp(-u) with u > 1e6 is far below the double range (and mpmath is very
# slow there); return a tiny exact sentinel so such points are not measured.
_TINY = mp.mpf("1e-1000000")


def _expneg(u):
    return _TINY if u > 1e6 else mp.exp(-u)


def pdf(x, c):
    return mp.exp(mp.log(c) + x - _u(x, c)) if _u(x, c) < 1e6 else _TINY


def logpdf(x, c):
    return mp.log(c) + x - _u(x, c)


def cdf(x, c):
    return 1 - _expneg(_u(x, c))


def sf(x, c):
    return _expneg(_u(x, c))


def ppf(q, c):
    return mp.log(1 - mp.log(1 - q) / c)


def isf(q, c):
    return mp.log(1 - mp.log(q) / c)


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": lambda x, c: mp.log(cdf(x, c)),
    "sf": sf,
    "logsf": lambda x, c: -_u(x, c),
    "ppf": ppf,
    "isf": isf,
}


def x_points(c):
    # small x (logpdf cancellation for c ~ 1), expm1 overflow at 709.78,
    # and where c*e^x reaches ~1 and ~745 (sf underflow).
    import numpy as np
    pts = [1e-12, 1e-8, 1e-5, 1e-3, 0.1, 0.5, 1.0, 709.0, 709.7, 709.78, 709.79, 709.8, 710.0, 720.0, 740.0]
    for t in (1.0, 50.0, 700.0, 745.0):
        v = np.log(t / c) if t / c < 1e300 else np.log(t) - np.log(c)
        if v > 0:
            pts += [v * 0.999, v, v * 1.001]
    return np.array(pts)
