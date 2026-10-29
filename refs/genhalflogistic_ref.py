"""Reference module for scipy.stats.genhalflogistic.

Docstring pdf (standard form, 0 <= x <= 1/c, c > 0):
    f(x; c) = 2 (1 - c x)**(1/c - 1) / (1 + (1 - c x)**(1/c))**2
cdf = (1 - t)/(1 + t) with t = (1 - c x)**(1/c);  sf = 2 t/(1 + t).
ppf: t = (1 - q)/(1 + q),  x = (1 - t**c)/c.   isf: t = q/(2 - q).
"""
import mpmath as mp
import numpy as np

DIST = "genhalflogistic"

# example; c = 1 (exponent 0); c < 1 / c > 1 (pdf finite / infinite at 1/c);
# small c (1/c large power, half-logistic limit); large c (tiny support).
PARAMS = [(0.7727472780992932,), (1.0,), (0.5,), (3.0,), (1e-3,), (1e-8,),
          (1e-14,), (1e6,)]


def _u(x, c):
    # the double 1/c (upper support end used by the harness) can exceed the
    # exact 1/c; such x lie outside the support, clamp to the endpoint.
    return max(1 - c * x, mp.mpf(0))


def _t(x, c):
    return _u(x, c) ** (1 / c)


def pdf(x, c):
    u = _u(x, c)
    if u == 0:
        return mp.inf if c > 1 else (mp.mpf(2) if c == 1 else mp.mpf(0))
    return 2 * u ** (1 / c - 1) / (1 + _t(x, c)) ** 2


def cdf(x, c):
    t = _t(x, c)
    return (1 - t) / (1 + t)


def sf(x, c):
    t = _t(x, c)
    return 2 * t / (1 + t)


def ppf(q, c):
    t = (1 - q) / (1 + q)
    return (1 - t ** c) / c


def isf(q, c):
    t = q / (2 - q)
    return (1 - t ** c) / c


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
    b = 1.0 / c
    pts = [0.0, 1e-300, 1e-20, 1e-17, 1e-16, 1e-12, 1e-8, 1e-4, 0.01, 0.5, 1.0, 2.0, 10.0, 30.0]
    # near the upper endpoint 1/c
    for r in (1e-15, 1e-12, 1e-8, 1e-4, 1e-2, 0.1, 0.5):
        pts.append(b * (1 - r))
    pts += [b * 1e-3, b * 1e-6]
    pts = np.asarray(pts, float)
    return pts[(pts >= 0) & (pts <= b)]


def q_points(c):
    return np.asarray([1e-300, 1e-100, 1e-20, 1e-17, 1e-16, 1e-12, 1e-8, 1e-4,
                       0.1, 0.5, 0.9, 1 - 1e-4, 1 - 1e-8, 1 - 1e-12, 1 - 1e-15], float)
