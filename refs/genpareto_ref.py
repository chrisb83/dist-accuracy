"""Reference module for scipy.stats.genpareto.

Parametrization (docstring pdf, standard form loc=0, scale=1):

    f(x, c) = (1 + c x)^(-1 - 1/c),   f(x, 0) = exp(-x)

support x >= 0 for c >= 0 and 0 <= x <= -1/c for c < 0.

    sf(x)  = (1 + c x)^(-1/c)       = exp(-log1p(c x)/c)
    cdf    = 1 - sf
    ppf(q) = ((1 - q)^(-c) - 1)/c   = expm1(-c log1p(-q))/c
    isf(q) = (q^(-c) - 1)/c         = expm1(-c log q)/c

log1p/expm1 forms are used only so that |c| down to 1e-300 needs no extra
precision (the plain power forms would need > 300 digits for c = 1e-300).
"""
import numpy as np
import mpmath as mp

DIST = "genpareto"

PARAMS = [
    (0.1,),        # inventory example
    (0.0,),        # c == 0 fill branch (exponential)
    (1.0,),        # heavy tail, |logpdf| large at x ~ 1e300
    (-0.5,),       # bounded support [0, 2]
    (-2.0,),       # bounded, pdf infinite at endpoint
    (1e-8,),       # near-exponential, c > 0
    (-1e-8,),      # near-exponential, c < 0
    (1e-300,),     # inv_boxcox1p |lmbda*x| < 1e-154 shortcut; c*x subnormal
    (-1e-160,),    # same shortcut, c < 0
    (1e3,),        # expm1 overflow window in isf/ppf
    (1e10,),       # c*x overflows for x > 1.8e298
]


def _u(x, c):
    u = 1 + c * x
    return u if u > 0 else mp.mpf(0)


def logsf(x, c):
    if c == 0:
        return -x
    u = _u(x, c)
    if u == 0:
        return mp.ninf
    return -mp.log1p(c * x) / c


def sf(x, c):
    return mp.exp(logsf(x, c))


def cdf(x, c):
    if c == 0:
        return -mp.expm1(-x)
    return -mp.expm1(logsf(x, c))


def logpdf(x, c):
    if c == 0:
        return -x
    if c == -1:
        return mp.mpf(0)
    u = _u(x, c)
    if u == 0:
        return mp.inf if c < -1 else mp.ninf
    return -(1 + 1 / c) * mp.log1p(c * x)


def pdf(x, c):
    return mp.exp(logpdf(x, c))


def ppf(q, c):
    if c == 0:
        return -mp.log1p(-q)
    return mp.expm1(-c * mp.log1p(-q)) / c


def isf(q, c):
    if c == 0:
        return -mp.log(q)
    return mp.expm1(-c * mp.log(q)) / c


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": lambda x, c: mp.log(cdf(x, c)),
    "sf": sf,
    "logsf": logsf,
    "ppf": ppf,
    "isf": isf,
}


def x_points(c):
    pts = [0.5, 1.0, 2.0, 10.0, 100.0]
    # inv_boxcox1p shortcut |c*x| < 1e-154: both sides
    if c == 0:
        return np.asarray(pts + [700.0, 708.0, 745.0], float)
    t = 1e-154 / abs(c)
    pts += [t * 0.5, t * 2]
    # c*x overflow (x > 1.8e308/c) and c*x subnormal (x < 2.2e-308/|c|)
    for e in (1.8e308 / abs(c), 2.2e-308 / abs(c), 4.9e-324 / abs(c)):
        pts += [e * 0.5, e * 0.99, e * 1.01, e * 2]
    if c > 1:
        pts += [1e299, 1e300, 1e305, 1e308]
    if c < 0:
        b = -1.0 / c
        pts += [b * (1 - 1e-6), b * (1 - 1e-12), b * (1 - 1e-15), b * 0.999]
    pts = np.asarray(pts, float)
    return pts[np.isfinite(pts) & (pts > 0)]


def q_points(c):
    pts = [0.25, 0.5, 0.75]
    if c > 1:
        # expm1(c*L) overflows for c*L > 709.78 while expm1(c*L)/c is finite
        # up to c*L = 709.78 + log(c); L = -log(q) (isf), -log1p(-q) (ppf)
        lo, hi = 709.78 / c, (709.78 + np.log(c)) / c
        for L in np.linspace(lo * 0.999, hi * 0.999, 6):
            pts += [np.exp(-L), -np.expm1(-L)]
    return np.asarray(pts, float)
