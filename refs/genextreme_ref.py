"""Reference module for scipy.stats.genextreme.

Parametrization (docstring pdf, standard form loc=0, scale=1):

    f(x, c) = exp(-(1 - c x)^(1/c)) (1 - c x)^(1/c - 1),   c != 0
    f(x, 0) = exp(-exp(-x)) exp(-x)                        (gumbel_r)

support x <= 1/c for c > 0, x >= 1/c for c < 0, the real line for c == 0.

With ll = log(-log F) = log1p(-c x)/c (= -x for c == 0):

    cdf    = exp(-exp(ll))
    logpdf = -exp(ll) + ll - log1p(-c x)
    ppf(q) = -expm1(-c y)/c,  y = -log(-log q)   (= (1 - (-log q)^c)/c)
    isf(q) = ppf(1 - q)

log1p/expm1 are used only so that tiny |c| needs no extra precision.
"""
import numpy as np
import mpmath as mp

DIST = "genextreme"

PARAMS = [
    (-0.1,),        # inventory example
    (0.0,),         # c == 0 fill branch (gumbel_r)
    (0.5,),         # bounded above, pdf -> 0 at endpoint
    (2.0,),         # bounded above, pdf -> inf at endpoint
    (1.00000001,),  # lpex2 - lex2 = L(1/c - 1) cancels (c near 1)
    (1e-12,),       # near gumbel, c > 0
    (-1e-12,),      # near gumbel, c < 0
    (1e3,),         # c*x overflow for x < -1.8e305; expm1 overflow window in ppf/isf
    (-1e3,),        # same, c < 0
]


def _u(x, c):
    return 1 - c * x


def loglog(x, c):
    if c == 0:
        return -x
    u = _u(x, c)
    if u <= 0:
        # at/beyond the endpoint: -log F = 0 (c > 0, F = 1) or inf (c < 0, F = 0)
        return mp.ninf if c > 0 else mp.inf
    return mp.log1p(-c * x) / c


def logcdf(x, c):
    ll = loglog(x, c)
    if ll > 1e6:
        return mp.ninf   # -exp(ll) is far below -1.8e308
    if ll == mp.ninf:
        return mp.mpf(0)
    if ll == mp.inf:
        return mp.ninf
    return -mp.exp(ll)


def _exp(v):
    # exp of a hugely negative argument is far below the double range; mpmath
    # would otherwise spend forever on exp(-exp(1e300)).
    if v < -1e6:
        return mp.mpf(0)
    return mp.exp(v)


def cdf(x, c):
    return _exp(logcdf(x, c))


def sf(x, c):
    return -mp.expm1(logcdf(x, c))


def logpdf(x, c):
    if c == 0:
        if -x > 1e6:
            return mp.ninf   # far below -1.8e308
        return -mp.exp(-x) - x
    u = _u(x, c)
    if u <= 0:
        if c > 0:
            if c == 1:
                return mp.mpf(0)
            return mp.ninf if c < 1 else mp.inf
        return mp.ninf
    ll = mp.log1p(-c * x) / c
    if ll > 1e6:
        return mp.ninf
    return -mp.exp(ll) + ll - mp.log1p(-c * x)


def pdf(x, c):
    return _exp(logpdf(x, c))


def _from_y(y, c):
    if c == 0:
        return y
    return -mp.expm1(-c * y) / c


def ppf(q, c):
    return _from_y(-mp.log(-mp.log(q)), c)


def isf(q, c):
    return _from_y(-mp.log(-mp.log1p(-q)), c)


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": logcdf,
    "sf": sf,
    "logsf": lambda x, c: mp.log(sf(x, c)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(c):
    pts = [-5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0, 20.0, 700.0, 708.0, 745.0]
    if c == 0:
        return np.asarray(pts + [-6.0, -6.6, -7.0], float)
    b = 1.0 / c
    # near the endpoint 1/c
    for r in (1e-15, 1e-12, 1e-8, 1e-4, 1e-2, 0.5):
        pts.append(b * (1 - r))   # inside the support for either sign of c
    # c*x overflow: |x| > 1.8e308/|c| on the unbounded side (x < 0 for c > 0)
    s = -1.0 if c > 0 else 1.0
    for f in (0.5, 0.99, 1.01, 2.0):
        pts.append(s * 1.79e308 / abs(c) * f)
    pts += [s * 1e300, s * 1e305, s * 1e307, s * 1e308]
    pts = np.asarray(pts, float)
    return pts[np.isfinite(pts)]


def q_points(c):
    pts = [0.25, np.exp(-1.0), 0.5, 0.75]
    if abs(c) > 1:
        # -expm1(-c y)/c with -c y in (709.78, 709.78 + log|c|): expm1 overflows
        # but the result is finite. y = -log(-log q) (ppf), sign(y) = -sign(c).
        lo, hi = 709.78 / abs(c), (709.78 + np.log(abs(c))) / abs(c)
        for L in np.linspace(lo * 1.0001, hi * 0.9999, 6):
            y = -np.sign(c) * L
            q = np.exp(-np.exp(-y))
            pts += [q, -np.expm1(-np.exp(-y))]   # ppf target, isf target (1 - q)
    return np.asarray(pts, float)
