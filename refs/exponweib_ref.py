"""Reference module for scipy.stats.exponweib.

Standard form, docstring pdf:
    f(x; a, c) = a c [1 - exp(-x**c)]**(a-1) exp(-x**c) x**(c-1),
    F(x; a, c) = [1 - exp(-x**c)]**a,  x > 0, a > 0, c > 0.
mpmath expm1/log1p are used so that x**c far below 10**-400 and
exp(-x**c) far below 10**-400 do not need more working digits.
"""
import mpmath as mp

DIST = "exponweib"

# Example params; a = 1 (weibull_min); small a (x**c tiny while cdf/sf
# moderate); large a (q**(1/a) near 1, subnormal exp(-x**c) lifted by a);
# large c (x**c underflow with normal quantiles); small c.
PARAMS = [
    (2.8923945291034436, 1.9505288745913174),
    (1.0, 1.0),
    (0.01, 1.0),
    (0.3, 50.0),
    (1e6, 1.0),
    (1e3, 0.5),
    (5.0, 1e3),
    (2.0, 0.01),
]


def _expneg(t):
    # exp(-t); far below any double for t > 1e6 (and slow in mpmath).
    return mp.mpf(0) if t > 1e6 else mp.exp(-t)


def _onemexp(t):
    # 1 - exp(-t)
    return -mp.expm1(-t) if t < 1e6 else mp.mpf(1)


def _log1mexp(z):
    # log(1 - exp(z)) for z < 0, accurate at any working precision
    # (the harness re-evaluates references at lower precision).
    if z < -1e6:
        # |log(1 - exp(z))| < exp(-1e6), far below any double; mpmath's
        # exp of a huge argument is slow.
        return -mp.mpf(0)
    if z > -mp.log(2):
        return mp.log(-mp.expm1(z))
    return mp.log1p(-mp.exp(z))


def pdf(x, a, c):
    t = x ** c
    return a * c * _onemexp(t) ** (a - 1) * _expneg(t) * x ** (c - 1)


def logpdf(x, a, c):
    t = x ** c
    return (mp.log(a) + mp.log(c) + (a - 1) * _log1mexp(-t) - t
            + (c - 1) * mp.log(x))


def cdf(x, a, c):
    return _onemexp(x ** c) ** a


def sf(x, a, c):
    # 1 - cdf, written so that cdf = 1 - 1e-500 needs no extra digits
    return -mp.expm1(a * _log1mexp(-x ** c))


def ppf(q, a, c):
    return (-_log1mexp(mp.log(q) / a)) ** (1 / c)


def isf(q, a, c):
    # ppf(1 - q)
    return (-_log1mexp(mp.log1p(-q) / a)) ** (1 / c)


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": lambda x, a, c: a * _log1mexp(-x ** c),
    "sf": sf,
    "logsf": lambda x, a, c: mp.log(sf(x, a, c)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(a, c):
    # t = x**c around eps (1 - exp(-t) cancellation), 1e-300..1e-330
    # (x**c underflow / subnormal), and t in [700, 745] (exp(-t) subnormal).
    import numpy as np
    t = np.array([1e-330, 1e-320, 1e-310, 1e-305, 1e-200, 1e-100, 1e-30, 1e-17,
                  1e-16, 1e-15, 1e-12, 1e-8, 1e-4, 0.1, 1.0, 10.0, 100.0,
                  700.0, 708.0, 720.0, 730.0, 740.0, 745.0, 760.0])
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        x = np.exp(np.log(t) / c)
    return x[np.isfinite(x) & (x > 0)]
