"""Reference module for scipy.stats.bradford.

Standard form, shape c > 0. Docstring pdf:
    f(x, c) = c / (log(1+c) (1 + c x)),  0 <= x <= 1.
cdf F(x) = log1p(c x) / log1p(c), sf = 1 - F,
ppf = expm1(q log1p(c)) / c, isf = ppf(1 - q).
"""
import mpmath as mp

DIST = "bradford"

# Example value, tiny shapes (c*x and q*log1p(c) underflow), c = 1,
# large shapes.
PARAMS = [(0.29891359763170633,), (1e-300,), (1e-10,), (1e-3,), (1.0,),
          (100.0,), (1e10,), (1e300,)]


def pdf(x, c):
    return c / (mp.log1p(c) * (1 + c * x))


def cdf(x, c):
    return mp.log1p(c * x) / mp.log1p(c)


def sf(x, c):
    return 1 - cdf(x, c)


def ppf(q, c):
    return mp.expm1(q * mp.log1p(c)) / c


def isf(q, c):
    return ppf(1 - q, c)


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
    # Ends of the support, x where c*x is ~1 (scale of the pdf), and x where
    # c*x drops below the normal range (subnormal intermediate in log1p(c*x)).
    import numpy as np
    pts = [0.5, 0.25, 0.75, 1.0, 1e-300, 1e-308, 1e-200, 1e-100, 1e-20]
    for d in [1e-16, 1e-15, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3]:
        pts += [d, 1 - d]
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        for t in [1e-300, 2.2e-308, 1e-310, 1e-320, 1e-16, 1.0]:
            v = t / c
            if 0 < v < 1:
                pts.append(v)
    return np.array(pts)


def q_points(c):
    # q*log1p(c) underflow (tiny c), q near 1.
    import numpy as np
    pts = [0.5, 1e-300, 1e-308, 1e-200, 1e-100, 1e-20, 1e-10]
    for d in [1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
        pts += [1 - d, d]
    return np.array(pts)
