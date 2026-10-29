"""Reference module for scipy.stats.fatiguelife (Birnbaum-Saunders).

Standard form, docstring pdf:
    f(x; c) = (x + 1) / (2 c sqrt(2 pi x**3)) * exp(-(x - 1)**2 / (2 x c**2)),  x > 0, c > 0.
cdf = Phi(z), sf = Phi(-z) with z = (sqrt(x) - 1/sqrt(x)) / c; Phi via erfc.
ppf/isf by numerical inversion of the cdf.
"""
import mpmath as mp
from accuracy import invert

DIST = "fatiguelife"

# example; c = 1; small c (narrow around x = 1); c ~ 1e-155 (c**2 subnormal);
# large c (cancellation in ppf lower tail, wide distribution).
PARAMS = [(29.0,), (1.0,), (0.1,), (1e-3,), (1e-155,), (1e3,), (1e6,), (1e100,)]


def _z(x, c):
    sx = mp.sqrt(x)
    return (sx - 1 / sx) / c


def logpdf(x, c):
    return (mp.log(x + 1) - (x - 1) ** 2 / (2 * x * c ** 2) - mp.log(2 * c)
            - mp.log(2 * mp.pi) / 2 - 3 * mp.log(x) / 2)


def pdf(x, c):
    return mp.exp(logpdf(x, c))


def cdf(x, c):
    return mp.erfc(-_z(x, c) / mp.sqrt(2)) / 2


def sf(x, c):
    return mp.erfc(_z(x, c) / mp.sqrt(2)) / 2


def ppf(q, c):
    return invert(lambda x: cdf(x, c), q)


def isf(q, c):
    return invert(lambda x: cdf(x, c), 1 - q)


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": lambda x, c: mp.log(cdf(x, c)),
    "sf": sf,
    "logsf": lambda x, c: mp.log(sf(x, c)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(c):
    import numpy as np
    near1 = [1 - 1e-8, 1 - 1e-15, 1.0, 1 + 2.3e-16, 1 + 1e-15, 1 + 1e-8]
    # (x-1)**2 overflows at x ~ 1.34e154
    big = [1e154, 1.3e154, 1.4e154, 1e160, 1e200]
    # z = +-k standard deviations: x = ((t + sqrt(t^2+4))/2)^2 with t = c*k
    k = np.array([-30.0, -10.0, -3.0, -1.0, 1.0, 3.0, 10.0, 30.0])
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        t = c * k
        xs = np.where(t < 0, 4.0 / (np.sqrt(t * t + 4) - t) ** 2,
                      0.25 * (t + np.sqrt(t * t + 4)) ** 2)
    return np.concatenate([near1, big, xs])


def q_points(c):
    return [1e-300, 1e-100, 1e-20, 1e-8, 0.01, 0.1, 0.4, 0.5, 0.6, 0.9, 0.99, 1 - 1e-8]
