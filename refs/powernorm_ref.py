"""Reference module for scipy.stats.powernorm.

Docstring pdf: f(x; c) = c phi(x) Phi(-x)**(c-1), x real, c > 0.
sf = Phi(-x)**c, cdf = 1 - sf. Written with log/log1p/expm1 so that values of
Phi(x) far below 10**-400 in the left tail do not get lost at 400 digits.
"""
import mpmath as mp
from accuracy import invert

DIST = "powernorm"

# example; small c (logpdf cancellation, pdf/isf underflow); c < 1; c = 1 (normal);
# large c (pattern 6, ndtri near 1).
PARAMS = [(4.445365225459078,), (1e-3,), (1e-8,), (0.5,), (1.0,), (50.0,), (1e4,), (1e6,)]


def logQ(x):
    # log Phi(-x), without forming a probability that rounds to 1 at 400 digits:
    # for x >= 0 Phi(-x) is small; for x < 0 use log1p(-Phi(x)).
    if x > mp.mpf(10) ** 60:
        # mp.ncdf overflows internally for x ~ 1e155; the asymptotic series
        # Q(x) = phi(x)/x * (1 - 1/x^2 + 3/x^4 - 15/x^6 + 105/x^8 - ...) has
        # truncation error < 1e-590 here, below the 400-digit working precision.
        u = 1 / (x * x)
        return (-x * x / 2 - mp.log(x) - mp.log(2 * mp.pi) / 2
                + mp.log1p(-u + 3 * u**2 - 15 * u**3 + 105 * u**4))
    if x >= 0:
        return mp.log(mp.ncdf(-x))
    return mp.log1p(-mp.ncdf(x))


def logsf(x, c):
    return c * logQ(x)


def sf(x, c):
    return mp.exp(logsf(x, c))


def cdf(x, c):
    return -mp.expm1(c * logQ(x))


def logpdf(x, c):
    return mp.log(c) - x * x / 2 - mp.log(2 * mp.pi) / 2 + (c - 1) * logQ(x)


def pdf(x, c):
    return mp.exp(logpdf(x, c))


def ppf(q, c):
    # upper half: invert the sf on 1 - q (exact at 400 digits); 1 - cdf would lose
    # everything once the sf is below 1e-400 (small c)
    if q <= mp.mpf(1) / 2:
        return invert(lambda x: cdf(x, c), q, positive=False)
    return invert(lambda x: sf(x, c), 1 - q, increasing=False, positive=False)


def isf(q, c):
    # invert the sf directly: 1 - cdf loses everything once sf < 1e-400
    return invert(lambda x: sf(x, c), q, increasing=False, positive=False)


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
    # ndtr(-x) underflow (~37.5..38.5), phi underflow (37.7), x**2 overflow (1.34e154),
    # near 0 (sf = 2**-c), moderate tails on both sides.
    return [-40.0, -38.0, -30.0, -20.0, -8.0, -1e-8, 1e-8, 1e-3, 0.3, 8.0, 20.0, 30.0,
            37.0, 37.6, 38.0, 38.5, 39.0, 40.0, 60.0, 100.0, 1e3, 1e5, 1e10, 1e100,
            1.3e154, 1.4e154, 1e155, 1e200, -1.3e154, -1.4e154]


def q_points(c):
    return [1e-20, 1e-17, 1e-16, 2e-16, 1e-15, 1e-12, 1e-8, 1e-5, 0.3, 0.7, 0.9, 0.999]
