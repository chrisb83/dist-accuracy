"""Reference module for scipy.stats.foldnorm.

Docstring pdf: f(x, c) = sqrt(2/pi) cosh(c x) exp(-(x**2 + c**2)/2), x >= 0, c >= 0
(= phi(x - c) + phi(x + c)).
cdf = Phi(x - c) - Phi(-x - c) = (erf((x-c)/sqrt2) + erf((x+c)/sqrt2))/2.
sf = Phi(c - x) + Phi(-x - c) = (erfc((x-c)/sqrt2) + erfc((x+c)/sqrt2))/2.
"""
import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".claude",
                                "skills", "dist-accuracy-review", "scripts"))
from accuracy import invert  # noqa: E402

DIST = "foldnorm"

# example; c = 0 (halfnorm); tiny c; c ~ 1; moderate/large c (cdf cancellation
# for x << c); huge c.
PARAMS = [(1.952125337355587,), (0.0,), (1e-3,), (1.0,), (5.0,), (30.0,), (1e3,)]


def pdf(x, c):
    return mp.sqrt(2 / mp.pi) * mp.cosh(c * x) * mp.exp(-(x**2 + c**2) / 2)


def cdf(x, c):
    s2 = mp.sqrt(2)
    return (mp.erf((x - c) / s2) + mp.erf((x + c) / s2)) / 2


def sf(x, c):
    s2 = mp.sqrt(2)
    return (mp.erfc((x - c) / s2) + mp.erfc((x + c) / s2)) / 2


def ppf(q, c):
    return invert(lambda x: cdf(x, c), q)


def isf(q, c):
    return invert(lambda x: cdf(x, c), 1 - q)


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
    # Around the mode x = c, small x << c (cdf cancellation), and the right
    # tail where the pdf/sf leave the normal range (x - c ~ 37.5 .. 38.6).
    pts = [1e-12, 1e-6, 1e-3, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 37.0, 38.0]
    if c > 0:
        pts += [c * 0.5, c * 0.9, c * (1 - 1e-6), c, c * (1 + 1e-6), c * 1.1,
                c + 1, c + 5, c + 20, c + 37, c + 38, max(c - 5, 0), max(c - 10, 0)]
    return pts
