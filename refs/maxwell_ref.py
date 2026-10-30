"""mpmath reference for scipy.stats.maxwell (standard form, loc=0, scale=1).

Docstring pdf (no shapes; chi with df = 3):

    f(x) = sqrt(2/pi) x**2 exp(-x**2/2),                 x >= 0
    F(x) = P(3/2, x**2/2)                                (regularized lower incomplete gamma)
    S(x) = erfc(x/sqrt(2)) + sqrt(2/pi) x exp(-x**2/2)  (= Q(3/2, x**2/2); both terms positive)

x**2/2 is formed in mpmath, so no overflow/underflow of x**2 in the reference.
ppf/isf by accuracy.invert on the cdf.
"""
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".claude", "skills",
                                "dist-accuracy-review", "scripts"))
from accuracy import invert  # noqa: E402

DIST = "maxwell"
DPS = 400

PARAMS = [()]


def logpdf(x):
    x = mp.mpf(x)
    if x == 0:
        return -mp.inf
    return mp.log(mp.sqrt(2 / mp.pi)) + 2 * mp.log(x) - x * x / 2


def pdf(x):
    x = mp.mpf(x)
    return mp.sqrt(2 / mp.pi) * x * x * mp.exp(-x * x / 2)


def cdf(x):
    x = mp.mpf(x)
    if x == 0:
        return mp.mpf(0)
    if x > 40:
        return 1 - sf(x)
    return mp.gammainc(mp.mpf(3) / 2, 0, x * x / 2, regularized=True)


def sf(x):
    x = mp.mpf(x)
    if x > 1e4:
        # far below any double; mp.erfc fails for huge arguments. Leading
        # asymptotic terms (relative error O(x**-4) < 1e-16).
        return mp.sqrt(2 / mp.pi) * mp.exp(-x * x / 2) * (x + 1 / x)
    return mp.erfc(x / mp.sqrt(2)) + mp.sqrt(2 / mp.pi) * x * mp.exp(-x * x / 2)


def ppf(q):
    return invert(cdf, q, positive=True)


def isf(q):
    return invert(cdf, 1 - q, positive=True)


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": lambda x: mp.log(cdf(x)),
    "sf": sf,
    "logsf": lambda x: mp.log(sf(x)),
    "ppf": ppf,
    "isf": isf,
}


def x_points():
    """Mode sqrt(2), bulk, the window where exp(-x**2/2) is subnormal but the
    pdf is normal (37.64 < x < 37.83), sf underflow (~38.6), the x**2
    overflow edge in logpdf (1.34e154 ... 1.9e154), the x**2 underflow edge
    (~1.5e-154) and where cdf ~ x**3 leaves the normal range (~1e-103)."""
    pts = [np.sqrt(2.0), 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0,
           37.0, 37.5, 37.6, 37.65, 37.7, 37.75, 37.8, 37.82, 37.83, 37.9,
           37.78, 37.79, 37.81, 37.815, 37.825, 37.828,
           38.0, 38.3, 38.5, 38.6, 38.7,
           1.3e154, 1.34e154, 1.35e154, 1.5e154, 1.8e154, 1.89e154, 1.9e154,
           1.6e-154, 1.7e-154, 2e-154, 1e-150, 1e-120, 1e-103, 1e-102,
           1e-100, 1e-50, 1e-20, 1e-10, 1e-5, 1e-3]
    return np.array(pts, dtype=float)


def q_points():
    return np.array([1e-300, 1e-290, 1e-200, 1e-100, 1e-17, 1e-16, 1e-8,
                     0.5 - 1e-16, 0.5, 0.5 + 1e-16, 0.9,
                     1 - 1e-16, 1 - 2e-16, 1 - 1e-13, 1 - 1e-9])
