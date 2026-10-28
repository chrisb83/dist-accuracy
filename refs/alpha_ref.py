"""mpmath reference for scipy.stats.alpha (standard form, loc=0, scale=1).

Docstring pdf (x > 0, a > 0):

    f(x; a) = phi(a - 1/x) / (x**2 * Phi(a))
    F(x; a) = Phi(a - 1/x) / Phi(a)
    S(x; a) = 1 - F(x; a)

phi, Phi: standard normal pdf / cdf (mp.npdf, mp.ncdf).
ppf/isf: numerical inversion of the cdf (accuracy.invert).
"""
import mpmath as mp
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".claude", "skills",
                                "dist-accuracy-review", "scripts"))
from accuracy import invert  # noqa: E402

DIST = "alpha"
DPS = 400

# example value, tiny / small / moderate a, a = 40 (phi(a) underflows,
# Phi(a) == 1 in double), a = 1e6 (a and 1/x cancel near the mode),
# a = 1e154 (mode near x = 1e-154 where x**2 is subnormal).
PARAMS = [(3.570477051665046,), (1e-3,), (0.5,), (2.0,), (40.0,), (1e6,),
          (1e154,)]


def _phi(z):
    return mp.exp(-z * z / 2) / mp.sqrt(2 * mp.pi)


def pdf(x, a):
    z = a - 1 / x
    return _phi(z) / (x * x * mp.ncdf(a))


def logpdf(x, a):
    z = a - 1 / x
    return -z * z / 2 - mp.log(mp.sqrt(2 * mp.pi)) - 2 * mp.log(x) - mp.log(mp.ncdf(a))


def cdf(x, a):
    return mp.ncdf(a - 1 / x) / mp.ncdf(a)


def sf(x, a):
    return 1 - cdf(x, a)


def ppf(q, a):
    return invert(lambda x: cdf(x, a), q)


def isf(q, a):
    return invert(lambda x: cdf(x, a), 1 - q)


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": lambda x, a: mp.log(cdf(x, a)),
    "sf": sf,
    "logsf": lambda x, a: mp.log(sf(x, a)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(a):
    # z = a - 1/x at selected values (around the mode x ~ 1/a and the
    # left tail where Phi(z) is tiny), the window where z**2 overflows in
    # _norm_logpdf (|z| ~ 1.34e154 .. 1.9e154), and x**2 subnormal/underflow.
    zs = np.array([-38.0, -30.0, -10.0, -3.0, -1.0, -1e-3, 0.0, 1e-3, 1.0, 3.0])
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        xs = 1.0 / (a - zs)
    window = [5.3e-155, 5.5e-155, 6e-155, 7e-155, 7.4e-155, 7.6e-155, 8e-155,
              1e-154, 1.4e-154, 1.5e-154, 1e-160, 1e-163]
    return np.concatenate([xs, window, [0.5, 1.0, 2.0]])


def q_points(a):
    return np.array([1e-300, 1e-100, 1e-20, 0.25, 0.5, 0.75, 1 - 1e-6,
                     1 - 1e-10, 1 - 1e-14, 1 - 2.2e-16])
