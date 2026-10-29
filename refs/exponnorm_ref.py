"""mpmath reference for scipy.stats.exponnorm (standard form, loc=0, scale=1).

Parametrization from the docstring pdf:

    f(x; K) = 1/(2K) exp(1/(2K^2) - x/K) erfc(-(x - 1/K)/sqrt(2)),  x real, K > 0
    F(x; K) = Phi(x) - exp(1/(2K^2) - x/K) Phi(x - 1/K)
    S(x; K) = Phi(-x) + exp(1/(2K^2) - x/K) Phi(x - 1/K)

(normal + exponential with mean K). S is written as the sum of positive terms
(no cancellation, and identical to 1 - F at any precision). ppf/isf are
inherited generic methods in SciPy and are not measured; references are given
via accuracy.invert for completeness.
"""
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".claude", "skills",
                                "dist-accuracy-review", "scripts"))
from accuracy import invert  # noqa: E402

DIST = "exponnorm"

# inventory example 1.5; K = 1 (branch threshold K < 1) and 0.999; small K
# (erfcx branch, -log K cancellation); K = 1e-200 (1/(2K^2) overflows);
# large K (cdf default branch cancels: Phi(x) - exp(-x/K) Phi(x - 1/K)).
PARAMS = [(1.5,), (0.999,), (1.0,), (0.1,), (1e-5,), (1e-200,), (10.0,),
          (1e4,), (1e8,), (1e17,)]


def _Phi(z):
    return mp.erfc(-z / mp.sqrt(2)) / 2


def _erfcx(w):
    """exp(w^2) erfc(w); asymptotic series for huge w (mpmath's erfc
    overflows internally for w ~ 1e100)."""
    if w > 1e10:
        t = 1 / (2 * w * w)
        s, term, n = mp.mpf(1), mp.mpf(1), 1
        while True:
            term *= -(2 * n - 1) * t
            if abs(term) < mp.mpf(10) ** (-mp.mp.dps - 5):
                break
            s += term
            n += 1
        return s / (w * mp.sqrt(mp.pi))
    return mp.exp(w * w) * mp.erfc(w)


def _corr(x, K):
    # exp(1/(2K^2) - x/K) Phi(x - 1/K) = exp(-x^2/2) erfcx(w) / 2
    w = (1 / K - x) / mp.sqrt(2)
    if w > 1e10:
        return mp.exp(-x * x / 2) * _erfcx(w) / 2
    return mp.exp(1 / (2 * K**2) - x / K) * _Phi(x - 1 / K)


def pdf(x, K):
    return _corr(x, K) / K


def cdf(x, K):
    return _Phi(x) - _corr(x, K)


def sf(x, K):
    return _Phi(-x) + _corr(x, K)


def ppf(q, K):
    return invert(lambda x: cdf(x, K), q, positive=False)


def isf(q, K):
    return invert(lambda x: cdf(x, K), 1 - q, positive=False)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, K: mp.log(pdf(x, K)),
    "cdf": cdf,
    "logcdf": lambda x, K: mp.log(cdf(x, K)),
    "sf": sf,
    "logsf": lambda x, K: mp.log(sf(x, K)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(K):
    # u = 0 threshold at x = 1/K (both sides), left normal tail near the
    # cdf/pdf underflow (x ~ -38), right exponential tail where x/K ~ 700.
    t = 1.0 / K
    pts = [t * (1 - 1e-6), t, t * (1 + 1e-6), t - 1, t + 1,
           -5.0, -10.0, -20.0, -30.0, -37.0, -38.0,
           0.5 * t, 2 * t, 5.0, 10.0, 30.0,
           700.0 * K, 740.0 * K, 1e3 * K, 1e-3 * K]
    pts = np.array(pts, dtype=float)
    return pts[np.isfinite(pts)]
