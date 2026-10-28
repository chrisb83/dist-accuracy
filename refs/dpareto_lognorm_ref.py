"""mpmath reference for scipy.stats.dpareto_lognorm (standard form).

Parametrization from the docstring pdf (shapes u, s, a, b = mu, sigma,
alpha, beta):

    z  = (log x - u)/s,  y1 = a s - z,  y2 = b s + z,  R(t) = Phic(t)/phi(t)
    f(x) = a b / ((a + b) x) * phi(z) * (R(y1) + R(y2))
    F(x) = Phi(z)  - phi(z) * (b R(y1) - a R(y2)) / (a + b)      [1] Eq. 2
    S(x) = Phic(z) + phi(z) * (b R(y1) - a R(y2)) / (a + b)

phi(z) R(y) is evaluated as Phic(y) * exp((y^2 - z^2)/2), so nothing
overflows. S is evaluated directly (not 1 - F) because in the right tail
S ~ x**-a can be far below 10**-400. All functions use erfc only (fast).
"""
import mpmath as mp

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".claude", "skills",
                                "dist-accuracy-review", "scripts"))
from accuracy import invert  # noqa: E402

DIST = "dpareto_lognorm"

# inventory example; small s (|z| large, log-space cancellation); large s;
# large a, b (close to lognormal); tiny a, b (very heavy tails);
# large u (log x - u cancels); a == b.
PARAMS = [(3.0, 1.2, 1.5, 2.0),
          (0.0, 0.01, 1.0, 1.0),
          (0.0, 10.0, 2.0, 3.0),
          (0.0, 1.0, 100.0, 100.0),
          (0.0, 1.0, 1e-3, 1e-3),
          (500.0, 0.1, 2.0, 3.0),
          (-2.0, 0.5, 5.0, 0.5)]


def _Phic(t):
    return mp.erfc(t / mp.sqrt(2)) / 2


def _phiR(z, y):
    """phi(z) * R(y) = Phic(y) * phi(z) / phi(y)."""
    return _Phic(y) * mp.exp((y * y - z * z) / 2)


def _zy(x, u, s, a, b):
    z = (mp.log(x) - u) / s
    return z, a * s - z, b * s + z


def pdf(x, u, s, a, b):
    z, y1, y2 = _zy(x, u, s, a, b)
    return a * b / ((a + b) * x) * (_phiR(z, y1) + _phiR(z, y2))


def cdf(x, u, s, a, b):
    z, y1, y2 = _zy(x, u, s, a, b)
    return 1 - _Phic(z) - (b * _phiR(z, y1) - a * _phiR(z, y2)) / (a + b)


def sf(x, u, s, a, b):
    z, y1, y2 = _zy(x, u, s, a, b)
    return _Phic(z) + (b * _phiR(z, y1) - a * _phiR(z, y2)) / (a + b)


def ppf(q, u, s, a, b):
    return invert(lambda x: cdf(x, u, s, a, b), q)


def isf(q, u, s, a, b):
    return invert(lambda x: cdf(x, u, s, a, b), 1 - q)


REFS = {
    "pdf": pdf,
    "logpdf": lambda *t: mp.log(pdf(*t)),
    "cdf": cdf,
    "logcdf": lambda *t: mp.log(cdf(*t)),
    "sf": sf,
    "logsf": lambda *t: mp.log(sf(*t)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(u, s, a, b):
    # points around the mode region exp(u + k s), including far |z|
    import numpy as np
    ks = np.array([-300, -100, -30, -10, -3, -1, -0.1, 0, 0.1, 1, 3, 10, 30,
                   100, 300], float)
    with np.errstate(over="ignore", under="ignore"):
        xs = np.exp(u + ks * s)
    return xs[np.isfinite(xs) & (xs > 0)]
