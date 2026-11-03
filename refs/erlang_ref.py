"""mpmath reference for scipy.stats.erlang (standard form, loc=0, scale=1).

erlang = gamma restricted to integer a (gamma_gen methods, unchanged).

Parametrization from the docstring pdf:

    f(x; a) = x**(a-1) * exp(-x) / Gamma(a),     x >= 0, a > 0
    F(x; a) = P(a, x) = gammainc(a, 0, x, regularized=True)
    S(x; a) = Q(a, x) = 1 - P(a, x)

Speed notes (mpmath 1.4.1, dps = 400):

* ``mp.gammainc(a, 0, x, regularized=True)`` (lower, = cdf) is fast for
  x <~ a but raises NoConvergence for large a and x > a
  (e.g. a = 1e6, x = 1.1e6).
* ``mp.gammainc(a, x, mp.inf, regularized=True)`` (upper, = sf) is fast for
  x >~ a but takes seconds for small x.

So the reference evaluates whichever of the two is on its fast/robust side
and gets the other by subtracting from 1 (exact enough at 400 digits for
every value that is a normal double).

ppf/isf are obtained from the cdf with ``accuracy.invert``; the bracket is
seeded from the (possibly inaccurate) SciPy value, which only speeds up the
search -- ``invert`` widens the bracket by itself and checks the residual.
"""
import mpmath as mp
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".claude", "skills",
                                "dist-accuracy-review", "scripts"))
from accuracy import invert  # noqa: E402

DIST = "erlang"
DPS = 400

# erlang_gen subclasses gamma_gen and overrides no numerical method, so the
# reference is gamma's with integer shapes only: the inventory example (10),
# a = 1 (exponential; xlogy(0, x) == 0), a = 2, and the large integer shapes
# where gamma's findings live (1e3: logpdf/pdf cancellation and the igam_fac
# prefactor; 1e6: unconverged igam_series and its inverses).
PARAMS = [(10.0,), (1.0,), (2.0,), (1e3,), (1e6,)]


def _loggamma(a):
    return mp.loggamma(a)


def pdf(x, a):
    if x == 0:
        if a == 1:
            return mp.mpf(1)
        return mp.inf if a < 1 else mp.mpf(0)
    return mp.exp((a - 1) * mp.log(x) - x - _loggamma(a))


def cdf(x, a):
    if x == 0:
        return mp.mpf(0)
    if x >= a:
        return 1 - mp.gammainc(a, x, mp.inf, regularized=True)
    return mp.gammainc(a, 0, x, regularized=True)


def sf(x, a):
    if x == 0:
        return mp.mpf(1)
    if x >= a:
        return mp.gammainc(a, x, mp.inf, regularized=True)
    return 1 - mp.gammainc(a, 0, x, regularized=True)


def _seed(method, q, a):
    """Bracket for the root, from the SciPy value (only a speed-up)."""
    from scipy import stats
    with np.errstate(all="ignore"):
        v = float(getattr(stats.erlang, method)(q, a))
    if not (np.isfinite(v) and v > 0):
        return None, None
    return v / 4, v * 4


def ppf(q, a):
    lo, hi = _seed("ppf", float(q), float(a))
    return invert(lambda x: cdf(x, a), q, lo=lo, hi=hi, positive=True)


def isf(q, a):
    lo, hi = _seed("isf", float(q), float(a))
    return invert(lambda x: cdf(x, a), 1 - q, lo=lo, hi=hi, positive=True)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, a: mp.log(pdf(x, a)),
    "cdf": cdf,
    "logcdf": lambda x, a: mp.log(cdf(x, a)),
    "sf": sf,
    "logsf": lambda x, a: mp.log(sf(x, a)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(a):
    """Points near the mode and around the transitions of `igam`/`igamc`.

    The default grid is decades 1e-300 ... 1e300, which misses the bulk of
    the distribution entirely for large a; scipy's incomplete-gamma code
    switches regimes near x = a (and at x = 1, and for a > 1e4 uses the
    uniform asymptotic expansion for |x - a| small), so both sides of
    x = a, the mode a - 1 and a few standard deviations sqrt(a) around it
    are added.
    """
    s = np.sqrt(a)
    pts = np.array([
        a * 0.5, a * 0.9, a * 0.99, a * 0.999, a, a * 1.001, a * 1.01,
        a * 1.1, a * 2.0, a * 10.0,
        a - 1.0, a - s, a + s, a - 3 * s, a + 3 * s, a + 10 * s,
        1.0 - 1e-8, 1.0, 1.0 + 1e-8, 0.5, 2.0, 710.0, 745.0, 1e-8,
    ], dtype=float)
    return pts[np.isfinite(pts) & (pts > 0)]


def q_points(a):
    return np.array([1e-17, 1e-16, 1e-8, 0.5 - 1e-16, 0.5, 0.5 + 1e-16,
                     1 - 1e-16, 1 - 2e-16, 1 - 1e-13, 1 - 1e-9])
