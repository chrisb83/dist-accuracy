"""mpmath reference for scipy.stats.chi2 (standard form, loc=0, scale=1).

Parametrization from the docstring pdf:

    f(x; k) = x**(k/2-1) * exp(-x/2) / (2**(k/2) * Gamma(k/2)),   x > 0, k > 0
    F(x; k) = P(k/2, x/2) = gammainc(k/2, 0, x/2, regularized=True)
    S(x; k) = 1 - F(x; k)

Speed (mpmath, dps = 400): the lower regularized gammainc is fast for
x/2 <~ k/2, the upper one is fast for x/2 >~ k/2 (and the lower one may not
converge there for large k), so whichever is on its fast side is evaluated
and the other obtained by subtraction from 1 (exact enough at 400 digits).
mpmath's gammainc does not converge for k/2 >~ 1e7; large df beyond 2e6 is
measured with the auxiliary scratch/chi2/chi2_df2e8_ref.py (Kummer series).

ppf/isf: accuracy.invert on the cdf, bracket seeded from SciPy (speed only).
"""
import mpmath as mp
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".claude", "skills",
                                "dist-accuracy-review", "scripts"))
from accuracy import invert  # noqa: E402

DIST = "chi2"
DPS = 400

# 55: inventory example; 2e-3 (a = 1e-3, cdf ~ x**a); 1 (a = 1/2); 2 (a = 1,
# exponential, xlogy(0, x) = 0); 3.98646... (a = 1.993, generic); 2e3
# (a = 1e3, igam_fac prefactor, logpdf cancellation); 2e6 (a = 1e6: igam
# 2000-term cap regime, large logpdf terms).
PARAMS = [(55.0,), (2e-3,), (1.0,), (2.0,), (3.9864610967601556,),
          (2e3,), (2e6,)]


def pdf(x, k):
    if x == 0:
        if k == 2:
            return mp.mpf(1) / 2
        return mp.inf if k < 2 else mp.mpf(0)
    a = k / 2
    return mp.exp((a - 1) * mp.log(x) - x / 2 - mp.loggamma(a) - a * mp.log(2))


def cdf(x, k):
    if x == 0:
        return mp.mpf(0)
    a, y = k / 2, x / 2
    if y >= a:
        return 1 - mp.gammainc(a, y, mp.inf, regularized=True)
    return mp.gammainc(a, 0, y, regularized=True)


def sf(x, k):
    if x == 0:
        return mp.mpf(1)
    a, y = k / 2, x / 2
    if y >= a:
        return mp.gammainc(a, y, mp.inf, regularized=True)
    return 1 - mp.gammainc(a, 0, y, regularized=True)


def _seed(method, q, k):
    from scipy import stats
    with np.errstate(all="ignore"):
        v = float(getattr(stats.chi2, method)(q, k))
    if not (np.isfinite(v) and v > 0):
        return None, None
    return v / 4, v * 4


def ppf(q, k):
    lo, hi = _seed("ppf", float(q), float(k))
    return invert(lambda x: cdf(x, k), q, lo=lo, hi=hi, positive=True)


def isf(q, k):
    lo, hi = _seed("isf", float(q), float(k))
    return invert(lambda x: cdf(x, k), 1 - q, lo=lo, hi=hi, positive=True)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, k: mp.log(pdf(x, k)),
    "cdf": cdf,
    "logcdf": lambda x, k: mp.log(cdf(x, k)),
    "sf": sf,
    "logsf": lambda x, k: mp.log(sf(x, k)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(k):
    """Bulk points around the mean df and the igam/igamc branch edges.

    igam(a, y) with a = df/2, y = x/2 switches to the uniform asymptotic
    expansion for |y - a|/a < 0.3 (20 < a < 200) or < 4.5/sqrt(a) (a > 200);
    in x these are x = df*(1 +- 0.3) and x = df +- 4.5*sqrt(2*df). Also
    x = df (igam vs 1 - igamc switch), x = 2 (y = 1), and the 2000-term series
    window df - (5..15)*sqrt(2*df).
    """
    s = np.sqrt(2 * k)
    pts = [k * 0.5, k * 0.69, k * 0.71, k * 0.9, k * 0.99, k, k * 1.01,
           k * 1.1, k * 1.29, k * 1.31, k * 2.0, k * 10.0, k - 2.0,
           k - s, k + s, k - 3 * s, k + 3 * s, k - 4.4 * s, k - 4.6 * s,
           k + 4.4 * s, k + 4.6 * s, k - 5 * s, k - 7 * s, k - 10 * s,
           k - 15 * s, k + 10 * s, k + 30 * s,
           2.0 - 1e-8, 2.0, 2.0 + 1e-8, 1.0, 4.0, 1420.0, 1490.0, 1e-8]
    pts = np.array(pts, dtype=float)
    return pts[np.isfinite(pts) & (pts > 0)]


def q_points(k):
    return np.array([1e-17, 1e-16, 1e-8, 1e-6, 0.5 - 1e-16, 0.5, 0.5 + 1e-16,
                     1 - 1e-6, 1 - 1e-8, 1 - 1e-13, 1 - 1e-16, 1 - 2e-16])
