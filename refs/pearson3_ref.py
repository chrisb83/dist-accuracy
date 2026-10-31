"""mpmath reference for scipy.stats.pearson3 (standard form, loc=0, scale=1).

Parametrization from the docstring pdf (shape ``skew`` = kappa, kappa != 0):

    beta  = 2/kappa,  alpha = beta**2 = 4/kappa**2,  zeta = -alpha/beta = -2/kappa
    t     = beta*(x - zeta) = alpha + beta*x
    f(x)  = |beta| / Gamma(alpha) * t**(alpha-1) * exp(-t)      (t > 0, else 0)
    kappa > 0:  F(x) = P(alpha, t),  S(x) = Q(alpha, t)   (support x > zeta)
    kappa < 0:  F(x) = Q(alpha, t),  S(x) = P(alpha, t)   (support x < zeta)

i.e. X = (G - alpha)/beta with G ~ Gamma(alpha): mean 0, variance 1, skewness
kappa. kappa == 0 is the standard normal (only used by SciPy's |kappa| < 1.6e-5
branch; the reference never approximates: tiny kappa is evaluated with the
exact gamma form).

Two routes for P/Q:

* alpha <= ALPHA_QUAD = 1e5 (|kappa| >= 6.3e-3): ``mp.gammainc`` on its robust side
  (lower for t < alpha, upper for t >= alpha), the other side by 1 - (...).
* alpha > ALPHA_QUAD: mpmath's gammainc / hyp1f1 raise NoConvergence for
  alpha >= 1e7, so the *smaller* tail integral of the pdf is computed with
  ``mp.quad`` from x outwards (integrand pdf(u)/pdf(x), result times pdf(x),
  so tiny tails keep full relative accuracy) (the integrand is log-concave and maximal at x,
  so geometric breakpoints x, x -/+ 2**k * w suffice), and the other side is
  1 - (...). Used with DPS = 100 (``scratch/pearson3/pearson3_tiny_ref.py``);
  validated against ``mp.gammainc`` at alpha = 4e6 (``scratch/pearson3/``).

cdf and sf each compute the smaller of P/Q directly (no 1 - (...) in the tail
that is returned), so the module also works at 100 digits (used for the ppf
grid: ``--dps 100``). ppf uses ``accuracy.invert`` on the cdf, isf on the sf
(support contains negative values, so ``positive=False``).
"""
import mpmath as mp
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".claude", "skills",
                                "dist-accuracy-review", "scripts"))
from accuracy import invert  # noqa: E402

DIST = "pearson3"
DPS = 400
ALPHA_QUAD = mp.mpf(1e5)

# inventory examples (0.1, -2); |kappa| = 1 and 0.1 both signs; small kappa
# (alpha = 4e4); large kappa
# (alpha = 0.04 and 4e-4, gamma's small-a regime).
# |kappa| <= 1e-3 (alpha >= 4e6) is measured with the auxiliary module
# scratch/pearson3/pearson3_tiny_ref.py (100 digits, quadrature route).
# (0.01,) and (-0.01,) (alpha = 4e4) are measured separately for
# pdf/logpdf/cdf/sf (--params "0.01;-0.01"), ppf there with --at points only:
# the bisection ppf reference makes the condition-number step take hours.
PARAMS = [(0.1,), (-2.0,), (1.0,), (-0.1,), (10.0,), (-100.0,)]


def _abt(k):
    k = mp.mpf(k)
    beta = 2 / k
    alpha = beta ** 2
    return alpha, beta


_LG = {}


def _const(k):
    """log|beta| - loggamma(alpha), cached per (kappa, precision)."""
    key = (k, mp.mp.prec)
    if key not in _LG:
        alpha, beta = _abt(k)
        _LG[key] = mp.log(abs(beta)) - mp.loggamma(alpha)
    return _LG[key]


def logpdf(x, k):
    alpha, beta = _abt(k)
    t = alpha + beta * x
    if t <= 0:
        return mp.ninf
    return (alpha - 1) * mp.log(t) - t + _const(k)


def pdf(x, k):
    v = logpdf(x, k)
    return mp.mpf(0) if v == mp.ninf else mp.exp(v)


def _PQ(alpha, t):
    """Regularized (P, Q)(alpha, t) for moderate alpha; the smaller of the two
    is always computed directly by mp.gammainc on its robust side (lower for
    t < alpha, upper for t >= alpha), the other as 1 - (...), so tail values
    keep full relative accuracy even at 100 digits."""
    if t <= 0:
        return mp.mpf(0), mp.mpf(1)
    if t >= alpha:
        Q = mp.gammainc(alpha, t, mp.inf, regularized=True)
        return 1 - Q, Q
    P = mp.gammainc(alpha, 0, t, regularized=True)
    return P, 1 - P


def _tail_quad(x, k, side):
    """Integral of the pdf from x to the support end on `side` (-1 left, +1 right)."""
    alpha, beta = _abt(k)
    zeta = -alpha / beta
    L0 = logpdf(x, k)
    if L0 == mp.ninf:
        return mp.mpf(0)
    # local scale: 1/|d logpdf/dx| capped to [1e-3, 1]
    t = alpha + beta * x
    slope = abs(beta * ((alpha - 1) / t - 1))
    w = min(mp.mpf(1), max(mp.mpf("1e-3"), 1 / slope if slope else mp.mpf(1)))
    drop = (mp.mp.dps + 30) * mp.log(10)
    pts = [x]
    # the step must change x at the working precision (x can be +-1e300)
    step = max(w, abs(x) * mp.mpf(10) ** (10 - mp.mp.dps))
    for _ in range(3000):
        y = x + side * step
        if (side < 0 and k > 0 and y <= zeta) or (side > 0 and k < 0 and y >= zeta):
            pts.append(zeta)
            break
        pts.append(y)
        if logpdf(y, k) < L0 - drop:
            break
        step *= 2
    if side < 0:
        pts = pts[::-1]
    # integrate pdf(u)/pdf(x) (O(1) near x) and rescale: mp.quad's error
    # control is effectively absolute, so integrating the unscaled pdf loses
    # relative accuracy for tiny tails (e.g. 5e-12 relative at 1e-198 with
    # 100 digits).
    def g(u):
        v = logpdf(u, k)
        return mp.mpf(0) if v == mp.ninf else mp.exp(v - L0)
    return mp.exp(L0) * mp.quad(g, pts)


def _mode(k):
    alpha, beta = _abt(k)
    return (alpha - 1 - alpha) / beta  # t = alpha - 1  ->  x = -1/beta


def _cdf_sf(x, k):
    alpha, beta = _abt(k)
    if alpha > ALPHA_QUAD:
        if x <= _mode(k):
            F = _tail_quad(x, k, -1)
            return F, 1 - F
        S = _tail_quad(x, k, +1)
        return 1 - S, S
    t = alpha + beta * x
    P, Q = _PQ(alpha, t)
    return (P, Q) if k > 0 else (Q, P)


def cdf(x, k):
    return _cdf_sf(x, k)[0]


def sf(x, k):
    return _cdf_sf(x, k)[1]


def _seed(method, q, k):
    from scipy import stats
    with np.errstate(all="ignore"):
        v = float(getattr(stats.pearson3, method)(q, k))
    if not np.isfinite(v):
        return None, None
    return v - 1 - abs(v) / 4, v + 1 + abs(v) / 4


def ppf(q, k):
    lo, hi = _seed("ppf", float(q), float(k))
    return invert(lambda x: cdf(x, k), q, lo=lo, hi=hi, positive=False)


def isf(q, k):
    lo, hi = _seed("isf", float(q), float(k))
    return invert(lambda x: sf(x, k), q, lo=lo, hi=hi, positive=False,
                  increasing=False)


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": lambda x, k: mp.log(cdf(x, k)),
    "sf": sf,
    "logsf": lambda x, k: mp.log(sf(x, k)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(k):
    """Support edge zeta = -2/k, mean 0, mode -k/2, and +-z standard deviations
    (the default decade grid misses the bulk except near 0); for small |k| the
    window where scipy.special.gammainc stops converging is
    4.5 < -(t - alpha)/sqrt(alpha) < 15, i.e. 4.5 < -sign(k)*x < 15."""
    zeta = -2.0 / k
    z = np.array([0.0, 0.3, -0.3, 1, -1, 2, -2, 3, -3, 4, -4, 4.6, -4.6, 5, -5,
                  6, -6, 7, -7, 8, -8, 10, -10, 12, -12, 15, -15, 20, -20, 30,
                  -30, 37, -37])
    edge = zeta + np.sign(-zeta) * np.abs(zeta) * np.array(
        [1e-12, 1e-8, 1e-4, 1e-2, 0.5])
    return np.concatenate([z, [-k / 2, zeta], edge])


def q_points(k):
    return np.array([1e-20, 1e-12, 1e-8, 1e-5, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9,
                     0.99, 1 - 1e-5, 1 - 1e-8, 1 - 1e-12])
