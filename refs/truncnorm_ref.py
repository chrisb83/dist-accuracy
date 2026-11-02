"""Reference module for scipy.stats.truncnorm.

Standard form: the standard normal truncated to [a, b] (a < b, either may be
infinite):
    f(x; a, b) = phi(x) / Z,   Z = Phi(b) - Phi(a),   a <= x <= b
    F(x; a, b) = (Phi(x) - Phi(a)) / Z
    S(x; a, b) = (Phi(b) - Phi(x)) / Z
Differences Phi(hi) - Phi(lo) are evaluated as Phi(-lo) - Phi(-hi) when
lo > 0, so intervals far in the right tail (Phi(-a) < 10**-400) stay exact.
ppf/isf are obtained by inverting the cdf (no closed-form inverse in mpmath).
"""
import mpmath as mp
from accuracy import invert

DIST = "truncnorm"

INF = float("inf")

# Example values; a = 0 and b = 0 (quantiles/cdf near 0 with cond ~ 1);
# tiny negative a; b = inf (isf right branch for small q); narrow intervals
# (left/right and central cases of _log_gauss_mass); far tails.
PARAMS = [(-1.0978730080013919, 2.730675410903198), (0.1, 2.0),
          (0.0, INF), (-INF, 0.0), (-1.0, INF), (-1e-3, 10.0),
          (1.0, 1.00000001), (-1e-6, 1e-6), (30.0, INF), (-50.0, -10.0)]


def Phi(x):
    return mp.ncdf(x)


def mass(lo, hi):
    if lo > 0:
        return Phi(-lo) - Phi(-hi)
    return Phi(hi) - Phi(lo)


def pdf(x, a, b):
    if x < a or x > b:
        return mp.mpf(0)
    return mp.npdf(x) / mass(a, b)


def cdf(x, a, b):
    if x <= a:
        return mp.mpf(0)
    if x >= b:
        return mp.mpf(1)
    return mass(a, x) / mass(a, b)


def sf(x, a, b):
    if x <= a:
        return mp.mpf(1)
    if x >= b:
        return mp.mpf(0)
    return mass(x, b) / mass(a, b)


def _quantile(p, a, b):
    # Solve F(x) = p. Near a finite endpoint, solve for the offset t > 0 from
    # that endpoint in log scale (x = a + t or x = b - t): fast and with full
    # relative accuracy of x - a, also when x is extremely close to a or b.
    # 1 - p is exact at 400 digits.
    if p <= mp.mpf(0.5) and not mp.isinf(a):
        return a + invert(lambda t: cdf(a + t, a, b), p, positive=True)
    if p > mp.mpf(0.5) and not mp.isinf(b):
        return b - invert(lambda t: sf(b - t, a, b), 1 - p, positive=True)
    return invert(lambda x: cdf(x, a, b), p, positive=False)


def ppf(q, a, b):
    return _quantile(q, a, b)


def isf(q, a, b):
    return _quantile(1 - q, a, b)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, a, b: mp.log(pdf(x, a, b)),
    "cdf": cdf,
    "logcdf": lambda x, a, b: mp.log(cdf(x, a, b)),
    "sf": sf,
    "logsf": lambda x, a, b: mp.log(sf(x, a, b)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(a, b):
    # Near the ends (mass(a, x) / mass(x, b) over narrow intervals), around
    # 0 (central case of _log_gauss_mass and the b <= 0 / a > 0 switches),
    # and both sides of the logcdf/logsf switch at log F = -0.1.
    import numpy as np
    from scipy import stats
    pts = [0.0, 1e-300, -1e-300, 1e-100, -1e-100, 1e-20, -1e-20, 1e-12,
           -1e-12, 1e-8, -1e-8, 1e-4, -1e-4]
    for e in [a, b]:
        if np.isfinite(e):
            s = max(1.0, abs(e))
            for d in [1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
                pts += [e + s * d, e - s * d]
    for p in [np.exp(-0.1), 1 - np.exp(-0.1)]:
        x0 = float(stats.truncnorm.ppf(p, a, b))
        if np.isfinite(x0):
            pts += [x0 * (1 - 1e-3), x0, x0 * (1 + 1e-3)]
    return np.array(pts)


def q_points(a, b):
    # Quantiles near 0 (q around F(0)), the switch q = exp(-0.1).
    import numpy as np
    from scipy import stats
    pts = [1e-20, 1e-12, 1e-8, np.exp(-0.1), 1 - np.exp(-0.1)]
    q0 = float(stats.truncnorm.cdf(0.0, a, b))
    for d in [0, 1e-12, 1e-8, 1e-4]:
        pts += [q0 + d, q0 - d]
    return np.array(pts)
