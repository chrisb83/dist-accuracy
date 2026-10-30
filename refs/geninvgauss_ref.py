"""Reference module for scipy.stats.geninvgauss.

Docstring pdf (standard form):
    f(x; p, b) = x^(p-1) exp(-b (x + 1/x) / 2) / (2 K_p(b)),  x > 0, b > 0.
cdf/sf: with x = e^t the density in t is exp(p t - b cosh t) / (2 K_p(b)), so
    F(x) = int_{-inf}^{log x} exp(p t - b cosh t) dt / (2 K_p(b)),
    S(x) = int_{log x}^{inf}  exp(p t - b cosh t) dt / (2 K_p(b)),
evaluated with mp.quad (no closed form for general p), split at the peak
t* = asinh(p/b) and at +-k peak widths so tanh-sinh resolves sharp peaks.
Only the tail away from the peak is integrated (cdf left of t*, sf right of
it); the other is 1 - tail, so tiny cdf and sf both keep relative accuracy.
"""
import functools

import mpmath as mp
from accuracy import invert

DIST = "geninvgauss"
DPS = 100

# example; invgauss-like p=-1/2; tiny b (very wide distribution, p=0 log-like);
# small b with moderate p; large b (sharp peak at 1); kve overflow (|p| large, b small);
# large p and b (cancellation of large log terms).
# b >= 2**30 - 0.5: scipy.special.kve returns nan.
PARAMS = [(2.3, 1.5), (-0.5, 1.0), (0.0, 1e-3), (3.0, 1e-6), (0.5, 1e4),
          (-1.5, 1e6), (100.0, 0.01), (1e3, 1e3), (0.5, 2e9)]


@functools.lru_cache(maxsize=256)
def _K2(p, b, dps):
    # 2 K_p(b), cached per shape set and precision (besselk is slow for large p)
    return 2 * mp.besselk(p, b)


def _logK2(p, b):
    return mp.log(_K2(p, b, mp.mp.dps))


def logpdf(x, p, b):
    return (p - 1) * mp.log(x) - b * (x + 1 / x) / 2 - _logK2(p, b)


def pdf(x, p, b):
    return mp.exp(logpdf(x, p, b))


def _breaks(p, b):
    ts = mp.asinh(mp.mpf(p) / b)
    # width of the peak in t: 1/sqrt(b cosh t*)
    w = 1 / mp.sqrt(b * mp.cosh(ts))
    return ts, w


def _h(t, p, b, g0):
    # minus the log of the scaled integrand
    return b * mp.cosh(t) - p * t + g0


def _trunc(ts, w, p, b, g0, sgn, cut):
    # first point ts + sgn*w*2^k where the scaled integrand is below e^-cut
    k = 1
    while _h(ts + sgn * w * k, p, b, g0) < cut:
        k *= 2
    return ts + sgn * w * k


def _integral(lo, hi, p, b):
    # int_lo^hi exp(p t - b cosh t) dt.  The integrand is scaled by its peak
    # value and the infinite ends are truncated where it is below
    # 10^-(dps + 30) relative to the peak (tanh-sinh is very slow on
    # double-exponentially decaying integrands over infinite intervals).
    ts, w = _breaks(p, b)
    g0 = p * ts - b * mp.cosh(ts)
    cut = (mp.mp.dps + 30) * mp.log(10)
    if lo == mp.ninf:
        lo = min(_trunc(ts, w, p, b, g0, -1, cut), hi - w)
    if hi == mp.inf:
        hi = max(_trunc(ts, w, p, b, g0, 1, cut), lo + w)
    f = lambda t: mp.exp(p * t - b * mp.cosh(t) - g0)
    pts = [ts + k * w for k in (-20, -5, -1, 0, 1, 5, 20)]
    nodes = [lo] + [t for t in pts if lo < t < hi] + [hi]
    return mp.quad(f, nodes) * mp.exp(g0)


def _norm(p, b):
    return _K2(p, b, mp.mp.dps)


def _far_tail(T, p, b):
    # If the integrand at the endpoint T is below e^-2000 times its peak, the
    # tail integral is far below the double range (never measured) and
    # tanh-sinh cannot resolve the endpoint spike. Return the leading Laplace
    # term exp(p T - b cosh T) / |b sinh T - p| instead (relative error
    # O(1/h), deterministic, hence stable in precision); else None.
    ts, w = _breaks(p, b)
    g0 = p * ts - b * mp.cosh(ts)
    if _h(T, p, b, g0) > 2000:
        return mp.exp(p * T - b * mp.cosh(T)) / abs(b * mp.sinh(T) - p)
    return None


def _tail(x, p, b):
    # Integrate only the tail away from the peak t* (left of it: cdf, right
    # of it: sf); the other function is 1 - tail, so cdf + sf = 1 exactly and
    # the small one keeps full relative accuracy.  Returns (cdf, sf).
    T = mp.log(x)
    ts, _ = _breaks(p, b)
    v = _far_tail(T, p, b)
    if T < ts:
        if v is None:
            v = _integral(mp.ninf, T, p, b)
        c = v / _norm(p, b)
        return c, 1 - c
    if v is None:
        v = _integral(T, mp.inf, p, b)
    s = v / _norm(p, b)
    return 1 - s, s


def cdf(x, p, b):
    return _tail(x, p, b)[0]


def sf(x, p, b):
    return _tail(x, p, b)[1]


def ppf(q, p, b):
    return invert(lambda x: cdf(x, p, b), q)


def isf(q, p, b):
    return invert(lambda x: cdf(x, p, b), 1 - q)


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": lambda x, p, b: mp.log(cdf(x, p, b)),
    "sf": sf,
    "logsf": lambda x, p, b: mp.log(sf(x, p, b)),
}
# ppf and isf are inherited generic methods (not measured). Their references
# (ppf, isf above) need hundreds of quad calls each, which made the harness
# self-checks far too slow, so they are left out of REFS. Add them back when
# a custom ppf/isf is being tested.


def x_points(p, b):
    import numpy as np
    # mode and its neighbourhood (cancellation b - b(x+1/x)/2 near x = 1 for
    # large b, (p-1)log x vs log K for large p), plus x = 1 neighbourhood.
    if p < 1:
        m = b / (np.sqrt((p - 1) ** 2 + b ** 2) + 1 - p)
    else:
        m = (np.sqrt((1 - p) ** 2 + b ** 2) - (1 - p)) / b
    w = 1 / np.sqrt(b * np.cosh(np.arcsinh(p / b)))
    rel = [1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.5]
    pts = [m] + [m * (1 + s * r) for r in rel for s in (-1, 1)]
    pts += [m * np.exp(k * w) for k in (-10, -5, -3, -1, 1, 3, 5, 10)]
    pts += [1 - 1e-3, 1 + 1e-3, 1 - 1e-8, 1 + 1e-8]
    return np.array([x for x in pts if x > 0])
