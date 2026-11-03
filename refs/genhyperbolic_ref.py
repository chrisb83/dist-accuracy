"""Reference module for scipy.stats.genhyperbolic.

Docstring pdf (standard form, loc = 0, scale = 1, delta = 1):
    f(x; p, a, b) = g^p / (sqrt(2 pi) a^(p-1/2) K_p(g)) e^(b x)
                    * K_(p-1/2)(a s) s^(p-1/2),
    s = sqrt(1 + x^2),  g = sqrt(a^2 - b^2) = sqrt((a+b)(a-b)).
|b| < a for p >= 0, |b| <= a for p < 0.  For g = 0 (p < 0, |b| = a) the
factor g^p / K_p(g) is replaced by its limit 1 / (Gamma(-p) 2^(-p-1)).

cdf/sf: normal variance-mean mixture (as in SciPy's _rvs):
    X = b V + sqrt(V) Z,  Z ~ N(0, 1),
    V ~ GIG with density  g^p / (2 K_p(g)) v^(p-1) exp(-(g^2 v + 1/v)/2)
        (g = 0: v^(p-1) exp(-1/(2v)) / (2^(-p) Gamma(-p)), an inverse gamma),
    F(x) = int_0^inf Phi((x - b v)/sqrt(v)) gig(v) dv.
The integral is taken in u = log v by the trapezoidal rule (analytic integrand
with double-exponentially or exponentially decaying ends, so the rule
converges geometrically). The integrand is scaled by its maximum (found by a
coarse scan and golden-section search), truncated where it is below
10^-(dps+20) of the maximum, and the step is halved until two successive sums
agree to 10^-(0.9 dps). Wide heavy-tail integrands (sharp step of Phi at
v = x/b, then a plateau) use mp.quad split at the maximum and the step.
sf(x; p, a, b) = F(-x; p, a, -b). The tail side (cdf for x <= 0, sf for
x >= 0) is integrated directly, the other side is 1 - tail, so tiny values
keep full relative accuracy.

pdf uses mp.besselk; 100 digits because mp.besselk with integer order
(p = 0.5 gives K_0) takes up to 14 s at 400 digits.
"""
import functools
import os

import mpmath as mp
import numpy as np

from accuracy import invert

DIST = "genhyperbolic"
DPS = 100
REPLACE_GRID = True

# example; symmetric hyperbolic (p = 1); |b| = a with p < 0 (allowed, g = 0);
# b/a = 1 - 1e-10 (near-polynomial right tail: pdf normal where a*s > 2**30);
# large a (sharp peak, a - a*s cancellation); large p with small g (kve
# overflow); tiny a (very wide); negative p with skew.
PARAMS = [(0.5, 1.5, -0.5), (1.0, 2.0, 0.0), (-1.0, 1.0, 1.0),
          (-1.5, 1.0, 0.9999999999), (1.0, 1e6, 0.0), (100.0, 0.01, 0.0),
          (0.5, 1e-3, 0.0), (-2.0, 3.0, -2.0)]

if os.environ.get("GH_PARAMS"):        # e.g. GH_PARAMS="0,1,2" to run a subset
    PARAMS = [PARAMS[int(i)] for i in os.environ["GH_PARAMS"].split(",")]


@functools.lru_cache(maxsize=512)
def _log_norm_parts(p, a, b, dps):
    """log of g^p / K_p(g) (or its g -> 0 limit)."""
    with mp.workdps(dps):
        p, a, b = mp.mpf(p), mp.mpf(a), mp.mpf(b)
        g2 = (a + b) * (a - b)
        if g2 == 0:
            if p >= 0:
                raise ValueError("|b| = a requires p < 0")
            return -(mp.loggamma(-p) + (-p - 1) * mp.log(2)), g2
        g = mp.sqrt(g2)
        return p * mp.log(g) - mp.log(mp.besselk(p, g)), g2


def _lognorm(p, a, b):
    v, _ = _log_norm_parts(p, a, b, mp.mp.dps)
    return v


def logpdf(x, p, a, b):
    s = mp.sqrt(1 + x * x)
    return (_lognorm(p, a, b) - mp.log(2 * mp.pi) / 2 - (p - mp.mpf(1) / 2) * mp.log(a)
            + b * x + (p - mp.mpf(1) / 2) * mp.log(s) + mp.log(mp.besselk(p - mp.mpf(1) / 2, a * s)))


def pdf(x, p, a, b):
    return mp.exp(logpdf(x, p, a, b))


def _log_gig_const(p, a, b):
    # log of the GIG normalization: g^p / (2 K_p(g)) == exp(_lognorm) / 2
    return _lognorm(p, a, b) - mp.log(2)


def _logPhi(z):
    # mp.ncdf raises OverflowError for astronomically large |z| (reached only
    # in the coarse scan, far from the maximum of the integrand). Beyond
    # |z| = 1e5 use log Phi = 0 resp. the asymptotic series; the omitted
    # terms are below exp(-5e9) resp. 1e-29 relative and such points never
    # carry weight in a normal-double result.
    if z > 100000:
        return mp.mpf(0)
    if z < -100000:
        return -z * z / 2 - mp.log(-z) - mp.log(2 * mp.pi) / 2 + mp.log(1 - 1 / z ** 2 + 3 / z ** 4)
    return mp.log(mp.ncdf(z))


def _mixture_cdf(x, p, a, b):
    """F(x) = int Phi((x - b v)/sqrt(v)) gig(v) dv, in u = log v."""
    g2 = (a + b) * (a - b)
    c0 = _log_gig_const(p, a, b)

    def L(u):
        e = mp.exp(u)
        z = x / mp.sqrt(e) - b * mp.sqrt(e)
        return c0 + p * u - (g2 * e + 1 / e) / 2 + _logPhi(z)

    # coarse scan for the maximum (log integrand is unimodal in u here)
    us = [mp.mpf(k) for k in range(-760, 761, 8)]
    vals = [L(u) for u in us]
    i = max(range(len(us)), key=lambda k: vals[k])
    lo, hi = us[max(i - 1, 0)], us[min(i + 1, len(us) - 1)]
    # golden-section refinement
    gr = (mp.sqrt(5) - 1) / 2
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fc, fd = L(c), L(d)
    for _ in range(90):
        if fc > fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = L(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = L(d)
    us_ = (lo + hi) / 2
    Lmax = L(us_)
    hstep = mp.mpf("1e-4")
    d2 = (L(us_ + hstep) - 2 * Lmax + L(us_ - hstep)) / hstep ** 2
    w = 1 / mp.sqrt(-d2) if d2 < 0 else mp.mpf(1)
    w = min(w, mp.mpf(1))
    if w < mp.mpf("1e-12"):
        # Peak narrower than the golden-section resolution (~1e-17). This
        # happens only for |x| so large that the Gaussian factor is
        # astronomically small (width ~ 1/sqrt(a |x|)): the value is far
        # outside the double range and is never measured. Raise so the
        # harness skips the point instead of hanging.
        raise ValueError(f"integrand peak unresolved (width {mp.nstr(w, 3)}, "
                         f"log integrand {mp.nstr(Lmax, 5)}); value not a normal double")
    cut = (mp.mp.dps + 20) * mp.log(10)
    k = 1
    while Lmax - L(us_ - w * k) < cut:
        k *= 2
        if k > 2 ** 40:
            raise RuntimeError("left truncation not found")
    left = us_ - w * k
    k = 1
    while Lmax - L(us_ + w * k) < cut:
        k *= 2
        if k > 2 ** 40:
            raise RuntimeError("right truncation not found")
    right = us_ + w * k
    if Lmax + mp.log(right - left) < -709 and not os.environ.get("GH_INHERITED"):
        # integrand <= exp(Lmax) on [left, right]: F < 2.2e-308, not measured
        # (skipped with GH_INHERITED so that logcdf/logsf demos work)
        raise ValueError(f"value below the normal range (log integrand max {mp.nstr(Lmax, 5)})")

    f = lambda u: mp.exp(L(u) - Lmax)
    if (right - left) / w > 2 ** 14:
        # Heavy tail with a sharp step of Phi at v = x/b (width ~ 1/sqrt(x b)
        # in u) followed by a long plateau: tanh-sinh on pieces split at the
        # maximum and at the step (nodes cluster at the piece ends).
        pts = {left, right, us_}
        if b != 0 and x / b > 0:
            u0 = mp.log(x / b)
            if left < u0 < right:
                pts.add(u0)
        pts = sorted(pts)
        extra = []
        for lo_, hi_ in zip(pts[:-1], pts[1:]):
            # geometric subdivision away from the sharp ends
            m = hi_ - lo_
            for j in range(1, 12):
                t = m * mp.mpf(2) ** (-3 * j)
                if t > w:
                    extra += [lo_ + t, hi_ - t]
        pts = sorted(set(pts) | set(extra))
        T = mp.quad(f, pts)
        return mp.exp(Lmax) * T
    n = int(mp.ceil((right - left) / (w / 2)))
    h = (right - left) / n
    S = sum(f(left + j * h) for j in range(1, n))
    T = h * S
    tol = mp.mpf(10) ** (-int(0.9 * mp.mp.dps))
    for _ in range(14):
        h2 = h / 2
        S += sum(f(left + (2 * j + 1) * h2) for j in range(n))
        n *= 2
        h = h2
        Tn = h * S
        if abs(Tn - T) <= tol * abs(Tn):
            return mp.exp(Lmax) * Tn
        T = Tn
        if n > 2 ** 19:
            break
    raise RuntimeError("trapezoid did not converge")


def cdf(x, p, a, b):
    # integrate the tail side directly; the other side as 1 - tail (exact
    # enough at DPS digits; avoids the long plateau Phi ~ 1 of heavy tails).
    # Split at x = 0: cdf(0) is between 0.18 and 0.83 for all PARAMS.
    if x <= 0:
        return _mixture_cdf(x, p, a, b)
    return 1 - _mixture_cdf(-x, p, a, -b)


def sf(x, p, a, b):
    if x >= 0:
        return _mixture_cdf(-x, p, a, -b)
    return 1 - _mixture_cdf(x, p, a, b)


def ppf(q, p, a, b):
    return invert(lambda x: cdf(x, p, a, b), q, positive=False)


def isf(q, p, a, b):
    return invert(lambda x: cdf(x, p, a, b), 1 - q, positive=False)


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "sf": sf,
}
# The inherited methods (logcdf, logsf, ppf, isf) are not measured. Their
# references are slow (ppf/isf invert the quadrature cdf, and the harness
# self-checks evaluate every entry of REFS at DPS + 100), so they are only
# exposed for single-point demonstrations with --at:  GH_INHERITED=1
if os.environ.get("GH_INHERITED"):
    REFS.update({
        "logcdf": lambda x, p, a, b: mp.log(cdf(x, p, a, b)),
        "logsf": lambda x, p, a, b: mp.log(sf(x, p, a, b)),
        "ppf": ppf,
        "isf": isf,
    })


def x_points(p, a, b):
    # bulk, small |x|, tails, and both sides of a*sqrt(1+x^2) = 2**30 - 0.5
    # where scipy.special.kve returns nan (INDEX item 16), and of x**2 overflow.
    t = (2.0 ** 30 - 0.5) / a
    thr = [t * 0.99, t * 1.01] if t > 1 else []
    mags = [1e-300, 1e-20, 1e-3, 0.3, 20, 30, 50, 100, 200, 500, 1e3, 1e4, 1e6,
            1e9, 1e12, 1e20, 1e50, 1e100, 1.3e154, 1.4e154, 1e200, 1e300]
    pts = list(np.linspace(-10, 10, 21)) + [0.0]
    for m in mags + thr:
        pts += [m, -m]
    if a >= 100:   # sharp peak of width ~ 1/sqrt(a)
        w = 1 / np.sqrt(a)
        pts += list(w * np.array([-8, -5, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 5, 8]))
    return np.array(pts)
