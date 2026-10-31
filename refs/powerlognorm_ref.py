"""Reference module for scipy.stats.powerlognorm.

Standard form, docstring pdf
    f(x; c, s) = c/(x s) phi(log(x)/s) Phi(-log(x)/s)**(c-1),  x > 0, c, s > 0.
With z = log(x)/s: sf = Phi(-z)**c, cdf = 1 - sf (via expm1 of the log),
ppf/isf in closed form x = exp(-s Phi^-1(q_sf**(1/c))), Phi^-1 by Newton on
log Phi in mpmath (cross-checked against invert() of the cdf: ppf_invert/isf_invert).
"""
import mpmath as mp
from accuracy import invert

DIST = "powerlognorm"
# All formulas avoid 1 - (number close to 1), so moderate precision suffices
# (checked against 400 digits); keeps the ppf/isf condition numbers fast.
DPS = 120

# Example; small c (cancellation in logpdf, q**(1/c) underflow in isf);
# large c (q**(1/c) near 1 in isf, large exponents in sf/pdf); large s;
# tiny s (z**2 overflow in _norm_logpdf), small s (large z near x = 1);
# huge c (log_ndtr underflow multiplied by c).
PARAMS = [(2.1413923530064087, 0.44639540782048337), (1e-3, 1.0), (1e3, 0.5),
          (1e6, 1.0), (2.0, 1e3), (0.5, 1e-152), (5.0, 1e-3), (1e20, 1.0)]


def _z(x, s):
    return mp.log(x) / s


def _logPhi(t):
    # log of the normal cdf; for t > 0 via log1p of the (small) upper tail so
    # that no digits are lost at moderate precision.
    if abs(t) > 1e100:
        # mpmath's erfc overflows internally for |t| > ~1e154; use the
        # asymptotic Mills-ratio form (relative corrections ~1/t**2 < 1e-200).
        a = abs(t)
        logtail = -a ** 2 / 2 - mp.log(a) - mp.log(2 * mp.pi) / 2
        return logtail if t < 0 else -mp.exp(logtail)
    if t > 0:
        return mp.log1p(-mp.erfc(t / mp.sqrt(2)) / 2)
    return mp.log(mp.erfc(-t / mp.sqrt(2)) / 2)


def logsf(x, c, s):
    return c * _logPhi(-_z(x, s))


def logpdf(x, c, s):
    z = _z(x, s)
    return (mp.log(c) - mp.log(x) - mp.log(s) - z ** 2 / 2
            - mp.log(2 * mp.pi) / 2 + (c - 1) * _logPhi(-z))


def pdf(x, c, s):
    return mp.exp(logpdf(x, c, s))


def sf(x, c, s):
    return mp.exp(logsf(x, c, s))


def cdf(x, c, s):
    return -mp.expm1(logsf(x, c, s))


def _ninv_log(L):
    """Phi^-1(u) for log(u) = L <= log(1/2): Newton on log Phi(z) = L."""
    # starting value from double precision (speed only; Newton refines it)
    import math
    from scipy.special import ndtri_exp
    Lf = float(L)
    z0 = ndtri_exp(Lf) if Lf > -1e300 else -math.sqrt(-2 * Lf) if math.isfinite(Lf) else None
    z = mp.mpf(z0) if z0 is not None and math.isfinite(z0) else -mp.sqrt(-2 * L)
    for _ in range(200):
        lp = _logPhi(z)
        # d/dz log Phi(z) = phi(z)/Phi(z)
        d = mp.exp(-z ** 2 / 2 - mp.log(2 * mp.pi) / 2 - lp)
        step = (lp - L) / d
        z -= step
        if abs(step) <= abs(z) * mp.mpf(10) ** (-mp.mp.dps + 20) + mp.mpf(10) ** (-mp.mp.dps + 20):
            break
    return z


def _ninv_from_log(logu):
    """Phi^-1(u) given log(u), accurate on both sides (1 - u = -expm1(logu))."""
    if logu <= -mp.log(2):
        return _ninv_log(logu)
    return -_ninv_log(mp.log(-mp.expm1(logu)))


def _x_from_logsf(logsf_target, c, s):
    # sf(x) = Phi(-z)**c  ->  -z = Phi^-1(sf**(1/c)),  x = exp(-s Phi^-1(sf**(1/c)))
    return mp.exp(-s * _ninv_from_log(logsf_target / c))


def isf(q, c, s):
    return _x_from_logsf(mp.log(q), c, s)


def ppf(q, c, s):
    return _x_from_logsf(mp.log1p(-q), c, s)


def ppf_invert(q, c, s):
    return invert(lambda x: cdf(x, c, s), q)


def isf_invert(q, c, s):
    return invert(lambda x: cdf(x, c, s), 1 - q)


def logcdf(x, c, s):
    # log(1 - sf): log1p(-sf) when sf is small (cdf near 1), log(-expm1(logsf))
    # when sf is near 1 (cdf tiny); both keep full relative accuracy.
    L = logsf(x, c, s)
    if L < -mp.log(2):
        return mp.log1p(-mp.exp(L))
    return mp.log(-mp.expm1(L))


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": lambda x, c, s: logcdf(x, c, s),
    "sf": sf,
    "logsf": logsf,
    "ppf": ppf,
    "isf": isf,
}


def x_points(c, s):
    # x = 1 and neighbours, x where z = log(x)/s hits the ndtr/log_ndtr
    # tail regions (+-8, 30, 37.5, 38.4, 38.6, 40), x where |z| passes the
    # z**2 overflow threshold 1.34e154 (tiny s).
    import numpy as np
    pts = [1.0, 1 + 2**-52, 1 - 2**-53, 1 + 1e-8, 1 - 1e-8, 0.5, 2.0]
    with np.errstate(over="ignore", under="ignore"):
        for z in [-40.0, -38.6, -38.4, -37.5, -30.0, -8.0, 8.0, 30.0, 37.5,
                  38.4, 100.0, 1e3, 1e4]:
            if abs(z * s) < 700:
                pts.append(np.exp(z * s))
        for z in [1.2e154, 1.4e154, 1.6e154, 1.85e154, 2e154]:
            for sg in (1, -1):
                if abs(z * s) < 700:
                    pts.append(np.exp(sg * z * s))
    return np.array(pts)


def q_points(c, s):
    import numpy as np
    # isf: q where log(q)/c < -745 (q**(1/c) underflows) for small c
    pts = [0.5, 0.1, 0.9, 1e-3, 1 - 1e-3, 1e-10, 1 - 1e-10]
    with np.errstate(over="ignore", under="ignore"):
        for t in [-700.0, -750.0]:
            v = np.exp(t * c)
            if 0 < v < 1:
                pts.append(v)
    return np.array(pts)
