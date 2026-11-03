"""Reference module for scipy.stats.vonmises_line (also covers vonmises on [-pi, pi]).

Standard form, docstring pdf: f(x; kappa) = exp(kappa cos x) / (2 pi I0(kappa)),
-pi <= x <= pi, kappa >= 0.

cdf: F(x) = int_{-pi}^{x} f(t) dt by mp.quad, using the symmetry
F(x) = 1 - F(-x) for x > 0 so that the integrand is monotone on the
integration interval [-pi, min(x, -x)].  The interval is split into pieces
scaled to the peak width 1/sqrt(kappa) so tanh-sinh resolves it; in the
tail (kappa sin^2(x/2) >= 1) the substitution u = sin^2(t/2) is used instead
(see _lower_tail).
sf(x) = cdf(-x) and isf(q) = -ppf(q) by symmetry, ppf by inversion.
"""
import mpmath as mp
import numpy as np
from accuracy import invert

DIST = "vonmises_line"
DPS = 100

# example; kappa = 0 (uniform); tiny; moderate; both sides of the CK = 50
# switch between the series and the normal approximation; large kappa
# (exp(kappa) overflow in a naive normal approximation for kappa > 709.8).
PARAMS = [(3.99390425810714,), (0.0,), (1e-3,), (0.5,), (20.0,), (49.9,), (50.0,), (200.0,), (1000.0,)]


def _scaled_integrand(k):
    # exp(k (cos t - 1)), monotone increasing on [-pi, 0]
    return lambda t: mp.exp(k * (mp.cos(t) - 1))


def _norm(k):
    # 2 pi I0(k) exp(-k)
    return 2 * mp.pi * mp.besseli(0, k) * mp.exp(-k)


def pdf(x, k):
    if abs(x) > mp.pi:
        return mp.mpf(0)
    return mp.exp(k * (mp.cos(x) - 1)) / _norm(k)


def _lower_tail(x, k):
    # int_{-pi}^{x} f for x in the tail, by u = sin^2(t/2):
    # exp(k cos t) dt = exp(k - 2 k u) du / sqrt(u (1 - u)), u in [u0, 1].
    # The integrand decays on the scale 1/(2k) in u away from u0, i.e.
    # ~1/(k sin|x|) in t, much narrower than 1/sqrt(k); breakpoints at
    # u0 + 2^j/(2k).  Split at the midpoint h and integrate the upper half
    # in s = sqrt(1 - u), which removes the 1/sqrt(1 - u) endpoint
    # singularity (w = 1 - u directly made tanh-sinh slow and inaccurate).
    u0 = mp.sin(x / 2) ** 2
    c = 1 - u0
    h = c / 2
    fa = lambda v: mp.exp(-2 * k * v) / mp.sqrt((u0 + v) * (c - v))
    fb = lambda s: 2 * mp.exp(-2 * k * (c - s * s)) / mp.sqrt(1 - s * s)
    bps = []
    j = 0
    while mp.mpf(2) ** j / (2 * k) < h:
        bps.append(mp.mpf(2) ** j / (2 * k))
        j += 1
    integral = mp.quad(fa, [0] + bps + [h]) + mp.quad(fb, [0, mp.sqrt(h)])
    return integral * mp.exp(-2 * k * u0) / _norm(k)


def _lower(x, k):
    # int_{-pi}^{x} f, x <= 0
    if x <= -mp.pi:
        return mp.mpf(0)
    if k > 1 and k * mp.sin(x / 2) ** 2 >= 1:
        # tail: the t-quad below with 1/sqrt(k) breakpoints does not resolve
        # the endpoint layer (cdf(-1, 1000) was off by 5e-7 relative)
        return _lower_tail(x, k)
    a = -mp.pi
    if k > 1:
        w = 1 / mp.sqrt(k)
        pts = [a]
        # breakpoints at -w*2^j (j = 0..) between a and x
        j = 0
        bps = []
        while True:
            b = -w * 2 ** j
            if b <= a:
                break
            bps.append(b)
            j += 1
        bps = sorted(b for b in bps if a < b < x)
        pts += bps + [x]
    else:
        pts = [a, x]
    return mp.quad(_scaled_integrand(k), pts) / _norm(k)


def cdf(x, k):
    if x <= 0:
        return _lower(x, k)
    if x >= mp.pi:
        return mp.mpf(1)
    return 1 - _lower(-x, k)


def sf(x, k):
    # symmetry of the density: S(x) = F(-x)
    return cdf(-x, k)


def ppf(q, k):
    return invert(lambda x: cdf(x, k), q, positive=False)


def isf(q, k):
    # symmetry: isf(q) = -ppf(q); avoids 1 - q rounding to 1 at DPS = 100
    return -ppf(q, k)


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
    pi = np.pi
    pts = [-pi, np.nextafter(-pi, 0), -pi + 1e-8, -pi + 1e-4, -3.1, -3.0, -2.5, -2.0,
           -1.5, -1.0, -0.5, -0.2, -0.1, -1e-3, -1e-8, -1e-20, 0.0, 1e-20, 1e-8, 1e-3,
           0.1, 0.5, 1.0, 2.0, 3.0, 3.1, pi - 1e-8]
    if k > 1:
        s = 1 / np.sqrt(k)
        pts += [-m * s for m in (1, 2, 4, 6, 8, 10, 14, 20)] + [m * s for m in (1, 3, 6, 10)]
    return np.array([p for p in pts if -pi <= p <= pi])
