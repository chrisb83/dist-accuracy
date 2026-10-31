"""Reference module for scipy.stats.recipinvgauss.

Docstring pdf (standard form):
    f(x; mu) = (2 pi x)^(-1/2) exp(-(1 - mu x)^2 / (2 mu^2 x)),  x > 0.
X = 1/Y with Y ~ invgauss(mu), so cdf_X(x) = sf_Y(1/x):
    z1 = (1/mu - x)/sqrt(x),  z2 = (1/mu + x)/sqrt(x)
    sf  = Phi(z1)  + exp(2/mu) Phi(-z2)      (sum of positive terms)
    cdf = Phi(-z1) - exp(2/mu) Phi(-z2)      (cancels for x << 1/mu; computed with
                                              extra digits ~ log10((1/mu + x)/(2x)))
"""
import mpmath as mp
from accuracy import invert

DIST = "recipinvgauss"

# example; mu = 1; 0.1 and 0.01 (exp(2/mu) large, Phi(-z2) underflow);
# 3e-3 just above and 1e-3 below the exp(2/mu) overflow at mu = 2.818e-3;
# large mu 1e3, 1e8; 1e160 (mu**2 overflows in logpdf); 1e-200 (mu**2 underflows).
PARAMS = [(0.6300426780936912,), (1.0,), (0.1,), (0.01,), (3e-3,), (1e-3,),
          (1e3,), (1e8,), (1e160,), (1e-200,)]


def _erfc(t):
    # asymptotic series for huge |t| (mpmath erfc is slow/overflows there)
    if t > 10**4:
        u = 1 / (2 * t * t)
        term, tot = mp.mpf(1), mp.mpf(1)
        for n in range(1, 80):
            term *= -(2 * n - 1) * u
            tot += term
        return mp.exp(-t * t) / (t * mp.sqrt(mp.pi)) * tot
    if t < -10**4:
        return 2 - _erfc(-t)
    return mp.erfc(t)


def _Phi(z):
    return _erfc(-z / mp.sqrt(2)) / 2


def pdf(x, mu):
    return 1 / mp.sqrt(2 * mp.pi * x) * mp.exp(-(1 - mu * x) ** 2 / (2 * mu**2 * x))


def logpdf(x, mu):
    return -(1 - mu * x) ** 2 / (2 * mu**2 * x) - mp.log(2 * mp.pi * x) / 2


def sf(x, mu):
    s = mp.sqrt(x)
    return _Phi((1 / mu - x) / s) + mp.exp(2 / mu) * _Phi(-(1 / mu + x) / s)


def cdf(x, mu):
    x, mu = mp.mpf(x), mp.mpf(mu)
    ratio = (1 / mu + x) / (2 * x)
    extra = int(max(0, mp.log10(ratio))) + 20 if ratio > 1 else 20
    with mp.workdps(mp.mp.dps + extra):
        s = mp.sqrt(x)
        r = _Phi(-(1 / mu - x) / s) - mp.exp(2 / mu) * _Phi(-(1 / mu + x) / s)
    return +r


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": lambda x, mu: mp.log(cdf(x, mu)),
    "sf": sf,
    "logsf": lambda x, mu: mp.log(sf(x, mu)),
    "ppf": lambda q, mu: invert(lambda x: cdf(x, mu), q),
    "isf": lambda q, mu: invert(lambda x: cdf(x, mu), 1 - q),
}


def x_points(mu):
    import numpy as np
    m = float(mu)
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        im = 1.0 / m
        pts = [im * f for f in (1e-6, 1e-3, 0.1, 0.5, 1 - 1e-8, 1.0, 1 + 1e-8, 2.0, 10.0, 1e3)]
        # left tail where cdf is still normal: 1/(2 mu^2 x) up to ~700
        pts += [im * im / f for f in (1400.0, 1000.0, 100.0, 10.0)]
        # right tail where sf is still normal (x/2 up to ~700)
        pts += [100.0, 500.0, 1000.0, 1300.0, 1400.0]
        # (1 - mu x)**2 overflow: mu x ~ 1.34e154
        pts += [1.3e154 * im, 1.4e154 * im]
        return np.array([p for p in pts if np.isfinite(p) and p > 0], float)
