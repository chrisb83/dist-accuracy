"""Reference module for scipy.stats.rel_breitwigner.

Docstring pdf: f(x; rho) = k / ((x^2 - rho^2)^2 + rho^2), x >= 0, with
k = 2 sqrt(2) rho^2 sqrt(rho^2+1) / (pi sqrt(rho^2 + rho sqrt(rho^2+1))).

Partial fractions with b = sqrt(-rho^2 - i rho) (Re b > 0):
  cdf(x) = (k/rho) Im(atan(x/b)/b),
  sf(x)  = (k/rho) Im(atan(b/x)/b)   (atan z + atan 1/z = pi/2, Re z > 0).
sf is computed directly so the far right tail (sf << 1e-400) stays accurate.
"""
import mpmath as mp

DIST = "rel_breitwigner"

PARAMS = [(36.545206797050334,), (1.0,), (1e-3,), (1e-8,), (1e3,), (1e6,), (1e20,), (1e-160,), (1e-154,), (1e-150,), (1e150,), (1e152,), (1e154,), (1e160,)]


def _k(rho):
    rho = mp.mpf(rho)
    s = mp.sqrt(rho**2 + 1)
    return 2 * mp.sqrt(2) * rho**2 * s / (mp.pi * mp.sqrt(rho**2 + rho * s))


def _b(rho):
    rho = mp.mpf(rho)
    return mp.sqrt(-rho**2 - mp.mpc(0, 1) * rho)


def pdf(x, rho):
    x, rho = mp.mpf(x), mp.mpf(rho)
    return _k(rho) / ((x**2 - rho**2) ** 2 + rho**2)


def cdf(x, rho):
    if x == 0:
        return mp.mpf(0)
    b = _b(rho)
    return _k(rho) / rho * mp.im(mp.atan(x / b) / b)


def sf(x, rho):
    if x == 0:
        return mp.mpf(1)
    x, rho = mp.mpf(x), mp.mpf(rho)
    b = _b(rho)
    if abs(b) / x > mp.mpf("1e-5"):
        return _k(rho) / rho * mp.im(mp.atan(b / x) / b)
    # Large x: atan(z)/b = sum_n (-1)^n b^(2n) / ((2n+1) x^(2n+1)); the n = 0
    # term 1/x is real, so sum n >= 1 with b^2 = -rho^2 - i rho exactly.
    b2 = mp.mpc(-rho**2, -rho)
    t = mp.mpc(1) / x
    tot = mp.mpf(0)
    n = 0
    while True:
        n += 1
        t = -t * b2 / x**2
        term = mp.im(t) / (2 * n + 1)
        tot += term
        if n > 3 and abs(term) < abs(tot) * mp.mpf(10) ** (-mp.mp.dps - 5):
            break
    return _k(rho) / rho * tot



REFS = {
    "pdf": pdf,
    "logpdf": lambda x, rho: mp.log(pdf(x, rho)),
    "cdf": cdf,
    "logcdf": lambda x, rho: mp.log(cdf(x, rho)),
    "sf": sf,
    "logsf": lambda x, rho: mp.log(sf(x, rho)),
}


def x_points(rho):
    import numpy as np
    r = float(rho)
    pts = [r * (1 - 1e-8), r, r * (1 + 1e-8), r * 0.5, r * 2, r - 1, r + 1,
           r - 0.5, r + 0.5, r - 0.25, r + 0.25, r - 2, r + 2, r * 1e-3, r * 1e3,
           1.16e77 * np.sqrt(r) * 0.9, 1.16e77 * np.sqrt(r) * 1.1]
    return np.array([p for p in pts if np.isfinite(p) and p > 0])
