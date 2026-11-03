"""Reference module for scipy.stats.semicircular.

Standard form, no shapes. Docstring pdf: f(x) = 2/pi * sqrt(1 - x**2), -1 <= x <= 1.
cdf F(x) = 1/2 + (x sqrt(1 - x**2) + asin(x))/pi, sf = 1 - F, ppf/isf by
numerical inversion of F (no closed form).
"""
import mpmath as mp
from accuracy import invert

DIST = "semicircular"

PARAMS = [()]


def pdf(x):
    return 2 / mp.pi * mp.sqrt(1 - x * x)


def cdf(x):
    return mp.mpf(1) / 2 + (x * mp.sqrt(1 - x * x) + mp.asin(x)) / mp.pi


def sf(x):
    return 1 - cdf(x)


def _center(t):
    # Solve (x sqrt(1-x^2) + asin x)/pi = t for |t| < 1/4 directly, so that
    # the root keeps full relative accuracy near x = 0 (ppf(1/2) = 0 exactly).
    if t == 0:
        return mp.mpf(0)
    return mp.findroot(lambda x: (x * mp.sqrt(1 - x * x) + mp.asin(x)) / mp.pi - t,
                       t * mp.pi / 2)


def ppf(q):
    q = mp.mpf(q)
    if abs(q - mp.mpf(1) / 2) < mp.mpf(1) / 4:
        return _center(q - mp.mpf(1) / 2)
    return invert(cdf, q, lo=-1, hi=1, positive=False)


def isf(q):
    return -ppf(q)  # symmetry: F(-x) = 1 - F(x)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x: mp.log(pdf(x)),
    "cdf": cdf,
    "logcdf": lambda x: mp.log(cdf(x)),
    "sf": sf,
    "logsf": lambda x: mp.log(sf(x)),
    "ppf": ppf,
    "isf": isf,
}


def x_points():
    # Both endpoints approached from inside (1 ulp ... 1e-2), x = 0 and tiny x
    # (logpdf, cdf near 1/2), and points around 0.5.
    import numpy as np
    pts = [0.0, 0.5, -0.5, 1e-300, -1e-300, 1e-20, -1e-20, 1e-8, -1e-8]
    for k in [1, 2, 3, 10, 100, 1e4, 1e8]:
        pts += [1 - k * np.spacing(0.5), -1 + k * np.spacing(0.5)]
    for d in [1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7,
              1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]:
        pts += [1 - d, -1 + d]
    return np.array(pts)


def q_points():
    # Tail q (cdf ~ d**1.5, so q = 1e-300 means 1 + x ~ 1e-200), q near 1/2
    # (2p - 1 cancellation) and q near 1.
    import numpy as np
    pts = [0.5, 0.25, 0.75]
    for e in [-300, -250, -200, -150, -100, -50, -40, -33, -30, -25, -20, -17,
              -16, -15, -12, -10, -8, -6, -4, -3, -2]:
        pts.append(10.0 ** e)
    for d in [1e-16, 1e-15, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
        pts += [1 - d, 0.5 + d, 0.5 - d]
    return np.array(pts)
