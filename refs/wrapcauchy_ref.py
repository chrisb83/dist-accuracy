"""Reference module for scipy.stats.wrapcauchy.

Standard form, shape 0 < c < 1, support [0, 2*pi]:
    f(x, c) = (1 - c**2) / (2 pi (1 + c**2 - 2 c cos x))
    F(x, c) = arctan(r tan(x/2)) / pi            for 0 <= x < pi
            = 1 - arctan(r tan((2 pi - x)/2)) / pi  for pi <= x <= 2 pi
    with r = (1 + c)/(1 - c);  F(pi) = 1/2.
    ppf(q)  = 2 arctan(tan(pi q)/r)               for q < 1/2
            = 2 pi - 2 arctan(tan(pi (1 - q))/r)  for q >= 1/2
2 pi here is the exact value (mp.pi), not the double 2*np.pi.
"""
import mpmath as mp

DIST = "wrapcauchy"

# Example value, small c (r ~ 1), c = 1/2, and c approaching 1 (sharp peak at
# x = 0 / 2 pi, cancellation in 1 + c^2 - 2 c cos x and in 1 - c^2;
# 0.9999999870854271 maximizes the rounding error of c*c in 1 - c*c, found
# by scratch/wrapcauchy/scan_num.py).
PARAMS = [(0.031071279018614728,), (1e-8,), (0.5,), (0.9,), (0.999,),
          (1 - 1e-6,), (0.9999999870854271,), (1 - 1e-10,), (1 - 2**-52,)]


def pdf(x, c):
    return (1 - c * c) / (2 * mp.pi * (1 + c * c - 2 * c * mp.cos(x)))


def cdf(x, c):
    r = (1 + c) / (1 - c)
    if x < mp.pi:
        return mp.atan(r * mp.tan(x / 2)) / mp.pi
    return 1 - mp.atan(r * mp.tan((2 * mp.pi - x) / 2)) / mp.pi


def sf(x, c):
    return 1 - cdf(x, c)


def ppf(q, c):
    v = (1 - c) / (1 + c)
    if q == mp.mpf(1) / 2:
        return mp.pi
    if q < mp.mpf(1) / 2:
        return 2 * mp.atan(v * mp.tan(mp.pi * q))
    return 2 * mp.pi - 2 * mp.atan(v * mp.tan(mp.pi * (1 - q)))


def isf(q, c):
    return ppf(1 - q, c)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, c: mp.log(pdf(x, c)),
    "cdf": cdf,
    "logcdf": lambda x, c: mp.log(cdf(x, c)),
    "sf": sf,
    "logsf": lambda x, c: mp.log(sf(x, c)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(c):
    # Peak region near 0 and 2 pi (denominator cancellation for c -> 1),
    # the cdf branch switch at x = pi (both sides), and x ~ (1 - c) scale.
    import numpy as np
    pi, tpi = np.pi, 2 * np.pi
    pts = [pi, np.nextafter(pi, 0), np.nextafter(pi, 4), pi - 1e-8, pi + 1e-8,
           1.0, 2.0, 4.0, 5.0, tpi, np.nextafter(tpi, 0)]
    for d in [1e-300, 1e-200, 1e-100, 1e-20, 1e-16, 1e-14, 1e-12, 1e-10,
              1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1]:
        pts += [d, tpi - d]
    w = 1 - c
    for f in [1e-3, 0.1, 1.0, 10.0]:
        if w * f < 1:
            pts += [w * f, tpi - w * f]
    pts = np.array(pts)
    return pts[(pts >= 0) & (pts <= tpi)]


def q_points(c):
    # Branch switch at q = 1/2 (both sides), tails.
    import numpy as np
    pts = [0.5, np.nextafter(0.5, 0), np.nextafter(0.5, 1), 0.5 - 1e-8,
           0.5 + 1e-8, 0.25, 0.75, 0.1, 0.9]
    for d in [1e-300, 1e-200, 1e-100, 1e-20, 1e-16, 1e-14, 1e-12, 1e-10,
              1e-8, 1e-6, 1e-4, 1e-2]:
        pts += [d, 1 - d]
    return np.array(pts)
