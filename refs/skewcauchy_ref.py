"""Reference module for scipy.stats.skewcauchy.

Standard form, shape -1 < a < 1. Docstring pdf:
    f(x) = 1 / (pi * (x**2 / (a*sign(x) + 1)**2 + 1)).
cdf: x <= 0: (1-a)/2 + (1-a)/pi * atan(x/(1-a));  x > 0: (1-a)/2 + (1+a)/pi * atan(x/(1+a)).
ppf: closed-form inverse of the two branches; sf = 1 - cdf, isf = ppf(1 - q).
"""
import mpmath as mp

DIST = "skewcauchy"

# Example 0.5, symmetric 0, negative, |a| near 1 (1 -+ a small), and a as close
# to +-1 as practical.
PARAMS = [(0.5,), (0.0,), (-0.5,), (0.9,), (-0.9,), (0.99999999,), (-0.99999999,),
          (0.999999999999999,), (-0.999999999999999,)]


def pdf(x, a):
    s = mp.sign(x)
    return 1 / (mp.pi * (x * x / (a * s + 1) ** 2 + 1))


def cdf(x, a):
    if x <= 0:
        return (1 - a) / 2 + (1 - a) / mp.pi * mp.atan(x / (1 - a))
    return (1 - a) / 2 + (1 + a) / mp.pi * mp.atan(x / (1 + a))


def sf(x, a):
    return 1 - cdf(x, a)


def ppf(q, a):
    q0 = (1 - a) / 2
    if q < q0:
        return mp.tan(mp.pi / (1 - a) * (q - q0)) * (1 - a)
    return mp.tan(mp.pi / (1 + a) * (q - q0)) * (1 + a)


def isf(q, a):
    return ppf(1 - q, a)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, a: mp.log(pdf(x, a)),
    "cdf": cdf,
    "logcdf": lambda x, a: mp.log(cdf(x, a)),
    "sf": sf,
    "logsf": lambda x, a: mp.log(sf(x, a)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(a):
    # x = 0 branch (both sides), the arctan saturation scale (1 -+ a)/eps on
    # both sides, x**2 overflow (1.34e154), both signs.
    import numpy as np
    pts = [0.0, 1e-300, 1e-100, 1e-20, 1e-8, 1e-3, 0.5, 1.0, 2.0, 1e3, 1e8,
           1e12, 1e15, 1e16, 1e17, 1e20, 1e50, 1e100, 1e150, 1.3e154, 1.4e154]
    for s in (1 - a, 1 + a):
        pts += [s, s * 1e8, s * 1e14, s * 1e16, s * 1e18]
    pts = pts + [-p for p in pts if p != 0]
    return np.array(pts)


def q_points(a):
    # Both sides of q0 = cdf(0) = (1-a)/2 (branch in _ppf), tiny q, q near 1.
    import numpy as np
    q0 = (1 - a) / 2
    pts = [q0]
    for d in [1e-16, 1e-12, 1e-8, 1e-4, 1e-2]:
        pts += [q0 * (1 + d), q0 * (1 - d)]
    pts += [1e-300, 1e-100, 1e-30, 1e-20, 1e-17, 1e-16, 1e-15, 1e-12, 1e-8,
            1 - 1e-4, 1 - 1e-8, 1 - 1e-12, 1 - 1e-15, 1 - 2.2e-16]
    return np.array([p for p in pts if 0 < p < 1])
