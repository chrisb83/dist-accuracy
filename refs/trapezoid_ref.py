"""Reference module for scipy.stats.trapezoid.

Standard form on [0, 1]; pdf rises linearly on [0, c], is flat (height
u = 2/(d - c + 1)) on [c, d], falls linearly on [d, 1].
All functions are closed forms evaluated at the harness precision.
"""
import mpmath as mp

DIST = "trapezoid"

# example, tiny c (left branch underflow), c = d (triang), c = 0 / d = 1
# (uniform edge), d close to 1, c close to d.
PARAMS = [(0.2, 0.8), (1e-200, 0.5), (1e-20, 0.9), (0.5, 0.5),
          (0.0, 1.0), (0.1, 1 - 1e-12), (0.3, 0.3 + 1e-9)]


def pdf(x, c, d):
    u = 2 / (d - c + 1)
    if x < c:
        return u * x / c
    if x <= d:
        return u
    return u * (1 - x) / (1 - d)


def cdf(x, c, d):
    h = d - c + 1
    if x < c:
        return x ** 2 / (c * h)
    if x <= d:
        return (c + 2 * (x - c)) / h
    return 1 - (1 - x) ** 2 / (h * (1 - d))


def sf(x, c, d):
    return 1 - cdf(x, c, d)


def ppf(q, c, d):
    h = d - c + 1
    qc = c / h
    qd = (2 * d - c) / h
    if q < qc:
        return mp.sqrt(q * c * h)
    if q <= qd:
        return (q * h + c) / 2
    return 1 - mp.sqrt((1 - q) * h * (1 - d))


def isf(q, c, d):
    return ppf(1 - q, c, d)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, c, d: mp.log(pdf(x, c, d)),
    "cdf": cdf,
    "logcdf": lambda x, c, d: mp.log(cdf(x, c, d)),
    "sf": sf,
    "logsf": lambda x, c, d: mp.log(sf(x, c, d)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(c, d):
    import numpy as np
    pts = [c, d, c * (1 - 1e-9), c * (1 + 1e-9), d * (1 - 1e-9), d * (1 + 1e-9),
           0.5 * c, 1e-3 * c, 1e-100 * c, 1e-160, 1e-155, 1e-150, 1e-200 * c,
           1 - 1e-3, 1 - 1e-8, 1 - 1e-15, (d + 1) / 2]
    pts = [p for p in pts if np.isfinite(p) and 0 < p <= 1]
    return np.array(pts)


def q_points(c, d):
    import numpy as np
    h = d - c + 1
    qc, qd = c / h, (2 * d - c) / h
    pts = [qc, qd, qc * (1 - 1e-9), qc * (1 + 1e-9), qd * (1 - 1e-9),
           qd * (1 + 1e-9), 1e-3 * qc, 1e-100 * qc, 1e-150 * qc, 1e-300,
           1 - 1e-12, 1 - 1e-15]
    return np.array([p for p in pts if 0 < p < 1])
