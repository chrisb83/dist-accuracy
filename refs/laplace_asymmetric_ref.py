"""Reference module for scipy.stats.laplace_asymmetric.

Standard form, docstring pdf:
  f(x; k) = exp(-x k) / (k + 1/k)  for x >= 0,
          = exp(x / k) / (k + 1/k) for x < 0.
cdf(0) = k**2 / (1 + k**2).
"""
import mpmath as mp
import numpy as np

DIST = "laplace_asymmetric"

# example 2; laplace case 1; small/large kappa on both sides of 1 (cdf(0)
# far from 1/2 makes the 1 - ... branches cancel); extreme kappa.
PARAMS = [(2.0,), (1.0,), (0.5,), (1e-3,), (1e3,), (1e-8,), (1e8,), (1e-150,), (1e150,)]


def pdf(x, k):
    c = 1 / (k + 1 / k)
    return c * mp.exp(-x * k) if x >= 0 else c * mp.exp(x / k)


def logpdf(x, k):
    return (-x * k if x >= 0 else x / k) - mp.log(k + 1 / k)


def cdf(x, k):
    if x >= 0:
        return 1 - mp.exp(-x * k) / (k * (k + 1 / k))
    return mp.exp(x / k) * k / (k + 1 / k)


def sf(x, k):
    if x >= 0:
        return mp.exp(-x * k) / (k * (k + 1 / k))
    return 1 - mp.exp(x / k) * k / (k + 1 / k)


def ppf(q, k):
    q0 = k * k / (1 + k * k)
    if q >= q0:
        return -mp.log((1 - q) * (1 + k * k)) / k
    return k * mp.log(q / q0)


def isf(q, k):
    return ppf(1 - q, k)


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": lambda x, k: mp.log(cdf(x, k)),
    "sf": sf,
    "logsf": lambda x, k: mp.log(sf(x, k)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(k):
    # x = 0 threshold, small |x| on both sides (where cdf / sf stay near
    # their values at 0), and |x| scaled by the two rates.
    small = [1e-300, 1e-200, 1e-100, 1e-30, 1e-16, 1e-10, 1e-5, 1e-2, 0.5]
    pts = [0.0]
    for s in small:
        pts += [s, -s]
    for r in (k, 1 / k):
        for m in (1e-10, 1e-5, 1e-2, 0.3, 1, 3, 30, 300, 700):
            pts += [m / r, -m / r]
    return np.array(pts, float)


def q_points(k):
    k = float(k)
    with np.errstate(all="ignore"):
        q0 = k * k / (1 + k * k)
        p0 = 1 / (1 + k * k)
    pts = []
    for q in (q0, p0):
        for r in (1e-12, 1e-8, 1e-4, 1e-2, 0.3):
            pts += [q * (1 + r), q * (1 - r), q + r * (1 - q), q - r * q]
    pts = np.array(pts, float)
    return pts[(pts > 0) & (pts < 1)]
