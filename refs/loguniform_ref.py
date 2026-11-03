"""Reference module for scipy.stats.loguniform (also scipy.stats.reciprocal,
same class reciprocal_gen).

Standard form, shapes 0 < a < b, support [a, b]; docstring pdf
    f(x; a, b) = 1 / (x log(b/a)),
so F(x) = log(x/a) / log(b/a) and ppf(q) = a (b/a)**q. Plain defining
formulas at 400 digits.
"""
import mpmath as mp
import numpy as np

DIST = "loguniform"

_A1 = 1e10
_B1 = float(np.nextafter(1e10, np.inf))   # b = a(1 + 1.9e-16): log(b) == log(a) in double

# example; log a = 0 exactly; very wide; |log a| large with moderate b/a;
# b/a - 1 = 1e-6 and 1e-10 with |log a| = 23 and 690; b = nextafter(a);
# extreme a (subnormal) and b; b/a - 1 = 1e-12 with log a = 0 (control).
PARAMS = [(0.01, 1.25), (1.0, 10.0), (1e-300, 1e300), (1e-300, 2e-300),
          (1e10, 1.000001e10), (1e300, 1.0000000001e300), (_A1, _B1),
          (5e-324, 1e308), (1.0, 1.000000000001)]


def L(a, b):
    return mp.log(b / a)


def pdf(x, a, b):
    return 1 / (x * L(a, b))


def cdf(x, a, b):
    return mp.log(x / a) / L(a, b)


def sf(x, a, b):
    return 1 - cdf(x, a, b)


def ppf(q, a, b):
    return a * (b / a) ** q


def isf(q, a, b):
    return ppf(1 - q, a, b)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, a, b: mp.log(pdf(x, a, b)),
    "cdf": cdf,
    "logcdf": lambda x, a, b: mp.log(cdf(x, a, b)),
    "sf": sf,
    "logsf": lambda x, a, b: mp.log(sf(x, a, b)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(a, b):
    # a few ulp steps away from each endpoint, the geometric midpoint, and
    # relative offsets 1e-12 ... 1e-3 in log scale from each end.
    pts = [a, b, float(np.sqrt(a) * np.sqrt(b))]
    lo, hi = a, b
    for _ in range(3):
        lo = float(np.nextafter(lo, np.inf))
        hi = float(np.nextafter(hi, 0))
        pts += [lo, hi]
    for r in (1e-12, 1e-8, 1e-5, 1e-3):
        la, lb = np.log(a), np.log(b)
        pts += [np.exp(la + r * (lb - la)), np.exp(lb - r * (lb - la))]
    return np.array(pts, float)


def q_points(a, b):
    return np.array([0.5, 0.25, 0.75])
