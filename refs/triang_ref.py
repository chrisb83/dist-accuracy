"""Reference module for scipy.stats.triang.

Standard form, shape 0 <= c <= 1 (the mode), support [0, 1]:
    f(x, c) = 2 x / c                 for 0 <= x < c
            = 2 (1 - x) / (1 - c)     for c <= x <= 1
    F(x, c) = x**2 / c                for x < c
            = 1 - (1 - x)**2 / (1 - c) for x >= c
    ppf(q)  = sqrt(c q)               for q < c
            = 1 - sqrt((1 - c)(1 - q)) otherwise
c = 0 and c = 1 are the limiting cases (only one branch exists).

Note: the harness self-check "integral of pdf != cdf" FAILs for the example
c = 0.15785 because mp.quad integrates across the kink of the pdf at x = c
without splitting there. Integrating on [x1, c, x2] matches the cdf difference
to 30 digits (checked in scratch/triang/), so this FAIL is a quadrature
artifact, not a reference error.
"""
import mpmath as mp

DIST = "triang"

# Example value, c = 1/2, tiny c (x*x and c*q underflow; cancellation in the
# ppf right branch), c = 0 and c = 1 (edge branches), c close to 1
# (cancellation in the cdf right branch).
PARAMS = [(0.15785029824528218,), (0.5,), (0.0,), (1e-10,), (1e-200,),
          (0.999,), (1 - 1e-6,), (1 - 1e-8,), (1 - 1e-10,), (1.0,)]


def pdf(x, c):
    if x < c or c == 1:
        return 2 * x / c
    return 2 * (1 - x) / (1 - c)


def cdf(x, c):
    if x < c or c == 1:
        return x * x / c
    return 1 - (1 - x) ** 2 / (1 - c)


def sf(x, c):
    return 1 - cdf(x, c)


def ppf(q, c):
    if q < c:
        return mp.sqrt(c * q)
    return 1 - mp.sqrt((1 - c) * (1 - q))


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
    # Both sides of the mode c, the ends of the support, points just above c
    # (x >= c branch where c is close to 1), x where x*x underflows
    # (x < 1.5e-154) while x*x/c is still normal for tiny c.
    import numpy as np
    pts = [0.5, 0.25, 0.75, 1.0, 1e-300, 1e-200, 1e-160, 1.4e-154, 1e-154,
           1e-150, 1e-100, 1e-20]
    for d in [1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]:
        pts += [d, 1 - d]
    if c > 0:
        for f in [0.5, 0.9, 1 - 1e-6]:
            pts.append(c * f)
    if c < 1:
        for f in [1e-6, 1e-3, 0.1, 0.5, 0.9]:
            pts.append(c + (1 - c) * f)
    pts.append(c)
    return np.array(pts)


def q_points(c):
    # Both sides of the branch switch q = c; c*q underflow for tiny c;
    # q just above c (right branch, cancellation for small c); q near 1.
    import numpy as np
    pts = [0.5, 1e-300, 1e-200, 1e-150, 1e-100, 1e-20, 1e-10]
    for d in [1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
        pts += [1 - d, d]
    if c > 0:
        pts += [c * 0.5, c * (1 - 1e-6)]
    if c < 1:
        pts += [c, c * 1.5 + 1e-300, c + (1 - c) * 1e-6, c + (1 - c) * 0.1]
    return np.array(pts)
