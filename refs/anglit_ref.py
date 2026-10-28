"""Reference module for scipy.stats.anglit.

Standard form, no shapes. Docstring pdf: f(x) = cos(2x), -pi/4 <= x <= pi/4.
cdf F(x) = sin(x + pi/4)**2 (integral of cos(2t) from -pi/4 is
(sin(2x) + 1)/2 = sin(x + pi/4)**2), sf = 1 - F, ppf = arcsin(sqrt q) - pi/4.
"""
import mpmath as mp

DIST = "anglit"

PARAMS = [()]


def pdf(x):
    return mp.cos(2 * x)


def cdf(x):
    return (1 + mp.sin(2 * x)) / 2


def sf(x):
    return 1 - cdf(x)


def ppf(q):
    return mp.asin(mp.sqrt(q)) - mp.pi / 4


def isf(q):
    return ppf(1 - q)


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
    # Near both ends of the support (a few ulp up to 1e-3 inside), small |x|
    # (pdf near 1), and points on both sides of 0.
    import numpy as np
    a = -np.pi / 4
    b = np.pi / 4
    pts = [0.0, 1e-300, -1e-300, 1e-20, -1e-20, 1e-9, -1e-9, 1e-5, -1e-5]
    for k in [1, 2, 3, 10, 100, 1e4, 1e8]:
        pts.append(a + k * np.spacing(a))
        pts.append(b - k * np.spacing(b))
    for d in [1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3]:
        pts += [a + d, b - d]
    return np.array(pts)


def q_points():
    # Both sides of q = 0.5 (ppf cancels to 0 there).
    import numpy as np
    d = [1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
    return np.array([0.5] + [0.5 + v for v in d] + [0.5 - v for v in d])
