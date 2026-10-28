"""Reference module for scipy.stats.cosine.

Standard form, no shapes. Docstring pdf: f(x) = (1 + cos x)/(2 pi), -pi <= x <= pi.
cdf F(x) = (pi + x + sin x)/(2 pi), sf = 1 - F; ppf/isf by numerical inversion
of the cdf (no closed form).
"""
import mpmath as mp
from accuracy import invert

DIST = "cosine"

PARAMS = [()]


def pdf(x):
    return (1 + mp.cos(x)) / (2 * mp.pi)


def cdf(x):
    return (mp.pi + x + mp.sin(x)) / (2 * mp.pi)


def sf(x):
    return 1 - cdf(x)


def ppf(q):
    # The median is exactly 0 (the pdf is even); numerical inversion would
    # return a residual of ~1e-60 there instead.
    if q == mp.mpf(1) / 2:
        return mp.zero
    return invert(cdf, q, positive=False)


def isf(q):
    return -ppf(q)


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
    # Near both ends of the support (pdf ~ (pi - |x|)**2/(4 pi), cdf ~ (x + pi)**3/(12 pi)),
    # the double nearest pi, small |x| and both sides of 0.
    import numpy as np
    a, b = -np.pi, np.pi
    pts = [0.0, 1e-300, -1e-300, 1e-20, -1e-20, 1e-9, -1e-9, 1e-5, -1e-5, 1.0, -1.0,
           2.0, -2.0, 3.0, -3.0, a, b]
    for k in [1, 2, 3, 10, 100, 1e4, 1e8]:
        pts.append(a + k * np.spacing(np.pi))
        pts.append(b - k * np.spacing(np.pi))
    for d in [1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5]:
        pts += [a + d, b - d]
    return np.array(pts)


def q_points():
    import numpy as np
    d = [1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1]
    return np.array([0.5] + [0.5 + v for v in d] + [0.5 - v for v in d]
                    + [1e-50, 1e-100, 1e-150, 1e-200, 1e-250, 1e-300, 1e-305])
