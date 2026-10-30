"""Reference module for scipy.stats.levy_l.

Standard form, no shapes. Docstring pdf (x < 0, ax = |x|):
    f(x) = exp(-1/(2 ax)) / (ax sqrt(2 pi ax)).
levy_l(x) = levy(-x), so cdf F(x) = erf(1/sqrt(2 ax)), sf = erfc(1/sqrt(2 ax));
ppf(q) = -1/(2 erfinv(q)^2); isf(q) = ppf(1 - q) = -1/(2 erfinv(1 - q)^2)
(1 - q exact at 400 digits).
"""
import mpmath as mp

DIST = "levy_l"

PARAMS = [()]


def pdf(x):
    ax = -x
    return mp.exp(-1 / (2 * ax)) / (ax * mp.sqrt(2 * mp.pi * ax))


def cdf(x):
    return mp.erf(1 / mp.sqrt(-2 * x))


def sf(x):
    return 1 - cdf(x)


def ppf(q):
    return -1 / (2 * mp.erfinv(q) ** 2)


def isf(q):
    return -1 / (2 * mp.erfinv(1 - q) ** 2)


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
    # pdf: exp(-1/(2ax)) subnormal/underflows for ax ~ 6.7e-4 ... 7.06e-4.
    # sf: erfc(z) underflows near ax ~ 7.1e-4.
    # cdf: 2*ndtr(1/sqrt(ax)) - 1 cancels for large ax; exactly 0 once
    # 1/sqrt(ax) < ~1e-16 (ax > ~1e32). 2*pi*ax overflows for ax > 2.86e307.
    import numpy as np
    pts = [6.0e-4, 6.5e-4, 6.7e-4, 6.8e-4, 7.0e-4, 7.06e-4, 7.1e-4, 7.5e-4,
           1e-3, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 1e3, 1e4, 1e6, 1e8,
           1e10, 1e14, 1e16, 1e20, 1e25, 1e30, 1e31, 1e32, 3e32, 1e33, 1e35,
           1e50, 1e100, 1e200, 1e300, 1e306, 2.8e307, 3e307, 1e308, 1.7e308]
    return -np.array(pts)


def q_points():
    # ppf: (q + 1)/2 rounds q away for q < 1.1e-16; isf: p/2 subnormal
    # for p < 4.45e-308; isf near p = 1: ndtri near 0.5.
    import numpy as np
    pts = [1e-300, 1e-200, 1e-160, 1e-100, 1e-50, 1e-20, 1e-17, 1e-16,
           2e-16, 1e-15, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 0.01,
           0.1, 0.3, 0.5, 0.9, 2.3e-308, 3e-308, 4.4e-308, 4.5e-308, 1e-307,
           1 - 1e-6, 1 - 1e-10, 1 - 1e-14, 1 - 2.2e-16]
    return np.array(pts)
