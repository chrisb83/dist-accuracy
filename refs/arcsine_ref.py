"""Reference module for scipy.stats.arcsine.

Standard form, no shapes. Docstring pdf: f(x) = 1/(pi*sqrt(x(1-x))), 0 < x < 1.
cdf F(x) = 2/pi * asin(sqrt(x)), sf = 1 - F, ppf = sin(pi q/2)**2,
isf = ppf(1 - q).
"""
import mpmath as mp

DIST = "arcsine"

PARAMS = [()]


def pdf(x):
    return 1 / (mp.pi * mp.sqrt(x * (1 - x)))


def cdf(x):
    return 2 / mp.pi * mp.asin(mp.sqrt(x))


def sf(x):
    return 1 - cdf(x)


def ppf(q):
    return mp.sin(mp.pi * q / 2) ** 2


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
    # Both ends of the support (a few ulp up to 1e-3 inside), x = 0.5
    # (where 1 - x stops being exact), subnormal/tiny x.
    import numpy as np
    pts = [0.5, 0.5 - 1e-16, 0.5 + 1e-16, 0.25, 0.75, 1e-300, 1e-308, 5e-324,
           1e-310, 1e-200, 1e-100, 1e-20]
    for k in [1, 2, 3, 10, 100, 1e4, 1e8]:
        pts.append(1 - k * np.spacing(0.5) / 2 * 2)
        pts.append(k * 5e-324)
    for d in [1e-15, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3]:
        pts += [d, 1 - d]
    return np.array(pts)


def q_points():
    # Underflow of sin(pi q/2)**2 (normal result needs q > ~9.5e-155),
    # q near 1 and 0.5.
    import numpy as np
    pts = [0.5, 1e-150, 1e-154, 9.5e-155, 1e-155, 1e-160, 1e-100, 1e-50]
    for d in [1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
        pts += [1 - d, 0.5 + d, 0.5 - d]
    return np.array(pts)
