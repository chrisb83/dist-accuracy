"""Reference module for scipy.stats.hypsecant.

Standard form, no shapes. Docstring pdf: f(x) = sech(x) / pi, x real.
cdf F(x) = (2/pi) atan(exp(x)), sf = (2/pi) atan(exp(-x)) (symmetry, exact),
ppf = log(tan(pi q / 2)), isf = -ppf(q) = ppf(1 - q).
"""
import mpmath as mp

DIST = "hypsecant"

PARAMS = [()]


def pdf(x):
    return 1 / (mp.pi * mp.cosh(x))


def logpdf(x):
    return mp.log(pdf(x))


def cdf(x):
    return 2 / mp.pi * mp.atan(mp.exp(x))


def sf(x):
    return 2 / mp.pi * mp.atan(mp.exp(-x))


def ppf(q):
    return mp.log(mp.tan(mp.pi * q / 2))


def isf(q):
    return -ppf(q)


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": lambda x: mp.log(cdf(x)),
    "sf": sf,
    "logsf": lambda x: mp.log(sf(x)),
    "ppf": ppf,
    "isf": isf,
}


def x_points():
    # cosh/exp overflow and underflow edges (|x| ~ 708..745), tails where
    # cdf/sf saturate (|x| ~ 37), small |x| around the median.
    pts = [1e-300, 1e-20, 1e-8, 1e-3, 0.5, 1.0, 2.0, 20.0, 37.0, 38.0, 40.0,
           300.0, 700.0, 707.0, 708.0, 709.0, 709.8, 710.0, 710.6, 740.0,
           744.0, 745.0]
    return pts + [-p for p in pts]


def q_points():
    # around the median (tan ~ 1, log cancels) and near 1 (tan pole)
    return [0.5 - 2**-53, 0.5, 0.5 + 2**-52, 0.5 + 1e-10, 0.5 - 1e-10,
            0.5 + 1e-5, 0.5 - 1e-5, 0.25, 0.75, 1 - 1e-5, 1 - 1e-10,
            1 - 1e-14, 1 - 2**-53, 1e-308, 1e-300]
