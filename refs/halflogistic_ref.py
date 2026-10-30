"""Reference module for scipy.stats.halflogistic.

Standard form, no shapes. Docstring pdf: f(x) = 2 exp(-x) / (1 + exp(-x))**2,
x >= 0. cdf F(x) = tanh(x/2) = (1 - exp(-x)) / (1 + exp(-x)),
sf = 2 / (1 + exp(x)) (written directly: at 400 digits 1 - tanh(x/2) would
round to 0 for x > ~920, where the true sf is still representable in mpmath
but below the double range anyway; the direct form keeps it exact),
ppf = 2 atanh(q), isf = log((2 - q) / q) (= 2 atanh(1 - q)).
"""
import mpmath as mp

DIST = "halflogistic"

PARAMS = [()]


def pdf(x):
    e = mp.exp(-x)
    return 2 * e / (1 + e) ** 2


def logpdf(x):
    return mp.log(2) - x - 2 * mp.log(1 + mp.exp(-x))


def cdf(x):
    return mp.tanh(x / 2)


def sf(x):
    return 2 / (1 + mp.exp(x))


def ppf(q):
    return 2 * mp.atanh(q)


def isf(q):
    return mp.log((2 - q) / q)


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
    # sf/pdf underflow edges (x ~ 708..745), tanh saturation (x ~ 37),
    # small x where tanh(x/2) ~ x/2.
    return [1e-310, 1e-20, 1e-8, 0.5, 1.0, 2.0, 20.0, 37.0, 38.0, 40.0,
            700.0, 708.0, 709.0, 709.8, 710.0, 740.0, 744.0, 745.0]


def q_points():
    # both sides of the isf branch at q = 0.5
    return [0.5 - 2**-53, 0.5, 0.5 + 2**-52, 0.25, 0.75, 1 - 1e-10]
