"""Reference module for scipy.stats.gumbel_l.

Standard form, no shapes. Docstring pdf: f(x) = exp(x - e^x), x real.
cdf F(x) = 1 - exp(-e^x), sf = exp(-e^x), ppf = log(-log(1 - q)),
isf = log(-log(q)).
"""
import mpmath as mp

DIST = "gumbel_l"

PARAMS = [()]


# For e^x > 1e6 the pdf and sf are below exp(-1e6), far outside the double
# range; returning 0 there avoids mpmath evaluating exp(-e^x) with e^x ~ 1e300
# (which needs ~1e300 digits and never finishes).
_BIG = mp.mpf(10) ** 6


def sf(x):
    t = mp.exp(x)
    if t > _BIG:
        return mp.mpf(0)
    return mp.exp(-t)


def pdf(x):
    t = mp.exp(x)
    if t > _BIG:
        return mp.mpf(0)
    return mp.exp(x - t)


def cdf(x):
    # -expm1 keeps relative accuracy for e^x below 10**-dps (x < -920).
    t = mp.exp(x)
    if t > _BIG:
        return mp.mpf(1)
    return -mp.expm1(-t)


def ppf(q):
    return mp.log(-mp.log(1 - q))


def isf(q):
    return mp.log(-mp.log(q))


REFS = {
    "pdf": pdf,
    "logpdf": lambda x: x - mp.exp(x),
    "cdf": cdf,
    "logcdf": lambda x: mp.log(cdf(x)),
    "sf": sf,
    "logsf": lambda x: -mp.exp(x),
    "ppf": ppf,
    "isf": isf,
}


def x_points():
    import numpy as np
    # x where exp(x) overflows (709.78), where pdf/sf underflow (e^x ~ 708, x ~ 6.56),
    # left tail where cdf ~ e^x underflows (x ~ -708), around 0 (mode).
    return np.array([-745.0, -708.0, -700.0, -40.0, -37.0, -20.0, -1e-8, 0.0, 1e-8,
                     0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.5, 6.55, 6.56, 6.6,
                     100.0, 700.0, 709.0, 709.7, 709.8])


def q_points():
    import numpy as np
    e = float(mp.e)
    return np.array([1 - 1 / e, 1 / e, 1 / e + 1e-10, 1 / e - 1e-10,
                     1 - 1 / e + 1e-10, 1 - 1 / e - 1e-10, 0.5, 1e-16, 1e-300,
                     1 - 1e-10, 1 - 1e-15])
