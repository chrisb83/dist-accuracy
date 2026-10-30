"""Reference module for scipy.stats.moyal.

Standard form, no shapes. Docstring pdf:
    f(x) = exp(-(x + exp(-x))/2) / sqrt(2 pi),  x real.
With z = exp(-x/2)/sqrt(2): cdf F(x) = erfc(z), sf = 1 - F = erf(z).
ppf(q) = -log(2 erfcinv(q)^2), erfcinv(q) = erfinv(1 - q) (1 - q exact at
400 digits); isf(q) = ppf(1 - q) = -log(2 erfinv(q)^2).
"""
import mpmath as mp

DIST = "moyal"

PARAMS = [()]


# For x < -40, pdf and cdf are below exp(-e^40/2) ~ exp(-1.2e17): never a
# normal double (never measured), and mpmath's erfc/exp of exp(-x) overflows
# or hangs for x ~ -1e3 ... -1e300. Return the exact limits there.
XMIN = -40


def pdf(x):
    if x < XMIN:
        return mp.mpf(0)
    return mp.exp(-(x + mp.exp(-x)) / 2) / mp.sqrt(2 * mp.pi)


def _z(x):
    return mp.exp(-x / 2) / mp.sqrt(2)


def cdf(x):
    if x < XMIN:
        return mp.mpf(0)
    return mp.erfc(_z(x))


def sf(x):
    return 1 - cdf(x)


def ppf(q):
    return -mp.log(2 * mp.erfinv(1 - q) ** 2)


def isf(q):
    return -mp.log(2 * mp.erfinv(q) ** 2)


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
    # pdf/cdf underflow for x < ~-7.3 (e^-x/2 > ~745); probe the left edge.
    # pdf ~ exp(-x/2)/sqrt(2 pi) and sf ~ sqrt(2/pi) exp(-x/2) leave the
    # normal range near x ~ 1415-1418; z = exp(-x/2)/sqrt(2) becomes
    # subnormal near x ~ 1417.5 (sf window, pattern 4).
    import numpy as np
    pts = [-7.4, -7.3, -7.2, -7.0, -6.5, -6.0, -5.0, -4.0, -3.0, -2.0, -1.5,
           -1.0, -0.5, -0.1, -1e-3, -1e-8, 0.0, 1e-8, 1e-3, 0.1, 0.5, 1.0,
           2.0, 5.0, 10.0, 20.0, 40.0, 75.0, 100.0, 300.0, 700.0, 1000.0,
           1400.0, 1410.0, 1414.0, 1415.0, 1415.5, 1416.0, 1416.3, 1416.6,
           1417.0, 1417.3, 1417.5, 1417.7, 1418.0, 1418.3, 1418.6]
    return np.array(pts)


def q_points():
    # ppf zero crossing at q = erfc(1/sqrt 2) = 0.3173105078629141; dense
    # points around it map where -log(y), y ~ 1, exceeds the conditioning.
    import numpy as np
    q0 = 0.3173105078629141
    pts = [2.3e-308, 1e-307, 1e-300, 1e-200, 1e-100, 1e-50, 1e-20, 1e-10,
           1e-5, 0.01, 0.1, 0.15, 0.2, 0.25, 0.28, 0.3, 0.31, 0.315,
           0.33, 0.35, 0.4, 0.45, 0.6, 0.7, q0 * (1 - 1e-6), q0 * (1 - 1e-10),
           np.nextafter(q0, 0), q0, np.nextafter(q0, 1), q0 * (1 + 1e-10),
           q0 * (1 + 1e-6), 0.32, 0.5, 0.9, 0.99, 1 - 1e-6, 1 - 1e-10,
           1 - 1e-14, 1 - 1e-15, 1 - 2.2e-16]
    return np.array(pts)
