"""Reference module for scipy.stats.powerlaw.

Standard form, SciPy's parametrization from the docstring:
f(x; a) = a * x**(a - 1), 0 <= x <= 1, a > 0.
cdf = x**a, sf = 1 - cdf, ppf = q**(1/a), isf = (1 - q)**(1/a).
"""
import mpmath as mp

DIST = "powerlaw"

# Inventory examples; a = 1 (exact exponents); tiny a (inexact a - 1.0,
# 1/a huge in ppf); a = 3 (inexact 1/a); large a (x**(a-1) subnormal while
# a*x**(a-1) is normal; sf = -powm1 with large a).
PARAMS = [(1.659113328990585,), (0.6591133289905851,), (1.0,), (1e-3,),
          (0.1,), (3.0,), (1e6,), (1e-12,)]


def pdf(x, a):
    return a * x ** (a - 1)


def cdf(x, a):
    return x ** a


def sf(x, a):
    return 1 - cdf(x, a)


def ppf(q, a):
    return q ** (1 / a)


def isf(q, a):
    return (1 - q) ** (1 / a)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, a: mp.log(a) + (a - 1) * mp.log(x),
    "cdf": cdf,
    "logcdf": lambda x, a: a * mp.log(x),
    "sf": sf,
    "logsf": lambda x, a: mp.log(sf(x, a)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(a):
    # Near the right end (sf -> a*(1-x)), the zero of logpdf (pdf = 1, at
    # x = a**(-1/(a-1))), and for large a the region where x**(a-1) is
    # subnormal (log x in (-745, -708)/(a-1)) while a*x**(a-1) is normal.
    import numpy as np
    near1 = 1 - np.array([1e-16, 1.1e-16, 3e-16, 1e-15, 1e-12, 1e-9, 1e-6, 1e-3])
    pts = list(near1) + [0.5, 0.9, 0.99]
    if a != 1:
        z = np.exp(-np.log(a) / (a - 1))
        if 0 < z < 1:
            pts += [z, z * (1 - 1e-6), z * (1 + 1e-6)]
    if a > 1:
        for L in (-700.0, -709.0, -715.0, -725.0, -740.0):
            pts.append(np.exp(L / (a - 1)))
    return np.array(pts)


def q_points(a):
    import numpy as np
    return np.array([0.5, 0.9, 1 - 1e-10, 1 - 1e-14, 1e-200, 1e-300])
