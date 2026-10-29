"""Reference module for scipy.stats.genexpon.

Standard form, SciPy's parametrization (docstring pdf):
    f(x; a, b, c) = (a + b (1 - exp(-c x))) exp(-a x - b x + b/c (1 - exp(-c x)))
for x >= 0, a, b, c > 0.  With E(x) = -(a+b) x + b/c (1 - exp(-c x)):
    sf = exp(E), cdf = 1 - exp(E).
"""
import sys, os
import mpmath as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".claude",
                                "skills", "dist-accuracy-review", "scripts"))
from accuracy import invert  # noqa: E402

DIST = "genexpon"

# example; moderate; a << b (cancellation in E for small x, lambertw near the
# branch point); tiny b; large c; small c; large a, b.
PARAMS = [
    (9.13259764654189, 16.23195660059063, 3.2819552690843983),
    (1.0, 1.0, 1.0),
    (1e-3, 1.0, 1.0),
    (1e-6, 10.0, 2.0),
    (1.0, 1e-6, 1.0),
    (0.5, 2.0, 1e3),
    (1.0, 2.0, 1e-3),
    (1e3, 1e4, 0.1),
]


def E(x, a, b, c):
    return -(a + b) * x + b / c * (1 - mp.exp(-c * x))


def pdf(x, a, b, c):
    return (a + b * (1 - mp.exp(-c * x))) * mp.exp(E(x, a, b, c))


def cdf(x, a, b, c):
    return 1 - mp.exp(E(x, a, b, c))


def sf(x, a, b, c):
    return mp.exp(E(x, a, b, c))


def ppf(q, a, b, c):
    return invert(lambda x: cdf(x, a, b, c), q)


def isf(q, a, b, c):
    return invert(lambda x: cdf(x, a, b, c), 1 - q)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, a, b, c: mp.log(a + b * (1 - mp.exp(-c * x))) + E(x, a, b, c),
    "cdf": cdf,
    "logcdf": lambda x, a, b, c: mp.log(cdf(x, a, b, c)),
    "sf": sf,
    "logsf": E,
    "ppf": ppf,
    "isf": isf,
}


def x_points(a, b, c):
    # Both sides of c*x ~ 1 (expm1 regime of the code), where the two terms of
    # E cancel (small x), and where sf underflows ((a+b) x ~ 745).
    import numpy as np
    s = a + b
    pts = [1e-3 / c, 0.1 / c, 1.0 / c, 10.0 / c, 1e-8, 1e-4,
           700.0 / s, 745.0 / s, 760.0 / s]
    return np.array(pts)
