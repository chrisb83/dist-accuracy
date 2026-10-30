"""Reference module for scipy.stats.johnsonsb.

Standard form, docstring pdf
    f(x; a, b) = b / (x (1 - x)) * phi(a + b log(x / (1 - x))),  0 < x < 1, b > 0.
cdf = Phi(z), sf = Phi(-z), z = a + b logit(x);
ppf/isf = expit((Phi^-1(q) - a)/b) with Phi^-1 by numerical inversion.
"""
import mpmath as mp
from accuracy import invert

DIST = "johnsonsb"

# Example parameters; symmetric (0, 1); large |a| (z large in the bulk: pdf
# exp(-z**2/2) rounding, tail underflow inside the support); small b (heavy
# logit tails reach x = 1e-300); large b (narrow mass, large z for moderate
# x); a = -b (mass centred near x = expit(1)).
PARAMS = [(4.317267509914106, 3.1837781130785063), (0.0, 1.0), (30.0, 0.5),
          (-30.0, 0.5), (0.5, 1e-3), (2.0, 100.0), (-50.0, 50.0)]


def _z(x, a, b):
    return a + b * mp.log(x / (1 - x))


def pdf(x, a, b):
    z = _z(x, a, b)
    return b / (x * (1 - x)) * mp.npdf(z)


def cdf(x, a, b):
    return mp.ncdf(_z(x, a, b))


def sf(x, a, b):
    return mp.ncdf(-_z(x, a, b))


def _expit(t):
    return 1 / (1 + mp.exp(-t))


def ppf(q, a, b):
    # x = expit((Phi^-1(q) - a) / b); Phi^-1 by inversion on the real line
    z = invert(mp.ncdf, q, positive=False)
    return _expit((z - a) / b)


def isf(q, a, b):
    z = invert(mp.ncdf, 1 - q, positive=False)
    return _expit((z - a) / b)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, a, b: mp.log(pdf(x, a, b)),
    "cdf": cdf,
    "logcdf": lambda x, a, b: mp.log(cdf(x, a, b)),
    "sf": sf,
    "logsf": lambda x, a, b: mp.log(sf(x, a, b)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(a, b):
    # Mode region expit(-a/b), the point 0.5 (logit = 0), points close to 1
    # (1 - x exact, logit large), and x where |z| = 37..38.5 (pdf/cdf tails).
    import numpy as np
    pts = [0.5, 0.5 + 1e-9, 0.5 - 1e-9, 0.3, 0.65, 1 - 1e-6, 1 - 1e-10,
           1 - 1e-14, 1 - 2**-52, 1 - 2**-53, 1e-300, 1e-100, 1e-10]
    for z in [-38.4, -37.5, -30.0, -8.0, -1.0, 0.0, 1.0, 8.0, 30.0, 37.5, 38.4]:
        t = (z - a) / b
        if abs(t) < 700:
            pts.append(1.0 / (1.0 + np.exp(-t)))
    pts = np.array(pts)
    return pts[(pts > 0) & (pts < 1)]


def q_points(a, b):
    import numpy as np
    return np.array([1e-300, 1e-100, 1e-20, 1e-16, 1e-12, 1e-8, 1e-4, 1e-2,
                     0.3, 0.5, 0.9, 1 - 1e-6, 1 - 1e-10, 1 - 1e-14,
                     1 - 2**-52, 1 - 2**-53])
