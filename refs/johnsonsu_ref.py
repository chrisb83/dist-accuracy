"""Reference module for scipy.stats.johnsonsu.

Standard form, docstring pdf
    f(x; a, b) = b / sqrt(x**2 + 1) * phi(a + b * log(x + sqrt(x**2 + 1))),  b > 0.
cdf = Phi(z), sf = Phi(-z), z = a + b asinh(x);
ppf/isf = sinh((Phi^-1(q) - a)/b) with Phi^-1 by numerical inversion.
"""
import mpmath as mp
from accuracy import invert

DIST = "johnsonsu"

# Example parameters; symmetric (0, 1); large |a| (z large in the bulk);
# small b (heavy tails: |x| up to 1e300 has moderate z, so x*x overflow in
# the pdf matters); large b (narrow mass, phi(z) underflow with large
# prefactor b, b = 1e100 makes the band where phi(z) underflows but the pdf
# is normal wide); a = -b relation.
PARAMS = [(2.554395574161155, 2.2482281679651965), (0.0, 1.0), (30.0, 0.5),
          (-30.0, 0.5), (0.5, 1e-3), (0.0, 0.01), (0.0, 1e6), (-50.0, 50.0),
          (0.0, 1e100)]


def _z(x, a, b):
    # mp.asinh, not log(x + sqrt(x**2 + 1)): the latter cancels completely for
    # x < -1e200 even at 400 digits.
    return a + b * mp.asinh(x)


def pdf(x, a, b):
    return b / mp.sqrt(x * x + 1) * mp.npdf(_z(x, a, b))


def cdf(x, a, b):
    return mp.ncdf(_z(x, a, b))


def sf(x, a, b):
    return mp.ncdf(-_z(x, a, b))


def _ndtri(q):
    # erfinv in the centre (exact 0 at q = 1/2, where invert leaves ~1e-60);
    # numerical inversion in the tails, where erfinv(2q - 1) loses digits.
    if abs(2 * q - 1) < mp.mpf("0.5"):
        return mp.sqrt(2) * mp.erfinv(2 * q - 1)
    return invert(mp.ncdf, q, positive=False)


def ppf(q, a, b):
    return mp.sinh((_ndtri(q) - a) / b)


def isf(q, a, b):
    return mp.sinh((_ndtri(1 - q) - a) / b)


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
    # x where |z| = 0, 8, 30, 37.5, 38.4, 38.8 (normal tails, phi underflow),
    # both sides of the x*x overflow at |x| = 1.34e154, and large |x|.
    import numpy as np
    pts = [0.0, 1e-300, -1e-300, 1e-10, -1e-10, 0.5, -0.5, 1e150, 1.3e154,
           1.35e154, 1e160, 1e200, 1e300, -1e200, -1e300]
    for z in [-43.0, -41.0, -39.5, -38.8, -38.4, -37.5, -30.0, -8.0, 0.0, 8.0,
              30.0, 37.5, 38.0, 38.4, 38.8, 39.5, 41.0, 43.0]:
        t = (z - a) / b
        if abs(t) < 700:
            pts.append(np.sinh(t))
    return np.array(pts)


def q_points(a, b):
    # standard grid plus q around Phi(a) (ndtri(q) - a cancels, x ~ 0)
    import numpy as np
    from scipy.special import ndtr
    pts = [1e-300, 1e-100, 1e-20, 1e-16, 1e-12, 1e-8, 1e-4, 1e-2, 0.3, 0.5,
           0.9, 1 - 1e-6, 1 - 1e-10, 1 - 1e-14, 1 - 2**-52, 1 - 2**-53]
    qa = ndtr(a)
    for f in [1 - 1e-6, 1 - 1e-9, 1 + 1e-9, 1 + 1e-6]:
        q = qa * f
        if 0 < q < 1:
            pts.append(q)
    qb = ndtr(-a)
    for f in [1 - 1e-6, 1 - 1e-9, 1 + 1e-9, 1 + 1e-6]:
        q = qb * f
        if 0 < q < 1:
            pts.append(q)
    return np.array(pts)
