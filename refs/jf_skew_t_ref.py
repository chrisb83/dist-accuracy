"""Reference module for scipy.stats.jf_skew_t.

Standard form. Docstring pdf:
  f(x; a, b) = C^{-1} (1 + t)^(a+1/2) (1 - t)^(b+1/2),  t = x / sqrt(a+b+x^2),
  C = 2^(a+b-1) B(a, b) sqrt(a+b).
cdf F(x) = I_y(a, b) with y = (1 + t)/2; sf = I_{1-y}(b, a).

1 + t and 1 - t are formed without cancellation (r = sqrt(a+b+x^2)):
  x >= 0: 1 + t = (r + x)/r,  1 - t = (a+b)/(r (r + x))
  x <  0: 1 - t = (r - x)/r,  1 + t = (a+b)/(r (r - x))
This is needed because at |x| = 1e300 the small quantity is ~1e-600, below
the 400-digit resolution of 1 - t.
"""
import mpmath as mp
from accuracy import invert

DIST = "jf_skew_t"

PARAMS = [(8.0, 4.0), (0.5, 0.5), (2.0, 0.5), (0.5, 2.0), (1e-3, 1e-3),
          (1000.0, 5.0), (600.0, 600.0)]


def _opm(x, a, b):
    """Return (1 + t, 1 - t) without cancellation."""
    s = a + b
    r = mp.sqrt(s + x * x)
    if x >= 0:
        return (r + x) / r, s / (r * (r + x))
    return s / (r * (r - x)), (r - x) / r


_CC = {}


def _C(a, b):
    key = (a, b, mp.mp.prec)
    if key not in _CC:
        _CC[key] = mp.mpf(2) ** (a + b - 1) * mp.beta(a, b) * mp.sqrt(a + b)
    return _CC[key]


def pdf(x, a, b):
    p, m = _opm(x, a, b)
    return p ** (a + mp.mpf(0.5)) * m ** (b + mp.mpf(0.5)) / _C(a, b)


def cdf(x, a, b):
    p, m = _opm(x, a, b)
    if x <= 0:
        return mp.betainc(a, b, 0, p / 2, regularized=True)
    return 1 - mp.betainc(b, a, 0, m / 2, regularized=True)


def sf(x, a, b):
    return cdf(-x, b, a)


def _x_from_u(u, v, a, b):
    """x for beta variable u = (1 + t)/2, v = 1 - u (passed separately)."""
    return (u - v) * mp.sqrt(a + b) / (2 * mp.sqrt(u * v))


def ppf(q, a, b):
    # Invert in the beta variable (solved in log space, bracket (0, 1]):
    # u = (1 + t)/2 with I_u(a, b) = q for q <= 1/2, otherwise
    # v = 1 - u with I_v(b, a) = 1 - q. Inverting the cdf in x directly is
    # limited by invert's bracket widening (|x| <= 2**81).
    if a == b and q == mp.mpf(1) / 2:
        return mp.mpf(0)
    if q <= mp.mpf(1) / 2:
        u = invert(lambda u: mp.betainc(a, b, 0, u, regularized=True), q,
                   hi=1, digits=100)
        return _x_from_u(u, 1 - u, a, b)
    v = invert(lambda v: mp.betainc(b, a, 0, v, regularized=True), 1 - q,
               hi=1, digits=100)
    return _x_from_u(1 - v, v, a, b)


def isf(q, a, b):
    return ppf(1 - q, a, b)


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
    # where (a+b)/(2 x^2) ~ eps (1 -/+ t cancels completely), x**2 overflow
    # edge (1.34e154), and moderate tails.
    s = float(a + b) ** 0.5
    pts = [s * k for k in (1e2, 1e4, 1e6, 3e7, 1e8, 1e9)]
    pts += [1e150, 1.3e154, 1.4e154, 1e200, 1e-3, 1e-10]
    return pts + [-p for p in pts]


def q_points(a, b):
    return [0.5, 0.5 + 1e-10, 0.5 - 1e-10, 1 - 1e-6, 1 - 1e-10, 1 - 1e-14,
            1 - 2**-53, 1e-10, 1e-50, 1e-300]
