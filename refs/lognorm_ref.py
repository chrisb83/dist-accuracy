"""Reference module for scipy.stats.lognorm.

Standard form, docstring pdf
    f(x; s) = 1/(s x sqrt(2 pi)) exp(-log(x)**2 / (2 s**2)),  x > 0, s > 0.
cdf = Phi(log(x)/s), sf = Phi(-log(x)/s);
ppf = exp(s Phi^-1(q)), isf = exp(-s Phi^-1(q)), Phi^-1 by numerical inversion.
"""
import mpmath as mp
from accuracy import invert

DIST = "lognorm"

# Example parameter; s = 1 (gibrat); small s (narrow, large z near x = 1);
# large s (heavy tails, s*x overflow in logpdf); s near and below the
# s**2 underflow threshold (1.5e-154) and in the subnormal s**2 range.
PARAMS = [(0.9536822696057533,), (1.0,), (1e-3,), (10.0,), (1e3,), (1e10,),
          (1e-150,), (1e-158,), (1e-200,)]


def _z(x, s):
    return mp.log(x) / s


def pdf(x, s):
    return mp.exp(-mp.log(x) ** 2 / (2 * s ** 2)) / (s * x * mp.sqrt(2 * mp.pi))


def logpdf(x, s):
    return -mp.log(x) ** 2 / (2 * s ** 2) - mp.log(s * x * mp.sqrt(2 * mp.pi))


def cdf(x, s):
    return mp.ncdf(_z(x, s))


def sf(x, s):
    return mp.ncdf(-_z(x, s))


def ppf(q, s):
    z = invert(mp.ncdf, q, positive=False)
    return mp.exp(s * z)


def isf(q, s):
    z = invert(mp.ncdf, 1 - q, positive=False)
    return mp.exp(s * z)


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": lambda x, s: mp.log(cdf(x, s)),
    "sf": sf,
    "logsf": lambda x, s: mp.log(sf(x, s)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(s):
    # x = 1 (log x = 0; 0/0 when s**2 underflows), x next to 1, the mode
    # exp(-s**2), x where |z| = log(x)/s is 8, 30, 37.5, 38.4 (ndtr tails),
    # x where s*x*sqrt(2 pi) passes 1.8e308, and x where logpdf crosses 0.
    import numpy as np
    pts = [1.0, 1 + 2**-52, 1 - 2**-53, 1 + 1e-12, 1 - 1e-12, 1 + 1e-8, 0.5, 2.0,
           1e300, 1e305, 1e307, 1.7e308]
    with np.errstate(over="ignore", under="ignore"):
        pts.append(np.exp(-s * s))
        for z in [-38.4, -37.5, -30.0, -8.0, 8.0, 30.0, 37.5, 38.4]:
            if abs(z * s) < 708:
                pts.append(np.exp(z * s))
        t = 1.8e308 / (s * np.sqrt(2 * np.pi))
        if 0 < t < 1.8e308:
            pts += [t * 0.5, t * 2.0]
        # zero crossings of logpdf: -L**2/(2 s**2) - L - log(s sqrt(2pi)) = 0
        c = np.log(s * np.sqrt(2 * np.pi))
        disc = 1 - 2 * c / (s * s)
        if np.isfinite(disc) and disc >= 0:
            for L in (s * s * (-1 + np.sqrt(disc)), s * s * (-1 - np.sqrt(disc))):
                if abs(L) < 700:
                    pts += [np.exp(L), np.exp(L) * (1 + 1e-6)]
    pts = np.array(pts, dtype=float)
    return pts[np.isfinite(pts) & (pts > 0)]


def q_points(s):
    import numpy as np
    return np.array([1e-300, 1e-200, 1e-100, 1e-20, 1e-16, 1e-12, 1e-8, 1e-4,
                     1e-2, 0.3, 0.5, 0.7, 0.9, 1 - 1e-6, 1 - 1e-10, 1 - 1e-14,
                     1 - 2**-52, 1 - 2**-53])
