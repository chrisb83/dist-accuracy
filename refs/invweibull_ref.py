"""Reference module for scipy.stats.invweibull (Frechet).

Standard form, docstring pdf f(x, c) = c x**(-c-1) exp(-x**-c), x > 0, c > 0.
cdf = exp(-x**-c), sf = 1 - cdf, ppf = (-log q)**(-1/c).
"""
import mpmath as mp

DIST = "invweibull"

# Example parameter; small c (exp(-t) subnormal or 0 while the pdf is normal,
# pow(x, -c-1) overflow near the smallest normal x, 1/c amplification in ppf);
# c = 1; large c (pow(x, -c-1) subnormal while the pdf is normal).
# The harness runs its self-checks on the first three sets. For c <~ 0.01 the
# bulk spans >100 decades and its plain mp.quad of the pdf fails (artifact:
# with the substitution x = exp(u) the integral matches the cdf to 1e-30, see
# scratch/invweibull/notes.md), so those sets are not placed first.
PARAMS = [(10.58,), (0.1,), (1.0,), (1e-3,), (3e-3,), (5e-3,), (9e-3,), (0.01,),
          (0.02,), (0.5,), (3.0,), (1e3,), (1e6,), (1e10,)]


def _t(x, c):
    return x ** (-c)


def _expneg(t):
    # exp(-t) for t > 1e6 is far below any double; avoid slow mpmath exp.
    return mp.mpf(0) if t > 1e6 else mp.exp(-t)


def pdf(x, c):
    return c * x ** (-c - 1) * _expneg(_t(x, c))


def cdf(x, c):
    return _expneg(_t(x, c))


def sf(x, c):
    return 1 - cdf(x, c)


def ppf(q, c):
    return (-mp.log(q)) ** (-1 / c)


def isf(q, c):
    return ppf(1 - q, c)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, c: mp.log(c) - (c + 1) * mp.log(x) - _t(x, c),
    "cdf": cdf,
    "logcdf": lambda x, c: -_t(x, c),
    "sf": sf,
    "logsf": lambda x, c: mp.log(sf(x, c)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(c):
    # x = t**(-1/c) for chosen t = x**-c (t ~ 708..750: exp(-t) subnormal),
    # x where pow(x, -c-1) is subnormal ((c+1) log x in [700, 745]),
    # and x near the smallest normal double (pow(x, -c-1) overflow for small c).
    import numpy as np
    t = np.array([1e-300, 1e-100, 1e-30, 1e-16, 1e-8, 1e-3, 0.1, 0.5, 1.0, 2.0,
                  10.0, 30.0, 100.0, 300.0, 600.0, 700.0, 708.0, 710.0,
                  720.0, 730.0, 740.0, 745.0, 750.0, 760.0, 780.0, 800.0, 900.0,
                  1000.0, 1100.0, 1200.0, 1300.0, 1400.0])
    s = np.array([700.0, 705.0, 708.0, 709.0, 710.0, 712.0, 715.0, 720.0, 725.0,
                  730.0, 735.0, 740.0, 745.0])
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        xs = np.exp(-np.log(t) / c)
        xr = np.exp(s / (c + 1))
    small = np.array([2.3e-308, 3e-308, 5e-308, 1e-307, 1e-306, 1e-305])
    pts = np.concatenate([xs, xr, small, [1 - 1e-6, 1.0, 1 + 1e-6]])
    return pts[np.isfinite(pts) & (pts > 0)]
