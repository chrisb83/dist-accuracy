"""Reference module for scipy.stats.weibull_max.

Standard form, docstring pdf f(x, c) = c (-x)**(c-1) exp(-(-x)**c), x < 0, c > 0.
cdf = exp(-(-x)**c), closed-form inverse ppf = -(-log q)**(1/c).
"""
import mpmath as mp

DIST = "weibull_max"

# example parameter, tiny/small c (pattern 6 in ppf, cdf), c = 1 (exponential),
# large c (pdf subnormal exp(-t) with large prefactor).
PARAMS = [(2.8687961709100187,), (1e-3,), (0.1,), (0.5,), (1.0,), (30.0,), (1e3,), (1e6,), (1e8,)]


def _t(x, c):
    return (-x) ** c


def _exp_neg(t):
    # exp(-t) for t > 1e6 is far below the double range; clamping avoids
    # mpmath's very slow exp of huge arguments (e.g. t = 1e9000).
    return mp.exp(-min(t, mp.mpf(10) ** 6))


def pdf(x, c):
    return c * (-x) ** (c - 1) * _exp_neg(_t(x, c))


def cdf(x, c):
    return _exp_neg(_t(x, c))


def sf(x, c):
    return 1 - cdf(x, c)


def ppf(q, c):
    return -(-mp.log(q)) ** (1 / c)


def isf(q, c):
    return ppf(1 - q, c)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, c: mp.log(c) + (c - 1) * mp.log(-x) - _t(x, c),
    "cdf": cdf,
    "logcdf": lambda x, c: -_t(x, c),
    "sf": sf,
    "logsf": lambda x, c: mp.log(sf(x, c)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(c):
    # x = -t**(1/c) for chosen t = (-x)**c: t = 1 (x = -1), small t (sf small),
    # and t around 708..745 where exp(-t) becomes subnormal / underflows.
    import numpy as np
    t = np.array([1e-300, 1e-100, 1e-30, 1e-16, 1e-8, 1e-3, 0.1, 0.5, 1.0, 2.0,
                  10.0, 30.0, 100.0, 300.0, 600.0, 700.0, 705.0, 708.0, 710.0,
                  715.0, 720.0, 725.0, 730.0, 735.0, 740.0, 745.0, 750.0])
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        xs = -np.exp(np.log(t) / c)
    return xs[np.isfinite(xs) & (xs < 0)]
