"""Reference module for scipy.stats.kappa4 (four-parameter kappa, Hosking).

Parametrization (docstring pdf, standard form loc=0, scale=1):

    f(x, h, k) = (1 - k x)^(1/k - 1) * (1 - h (1 - k x)^(1/k))^(1/h - 1)

with the limiting forms for h = 0 and/or k = 0.  Writing

    y(x) = (1 - k x)^(1/k)     (k != 0),      y(x) = exp(-x)      (k = 0)

the cdf is

    F(x) = (1 - h y)^(1/h)     (h != 0),      F(x) = exp(-y)      (h = 0)

and the pdf is f = A * B with
    A = (1 - k x)^(1/k - 1)  (k != 0) or exp(-x)   (k = 0)
    B = (1 - h y)^(1/h - 1)  (h != 0) or exp(-y)   (h = 0)

The quantile function is closed form (invert F step by step):
    y = (1 - q^h)/h  (h != 0) or -log q  (h = 0),
    x = (1 - y^k)/k  (k != 0) or -log y  (k = 0).

These are the plain defining formulas; at 400 digits the cancellation in
1 - q^h, 1 - y^k, 1 - h y and 1 - cdf is harmless.  The only computational
care needed is that for h = 0 the cdf is a *double* exponential,
exp(-exp(-x)), and the grid reaches x = -1e300: evaluated naively, mpmath
tries to build an integer exponent with 1e299 digits there.  Everything is
therefore built from log quantities and exponentiated through the saturating
helper `_exp` (anything outside exp(+-1e5) is far outside the double range
and is never measured by the harness).  Saturation is a convenience, not a
necessity: writing the overflow-safe term as `L + log1p(exp(-L))` and
dropping the second part only for L > 1e4 reaches x = -1e300 exactly.
"""
import numpy as np
import mpmath as mp

DIST = "kappa4"

# Branch coverage of the _lazyselect trees (h == 0 / h != 0  x  k == 0 / k != 0)
# and of the sign combinations, plus the neighbourhoods of the h -> 0 and
# k -> 0 limits (where the 1/k and 1/h exponents amplify rounding) and shapes
# for which the intermediate (1 - k x)^(1/k) resp. exp(-x) overflows while the
# result is still a normal double.
PARAMS = [
    (0.0, 0.0),        # f3: Gumbel
    (0.0, 0.1),        # f1: generalized extreme value, k > 0 (bounded above)
    (0.1, 0.0),        # f2: h > 0, k = 0
    (-1.0, 0.0),       # f2: h < 0, k = 0  (logistic)
    (-0.1, 0.1),       # f0: h < 0, k > 0
    (2.0, 0.5),        # f0: h > 0, k > 0  (both endpoints finite)
    (1.0, -0.5),       # f0: h > 0, k < 0  (generalized Pareto)
    (-2.0, -0.5),      # f0: h < 0, k < 0
    (0.5, 1e-8),       # k -> 0 limit from above
    (1e-8, 0.5),       # h -> 0 limit from above
    (-1000.0, 0.1),    # (1 - k x)^(1/k) overflows in the left tail
    (-1e6, 0.0),       # exp(-x) overflows in the left tail
    (-1e6, -0.01),     # k < 0: (1 - k x)^(1/k) overflows near the lower endpoint 1/k
]

_BIG = mp.mpf("1e5")      # |log value| beyond this is never a normal double
_ASY = mp.mpf("1e4")      # use log1p(-h y) ~ log(-h) + log y beyond this


def _exp(v):
    """mp.exp, saturated outside the double range (keeps mpmath from building
    astronomically large integer exponents for the h = 0 double exponential)."""
    if v > _BIG:
        return mp.inf
    if v < -_BIG:
        return mp.mpf(0)
    return mp.e ** v


def _log_y(x, h, k):
    """log of y = (1 - k x)^(1/k), resp. exp(-x) for k = 0."""
    if k == 0:
        return -x
    return mp.log(1 - k * x) / k


def _log1p_mhy(ly, h):
    """log(1 - h y) with log y = ly;  h != 0."""
    if ly + mp.log(abs(h)) > _ASY:          # |h| y huge, so h < 0 here
        return mp.log(-h) + ly              # neglected term < exp(-1e4)
    return mp.log1p(-h * _exp(ly))


def logpdf(x, h, k):
    ly = _log_y(x, h, k)
    log_A = (1 - k) * ly                    # (1/k - 1) log(1 - k x) = (1 - k) log y
    log_B = -_exp(ly) if h == 0 else (1 / h - 1) * _log1p_mhy(ly, h)
    return log_A + log_B


def logcdf(x, h, k):
    ly = _log_y(x, h, k)
    return -_exp(ly) if h == 0 else _log1p_mhy(ly, h) / h


def logsf(x, h, k):
    return mp.log(-mp.expm1(logcdf(x, h, k)))


def pdf(x, h, k):
    return _exp(logpdf(x, h, k))


def cdf(x, h, k):
    return _exp(logcdf(x, h, k))


def sf(x, h, k):
    return -mp.expm1(logcdf(x, h, k))


def ppf(q, h, k):
    y = -mp.log(q) if h == 0 else (1 - q ** h) / h
    return -mp.log(y) if k == 0 else (1 - y ** k) / k


def isf(q, h, k):
    return ppf(1 - q, h, k)


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": logcdf,
    "sf": sf,
    "logsf": logsf,
    "ppf": ppf,
    "isf": isf,
}


def x_points(h, k):
    """Extra x values: just inside the support endpoints, and both sides of
    where the intermediate (1 - k x)^(1/k) (or exp(-x) for k = 0) overflows or
    underflows in double precision."""
    pts = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 10.0, -10.0]
    with np.errstate(all="ignore"):
        if k == 0:
            # exp(-x) crosses 1.8e308 / 2.2e-308 at x = -709.8 / +709.8
            pts += [-709.0, -710.0, -745.2, -746.0, -1e4, -1e6, -1e8,
                    709.0, 710.0, 745.0]
        else:
            # (1/k) * log1p(-k x) = L  <=>  x = (1 - exp(k L))/k
            for L in (-745.2, -709.8, 709.8, 745.2, 1e4, 1e6):
                for f in (0.99, 1.0, 1.01):
                    x = (1.0 - np.exp(k * L * f)) / k
                    if np.isfinite(x):
                        pts.append(float(x))
        if h > 0:       # lower endpoint a: h*y -> 1, cancellation in 1 - h y
            a = np.log(h) if k == 0 else (1.0 - np.float_power(h, -k)) / k
            if np.isfinite(a):
                w = max(abs(a), 1.0)
                pts += [a + w * 1e-16, a + w * 1e-12, a + w * 1e-8,
                        a + w * 1e-4, a * (1 + 1e-13), a * (1 - 1e-13)]
        if k > 0:       # upper endpoint b = 1/k
            b = 1.0 / k
            pts += [b - abs(b) * 1e-16, b - abs(b) * 1e-12, b - abs(b) * 1e-8,
                    b - abs(b) * 1e-4, b]
    return np.array([p for p in pts if np.isfinite(p)], dtype=float)


def q_points(h, k):
    """Extra q values: around exp(-1) (where 1 - (-log q)^k cancels for h = 0)
    and very close to both ends."""
    e1 = float(mp.e ** -1)
    return np.array([e1, e1 * (1 - 1e-8), e1 * (1 + 1e-8), 0.5,
                     1e-16, 1e-8, 1e-4, 1 - 1e-16, 1 - 2e-16, 1 - 1e-12,
                     1 - 1e-8, 1 - 1e-4])
