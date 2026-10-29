"""Reference module for scipy.stats.exponpow.

Standard form, docstring pdf:  f(x; b) = b x**(b-1) exp(1 + x**b - exp(x**b)),  x >= 0, b > 0.
sf = exp(1 - exp(x**b)), cdf = 1 - sf; closed-form ppf/isf.
expm1/log1p forms are used only where x**b or q can be below 10**-400.
"""
import mpmath as mp

DIST = "exponpow"

# example, b = 1 and b near 1 (cancellation in logpdf), small and large b.
PARAMS = [(2.697119160358469,), (1.0,), (1.0 + 1e-8,), (0.5,), (1e-3,), (0.1,), (10.0,), (1e3,)]


# For x**b > XB_MAX every value below is far outside the double range
# (pdf, sf < exp(-exp(1000)), logpdf, logsf < -exp(1000)); evaluating
# exp(-exp(x**b)) there would need ~exp(x**b) bits, so return stand-ins that
# are not normal doubles (never measured) and cdf = 1 (exact to ~exp(-exp(1000))).
XB_MAX = 1000
BIG = mp.mpf("1e400")


def _exp_m1_mx(t):
    """exp(t) - 1 - t without cancellation (series for small t)."""
    if t == 0:
        return mp.mpf(0)
    if abs(t) > mp.mpf("1e-3"):
        return mp.exp(t) - 1 - t
    term, total, k = t * t / 2, mp.mpf(0), 2
    while total == 0 or abs(term) > abs(total) * mp.eps:
        total += term
        k += 1
        term *= t / k
    return total


def logpdf(x, b):
    xb = x ** b
    if xb > XB_MAX:
        return -BIG
    return mp.log(b) + (b - 1) * mp.log(x) - _exp_m1_mx(xb)


def pdf(x, b):
    if x ** b > XB_MAX:
        return mp.mpf(0)
    return mp.exp(logpdf(x, b))


def sf(x, b):
    if x ** b > XB_MAX:
        return mp.mpf(0)
    return mp.exp(-mp.expm1(x ** b))


def cdf(x, b):
    # 1 - sf, written with expm1 so that x**b far below 10**-400 (e.g. b = 10,
    # x = 1e-300) does not cancel to 0 even at 400 digits.
    if x ** b > XB_MAX:
        return mp.mpf(1)
    return -mp.expm1(-mp.expm1(x ** b))


def ppf(q, b):
    return mp.log1p(-mp.log1p(-q)) ** (1 / b)


def isf(q, b):
    return mp.log1p(-mp.log(q)) ** (1 / b)


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": lambda x, b: mp.log(cdf(x, b)),
    "sf": sf,
    "logsf": lambda x, b: -BIG if x ** b > XB_MAX else -mp.expm1(x ** b),
    "ppf": ppf,
    "isf": isf,
}


def x_points(b):
    # x**b = t for t in a range of small/moderate/large values, including where
    # exp(x**b) overflows (t ~ 709.78) and where x**b is tiny (cancellation in
    # 1 + xb - exp(xb)).
    import numpy as np
    t = np.array([1e-300, 1e-100, 1e-20, 1e-10, 1e-5, 1e-3, 0.1, 0.5, 1.0, 2.0,
                  5.0, 20.0, 100.0, 700.0, 709.0, 709.7, 710.0])
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        return t ** (1.0 / b)
