"""mpmath reference for scipy.stats.mielke (Mielke Beta-Kappa / Dagum).

Parametrization from the docstring pdf (standard form, loc=0, scale=1):

    f(x; k, s) = k x**(k-1) / (1 + x**s)**(1 + k/s)      (x > 0, k, s > 0)

Integrating gives the cdf used in the code,

    F(x; k, s) = x**k / (1 + x**s)**(k/s) = (1 + x**(-s))**(-k/s),

i.e. mielke(k, s) is burr(c = s, d = k/s).  The quantile follows from
x**s = t/(1 - t) with t = q**(s/k).

Simplest defining forms are used everywhere; at 400 digits the cancellation
in 1 - qsk or x**k / x**k is harmless.  The two exceptions are sf and logcdf,
whose true values fall below 1e-400 in the far right tail; they are written
with log1p/expm1 so that the small quantity keeps full relative accuracy at
any x in the grid (sf/logsf/logcdf/isf are inherited generic methods in
SciPy and are not measured, but the references are needed for the
"Opportunities" demonstration points).
"""
import numpy as np
import mpmath as mp

DIST = "mielke"

# (10.4, 4.6)  inventory example (k/s = 2.26..., inexact)
# (2.0, 3.0)   plain case, k < s
# (1.0, 1.0)   k = 1 (the log(x)*(k-1) term vanishes), k/s = 1
# (0.5, 2.0)   k < 1: pdf -> inf at x = 0
# (1e-3, 1e-3) tiny shapes, k/s = 1
# (1e6, 1e-3)  k >> s: huge exponent k in cdf/pdf, s/k ~ 0 in ppf
# (1e-3, 1e6)  k << s: huge exponent s in logpdf's x**s
# (300.0, 4.6) large k with moderate s (early overflow of x**k in the cdf)
# (1000.0, 1.0) k/s = 1000 reachable inside the double range (rounding of
#               1 + x**s raised to a large power; log-space cancellation)
# (1.0, 1e-3)  s/k = 1e-3: q**(s/k) -> 1 in the ppf, error amplified by 1/s
# (1e6, 1.0)   k >> s with x**s harmless: pure log-space cancellation in
#              _logpdf ((k-1)log x against (1+k/s)log1p(x**s))
PARAMS = [
    (10.4, 4.6),
    (2.0, 3.0),
    (1.0, 1.0),
    (0.5, 2.0),
    (1e-3, 1e-3),
    (1e6, 1e-3),
    (1e-3, 1e6),
    (300.0, 4.6),
    (1000.0, 1.0),
    (1.0, 1e-3),
    (1e6, 1.0),
]


def pdf(x, k, s):
    return k * x ** (k - 1) / (1 + x ** s) ** (1 + k / s)


def cdf(x, k, s):
    return x ** k / (1 + x ** s) ** (k / s)


def logcdf(x, k, s):
    # -(k/s)*log1p(x**-s): no cancellation, valid for every x > 0.
    return -(k / s) * mp.log1p(x ** (-s))


def sf(x, k, s):
    return -mp.expm1(logcdf(x, k, s))


def ppf(q, k, s):
    t = q ** (s / k)
    return (t / (1 - t)) ** (1 / s)


def isf(q, k, s):
    lt = (s / k) * mp.log1p(-q)          # log t with t = (1-q)**(s/k)
    return (mp.exp(lt) / (-mp.expm1(lt))) ** (1 / s)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, k, s: mp.log(pdf(x, k, s)),
    "cdf": cdf,
    "logcdf": logcdf,
    "sf": sf,
    "logsf": lambda x, k, s: mp.log(sf(x, k, s)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(k, s):
    """Both sides of every overflow/underflow/rounding threshold in the code.

    Exponents appearing in the code: k (cdf numerator), k-1 (pdf numerator),
    s (x**s inside log1p), s+k (denominator (1+x**s)**(1+k/s) ~ x**(s+k)).
    """
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        logs = []
        for p in [k, abs(k - 1.0), s, s + k, s + k - 1.0]:
            if p == 0:
                continue
            for L in [709.8, -709.8, 745.0, -745.0, 36.7, -36.7, 708.0, 720.0]:
                logs.append(L / p)
        logs = np.array(logs)
        edges = np.exp(np.concatenate([logs * 0.98, logs, logs * 1.02]))
        edges = edges[np.isfinite(edges) & (edges > 0)]
    return np.concatenate([[0.5, 1 - 1e-8, 1.0, 1 + 1e-8, 2.0, 10.0], edges])


def q_points(k, s):
    # both sides of the thresholds in _ppf: q**(s/k) underflows / becomes
    # subnormal ((s/k) log q < -708 / -745) and rounds to 1 ((s/k)|log q| < eps)
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        u = np.exp(np.array([-708.0, -745.0, -690.0, -760.0]) * k / s)
        hi = 1 - np.exp(np.array([-36.0, -36.7, -38.0]) * k / s)
    pts = np.concatenate([
        [1 - 1e-16, 1 - 1e-14, 1 - 1e-13, 1 - 1e-11, 1 - 1e-10, 1 - 1e-8,
         1 - 1e-6, 1 - 1e-3, 1e-16, 1e-13, 1e-10, 1e-6, 1e-3, 0.5],
        u, u * 1e3, u * 1e-3, hi])
    return pts[np.isfinite(pts) & (pts > 0) & (pts < 1)]
