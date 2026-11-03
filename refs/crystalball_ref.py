"""Reference module for scipy.stats.crystalball.

Standard form, SciPy's shapes (beta > 0, m > 1), docstring pdf:
    f(x) = N exp(-x**2/2)            for x > -beta
    f(x) = N A (B - x)**(-m)         for x <= -beta
with A = (m/beta)**m exp(-beta**2/2), B = m/beta - beta and
1/N = (m/beta) exp(-beta**2/2)/(m-1) + sqrt(2 pi) Phi(beta).
cdf (x <= -beta) = N A (B - x)**(1-m)/(m-1);
sf  (x > -beta)  = N sqrt(2 pi) Phi(-x);  the other side via 1 - (.).
ppf/isf in closed form (power-law part) and via erfinv (Gaussian part).
"""
import mpmath as mp

DIST = "crystalball"

# example (2, 3); small beta (sf 1 - cdf, (m/beta)**m overflow); large beta;
# moderate m (power amplification); m = 140 / 150 at beta = 1 on both sides
# of the (m/beta)**m overflow (m*log(m/beta) = 709.8); large m ((m/beta)**m overflow, logpdf
# cancellation); m close to 1.
PARAMS = [(2.0, 3.0), (1e-3, 3.0), (1e-10, 2.0), (1e-150, 3.0), (10.0, 3.0),
          (2.0, 50.0), (1.0, 140.0), (1.0, 150.0), (1.0, 1000.0), (1.0, 1e6),
          (0.5, 1.001)]


def _N(beta, m):
    return 1 / ((m / beta) * mp.exp(-beta**2 / 2) / (m - 1)
                + mp.sqrt(2 * mp.pi) * mp.ncdf(beta))


def logpdf(x, beta, m):
    if x > -beta:
        return mp.log(_N(beta, m)) - x * x / 2
    return (mp.log(_N(beta, m)) + m * mp.log(m / beta) - beta**2 / 2
            - m * mp.log(m / beta - beta - x))


def pdf(x, beta, m):
    return mp.exp(logpdf(x, beta, m))


def cdf(x, beta, m):
    if x <= -beta:
        return mp.exp(mp.log(_N(beta, m)) + m * mp.log(m / beta) - beta**2 / 2
                      + (1 - m) * mp.log(m / beta - beta - x)) / (m - 1)
    return 1 - sf(x, beta, m)


def sf(x, beta, m):
    if x > -beta:
        return _N(beta, m) * mp.sqrt(2 * mp.pi) * mp.ncdf(-x)
    return 1 - cdf(x, beta, m)


def _quantile(q, s, beta, m):
    # q = lower mass, s = 1 - q = upper mass (both exact at this precision).
    N = _N(beta, m)
    pbeta = N * (m / beta) * mp.exp(-beta**2 / 2) / (m - 1)
    if q <= pbeta:   # power-law part: cdf = N A (B - x)**(1-m)/(m-1)
        logA = m * mp.log(m / beta) - beta**2 / 2
        return (m / beta - beta
                - mp.exp((mp.log(q) + mp.log(m - 1) - mp.log(N) - logA) / (1 - m)))
    u = s / (N * mp.sqrt(2 * mp.pi))  # = Phi(-x)
    return -mp.sqrt(2) * mp.erfinv(2 * u - 1)


def ppf(q, beta, m):
    return _quantile(q, 1 - q, beta, m)


def isf(q, beta, m):
    return _quantile(1 - q, q, beta, m)


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": lambda x, b, m: mp.log(cdf(x, b, m)),
    "sf": sf,
    "logsf": lambda x, b, m: mp.log(sf(x, b, m)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(beta, m):
    # branch at x = -beta (both sides), multiples of beta in the power-law
    # tail, x = 0, and the x**2 overflow window of logpdf (rhs).
    import numpy as np
    b = float(beta)
    B = float(m) / b - b
    pts = [-b * (1 + 1e-9), -b, -b * (1 - 1e-9), -b * 1.001, -b * 0.999,
           -1.5 * b, -2 * b, -10 * b, -b - B, -b - 10 * B, -b - 1e3 * B,
           0.0, 1.35e154, 1.5e154, 1.89e154]
    # (B - x)**(-m) and (B - x)**(1-m) underflow (subnormal / 0) while
    # (m/beta)**m (B - x)**(-m) can still be normal: B - x = 10**(k/m)
    for k in (290.0, 300.0, 310.0, 320.0, 340.0, 360.0):
        with np.errstate(over="ignore"):
            pts.append(B - np.power(10.0, k / float(m)))
    return np.array([p for p in pts if np.isfinite(p)])


def q_points(beta, m):
    # both sides of the ppf branch at pbeta = cdf(-beta)
    import numpy as np
    mp.mp.dps = 50
    pb = float(cdf(-mp.mpf(beta), mp.mpf(beta), mp.mpf(m)))
    pts = [pb, pb * (1 - 1e-9), pb * (1 + 1e-9), pb * 0.9, pb * 0.5,
           pb * 0.1, pb * 1e-3, 1 - (1 - pb) * 0.5, 1 - (1 - pb) * 0.9]
    return np.array([p for p in pts if 0 < p < 1])
