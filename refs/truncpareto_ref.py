"""Reference module for scipy.stats.truncpareto.

Standard form, shapes b != 0, c > 1, support [1, c] (docstring pdf):
    f(x; b, c) = b / (1 - c**-b) * x**(-b-1)
    F(x; b, c) = (1 - x**-b) / (1 - c**-b)
    ppf(q)     = (1 - (1 - c**-b) q)**(-1/b)
Valid for both signs of b. Plain defining formulas; at 400 digits the
cancellation in 1 - c**-b (|b| down to 1e-8, c - 1 down to 1e-8) is harmless.
"""
import mpmath as mp

DIST = "truncpareto"

# Example params; tiny |b| and c near 1 (1 - c**-b cancels); large b > 0
# (x**-b underflows); large negative b and huge c with b < 0 (c**-b overflows);
# moderate negative b near the overflow boundary.
PARAMS = [(1.8, 5.3), (2.0, 5.0), (-2.0, 5.0), (1e-8, 5.0), (-1e-8, 5.0),
          (2.0, 1.00000001), (1e3, 5.0), (-1e3, 5.0), (-2.0, 1e200), (-400.0, 5.0)]


def D(b, c):
    return 1 - c ** (-b)


def pdf(x, b, c):
    return b * x ** (-b - 1) / D(b, c)


def cdf(x, b, c):
    return (1 - x ** (-b)) / D(b, c)


def sf(x, b, c):
    return 1 - cdf(x, b, c)


def ppf(q, b, c):
    return (1 - D(b, c) * q) ** (-1 / b)


def isf(q, b, c):
    return ppf(1 - q, b, c)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, b, c: mp.log(pdf(x, b, c)),
    "cdf": cdf,
    "logcdf": lambda x, b, c: mp.log(cdf(x, b, c)),
    "sf": sf,
    "logsf": lambda x, b, c: mp.log(sf(x, b, c)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(b, c):
    # endpoints and points just inside them; where x**-b (or x**-(b+1))
    # leaves the normal range: |b| log x = 708 ... 745.
    import numpy as np
    pts = [1.0, 1 + 1e-15, 1 + 1e-12, 1 + 1e-8, 1 + 1e-4, c * (1 - 1e-15),
           c * (1 - 1e-12), c * (1 - 1e-8), c * (1 - 1e-4), c]
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        for L in (700.0, 708.0, 720.0, 745.0, 760.0):
            pts.append(np.exp(L / abs(b)))
    return np.array(pts)
