"""mpmath reference for scipy.stats.fisk (log-logistic), standard form.

Parametrization from the docstring pdf:

    f(x; c) = c x**(c-1) / (1 + x**c)**2
            = c x**(-c-1) / (1 + x**(-c))**2        (x > 0, c > 0)

fisk is burr with d = 1, and fisk_gen delegates every method to burr_gen,
so the reference formulas are the burr ones with d = 1.  Simplest defining
forms only; at 400 digits cancellation in 1 - cdf or 1/q - 1 is harmless.
"""
import numpy as np
import mpmath as mp

DIST = "fisk"

# example parameter from the inventory; c = 1 (c*d = 1, exact cancellation in
# burr._logpdf); c = 1 + 1e-10 (near-cancellation); c = 1.2 (x**-c overflows
# inside the x grid while the pdf is still a normal double); c = 2 (clean
# tails); c = 0.5 and 1e-3 (small shapes); c = 100 and 1e6 (large shapes,
# subnormal x**(-c-1)).
PARAMS = [
    (3.085754862225318,),
    (1.0,),
    (1.0000000001,),
    (1.2,),
    (2.0,),
    (0.5,),
    (1e-3,),
    (100.0,),
    (1e6,),
]


def pdf(x, c):
    return c * x ** (-c - 1) / (1 + x ** (-c)) ** 2


def cdf(x, c):
    return 1 / (1 + x ** (-c))


def sf(x, c):
    return 1 - cdf(x, c)


def ppf(q, c):
    return (1 / q - 1) ** (-1 / c)


def isf(q, c):
    return ppf(1 - q, c)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, c: mp.log(pdf(x, c)),
    "cdf": cdf,
    "logcdf": lambda x, c: mp.log(cdf(x, c)),
    "sf": sf,
    "logsf": lambda x, c: mp.log(sf(x, c)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(c):
    """Both sides of every threshold in burr's code, for d = 1."""
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        # x**(-c) overflows / underflows
        a = np.array([-709.8, -745.0, 709.8, 745.0]) / c
        # x**(-c-1) overflows / goes subnormal / underflows (pattern 4)
        b = np.array([-709.8, 708.0, 709.9, 711.0, 713.0, 715.0, 717.0, 719.0,
                      720.0, 721.0, 722.0, 723.0, 724.0, 735.0, 745.0]) / (c + 1.0)
        # cdf rounds to 1 / sf loses all digits: x**-c ~ eps
        e = np.array([-36.0, -36.7, -37.0, -38.0, -40.0]) / c
        logs = np.concatenate([a, b, e])
        edges = np.exp(np.concatenate([logs * 0.999, logs, logs * 1.001]))
    return np.concatenate([[0.5, 1 - 1e-8, 1.0, 1 + 1e-8, 2.0], edges])


def q_points(c):
    return np.array([1 - 1e-16, 1 - 1e-13, 1 - 1e-10, 1 - 1e-6, 1 - 1e-3,
                     1e-16, 1e-13, 1e-10, 1e-6, 1e-3, 0.5])
