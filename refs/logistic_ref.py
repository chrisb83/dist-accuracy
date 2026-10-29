"""Reference module for scipy.stats.logistic.

Standard form, no shapes. Docstring pdf: f(x) = exp(-x)/(1 + exp(-x))**2, real x.
cdf F(x) = 1/(1 + exp(-x)), sf = 1 - F = 1/(1 + exp(x)) (written in
this exact form so log(sf) stays representable at 400 digits for x > ~921), ppf = log(q/(1 - q)), isf = ppf(1 - q).
"""
import mpmath as mp

DIST = "logistic"

PARAMS = [()]


def pdf(x):
    return mp.exp(-x) / (1 + mp.exp(-x)) ** 2


def cdf(x):
    return 1 / (1 + mp.exp(-x))


def sf(x):
    return 1 / (1 + mp.exp(x))


def ppf(q):
    return mp.log(q / (1 - q))


def isf(q):
    return ppf(1 - q)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x: mp.log(pdf(x)),
    "cdf": cdf,
    "logcdf": lambda x: mp.log(cdf(x)),
    "sf": sf,
    "logsf": lambda x: mp.log(sf(x)),
    "ppf": ppf,
    "isf": isf,
}


def x_points():
    # Near 0 (logpdf = -log 4, logcdf/logsf = -log 2), the region where
    # exp(-|x|) < eps (|x| ~ 36.7), where the cdf/sf/pdf underflow
    # (|x| ~ 708 .. 745), and huge |x| for logpdf/logcdf/logsf. Both signs.
    import numpy as np
    pts = [0.0, 1e-300, 1e-100, 1e-20, 1e-10, 1e-5, 0.01, 0.1, 0.5, 1.0, 2.0,
           5.0, 10.0, 18.0, 20.0, 30.0, 33.3, 36.0, 36.7, 37.5, 40.0, 50.0,
           100.0, 300.0, 600.0, 700.0, 708.0, 709.0, 709.8, 720.0, 740.0,
           744.0, 745.0, 746.0, 800.0, 1e4, 1e10, 1e100, 1e300, 1.7e308]
    pts = pts + [-p for p in pts if p != 0]
    return np.array(pts)


def q_points():
    # Both sides of q = 0.5 (logit ~ 4(q - 1/2)), the log1p branch region of
    # logit, q near 1 and tiny q.
    import numpy as np
    d = [1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.2, 0.25]
    pts = [0.5] + [0.5 + v for v in d] + [0.5 - v for v in d]
    pts += [0.3, 0.29, 0.31, 0.65, 0.64, 0.66, 0.7,
            1 - 2.2e-16, 1 - 1e-15, 1 - 1e-12, 1 - 1e-8, 1 - 1e-4,
            1e-308, 1e-300, 1e-200, 1e-100, 1e-20, 1e-8, 1e-3]
    return np.array(pts)
