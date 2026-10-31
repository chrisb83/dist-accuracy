"""Reference module for scipy.stats.t.

Standard form, shape df (nu > 0). Docstring pdf:
  f(x, nu) = Gamma((nu+1)/2) / (sqrt(pi nu) Gamma(nu/2)) (1 + x**2/nu)**(-(nu+1)/2).
cdf: for x <= 0, F(x) = I_{nu/(nu+x^2)}(nu/2, 1/2) / 2; for x > 0, F(x) = 1 - F(-x).
sf(x) = F(-x) (symmetry). ppf by inversion of the cdf in log|x|; isf(q) = -ppf(q).
df = inf (SciPy's normal branch) is represented by the shape value inf.
"""
import mpmath as mp
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".claude",
                                "skills", "dist-accuracy-review", "scripts"))
from accuracy import invert  # noqa: E402

DIST = "t"

# example; Cauchy (df = 1); tiny df; small df; moderate; large df (both
# sides of nothing in t itself, but stdtr/stdtrit switch methods); huge df
# for the log(poch) - 0.5 log(df) cancellation; df = inf (normal branch).
PARAMS = [(2.7433514990818093,), (1.0,), (1e-3,), (0.1,), (30.0,), (1e6,),
          (1e15,), (1e100,), (float("inf"),)]


def _isinf(df):
    return mp.isinf(df)


def pdf(x, df):
    if _isinf(df):
        return mp.npdf(x)
    return (mp.gamma((df + 1) / 2) / (mp.sqrt(mp.pi * df) * mp.gamma(df / 2))
            * (1 + x * x / df) ** (-(df + 1) / 2))


def logpdf(x, df):
    if _isinf(df):
        return -x * x / 2 - mp.log(2 * mp.pi) / 2
    return (mp.loggamma((df + 1) / 2) - mp.loggamma(df / 2)
            - mp.log(mp.pi * df) / 2 - (df + 1) / 2 * mp.log1p(x * x / df))


def _lower(x, df):
    # F(x) for x <= 0
    if _isinf(df):
        return mp.ncdf(x)
    z = df / (df + x * x)
    L = df / 2 * mp.log(z)
    if L < -2000:
        # F < exp(-1600) (the prefactor is at most ~sqrt(df)): far below any double. Only reached while ppf's
        # bracket search probes far out; mp.betainc is very slow or fails
        # there, and the exact value is irrelevant (monotone stand-in).
        return mp.exp(L)
    return mp.betainc(df / 2, mp.mpf(1) / 2, 0, df / (df + x * x),
                      regularized=True) / 2


def cdf(x, df):
    if x <= 0:
        return _lower(x, df)
    return 1 - _lower(-x, df)


def sf(x, df):
    return cdf(-x, df)


def ppf(q, df):
    # Symmetric: solve in log|x| (positive=True) so huge or tiny roots
    # converge to full relative accuracy.
    q = mp.mpf(q)
    if q == mp.mpf(1) / 2:
        return mp.mpf(0)
    # Roots beyond the double range (tiny df) are never measured; return
    # +-inf at once instead of searching out to exp(1e24).
    big = mp.mpf("1.8e308")
    if q < mp.mpf(1) / 2 and cdf(-big, df) > q:
        return -mp.inf
    if q > mp.mpf(1) / 2 and cdf(big, df) < q:
        return mp.inf
    if q < mp.mpf(1) / 2:
        return -invert(lambda y: cdf(-y, df), q, increasing=False,
                       positive=True)
    return invert(lambda y: cdf(y, df), q, positive=True)


def isf(q, df):
    # isf(q) = ppf(1 - q) = -ppf(q) by symmetry
    return -ppf(q, df)


REFS = {
    "pdf": pdf,
    "logpdf": logpdf,
    "cdf": cdf,
    "logcdf": lambda x, df: mp.log(cdf(x, df)),
    "sf": sf,
    "logsf": lambda x, df: mp.log(sf(x, df)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(df):
    # x*x overflow at |x| ~ 1.34e154; x*x/df overflow at |x| = sqrt(1.8e308*df);
    # near 0; sqrt(df) scale; both signs.
    import numpy as np
    pts = [0.0, 1e-300, 1e-160, 1e-20, 1e-8, 1e-3, 0.5, 1.0, 2.0, 10.0, 1e8,
           1e16, 1e50, 1e100, 1e150, 1.3e154, 1.4e154, 1e200, 1e300]
    with np.errstate(over="ignore", invalid="ignore"):
        s = float(np.sqrt(df)) if np.isfinite(df) else 1e10
        for f in (1e-3, 1.0, 3.0, 10.0, 40.0):
            pts.append(f * s)
        lim = np.sqrt(1.79e308 * df) if np.isfinite(df) else np.inf
        if np.isfinite(lim):
            pts += [lim * 0.99, lim * 1.01]
    pts = [p for p in pts if np.isfinite(p) and p < 1.7e308]
    pts = pts + [-p for p in pts if p != 0]
    return np.array(sorted(set(pts)))


def q_points(df):
    import numpy as np
    d = [1e-16, 1e-12, 1e-8, 1e-4, 0.1]
    pts = [0.5] + [0.5 + v for v in d] + [0.5 - v for v in d]
    pts += [1 - 2.2e-16, 1 - 1e-12, 1e-308, 1e-300, 1e-200, 1e-100, 1e-20]
    return np.array(pts)
