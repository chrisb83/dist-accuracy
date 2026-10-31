"""mpmath reference for scipy.stats.ncx2 (standard form, loc=0, scale=1).

Parametrization (docstring): k = df, lam = nc,
    pdf(x) = 1/2 exp(-(lam + x)/2) (x/lam)**((k-2)/4) I_{(k-2)/2}(sqrt(lam x)).
Poisson mixture with a = k/2, y = x/2, mu = lam/2, w_j = exp(-mu) mu**j / j!:
    cdf(x) = sum_j w_j P(a + j, y),   sf(x) = sum_j w_j Q(a + j, y),
with P, Q the regularized incomplete gamma functions.  With
d_j = y**(a+j) exp(-y) / Gamma(a+j+1) = P(a+j, y) - P(a+j+1, y):
  * cdf: P_J from mp.gammainc once, then P_j = P_{j+1} + d_j backwards
    (positive terms only), truncation bound P_J * Poisson tail(J).
  * sf: Q_0 from mp.gammainc once, then Q_{j+1} = Q_j + d_j forwards
    (positive terms only), truncation bound (Q <= 1) Poisson tail(j).
Poisson tail(J) = sum_{i>J} w_i <= w_{J+1} / (1 - mu/(J+2)) for J + 2 > mu.
For x so far in the right tail that the series would need ~sqrt(lam x)
terms and the sf is certainly below exp(-800) (Chernoff bound), sf returns
that bound: an upper bound below the double range, so the harness classifies
the point as non-normal and skips it, as it would for the exact value.
ppf/isf invert the cdf or sf, whichever side has target <= 1/2.
"""
import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".claude", "skills",
                                "dist-accuracy-review", "scripts"))
from accuracy import invert  # noqa: E402

DIST = "ncx2"
DPS = 100

# inventory example; df < 2 (pdf pole at 0); tiny nc (continuity with chi2,
# Boost pdf l <= 50 path with huge x/l); nc in (50, 200) (Boost pdf series
# path, cdf Ding path); nc >= 200 (Boost cdf P series from the Poisson mode);
# large df with small nc (ive underflow in logpdf); large nc.
# Sets with nc/2 + df/2 >~ 171 ((3, 1e4), (21, 1000)) make sf raise
# OverflowError for x <~ 1e-8, which aborts the harness; they are run
# separately with --params ... --methods pdf,logpdf,cdf,ppf,isf.
PARAMS = [(21.0, 1.0560465975116415), (0.5, 1.0), (2.0, 1e-10), (5.0, 60.0),
          (10.0, 300.0), (1000.0, 1.0)]


def _eps():
    return mp.mpf(10) ** (-(mp.mp.dps + 3))


def _logw(j, mu):
    return -mu + j * mp.log(mu) - mp.loggamma(j + 1)


def _logd(j, a, y):
    return (a + j) * mp.log(y) - y - mp.loggamma(a + j + 1)


def _ptail_bound(J, mu):
    """Upper bound of sum_{i > J} w_i (requires J + 2 > mu)."""
    return mp.exp(_logw(J + 1, mu)) / (1 - mu / (J + 2))


def cdf(x, k, lam):
    x, k, lam = mp.mpf(x), mp.mpf(k), mp.mpf(lam)
    if x <= 0:
        return mp.mpf(0)
    a, y, mu = k / 2, x / 2, lam / 2
    if mu == 0:
        return mp.gammainc(a, 0, y, regularized=True)
    J = int(mp.ceil(mu + 30 * mp.sqrt(mu) + 60))
    while True:
        P = mp.gammainc(a + J, 0, y, regularized=True)
        d = mp.exp(_logd(J - 1, a, y))          # d_{J-1}
        w = mp.exp(_logw(J, mu))
        total = w * P
        for j in range(J - 1, -1, -1):
            # P_j = P_{j+1} + d_j ; w_j = w_{j+1} (j+1)/mu
            P = P + d
            w = w * (j + 1) / mu
            total += w * P
            if j > 0:
                d = d * (a + j) / y
        if P_bound_ok(J, mu, a, y, total):
            return total
        J = int(J * 1.5) + 10


def P_bound_ok(J, mu, a, y, total):
    PJ1 = mp.gammainc(a + J + 1, 0, y, regularized=True)
    return PJ1 * _ptail_bound(J, mu) <= _eps() * total


def _log_chernoff_sf(x, k, lam):
    best = mp.mpf(0)
    for i in range(1, 80):
        s = mp.mpf(10) ** (-mp.mpf(i) / 10)    # 1 - 2t, t in (0, 1/2)
        t = (1 - s) / 2
        v = -t * x - k / 2 * mp.log(s) + lam * t / s
        if v < best:
            best = v
    return best


def sf(x, k, lam):
    x, k, lam = mp.mpf(x), mp.mpf(k), mp.mpf(lam)
    if x <= 0:
        return mp.mpf(1)
    a, y, mu = k / 2, x / 2, lam / 2
    if mu == 0:
        return mp.gammainc(a, y, mp.inf, regularized=True)
    lb = _log_chernoff_sf(x, k, lam)
    if lb < -800 and mp.sqrt(lam * x) > 2e4:
        return mp.exp(lb)                     # far below the double range
    if y > a:
        Q = mp.gammainc(a, y, mp.inf, regularized=True)
    else:
        Q = 1 - mp.gammainc(a, 0, y, regularized=True)
    d = mp.exp(_logd(0, a, y))
    w = mp.exp(-mu)
    total = w * Q
    j = 0
    eps = _eps()
    while True:
        Q = Q + d                              # Q_{j+1}
        d = d * y / (a + j + 1)                # d_{j+1}
        w = w * mu / (j + 1)                   # w_{j+1}
        j += 1
        total += w * Q
        # Poisson tail beyond j: <= w_{j+1} / (1 - mu/(j+2))
        if j + 2 > mu and w * mu / (j + 1) / (1 - mu / (j + 2)) <= eps * total:
            return total
        if j > 3 * 10**5:
            raise mp.NoConvergence("sf series")


def pdf(x, k, lam):
    x, k, lam = mp.mpf(x), mp.mpf(k), mp.mpf(lam)
    if lam == 0:
        return mp.exp((k / 2 - 1) * mp.log(x) - x / 2 - k / 2 * mp.log(2)
                      - mp.loggamma(k / 2))
    z = mp.sqrt(lam * x)
    return (mp.exp(-(lam + x) / 2) * (x / lam) ** ((k - 2) / 4)
            * mp.besseli((k - 2) / 2, z) / 2)


def logpdf(x, k, lam):
    return mp.log(pdf(x, k, lam))


def _quantile(p, pc, k, lam):
    if p <= mp.mpf(0.5):
        return invert(lambda x: cdf(x, k, lam), p, positive=True, digits=30)
    return invert(lambda x: sf(x, k, lam), pc, increasing=False,
                  positive=True, digits=30)


def ppf(q, k, lam):
    q = mp.mpf(q)
    return _quantile(q, 1 - q, k, lam)


def isf(q, k, lam):
    q = mp.mpf(q)
    return _quantile(1 - q, q, k, lam)


def logcdf(x, k, lam):
    F = cdf(x, k, lam)
    return mp.log(F) if F < mp.mpf(0.5) else mp.log1p(-sf(x, k, lam))


def logsf(x, k, lam):
    S = sf(x, k, lam)
    return mp.log(S) if S < mp.mpf(0.5) else mp.log1p(-cdf(x, k, lam))


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


def x_points(k, lam):
    # x = k + lam (Boost cdf/sf switch between Q series and P/Ding),
    # points around (sqrt(lam) +- c)**2 (bulk and tails of the non-central
    # part), and the right tail where gamma_q(k/2 + lam/2, x/2) at the
    # Poisson mode underflows (x/2 > ~745 + ...).
    import numpy as np
    k, lam = float(k), float(lam)
    m = k + lam
    s = np.sqrt(lam)
    pts = [m * (1 - 1e-6), m, m * (1 + 1e-6), 0.5 * m, 2 * m,
           (s + 10) ** 2, (s + 20) ** 2, (s + 30) ** 2, (s + 36) ** 2,
           1500.0, 2000.0, 3000.0]
    if s > 12:
        pts += [(s - 10) ** 2, (s - 5) ** 2]
    pts = np.array(pts)
    return pts[(pts > 0) & np.isfinite(pts)]
