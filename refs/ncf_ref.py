"""mpmath reference for scipy.stats.ncf (standard form, loc=0, scale=1).

Parametrization (docstring / ncfdtr notes, n = dfn, m = dfd, lam = nc):
X ~ noncentral chi2(n, lam), Y ~ chi2(m), Z = (X/n)/(Y/m).  With
a = n/2, b = m/2, mu = lam/2, w_j = exp(-mu) mu**j / j!, t = n x / m,
y = t/(1+t), yc = 1/(1+t):

    pdf(x) = sum_j w_j (n/m) t**(a+j-1) (1+t)**(-(a+j+b)) / B(a+j, b)
    cdf(x) = sum_j w_j I_y(a+j, b)
    sf(x)  = sum_j w_j I_yc(b, a+j)

All three sums have positive terms only.  The incomplete betas are obtained
from one continued-fraction evaluation plus the two-term recurrence
I_y(a+j+1, b) = I_y(a+j, b) - d_j, d_j = y**(a+j) yc**b / ((a+j) B(a+j, b)),
run in its stable direction: backwards from a large J for the cdf (adding
d_j), forwards from j = 0 for the sf (S_{j+1} = S_j + d_j).  Truncation is
checked against a rigorous Poisson-tail bound.  ppf/isf invert the cdf or
sf, choosing the side where the target is <= 1/2.
"""
import mpmath as mp

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".claude", "skills",
                                "dist-accuracy-review", "scripts"))
from accuracy import invert  # noqa: E402

DIST = "ncf"
DPS = 100

# inventory example; moderate nc; dfn = dfd = 1 (heavy tails, pole at 0);
# small dfd (heavy right tail: large quantiles, pdf right tail); large nc
# (Poisson mode far from 0: left-tail underflow of the mode term);
# large df; tiny nc (continuity at nc -> 0).
PARAMS = [(27.0, 27.0, 0.41578441799226107), (2.0, 3.0, 10.0), (1.0, 1.0, 1.0),
          (5.0, 0.5, 2.0), (0.5, 10.0, 200.0), (100.0, 1000.0, 50.0),
          (10.0, 20.0, 1e-8)]


def _logbeta(a, b):
    return mp.loggamma(a) + mp.loggamma(b) - mp.loggamma(a + b)


def _betacf(a, b, x):
    """Continued fraction for I_x(a,b) (modified Lentz, Numerical Recipes)."""
    tiny = mp.mpf(10) ** (-(mp.mp.dps + 50))
    eps = mp.mpf(10) ** (-(mp.mp.dps + 5))
    qab, qap, qam = a + b, a + 1, a - 1
    c = mp.mpf(1)
    d = 1 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1 / d
    h = d
    m = 1
    while True:
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        de = d * c
        h *= de
        if abs(de - 1) < eps:
            return h
        m += 1
        if m > 10**7:
            raise mp.NoConvergence("betacf")


def _ibeta(a, b, y, yc, logy, logyc):
    """(I_y(a,b), I_yc(b,a)), the small one from the convergent CF side."""
    if y < (a + 1) / (a + b + 2):
        lp = a * logy + b * logyc - _logbeta(a, b) - mp.log(a)
        Iv = mp.exp(lp) * _betacf(a, b, y)
        return Iv, 1 - Iv
    lp = b * logyc + a * logy - _logbeta(a, b) - mp.log(b)
    Sv = mp.exp(lp) * _betacf(b, a, yc)
    return 1 - Sv, Sv


def _setup(x, n, m, lam):
    x, n, m, lam = mp.mpf(x), mp.mpf(n), mp.mpf(m), mp.mpf(lam)
    a, b, mu = n / 2, m / 2, lam / 2
    t = n * x / m
    l1p = mp.log1p(t)
    logy = mp.log(t) - l1p
    logyc = -l1p
    y = t / (1 + t)
    yc = 1 / (1 + t)
    return x, n, m, a, b, mu, t, y, yc, logy, logyc


def _logd(j, a, b, logy, logyc):
    # log d_j, d_j = I_y(a+j, b) - I_y(a+j+1, b)
    return (mp.loggamma(a + j + b) - mp.loggamma(a + j + 1) - mp.loggamma(b)
            + (a + j) * logy + b * logyc)


def _logw(j, mu):
    return -mu + j * mp.log(mu) - mp.loggamma(j + 1)


def cdf(x, n, m, lam):
    if mp.mpf(x) <= 0:
        return mp.mpf(0)
    if mp.isinf(mp.mpf(x)):
        return mp.mpf(1)
    x, n, m, a, b, mu, t, y, yc, logy, logyc = _setup(x, n, m, lam)
    if mu == 0:
        return _ibeta(a, b, y, yc, logy, logyc)[0]
    eps = mp.mpf(10) ** (-(mp.mp.dps + 5))
    J = int(2 * mu + 10 * mp.sqrt(mu) + 40)
    while True:
        IJ = _ibeta(a + J, b, y, yc, logy, logyc)[0]
        d = mp.exp(_logd(J, a, b, logy, logyc))      # d_J
        w = mp.exp(_logw(J, mu))
        I = IJ
        total = w * I
        for j in range(J - 1, -1, -1):
            d = d * (a + j + 1) / ((a + j + b) * y)  # d_j from d_{j+1}
            I = I + d                                 # I_j = I_{j+1} + d_j
            w = w * (j + 1) / mu
            total += w * I
        # tail sum_{j>J} w_j I_j <= I_J * P(Pois > J)
        wJ1 = mp.exp(_logw(J + 1, mu))
        bound = IJ * wJ1 / (1 - mu / (J + 2))
        if bound <= eps * total:
            return total
        J *= 2


def sf(x, n, m, lam):
    if mp.mpf(x) <= 0:
        return mp.mpf(1)
    if mp.isinf(mp.mpf(x)):
        return mp.mpf(0)
    x, n, m, a, b, mu, t, y, yc, logy, logyc = _setup(x, n, m, lam)
    if mu == 0:
        return _ibeta(a, b, y, yc, logy, logyc)[1]
    eps = mp.mpf(10) ** (-(mp.mp.dps + 5))
    S = _ibeta(a, b, y, yc, logy, logyc)[1]
    d = mp.exp(_logd(0, a, b, logy, logyc))
    w = mp.exp(-mu)
    total = w * S
    j = 0
    while True:
        S = S + d                                     # S_{j+1}
        d = d * y * (a + j + b) / (a + j + 1)         # d_{j+1}
        w = w * mu / (j + 1)                          # w_{j+1}
        j += 1
        total += w * S
        # remaining tail <= P(Pois > j) <= w_{j+1} / (1 - mu/(j+2))
        if j + 2 > 2 * mu:
            tail = 2 * w * mu / (j + 1)
            if tail <= eps * total:
                return total
            if total + tail < mp.mpf(10) ** -400:
                # far below the double range (only reached while bracketing
                # in invert); full accuracy is not needed there
                return total
        if j > 10**6:
            raise mp.NoConvergence("sf")


def pdf(x, n, m, lam):
    xm = mp.mpf(x)
    n, m, lam = mp.mpf(n), mp.mpf(m), mp.mpf(lam)
    a, b, mu = n / 2, m / 2, lam / 2
    if xm <= 0:
        if a < 1:
            return mp.inf
        if a == 1:
            return mp.exp(-mu) * (n / m) * b
        return mp.mpf(0)
    x, n, m, a, b, mu, t, y, yc, logy, logyc = _setup(x, n, m, lam)
    eps = mp.mpf(10) ** (-(mp.mp.dps + 5))
    l1p = -logyc
    lg = mp.log(n / m) + (a - 1) * mp.log(t) - (a + b) * l1p - _logbeta(a, b)
    g = mp.exp(lg - mu)                               # w_0 g_0
    total = g
    j = 0
    while True:
        if mu == 0:
            return total
        r = mu / (j + 1) * y * (a + j + b) / (a + j)
        g = g * r
        total += g
        j += 1
        rn = mu / (j + 1) * y * (a + j + b) / (a + j)
        if rn < mp.mpf(0.5) and g <= eps * total:
            return total
        if j > 10**6:
            raise mp.NoConvergence("pdf")


def _quantile(p, pc, n, m, lam):
    if p <= mp.mpf(0.5):
        return invert(lambda x: cdf(x, n, m, lam), p, positive=True,
                      digits=30)
    return invert(lambda x: sf(x, n, m, lam), pc, increasing=False,
                  positive=True, digits=30)


def ppf(q, n, m, lam):
    q = mp.mpf(q)
    return _quantile(q, 1 - q, n, m, lam)


def isf(q, n, m, lam):
    q = mp.mpf(q)
    return _quantile(1 - q, q, n, m, lam)


def logcdf(x, n, m, lam):
    F = cdf(x, n, m, lam)
    return mp.log(F) if F < mp.mpf(0.5) else mp.log1p(-sf(x, n, m, lam))


def logsf(x, n, m, lam):
    S = sf(x, n, m, lam)
    return mp.log(S) if S < mp.mpf(0.5) else mp.log1p(-cdf(x, n, m, lam))


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, n, m, lam: mp.log(pdf(x, n, m, lam)),
    "cdf": cdf,
    "logcdf": logcdf,
    "sf": sf,
    "logsf": logsf,
    "ppf": ppf,
    "isf": isf,
}


def x_points(n, m, lam):
    # n*x/m = 1 (beta side switch), t = n x / m ~ 1e8, 1e16 (y = t/(1+t)
    # rounds to 1 in Boost), 1e154 ((1+t)**2 overflows in Boost pdf),
    # t = 1e308 and 1e309 (t = x*alpha/beta overflows in Boost cdf/sf), and a few points around the mean.
    import numpy as np
    n, m = float(n), float(m)
    r = m / n
    pts = [0.5, 1.0, 2.0, r, r * 1e4, r * 1e8, r * 1e12, r * 1e16, r * 1e20,
           r * 1e100, r * 1e150, r * 1e155, r * 1e308, (r * 1e308) * 10,
           r * 1e-10, r * 1e-30]
    pts = np.array(pts)
    return pts[(pts > 0) & np.isfinite(pts) & (pts < 1.7e308)]
