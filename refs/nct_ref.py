"""mpmath reference for scipy.stats.nct (standard form, loc=0, scale=1).

Parametrization (docstring): X = (Y + c)/sqrt(V/k), Y ~ N(0, 1), V ~ chi2(k),
k = df > 0, c = nc real.  With mu = c**2/2, y = t**2/(k + t**2),
yc = k/(k + t**2), b = k/2 and the weights

    p_j = exp(-mu) mu**j / j!,
    q_j = c exp(-mu) mu**j / (sqrt(2) Gamma(j + 3/2))     (sum_j q_j = erf(c/sqrt 2)),

for t >= 0 (Guenther 1978; Lenth 1989, AS 243):

    F+(t; k, c) = Phi(-c) + 1/2 sum_j [p_j I_y(j+1/2, b) + q_j I_y(j+1, b)]
    S+(t; k, c) = 1/2 sum_j [p_j I_yc(b, j+1/2) + q_j I_yc(b, j+1)]

(S+ = 1 - F+ follows from sum p_j = 1 and 1/2 sum q_j = Phi(c) - 1/2.)
For t < 0 the reflection X(k, c) = -X(k, -c) gives cdf(t) = S+(-t; k, -c) and
sf(t) = F+(-t; k, -c).  So every tail value comes from a direct sum (no
1 - F), and all lower-tail sums have positive terms only when c >= 0; for
c < 0 extra digits (mu/ln 10) cover the cancellation between p and q terms.

The incomplete betas are obtained from one continued fraction plus the
a-recurrence I_y(a, b) = I_y(a+1, b) + d_a run in its stable direction
(backwards from a large J for F+, forwards for S+), as in refs/ncf_ref.py.

pdf (Wikipedia "Noncentral t-distribution", checked against the integral of
the pdf by the harness):

    f(x) = k**(k/2) Gamma(k+1) exp(-mu) / (2**k (k+x**2)**(k/2) Gamma(k/2))
           * [ sqrt(2) c x M(k/2+1, 3/2, z) / ((k+x**2) Gamma((k+1)/2))
               + M((k+1)/2, 1/2, z) / (sqrt(k+x**2) Gamma(k/2+1)) ],
    z = c**2 x**2 / (2 (k + x**2)),  M = 1F1.

For df > 1e5 and c**2 x**2/4 > 1e3 mpmath's hyp1f1 (a ~ df/2) does not finish, so the pdf uses the
exact identity f(x) = (k/x) [F_{k+2}(x sqrt(1+2/k)) - F_k(x)] (sf form for
x > 0) with log10(k) + 30 guard digits; it agrees with the 1F1 form to 20
digits at df = 1e5.

ppf/isf: the sign of the quantile is decided by comparing with cdf(0) =
Phi(-c); then the root is found in log|x| (positive=True), always inverting a
function whose target is <= 1/2.
"""
import mpmath as mp

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".claude", "skills",
                                "dist-accuracy-review", "scripts"))
from accuracy import invert  # noqa: E402

DIST = "nct"
DPS = 100

# inventory example; df = 1 (Cauchy-like tails, |x| > 1e154 normal values);
# negative nc; small df with large nc (heavy right tail); large nc with
# moderate/large df (left tail tiny, Poisson mode far from 0); large df;
# tiny nc (continuity to the central t).
PARAMS = [(14.0, 0.24045031331198066), (1.0, 1.0), (5.0, -3.0), (0.5, 5.0),
          (30.0, 20.0), (10.0, 30.0), (1e6, 2.0), (10.0, 1e-8)]


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


def _logd(a, b, logy, logyc):
    # d_a = I_y(a, b) - I_y(a+1, b)
    return (mp.loggamma(a + b) - mp.loggamma(a + 1) - mp.loggamma(b)
            + a * logy + b * logyc)


def _weights(mu, c):
    """[(a0, log w_0 function, ratio w_{j+1}/w_j function, sign)] for p and q."""
    lmu = mp.log(mu)
    out = [(mp.mpf(0.5), lambda j: -mu + j * lmu - mp.loggamma(j + 1),
            lambda j: mu / (j + 1), 1)]
    lc = mp.log(abs(c)) - mp.log(2) / 2
    out.append((mp.mpf(1), lambda j: lc - mu + j * lmu - mp.loggamma(j + mp.mpf(1.5)),
                lambda j: mu / (j + mp.mpf(1.5)), 1 if c > 0 else -1))
    return out


def _lower(a0, logw, ratio, b, mu, y, yc, logy, logyc):
    """sum_j w_j I_y(a0 + j, b), backwards from J (positive terms)."""
    eps = mp.mpf(10) ** (-(mp.mp.dps + 5))
    J = int(mu + 10 * mp.sqrt(mu) + 40)
    while True:
        IJ = _ibeta(a0 + J, b, y, yc, logy, logyc)[0]
        d = mp.exp(_logd(a0 + J, b, logy, logyc))   # d_{a0+J}
        w = mp.exp(logw(J))
        I = IJ
        total = w * I
        for j in range(J - 1, -1, -1):
            a = a0 + j
            d = d * (a + 1) / ((a + b) * y)          # d_a from d_{a+1}
            I = I + d                                 # I(a) = I(a+1) + d_a
            w = w / ratio(j)
            total += w * I
        wJ1 = mp.exp(logw(J + 1))
        bound = IJ * wJ1 / (1 - mu / (J + 2))
        if bound <= eps * total:
            return total
        J *= 2


def _upper(a0, logw, ratio, b, mu, y, yc, logy, logyc):
    """sum_j w_j I_yc(b, a0 + j), forwards (positive terms, increasing)."""
    eps = mp.mpf(10) ** (-(mp.mp.dps + 5))
    floor = mp.mpf(10) ** (-(mp.mp.dps + 400))
    S = _ibeta(a0, b, y, yc, logy, logyc)[1]
    d = mp.exp(_logd(a0, b, logy, logyc))
    w = mp.exp(logw(0))
    total = w * S
    j = 0
    while True:
        a = a0 + j
        S = S + d                                     # S(a+1) = S(a) + d_a
        d = d * y * (a + b) / (a + 1)                 # d_{a+1}
        w = w * ratio(j)
        j += 1
        total += w * S
        if j + 2 > 2 * mu:
            tail = 2 * w * ratio(j)                   # >= sum_{i>j} w_i (S <= 1)
            if tail <= eps * total or total + tail < floor:
                return total
        if j > 10**6:
            raise mp.NoConvergence("upper")


def _setup(t, k):
    t2 = t * t
    lden = mp.log(k + t2)
    logy = mp.log(t2) - lden
    logyc = mp.log(k) - lden
    return mp.exp(logy), mp.exp(logyc), logy, logyc


def _Fplus(t, k, c):
    """cdf for t > 0."""
    k, c = mp.mpf(k), mp.mpf(c)
    y, yc, logy, logyc = _setup(t, k)
    b = k / 2
    mu = c * c / 2
    if mu == 0:
        return (1 + _ibeta(mp.mpf(0.5), b, y, yc, logy, logyc)[0]) / 2
    s = 0
    for a0, logw, ratio, sg in _weights(mu, c):
        s += sg * _lower(a0, logw, ratio, b, mu, y, yc, logy, logyc)
    return mp.erfc(c / mp.sqrt(2)) / 2 + s / 2


def _Splus(t, k, c):
    """sf for t > 0."""
    k, c = mp.mpf(k), mp.mpf(c)
    y, yc, logy, logyc = _setup(t, k)
    b = k / 2
    mu = c * c / 2
    if mu == 0:
        return _ibeta(mp.mpf(0.5), b, y, yc, logy, logyc)[1] / 2
    s = 0
    for a0, logw, ratio, sg in _weights(mu, c):
        s += sg * _upper(a0, logw, ratio, b, mu, y, yc, logy, logyc)
    return s / 2


def _guard(c):
    # extra digits for the p/q cancellation in S+ with c < 0 (and in the pdf
    # when c*x < 0); the other sums have no cancellation relative to the result
    return 20 + int(mp.mpf(c) ** 2 / 2 / mp.log(10))


def cdf(x, k, c):
    x = mp.mpf(x)
    if mp.isinf(x):
        return mp.mpf(0) if x < 0 else mp.mpf(1)
    g = _guard(c) if (x < 0 and c > 0) else 10
    with mp.workdps(mp.mp.dps + g):
        if x == 0:
            r = mp.erfc(mp.mpf(c) / mp.sqrt(2)) / 2
        elif x > 0:
            r = _Fplus(x, k, c)
        else:
            r = _Splus(-x, k, -mp.mpf(c))
    return +r


def sf(x, k, c):
    x = mp.mpf(x)
    if mp.isinf(x):
        return mp.mpf(1) if x < 0 else mp.mpf(0)
    g = _guard(c) if (x > 0 and c < 0) else 10
    with mp.workdps(mp.mp.dps + g):
        if x == 0:
            r = mp.erfc(-mp.mpf(c) / mp.sqrt(2)) / 2
        elif x > 0:
            r = _Splus(x, k, c)
        else:
            r = _Fplus(-x, k, -mp.mpf(c))
    return +r


def _pdf_via_cdf(x, k, c):
    # f(x) = (k/x) [F_{k+2}(x sqrt(1+2/k)) - F_k(x)]  (exact identity), written
    # with sf for x > 0 so the difference is between two small tail values;
    # the cancellation costs ~log10(k) digits, covered by the guard.
    g = 30 + int(mp.log10(k))
    with mp.workdps(mp.mp.dps + g):
        x2 = x * mp.sqrt(1 + 2 / k)
        if x > 0:
            r = k / x * (sf(x, k, c) - sf(x2, k + 2, c))
        else:
            r = k / x * (cdf(x2, k + 2, c) - cdf(x, k, c))
    return +r


def pdf(x, k, c):
    x, k, c = mp.mpf(x), mp.mpf(k), mp.mpf(c)
    if k > 10**5 and c * c * x * x / 4 > 10**3:
        # mpmath hyp1f1 with a ~ k/2 and a*z = c**2 x**2/4 large hangs; here
        # |x| > 63/|c|, so the identity loses only ~log10(k) digits
        return _pdf_via_cdf(x, k, c)
    g = _guard(c) if c * x < 0 else 10
    with mp.workdps(mp.mp.dps + g):
        s = k + x * x
        z = c * c * x * x / (2 * s)
        pre = mp.exp(k / 2 * mp.log(k) + mp.loggamma(k + 1) - c * c / 2
                     - k * mp.log(2) - k / 2 * mp.log(s) - mp.loggamma(k / 2))
        t1 = (mp.sqrt(2) * c * x * mp.hyp1f1(k / 2 + 1, mp.mpf(1.5), z)
              / (s * mp.gamma((k + 1) / 2))) if c != 0 and x != 0 else 0
        t2 = mp.hyp1f1((k + 1) / 2, mp.mpf(0.5), z) / (mp.sqrt(s) * mp.gamma(k / 2 + 1))
        r = pre * (t1 + t2)
    return +r


def _quantile_lower(p, k, c):
    """x with cdf(x) = p, p <= 1/2 (lower-side target)."""
    p0 = cdf(0, k, c)
    if p < p0:   # negative root: u -> cdf(-u) decreasing in u
        return -invert(lambda u: cdf(-u, k, c), p, increasing=False,
                       positive=True, digits=30)
    return invert(lambda u: cdf(u, k, c), p, positive=True, digits=30)


def _quantile_upper(p, k, c):
    """x with sf(x) = p, p <= 1/2."""
    p0 = sf(0, k, c)
    if p < p0:   # positive root
        return invert(lambda u: sf(u, k, c), p, increasing=False,
                      positive=True, digits=30)
    return -invert(lambda u: sf(-u, k, c), p, positive=True, digits=30)


def ppf(q, k, c):
    q = mp.mpf(q)
    if q <= mp.mpf(0.5):
        return _quantile_lower(q, k, c)
    return _quantile_upper(1 - q, k, c)


def isf(q, k, c):
    q = mp.mpf(q)
    if q <= mp.mpf(0.5):
        return _quantile_upper(q, k, c)
    return _quantile_lower(1 - q, k, c)


def logcdf(x, k, c):
    F = cdf(x, k, c)
    return mp.log(F) if F < mp.mpf(0.5) else mp.log1p(-sf(x, k, c))


def logsf(x, k, c):
    S = sf(x, k, c)
    return mp.log(S) if S < mp.mpf(0.5) else mp.log1p(-cdf(x, k, c))


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, k, c: mp.log(pdf(x, k, c)),
    "cdf": cdf,
    "logcdf": logcdf,
    "sf": sf,
    "logsf": logsf,
    "ppf": ppf,
    "isf": isf,
}


def x_points(k, c):
    # x = 0 (cdf = Phi(-c)), both sides of |x| = 1.34e154 (x*x overflows),
    # the mode region ~ c, and moderate negative x (left tail for c > 0).
    import numpy as np
    c = float(c)
    pts = [0.0, -0.5, 0.5, c, -c, 2 * abs(c) + 1, -10.0, -100.0, 1e3, -1e3,
           1e150, -1e150, 1.3e154, -1.3e154, 1.4e154, -1.4e154, 1e160, -1e160]
    return np.array([p for p in pts if np.isfinite(p)])
