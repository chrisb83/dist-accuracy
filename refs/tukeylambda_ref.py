"""Reference module for scipy.stats.tukeylambda.

Standard form, shape lam real. The distribution is defined by its quantile
function
    Q(p) = (p**lam - (1 - p)**lam) / lam      (lam != 0)
    Q(p) = log(p) - log(1 - p)                (lam == 0)
F = Q^{-1}, f(x) = 1 / Q'(F(x)) = 1 / (F**(lam-1) + (1-F)**(lam-1)).
Support (-1/lam, 1/lam) for lam > 0, the real line otherwise.

Q is written with expm1/log1p, so it stays exact for tiny lam.
The cdf is computed for x <= 0 only (p in (0, 1/2]) by a safeguarded Newton
iteration in t = log p on a log-transformed equation (linear in t in the
tail), and F(x) = 1 - F(-x) for x > 0 (exact at the working precision).
"""
import mpmath as mp

DIST = "tukeylambda"

# example value; Cauchy (-1); heavy tails (-3); tiny lam (logistic branch
# |lam| < 1e-4 in tklmbda, both sides of the threshold); lam = 0 exactly;
# approx-normal 0.14; 0.5; lam > 1 (2 = uniform), large lam.
PARAMS = [(3.1321477856738267,), (-1.0,), (-3.0,), (1e-5,), (1e-3,), (0.0,),
          (0.14,), (0.5,), (2.0,), (50.0,)]


def _Q(p, lam):
    p = mp.mpf(p)
    if lam == 0:
        return mp.log(p) - mp.log1p(-p)
    return (mp.expm1(lam * mp.log(p)) - mp.expm1(lam * mp.log1p(-p))) / lam


def _dQdt(p, lam):
    # d Q(e^t) / dt = p * Q'(p)
    return p ** lam + p * (1 - p) ** (lam - 1)


def _p_left(x, lam):
    """p = F(x) for x <= 0 (so p <= 1/2)."""
    x = mp.mpf(x)
    lam = mp.mpf(lam)
    if x == 0:
        return mp.mpf(1) / 2
    if lam == 0:
        return 1 / (1 + mp.exp(-x))
    if lam > 0:
        D = x + 1 / lam            # distance from the lower endpoint
        if D <= 0:
            return mp.mpf(0)
        # R(p) = Q(p) + 1/lam > 0, increasing; solve log R = log D
        def phi(t):
            p = mp.exp(t)
            return mp.log(_Q(p, lam) + 1 / lam) - mp.log(D)

        def dphi(t):
            p = mp.exp(t)
            return _dQdt(p, lam) / (_Q(p, lam) + 1 / lam)
        if lam < 1:
            t0 = mp.log(lam * D) / lam
        else:
            t0 = mp.log(D)
    else:
        # -Q(p) > 0, decreasing in t; solve log(-Q) = log(-x)
        def phi(t):
            p = mp.exp(t)
            return mp.log(-x) - mp.log(-_Q(p, lam))

        def dphi(t):
            p = mp.exp(t)
            return -_dQdt(p, lam) / _Q(p, lam)
        t0 = mp.log(-lam * -x) / lam
    tb = mp.log(mp.mpf(1) / 2)
    # phi increasing in t, phi(tb) >= 0 (x <= Q(1/2) = 0)
    t0 = min(t0, tb - mp.mpf("1e-3"))
    a = t0
    step = mp.mpf(1)
    while phi(a) > 0:
        a -= step
        step *= 2
    b = tb
    t = a if phi(a) == 0 else t0
    if t <= a or t >= b:
        t = (a + b) / 2
    tol = mp.mpf(10) ** (-(mp.mp.dps - 15))
    for _ in range(500):
        v = phi(t)
        if v == 0:
            break
        if v < 0:
            a = t
        else:
            b = t
        d = dphi(t)
        tn = t - v / d if d != 0 else (a + b) / 2
        if not (a < tn < b):
            tn = (a + b) / 2
        if abs(tn - t) <= tol * (1 + abs(t)):
            t = tn
            break
        t = tn
    return mp.exp(t)


def cdf(x, lam):
    x = mp.mpf(x)
    if x <= 0:
        return _p_left(x, lam)
    return 1 - _p_left(-x, lam)


def sf(x, lam):
    return cdf(-mp.mpf(x), lam)


def pdf(x, lam):
    x = mp.mpf(x)
    lam = mp.mpf(lam)
    if lam > 0 and abs(x) >= 1 / lam:
        return mp.mpf(0)
    p = _p_left(-abs(x), lam)
    if p == 0:
        return mp.mpf(0)
    return 1 / (p ** (lam - 1) + (1 - p) ** (lam - 1))


def ppf(q, lam):
    q = mp.mpf(q)
    if q == 0:
        return -1 / mp.mpf(lam) if lam > 0 else -mp.inf
    if q == 1:
        return 1 / mp.mpf(lam) if lam > 0 else mp.inf
    return _Q(q, mp.mpf(lam))


def isf(q, lam):
    return -ppf(q, lam)


REFS = {
    "pdf": pdf,
    "logpdf": lambda x, lam: mp.log(pdf(x, lam)),
    "cdf": cdf,
    "logcdf": lambda x, lam: mp.log(cdf(x, lam)),
    "sf": sf,
    "logsf": lambda x, lam: mp.log(sf(x, lam)),
    "ppf": ppf,
    "isf": isf,
}


def x_points(lam):
    import numpy as np
    pts = [-1e-20, -1e-15, -1e-8, 1e-15, 1e-8, -0.5, 0.5, -3.0, 3.0, -10.0, 10.0,
           -30.0, 30.0, -100.0, 100.0, -700.0, 700.0]
    if lam > 0:
        e = 1.0 / lam
        pts += [s * e * f for s in (-1, 1) for f in (0.5, 0.9, 0.99, 1 - 1e-6, 1 - 1e-10, 1 - 1e-14)]
    return np.array(pts)


def q_points(lam):
    import numpy as np
    return np.array([0.5 - 1e-16, 0.5 + 1e-16, 0.5 - 1e-10, 0.5 + 1e-10, 0.49, 0.51,
                     1e-200, 1e-100, 1e-30])
