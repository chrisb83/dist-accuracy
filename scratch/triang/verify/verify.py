import mpmath as mp
import numpy as np
from scipy import stats

def pdf(x, c):
    if x < c:
        return 2*x/c
    return 2*(1-x)/(1-c)

def cdf_quad(x, c):
    # integrate pdf, split at the kink c
    if x <= c:
        return mp.quad(lambda t: 2*t/c, [0, x]) if c > 0 else mp.mpf(0)
    left = mp.mpf(c)  # integral of 2t/c on [0,c] = c (analytic triangle area)
    return left + mp.quad(lambda t: 2*(1-t)/(1-c), [c, x])

def ppf_ref(q, c):
    # different route: solve cdf(x)=q via findroot on the quad-based cdf,
    # starting from a bracket; verify cdf(x_ref)==q
    q = mp.mpf(q); c = mp.mpf(c)
    if q < c:
        x0 = mp.sqrt(c*q)
    else:
        # use log-form root to avoid cancellation: 1-x = sqrt((1-c)(1-q))
        x0 = -mp.expm1((mp.log1p(-c) + mp.log1p(-q))/2)
    # Newton refinement on quad cdf
    x = x0
    for _ in range(3):
        x = x - (cdf_quad(x, c) - q)/pdf(x, c)
    return x

cases = [
 ('ppf', 1e-20, 0.0), ('ppf', 1e-20, 1e-200), ('ppf', 3.4807005884285624e-14, 0.0),
 ('ppf', 1e-12, 0.0), ('ppf', 1e-10, 1e-10), ('ppf', 1e-08, 1e-10), ('ppf', 1e-06, 1e-10),
 ('ppf', 1e-250, 1e-200), ('ppf', 1e-300, 1e-200), ('ppf', 5e-201, 1e-200),
 ('cdf', 9e-201, 1e-200), ('cdf', 1e-250, 1e-200),
 ('cdf', 0.9999999909999999, 0.99999999), ('cdf', 0.999999995, 0.99999999),
 ('cdf', 0.99999999, 0.99999999), ('cdf', 0.9999999999, 0.9999999999),
 ('cdf', 0.9999991, 0.999999),
]

def ref(m, a, c, dps):
    with mp.workdps(dps):
        a_ = mp.mpf(a); c_ = mp.mpf(c)
        if m == 'cdf':
            return cdf_quad(a_, c_)
        x = ppf_ref(a_, c_)
        resid = (cdf_quad(x, c_) - a_)/a_
        return x, resid

for m, a, c in cases:
    if m == 'ppf':
        r1, res1 = ref(m, a, c, 100); r2, res2 = ref(m, a, c, 200)
    else:
        r1 = ref(m, a, c, 100); r2 = ref(m, a, c, 200); res1 = 0
    with mp.workdps(200):
        agree = abs(r1 - r2) / abs(r2) < mp.mpf(10)**-30
    s = getattr(stats.triang, m)(a, c)
    with mp.workdps(50):
        ulp = abs(mp.mpf(float(s)) - r2)/abs(r2)/mp.mpf(2.220446049250313e-16)
    print(f"{m}({a!r}, {c!r}) scipy={float(s)!r} ref={mp.nstr(r2, 17)} ulp={mp.nstr(ulp, 3)} agree30={agree} resid={mp.nstr(res1, 3)}")
