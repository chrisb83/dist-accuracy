import mpmath as mp
import numpy as np
from scipy import stats

# cdf via quadrature of pdf after t = log(x): F(x) = int_0^{log x} b exp(-b t) dt
def cdf_quad(x, b):
    L = mp.log(mp.mpf(x)); b = mp.mpf(b)
    return mp.quad(lambda t: b*mp.exp(-b*t), [0, L])

# ppf: closed form from F = 1 - x^-b, checked by cdf_quad(x_ref) == q
def ppf_ref(q, b):
    q = mp.mpf(q); b = mp.mpf(b)
    return mp.exp(-mp.log1p(-q)/b)

cdf_cases = [(2.0, 1e-17), (1.0000000000000002, 0.001), (1e20, 1e-17), (2.0, 1e-8),
             (1.0001, 0.001), (10000000001.0, 1e-8), (1.01, 0.001)]
ppf_cases = [(5e-16, 1e-17), (1e-16, 1e-17), (1e-17, 1e-17), (3.4807005884285624e-14, 1e-8),
             (1e-12, 1e-8), (1e-17, 1e-8)]

def at(dps, f, *a):
    with mp.workdps(dps):
        return f(*a)

def ulp(s, r):
    if r == 0: return float('nan')
    return float(abs(mp.mpf(s) - r) / (abs(r) * mp.mpf(2)**-52))

for x, b in cdf_cases:
    r1 = at(100, cdf_quad, x, b); r2 = at(200, cdf_quad, x, b)
    with mp.workdps(200):
        agree = abs(r1 - r2) <= abs(r2) * mp.mpf(10)**-30
    s = stats.pareto.cdf(x, b)
    print(f"cdf({x!r}, {b!r}) scipy={s!r} ref={mp.nstr(r2, 17)} ulp={ulp(s, r2):.3g} agree30={agree}")

for q, b in ppf_cases:
    r1 = at(100, ppf_ref, q, b); r2 = at(200, ppf_ref, q, b)
    with mp.workdps(200):
        agree = abs(r1 - r2) <= abs(r2) * mp.mpf(10)**-30
        back = cdf_quad(r2, b)
        inv_ok = abs(back - mp.mpf(q)) <= mp.mpf(q) * mp.mpf(10)**-30
    s = stats.pareto.ppf(q, b)
    print(f"ppf({q!r}, {b!r}) scipy={s!r} ref={mp.nstr(r2, 17)} ulp={ulp(s, r2):.3g} agree30={agree} cdf(ref)==q:{inv_ok}")
