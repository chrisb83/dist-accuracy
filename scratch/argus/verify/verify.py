# Independent reference for argus: integrates the unnormalized pdf
# g(x) = x sqrt(1-x^2) exp(-chi^2 (1-x^2)/2) with mp.quad after the substitution
# u = 1 - x^2:  int g dx = (1/2) int sqrt(u) exp(-a u) du, a = chi^2/2.
# Normalizer Z = (1/2) N(0, 1); cdf = N(y, 1)/N(0, 1); sf = N(0, y)/N(0, 1).
import sys
import mpmath as mp
import numpy as np
import scipy.stats as st

def N(lo, hi, a):
    # int_lo^hi sqrt(u) exp(-a u) du, rescaled to t in [0, 1]
    w = hi - lo
    L = a * w
    f = lambda t: mp.sqrt(lo + w * t) * mp.exp(-L * t)
    pts = [mp.mpf(0)]
    for k in (1, 10, 100, 1000):
        if k / L < 1:
            pts.append(k / L)
    pts.append(mp.mpf(1))
    return w * mp.exp(-a * lo) * mp.quad(f, pts)

def ref(method, x, chi):
    x = mp.mpf(x); chi = mp.mpf(chi)
    a = chi**2 / 2
    y = 1 - x**2
    Z = N(0, 1, a)
    if method == 'cdf':
        return N(y, 1, a) / Z
    if method == 'sf':
        return N(0, y, a) / Z
    lp = mp.log(x) + mp.log(y) / 2 - a * y - mp.log(Z / 2)
    if method == 'logpdf':
        return lp
    if method == 'pdf':
        return mp.exp(lp)

calls = [
    ('F1', 'cdf', 1e-10, 1), ('F1', 'cdf', 1e-8, 1), ('F1', 'cdf', 1e-5, 0.001),
    ('F1', 'cdf', 0.5, 10), ('F1', 'cdf', 0.95, 100), ('F1', 'cdf', 0.999999, 1e4),
    ('F2', 'logpdf', 0.5, 1e-103), ('F2', 'pdf', 0.5, 1e-103),
    ('F2', 'sf', 0.5, 1e-103), ('F2', 'cdf', 0.5, 1e-103),
    ('F2', 'pdf', 1e-300, 1e-110),
    ('F3', 'sf', 0.999999, 1e-100), ('F3', 'sf', 0.9999999999999999, 1e-100),
    ('F3', 'cdf', 0.999999, 1e-100),
    ('F4', 'logpdf', 0.99, 1e155), ('F4', 'logpdf', 0.9999999999999999, 1e155),
    ('F5', 'logpdf', 0.35, 1e-100), ('F5', 'logpdf', 0.35, 0.001),
    ('F5', 'logpdf', 0.5, 1e-102),
]
import warnings; warnings.simplefilter('ignore')
for fid, m, x, chi in calls:
    mp.mp.dps = 100; r1 = ref(m, x, chi)
    mp.mp.dps = 200; r2 = ref(m, x, chi)
    agree = mp.nstr(abs(r1 - r2) / abs(r2), 3) if r2 != 0 else 'r2=0'
    sp = getattr(st.argus, m)(x, chi)
    r = float(r2)
    ulp = abs(sp - r) / (abs(r) * 2.220446049250313e-16) if np.isfinite(sp) else np.inf
    print(f"{fid} {m}({x!r}, {chi!r}) scipy={sp!r} ref={r!r} relagree100v200={agree} ulp={ulp:.3g}")
