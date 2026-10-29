# Independent reference for foldcauchy cdf: integrate docstring pdf
# f(x,c) = 1/(pi(1+(x-c)^2)) + 1/(pi(1+(x+c)^2)), x >= 0, with mp.quad,
# cross-checked with cdf = atan2(2x, 1 + c^2 - x^2)/pi.
import mpmath as mp
import numpy as np
from scipy import stats

def pdf(x, c):
    return 1/(mp.pi*(1+(x-c)**2)) + 1/(mp.pi*(1+(x+c)**2))

def cdf_quad(x, c):
    x = mp.mpf(x); c = mp.mpf(c)
    pts = [0, x]
    if 0 < c < x:
        pts = [0, c, x]
    # substitute t = u*x for tiny x is unnecessary: quad handles scale
    return mp.quad(lambda t: pdf(t, c), pts)

def cdf_atan2(x, c):
    x = mp.mpf(x); c = mp.mpf(c)
    return mp.atan2(2*x, 1 + c**2 - x**2)/mp.pi

cases = [(1e-20, 1), (1e-300, 4.716467345583189), (1, 1e100), (5e99, 1e100),
         (10, 1e6), (1e-10, 1), (1e-10, 0.001), (1e-5, 4.716467345583189),
         (1, 1000), (1e-5, 1), (0.1, 4.716467345583189)]
for x, c in cases:
    vals = {}
    for d in (100, 200):
        mp.mp.dps = d
        vals[d] = (cdf_quad(x, c), cdf_atan2(x, c))
    mp.mp.dps = 200
    q1, a1 = vals[100]; q2, a2 = vals[200]
    agree = lambda u, v: float(abs(u-v)/abs(v))
    ref = a2
    s = stats.foldcauchy.cdf(x, c)
    ulp = float(abs(mp.mpf(s)-ref)/(ref*mp.mpf(2)**-52)) if True else None
    print(f"cdf({x!r}, {c!r}) scipy={s!r} ref={mp.nstr(ref,17)} ulp={ulp:.3g} "
          f"| quad vs atan2 rel={agree(q2,a2):.1e} 100vs200 atan2={agree(a1,a2):.1e} quad={agree(q1,q2):.1e}")
