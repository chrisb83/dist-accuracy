import mpmath as mp
import numpy as np
from scipy import stats

def pdf(t, a):
    s = 1 + a if t > 0 else 1 - a
    return 1 / (mp.pi * (1 + (t / s)**2))

def cdf_quad(x, a):
    # independent route: integrate the docstring pdf
    x = mp.mpf(x); a = mp.mpf(a)
    if x <= 0:
        # substitute to keep quad well-behaved: split at multiple scales
        pts = [-mp.inf] + [x * 10**k for k in (8, 4, 2, 1)] + [x]
        return mp.quad(lambda t: pdf(t, a), pts)
    else:
        return mp.quad(lambda t: pdf(t, a), [-mp.inf, -1, 0]) + mp.quad(lambda t: pdf(t, a), [0, x])

def ppf_ref(q, a):
    q = mp.mpf(q); a = mp.mpf(a)
    q0 = (1 - a) / 2
    if q < q0:
        return -(1 - a) / mp.tan(mp.pi * q / (1 - a))  # cot form
    else:
        # complementary form: sf = (1+a)/pi * acot(x/(1+a))
        return (1 + a) * mp.cot(mp.pi * (1 - q) / (1 + a))

cdf_calls = [(-1e300, .5), (-1e17, .5), (-1e8, .5), (-1000, .9), (-10, 0.999999999999999),
             (-1, 0.999999999999999), (-1, 0.99999999)]
ppf_calls = [(1e-300, .5), (1e-20, 0.0), (1e-12, .5), (1e-8, .5),
             (0.95, -0.9), (0.999999995, -0.99999999), (0.9999999999999996, -0.999999999999999),
             (0.9999999999956061, 0.999999999999999), (0.9991516571017559, 0.999999999999999)]

def ulp(c, r):
    if r == 0: return float('nan')
    return float(abs(mp.mpf(c) - r) / abs(r) / mp.mpf(2.0**-52))

for name, calls, f, sp in [('cdf', cdf_calls, cdf_quad, stats.skewcauchy.cdf),
                           ('ppf', ppf_calls, ppf_ref, stats.skewcauchy.ppf)]:
    for x, a in calls:
        mp.mp.dps = 100; r1 = f(x, a)
        mp.mp.dps = 200; r2 = f(x, a)
        agree = mp.nstr(abs(r1 - r2) / abs(r2), 3)
        extra = ''
        if name == 'ppf':
            mp.mp.dps = 100
            c = cdf_quad(r2, a)
            extra = ' cdfcheck_rel=' + mp.nstr(abs(c - mp.mpf(x)) / mp.mpf(x), 3)
        s = sp(x, a)
        print(f"{name}({x!r}, {a!r}) scipy={s!r} ref={mp.nstr(r2, 17)} ulp={ulp(s, r2):.3g} p100v200={agree}{extra}")
