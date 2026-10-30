import warnings
import mpmath as mp
import numpy as np
from scipy import stats
warnings.simplefilter("ignore")

cases = [  # (finding, x, c)
    ("F1", 1e-300, 0.01), ("F1", 1e-150, 0.02),
    ("F1", 1.8986165949114554e-29, 0.1), ("F1", 2.030873172573601e-29, 0.1),
    ("F2", 2.3e-308, 0.003), ("F2", 2.3e-308, 0.005),
    ("F2", 1e-306, 0.009), ("F2", 1e-307, 0.01), ("F2", 2.964e-323, 0.009),
]

def ref_t(x, c):
    # route 1: t-form c * t**(1+1/c) * exp(-t), t = x**-c
    x, c = mp.mpf(x), mp.mpf(c)
    t = x**(-c)
    return c * t**(1 + 1/c) * mp.exp(-t)

def ref_diff(x, c):
    # route 2: derivative of cdf exp(-x**-c) in u = log x: pdf = dF/du / x
    x, c = mp.mpf(x), mp.mpf(c)
    F = lambda u: mp.exp(-mp.exp(-c*u))
    return mp.diff(F, mp.log(x)) / x

for f, x, c in cases:
    vals = []
    for dps in (100, 200):
        mp.mp.dps = dps
        vals.append((ref_t(x, c), ref_diff(x, c)))
    mp.mp.dps = 200
    (a1, b1), (a2, b2) = vals
    agree = all(abs(p - q) <= abs(q) * mp.mpf(10)**-30 for p, q in [(a1, a2), (b1, b2), (a2, b2)])
    r = a2
    s = stats.invweibull.pdf(x, c)
    if np.isfinite(s) and s != 0:
        ulp = float(abs(mp.mpf(s) - r) / (abs(r) * mp.mpf(2)**-52))
    else:
        ulp = float("inf")
    print(f"{f} pdf({x!r}, {c}) scipy={s!r} ref={mp.nstr(r, 17)} ulp={ulp:.3g} agree30={agree}")
