# Independent check: cdf = integral_0^x of docstring pdf sqrt(2/pi) cosh(cx) exp(-(x^2+c^2)/2)
import mpmath as mp
import numpy as np
from scipy import stats

def pdf(t, c):
    return mp.sqrt(2/mp.pi)*mp.cosh(c*t)*mp.exp(-(t*t+c*c)/2)

def cdf(x, c):
    x = mp.mpf(x); c = mp.mpf(c)
    pts = [mp.mpf(0)]
    for d in (60, 20, 5, 1):
        if x - d > 0: pts.append(x - d)
    pts.append(x)
    return mp.quad(lambda t: pdf(t, c), pts)

calls = [(1e-20, 1.952125337355587), (1e-12, 1.952125337355587), (1e-10, 5.0),
         (1e-17, 0.001), (1e-10, 0.001), (1e-06, 1.0), (0.5, 5.0),
         (1.0, 30.0), (965.0, 1000.0), (25.0, 30.0), (995.0, 1000.0)]
for x, c in calls:
    mp.mp.dps = 100; r1 = cdf(x, c)
    mp.mp.dps = 200; r2 = cdf(x, c)
    agree = abs(r1 - r2) <= abs(r2)*mp.mpf(10)**-30
    s = stats.foldnorm.cdf(x, c)
    ref = float(r2)
    ulp = abs(s - ref)/(abs(ref)*2.220446049250313e-16)
    print(f"cdf({x!r}, {c!r}) scipy={s!r} ref={ref!r} ulp={ulp:.3g} agree30={agree}")
