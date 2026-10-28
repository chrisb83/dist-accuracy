import sys
sys.path.insert(0, 'C:/Users/chbau/Documents/research/dist-accuracy/scratch/kappa4/verify')
from mpmath import mp
import vref

mp.dps = 60
# 1) pdf from numerical derivative of cdf (different route)
pts = [(1.0, 2.0, 0.5), (-0.5, -2.0, -0.5), (0.3, 0.0, 0.1), (1.5, 0.1, 0.0),
       (0.7, -1.0, 0.0), (-2.0, 0.0, 0.0), (0.25, 1.0, -0.5)]
for x, h, k in pts:
    d = mp.diff(lambda t: vref.cdf(t, h, k), mp.mpf(x))
    f = vref.pdf(x, h, k)
    print(f"pdf  x={x} h={h} k={k}: formula={mp.nstr(f,20)} diff(cdf)={mp.nstr(d,20)} rel={mp.nstr(abs(f/d-1),3)}")
# 2) cdf from integral of pdf
for x, h, k in [(1.0, 2.0, 0.5), (-0.5, -2.0, -0.5), (0.3, 0.0, 0.1), (0.25, 1.0, -0.5)]:
    a = vref.support_lower(h, k)
    lo = a if a > -mp.inf else mp.mpf(-40)
    I = mp.quad(lambda t: vref.pdf(t, h, k), [lo, mp.mpf(x)])
    F = vref.cdf(x, h, k)
    print(f"cdf  x={x} h={h} k={k}: formula={mp.nstr(F,20)} quad(pdf)={mp.nstr(I,20)} rel={mp.nstr(abs(F/I-1),3)}")
# 3) ppf inverts cdf
for q, h, k in [(0.3, 2.0, 0.5), (0.9, -2.0, -0.5), (1e-5, 0.0, 0.1), (0.36, -1.0, 0.0),
                (0.01, 0.5, 1e-8), (0.1, -1000.0, 0.1), (1e-8, 1.0, -0.5)]:
    x = vref.ppf(q, h, k)
    F = vref.cdf(x, h, k)
    print(f"ppf  q={q} h={h} k={k}: x={mp.nstr(x,20)} cdf(x)={mp.nstr(F,20)} rel={mp.nstr(abs(F/mp.mpf(q)-1),3)}")
