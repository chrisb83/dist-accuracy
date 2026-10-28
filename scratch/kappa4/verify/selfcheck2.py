import sys
sys.path.insert(0, 'C:/Users/chbau/Documents/research/dist-accuracy/scratch/kappa4/verify')
from mpmath import mp
import vref
mp.dps = 50
def safe(t,h,k):
    try: return vref.pdf(t,h,k)
    except Exception: return mp.mpf(0)
for x, h, k in [(1.0, 2.0, 0.5), (-0.5, -2.0, -0.5), (0.3, 0.0, 0.1), (0.25, 1.0, -0.5), (1.5,0.1,0.0), (-2.0,0.0,0.0)]:
    a = vref.support_lower(h, k)
    lo = a if a > -mp.inf else mp.mpf(-50)
    I = mp.quad(lambda t: safe(t, h, k), [lo, mp.mpf(x)], maxdegree=8)
    F = vref.cdf(x, h, k)
    print(f"cdf  x={x} h={h} k={k}: formula={mp.nstr(F,20)} quad(pdf)={mp.nstr(I,20)} rel={mp.nstr(abs(F/I-1),3)}")
for q, h, k in [(0.3, 2.0, 0.5), (0.9, -2.0, -0.5), (1e-5, 0.0, 0.1), (0.36, -1.0, 0.0),
                (0.01, 0.5, 1e-8), (0.1, -1000.0, 0.1), (1e-8, 1.0, -0.5), (1e-300,-1.0,0.0),
                (0.9999999999999997, 0.1, 0.0), (0.9999999999999999, 0.5, 1e-8)]:
    x = vref.ppf(q, h, k)
    F = vref.cdf(x, h, k)
    print(f"ppf  q={q!r} h={h} k={k}: x={mp.nstr(x,20)} cdf(x)/q-1={mp.nstr(abs(F/mp.mpf(q)-1),3)}")
