import sys, math, warnings
sys.path.insert(0, r"C:/Users/chbau/Documents/research/dist-accuracy/scratch/mielke/verify")
from mpmath import mp
import ref_check as R
import numpy as np
from scipy.stats import mielke

EPS = 2.220446049250313e-16

CASES = [
 # (finding, method, arg, k, s)
 ("F1","cdf", 1e10, 1e6, 1.0),
 ("F1","cdf", 4.371176973533702e+29, 10.4, 4.6),
 ("F1","cdf", 11.0, 300.0, 4.6),
 ("F1","cdf", 2.0, 1000.0, 1.0),
 ("F1","cdf", 1.036583355609761, 1000.0, 1.0),
 ("F1","cdf", 1.001, 1e-3, 1e6),
 ("F2","pdf", 1e10, 1e6, 1.0),
 ("F2","pdf", 3.554797644010525e+20, 10.4, 4.6),
 ("F2","pdf", 11.0, 300.0, 4.6),
 ("F2","pdf", 2.0, 1000.0, 1.0),
 ("F2","pdf", 1.036583355609761, 1000.0, 1.0),
 ("F3","logpdf", 1e70, 10.4, 4.6),
 ("F3","logpdf", 1.001, 1e-3, 1e6),
 ("F3","logpdf", 1.35242013341704e+154, 0.5, 2.0),
 ("F4","ppf", 0.9999999999999999, 10.4, 4.6),
 ("F4","ppf", 0.9999999999522942, 1e6, 1.0),
 ("F4","ppf", 0.9999999999, 1e6, 1.0),
 ("F4","ppf", 0.9999999999, 1000.0, 1.0),
 ("F4","ppf", 0.85, 1e6, 1.0),
 ("F4","ppf", 0.99999280314327, 300.0, 4.6),
 ("F4","ppf", 0.9991516571017559, 1000.0, 1.0),
 ("F5","ppf", 1e-300, 2.0, 3.0),
 ("F5","ppf", 0.5, 1e-3, 1e6),
 ("F5","ppf", 3.162277660168379e-151, 0.5, 2.0),
 ("F5","ppf", 1.2961424830573168e-81, 0.5, 2.0),
 ("F5","ppf", 1.997134533661588e-216, 2.0, 3.0),
 ("F5","ppf", 0.9999992550002775, 1e-3, 1e6),
 ("F6","pdf", 1.31830766936289e-309, 0.001, 0.001),
 ("F6","pdf", 2.686618191037897e-309, 0.001, 1e6),
 ("F10","logpdf", 0.0, 1.0, 1.0),
 ("F10","logpdf", 0.0, 0.5, 2.0),
 ("F10","logpdf", 0.0, 2.0, 3.0),
]

def refval(method, arg, k, s, dps):
    mp.dps = dps
    f = getattr(R, method)
    return f(arg, k, s)

def ulp(scipy_v, ref):
    if ref == 0:
        return float('nan')
    if not np.isfinite(scipy_v):
        return float('inf')
    mp.dps = 60
    return float(abs(mp.mpf(scipy_v) - ref) / (abs(ref) * mp.mpf(EPS)))

print(f"{'id':4} {'method':7} {'arg':24} {'k':8} {'s':8} {'scipy':26} {'ref(100dps)':26} {'ulp':>10} {'stable'}")
for fid, method, arg, k, s in CASES:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = float(getattr(mielke, method)(arg, k, s))
    r100 = refval(method, arg, k, s, 100)
    r200 = refval(method, arg, k, s, 200)
    mp.dps = 60
    if mp.isinf(r100) or r100 == 0:
        stable = (r100 == r200)
        agree = 'exact' if stable else 'DIFF'
    else:
        rel = abs(r100 - r200)/abs(r200)
        stable = rel < mp.mpf(10)**(-30)
        agree = 'ok' if stable else f'DIFF {mp.nstr(rel,3)}'
    mp.dps = 25
    print(f"{fid:4} {method:7} {arg:<24.17g} {k:<8g} {s:<8g} {sv!s:26} {mp.nstr(r200,17):26} {ulp(sv,r200):10.3g} {agree}")
