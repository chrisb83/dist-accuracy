import sys
sys.path.insert(0, r"C:/Users/chbau/Documents/research/dist-accuracy/scratch/gamma/verify")
import numpy as np
from mpmath import mp
from scipy.stats import gamma
import scipy.special as sc
from quadref import P_quad

EPS = 2.220446049250313e-16

def ulp(scipy_val, ref):
    return abs(mp.mpf(scipy_val) - ref) / (abs(ref) * EPS)

def pdf_ref(x, a, dps=100):
    mp.dps = dps
    a = mp.mpf(a); x = mp.mpf(x)
    return mp.e ** ((a - 1) * mp.log(x) - x - mp.loggamma(a))

print("=== F1: cdf / sf ===")
cases = [(995000.0, 1e6), (992000.0, 1e6), (990000.0, 1e6),
         (99954000.0, 1e8), (99900000.0, 1e8)]
for x, a in cases:
    for dps in (100, 200):
        r = P_quad(a, x, dps=dps)
        if dps == 100:
            r100 = r
    mp.dps = 100
    agree = mp.nstr(abs(r100 - r)/abs(r), 3)
    s = gamma.cdf(x, a)
    mp.dps = 100
    print("cdf(%.10g, %g): scipy=%.17g ref=%s ulp=%.3g  (dps100vs200 rel diff %s)"
          % (x, a, s, mp.nstr(r100, 20), float(ulp(s, r100)), agree))
    sq = 1 - r100
    ss = gamma.sf(x, a)
    print("   sf: scipy=%.17g ref=%s ulp=%.3g" % (ss, mp.nstr(sq, 20), float(ulp(ss, sq))))

print()
print("=== headline: x = a - 5 sqrt(a) ===")
for a in (1e5, 3e5, 1e6, 1e7, 1e8, 1e9):
    mp.dps = 60
    x = float(mp.mpf(a) - 5*mp.sqrt(mp.mpf(a)))
    r = P_quad(a, x, dps=100)
    r2 = P_quad(a, x, dps=200)
    mp.dps = 100
    s = sc.gammainc(a, x)
    rel = abs(mp.mpf(s) - r)/abs(r)
    print("a=%g x=%.17g scipy=%.17g ref=%s relerr=%.3g  stab=%.2g"
          % (a, x, s, mp.nstr(r, 20), float(rel), float(abs(r-r2)/abs(r))))

print()
print("=== F2: pdf ===")
for x, a in [(1e6, 1e6), (1010000.0, 1e6), (1003000.0, 1e6), (999.0, 1000.0),
             (1e-280, 1.9932305483800778)]:
    r = pdf_ref(x, a, 100); r2 = pdf_ref(x, a, 200)
    mp.dps = 100
    s = gamma.pdf(x, a)
    print("pdf(%.10g, %g): scipy=%.17g ref=%s ulp=%.3g stab=%.2g"
          % (x, a, s, mp.nstr(r, 20), float(ulp(s, r)), float(abs(r-r2)/abs(r))))
