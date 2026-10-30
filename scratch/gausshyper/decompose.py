"""Relative error (ulp) of sc.beta(a,b) and sc.hyp2f1(c,a,a+b,-z) per PARAMS set."""
import sys
import mpmath as mp
import scipy.special as sc
sys.path.insert(0, r"C:\Users\chbau\Documents\research\dist-accuracy\refs")
import gausshyper_ref as R
mp.mp.dps = 60
for s in R.PARAMS:
    a, b, c, z = s
    A, B, C, Z = map(mp.mpf, s)
    rb = mp.beta(A, B)
    rh = mp.hyp2f1(C, A, A + B, -Z)
    eb = abs(mp.mpf(sc.beta(a, b)) / rb - 1) / 2.22e-16
    eh = abs(mp.mpf(sc.hyp2f1(c, a, a + b, -z)) / rh - 1) / 2.22e-16
    print(s, "beta ulp", mp.nstr(eb, 3), " hyp2f1 ulp", mp.nstr(eh, 3),
          " hyp2f1 args", (c, a, a + b, -z), repr(sc.hyp2f1(c, a, a + b, -z)), mp.nstr(rh, 17))
