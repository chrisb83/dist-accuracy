import sys, warnings
sys.path.insert(0, sys.argv[1]); sys.path.insert(1, r'C:/Users/Christoph/Documents/repos/dist-accuracy/refs')
import numpy as np, mpmath as mp
from scipy import stats
import argus_ref as R
mp.mp.dps = 60
def ulp(m, x, chi):
    ref = getattr(R, m)(mp.mpf(x), mp.mpf(chi)); got = float(getattr(stats.argus, m)(x, chi))
    return float(abs((mp.mpf(got)-ref)/ref))/2.22e-16, got, float(ref)
for x in [0.01, 0.1, 0.2, 0.3, 0.5]:
    print("pdf", x, 10, ulp("pdf", x, 10))
# cdf around the switch b = 1, and bulk
for chi in [1.4143, 1.5, 2, 3, 10, 30]:
    a = chi*chi/2
    xb = np.sqrt(1 - 1/a)
    res = []
    for x in [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 0.95, 0.99, xb*(1-1e-9), xb*(1+1e-9)]:
        if not 0 < x < 1: continue
        res.append((round(ulp("cdf", x, chi)[0], 1), x))
    print("cdf", chi, "switch x=%.4f" % xb, max(res), res)
