import sys, warnings
sys.path.insert(0, sys.argv[1]); sys.path.insert(1, r'C:/Users/Christoph/Documents/repos/dist-accuracy/refs')
import numpy as np, mpmath as mp, scipy
from scipy import stats
import argus_ref as R
mp.mp.dps = 60
print(scipy.__file__)
xs = [1e-300, 1e-150, 1e-20, 1e-10, 1e-5, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.35, 0.45, 0.5, 0.7, 0.9, 0.95, 0.99, 0.999999, 1-1e-12, 1-2**-53]
chis = [1e-300, 1e-200, 1e-110, 1e-103, 1e-100, 1e-50, 1e-3, 0.1, 0.5, 1, 1.414, 1.4143, 2, 3, 10, 100, 1e4, 1e155]
warnings.simplefilter("ignore")
for meth in ["pdf", "logpdf", "sf", "cdf"]:
    for chi in chis:
        worst = (0, None)
        for x in xs:
            ref = getattr(R, meth)(mp.mpf(x), mp.mpf(chi))
            got = float(getattr(stats.argus, meth)(x, chi))
            if abs(ref) < 1e-300 or (meth != "logpdf" and ref == 0): continue
            e = float(abs((mp.mpf(got)-ref)/ref))/2.22e-16 if np.isfinite(got) else np.inf
            if meth == "cdf" and x < 1e-3 and chi < 50: continue  # small-x part of F1: not fixed
            f = lambda t: getattr(R, meth)(t, mp.mpf(chi))
            cond = float(abs(mp.mpf(x)*mp.diff(f, mp.mpf(x))/ref)) if x < 1 - 1e-13 else 1e16
            e = e/(8 + cond)
            if e > worst[0]: worst = (e, x, got, float(ref))
        print(meth, chi, "%.3g ulp/(8+cond)" % worst[0], worst[1:] if worst[0] > 20 else "")
