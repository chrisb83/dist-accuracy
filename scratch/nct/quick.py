import sys, time, mpmath as mp
sys.path.insert(0, "refs")
import nct_ref as R
from scipy import stats
mp.mp.dps = 100
for k, c in R.PARAMS:
    for x in [-3.0, -0.5, 0.0, 0.7, 2.5, 30.0]:
        out = []
        for m in ["pdf", "cdf", "sf"]:
            t = time.time(); v = R.REFS[m](mp.mpf(x), mp.mpf(k), mp.mpf(c)); dt = time.time() - t
            s = getattr(stats.nct, m)(x, k, c)
            out.append(f"{m} {float(v):.6e} rel {abs(s/float(v)-1) if float(v) else 0:.1e} {dt:.3f}s")
        print(k, c, x, " | ".join(out))
    for q in [1e-10, 0.3, 0.9]:
        t = time.time(); v = R.ppf(q, k, c); v2 = R.isf(q, k, c); dt = time.time() - t
        print("  q", q, float(v), stats.nct.ppf(q, k, c), float(v2), stats.nct.isf(q, k, c), f"{dt:.2f}s")
