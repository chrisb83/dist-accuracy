import sys, time
sys.path.insert(0, "refs")
import mpmath as mp
import chi2_ref as R
mp.mp.dps = 400
for k, x, q in [(55.0, 50.0, 1e-8), (2e-3, 1e-100, 0.5), (2e6, 1.99e6, 1e-8), (2e6, 2.01e6, 1-1e-8), (2e3, 3000., 1e-16)]:
    k = mp.mpf(k); x = mp.mpf(x); q = mp.mpf(q)
    for name in ["pdf", "cdf", "sf", "ppf", "isf"]:
        t = time.time(); arg = q if name in ("ppf", "isf") else x
        v = R.REFS[name](arg, k)
        print(name, float(k), mp.nstr(v, 10), round(time.time() - t, 3))
