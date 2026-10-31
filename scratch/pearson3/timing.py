import sys, time
sys.path.insert(0, "refs")
import mpmath as mp
import pearson3_ref as R
mp.mp.dps = 400
for k in (0.1, -2.0, 1e-3, -1e-3, 10.0, -100.0):
    for m, a in (("pdf", 0.5), ("cdf", -3.0), ("cdf", 3.0), ("cdf", 1e10), ("sf", -6.0), ("ppf", 1e-8), ("isf", 1e-8)):
        t = time.time()
        try:
            v = R.REFS[m](mp.mpf(a), mp.mpf(k))
            print(k, m, a, mp.nstr(v, 12), round(time.time() - t, 3))
        except Exception as e:
            print(k, m, a, type(e).__name__, round(time.time() - t, 3))
# validate quadrature route at alpha = 4e6
mp.mp.dps = 100
for k in (1e-3, -1e-3):
    for x in (-8.0, -5.0, -1.0, 0.0, 2.0, 6.0, 12.0):
        t = time.time()
        g = R.cdf(mp.mpf(x), mp.mpf(k))
        R.ALPHA_QUAD = mp.mpf(1e5)
        q = R.cdf(mp.mpf(x), mp.mpf(k))
        R.ALPHA_QUAD = mp.mpf(1e7)
        print("val", k, x, mp.nstr(g, 15), mp.nstr(abs(q / g - 1), 3), mp.nstr(abs((1-q) / (1-g) - 1), 3), round(time.time() - t, 2))
