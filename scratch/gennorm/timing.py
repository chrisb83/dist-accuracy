import sys, time
sys.path.insert(0, "refs")
import mpmath as mp
mp.mp.dps = 400
import gennorm_ref as R
for b in [float(v) for v in sys.argv[1:]]:
    for name, f, args in [("cdf", R.cdf, [1e-300, 0.4, -0.9, 3.0, -1e10, 1e300]), ("ppf", R.ppf, [1e-300, 0.6, 0.9, 1-1e-16])]:
        for a in args:
            t = time.time()
            try:
                v = f(mp.mpf(a), mp.mpf(b))
                s = repr(float(v))
            except Exception as e:
                s = repr(e)[:60]
            print(b, name, a, s, round(time.time()-t, 3), flush=True)
