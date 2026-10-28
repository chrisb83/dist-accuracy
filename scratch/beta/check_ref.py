import sys, time
sys.path.insert(0, "refs")
import mpmath as mp
import beta_ref as R
mp.mp.dps = 120
for a, b, x in [(2.3, 0.63, 0.5), (2.3, 0.63, 0.99), (1e-3, 1e-3, 1e-100), (1e-3, 1e-3, 0.7),
                (0.5, 5, 0.3), (20, 50, 0.1), (20, 50, 0.6), (1e3, 1e3, 0.45)]:
    t = time.time(); v = R.cdf(x, a, b); t1 = time.time() - t
    w = mp.betainc(a, b, 0, x, regularized=True)
    print(a, b, x, mp.nstr(v, 20), mp.nstr(abs(v - w) / w, 3), round(t1, 4))
for a, b, x in [(1e6, 1e6, 0.4999), (1e5, 3e5, 0.25), (1e6, 2, 0.999999), (1e6, 2, 0.99999)]:
    t = time.time(); v = R.cdf(x, a, b); s = R.sf(x, a, b)
    print(a, b, x, mp.nstr(v, 20), mp.nstr(s, 20), round(time.time() - t, 4))
t = time.time(); print(mp.nstr(R.ppf(1e-100, 1e5, 3e5), 20), time.time() - t)
t = time.time(); print(mp.nstr(R.isf(1e-100, 2.3, 0.63), 20), time.time() - t)
