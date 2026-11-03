import sys, time
sys.path.insert(0, r"C:/Users/chbau/Documents/research/dist-accuracy/refs")
import mpmath as mp
mp.mp.dps = 400
import halfgennorm_ref as R
for b in (0.6748054997000371, 0.01, 2.0, 1e3, 1e300):
    for m, args in (("cdf", 0.5), ("sf", 3.0), ("logpdf", 0.5), ("ppf", 1e-8), ("ppf", 0.7), ("isf", 1e-8), ("isf", 0.7)):
        t = time.time()
        v = R.REFS[m](mp.mpf(args), mp.mpf(b))
        print(b, m, args, mp.nstr(v, 15), round(time.time() - t, 3))
