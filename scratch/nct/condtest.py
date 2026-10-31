import sys, time, mpmath as mp
sys.path.insert(0, "refs"); import nct_ref as R
mp.mp.dps = 100
k, c, q = 14.0, 0.24045031331198066, 0.3
t = time.time()
x = R.ppf(q, k, c)
an = q / (x * R.pdf(x, k, c))
g = lambda e: R.ppf(mp.mpf(q) * (1 + e), k, c)
num = mp.diff(g, 0) / x
print("analytic", mp.nstr(an, 15), "numeric", mp.nstr(num, 15), f"{time.time()-t:.1f}s")
