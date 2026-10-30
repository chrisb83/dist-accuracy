import sys; sys.path.insert(0, 'refs')
import mpmath as mp, scipy.special as sc
import maxwell_ref as r
mp.mp.dps = 400
for q in [1e-300, 1e-290, 1e-200, 1e-100, 1e-17]:
    y = sc.gammaincinv(1.5, q)
    yr = r.ppf(mp.mpf(q))**2 / 2
    print(f"q={q:g} gammaincinv(1.5,q) ulp={float(abs(y-yr)/yr)/2.22e-16:.1f}")
