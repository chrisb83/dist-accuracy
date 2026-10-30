import mpmath as mp, numpy as np, scipy.special as sc
mp.mp.dps = 60
a = 1.5
print("gammainc(1.5, y) at exact double y:")
for x in [1e-102, 1e-90, 1e-70, 1e-40, 1e-10, 1e-3]:
    y = x*x/2.0
    v = sc.gammainc(a, y); r = mp.gammainc(mp.mpf(a), 0, mp.mpf(y), regularized=True)
    # prefactor exp(a log y - y - lgam(a))
    print(f"  x={x:g} y={y:.3e} ulp={float(abs(v-r)/r)/2.22e-16:.1f}")
print("gammaincinv(1.5, q): ulp of y")
for q in [1e-300, 1e-290, 1e-200, 1e-100, 1e-17]:
    y = sc.gammaincinv(a, q)
    f = lambda t: mp.gammainc(mp.mpf(a), 0, t, regularized=True) - q
    r = mp.findroot(f, mp.mpf(y))
    print(f"  q={q:g} y={y!r} ulp={float(abs(y-r)/r)/2.22e-16:.1f}  sqrt(2y) ulp~half")
