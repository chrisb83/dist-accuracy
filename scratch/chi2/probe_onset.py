import mpmath as mp, numpy as np, scipy.special as sc
mp.mp.dps = 60
for k in [1e5, 2e5, 4e5, 6e5, 1e6, 2e6]:
    x = k - 5*np.sqrt(2*k)
    ref = mp.gammainc(mp.mpf(k)/2, 0, mp.mpf(x)/2, regularized=True)
    got = sc.chdtr(k, x)
    rel = abs(got - ref)/ref
    print(f"df={k:.0e} x={x!r} chdtr={got!r} ref={mp.nstr(ref,17)} rel={float(rel):.1e} ulp={float(rel)/2.22e-16:.1e}")
