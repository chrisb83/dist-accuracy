import numpy as np, mpmath as mp, scipy.special as sc
mp.mp.dps = 60
E = 2.220446049250313e-16
def ulp(a, r): return float(abs(mp.mpf(a)-r)/abs(r))/E
for z in [-30.0, -23.025850929940457, -20.0, -8.0, -5.0, -37.5]:
    y = (-z) * np.sqrt(0.5)          # what cephes forms: a*M_SQRT1_2
    exact = mp.ncdf(mp.mpf(z))
    via_y = mp.erfc(mp.mpf(y))/2      # exact erfc at the rounded y
    print("z=%g ndtr err %.1f ulp | 0.5*erfc(fl(y)) exact vs Phi(z) %.1f ulp | sc.erfc(y)/2 vs exact-at-y %.2f ulp | log_ndtr err %.2f ulp" % (
        z, ulp(sc.ndtr(z), exact), ulp(via_y, exact), ulp(sc.erfc(y)/2, via_y),
        ulp(sc.log_ndtr(z), mp.log(exact))))
