import numpy as np, mpmath as mp, scipy.special as sc
mp.mp.dps = 60
def ulp(a, r): return float(abs(mp.mpf(a)-r)/abs(r))/2.220446049250313e-16
for x in [1e-10, 9.357622968840175e-14, 1e10]:
    z = np.log(x)
    zt = mp.log(mp.mpf(x))
    print("cdf x=%g: ulp(log)=%.2f ndtr(z_double) vs ncdf(z_double): %.2f ulp; total %.1f ulp; z^2*dz=%.1f" % (
        x, ulp(z, zt), ulp(sc.ndtr(z), mp.ncdf(mp.mpf(z))), ulp(sc.ndtr(z), mp.ncdf(zt)),
        float(abs(zt)*abs(mp.mpf(z)-zt)/2.220446049250313e-16)))
for q in [2.872984833353539e-288, 5.6234132519034905e-226]:
    n = sc.ndtri(q)
    nt = mp.findroot(lambda t: mp.ncdf(t) - mp.mpf(q), mp.mpf(n))
    print("ppf q=%g: ndtri err %.2f ulp (rel), |t|=%.1f, predicted exp err %.1f ulp; exp(ndtri_double) vs exp(true) %.1f" % (
        q, ulp(n, nt), abs(n), ulp(n, nt)*abs(n), ulp(np.exp(n), mp.exp(nt))))
