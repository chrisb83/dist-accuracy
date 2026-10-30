import numpy as np, mpmath as mp
from scipy import special as sc
mp.mp.dps = 60
rng = np.random.default_rng(0)
zs = np.concatenate([-np.linspace(5, 37.5, 400), -rng.uniform(5, 37.5, 400)])
errs = []
for z in zs:
    r = mp.ncdf(mp.mpf(z)); errs.append(float(abs(mp.mpf(sc.ndtr(z)) - r)/r/2.0**-52))
errs = np.array(errs); a = np.abs(zs)
for lo, hi in [(5,10),(10,20),(20,30),(30,37.5)]:
    m = (a>=lo)&(a<hi); print(f"|z| in [{lo},{hi}): ndtr max {errs[m].max():.0f} ulp, median {np.median(errs[m]):.0f}, median/z^2 {np.median(errs[m]/a[m]**2):.3f}")
# log_ndtr for positive z (logcdf right tail) vs -ndtr(-z)
for z in [10., 20., 30., 36.841361487904734]:
    r = mp.log1p(-mp.ncdf(-mp.mpf(z)))
    print("log_ndtr", z, float(abs(mp.mpf(sc.log_ndtr(z))-r)/abs(r)/2.0**-52), "ndtr(-z)", float(abs(mp.mpf(sc.ndtr(-z))-mp.ncdf(-mp.mpf(z)))/mp.ncdf(-mp.mpf(z))/2.0**-52))
