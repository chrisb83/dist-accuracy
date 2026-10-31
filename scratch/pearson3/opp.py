import numpy as np, mpmath as mp, warnings
from scipy import stats
warnings.simplefilter("ignore")
mp.mp.dps = 60
def ab(k):
    k = mp.mpf(k); b = 2/k; return b**2, b
# logsf right tail, skew 0.1
k, x = 0.1, 200.0
a, b = ab(k); t = a + b*x
print("logsf", x, k, stats.pearson3.logsf(x, k), mp.nstr(mp.log(mp.gammainc(a, t, mp.inf, regularized=True)), 17))
# logcdf left tail near support edge, skew 0.1
x = -19.9; t = a + b*mp.mpf(x)
print("logcdf", x, k, stats.pearson3.logcdf(x, k), mp.nstr(mp.log(mp.gammainc(a, 0, t, regularized=True)), 17))
# isf small q, skew 0.1 and -0.1
for k in (0.1, -0.1):
    a, b = ab(k); q = mp.mpf(1e-20)
    # solve upper tail: skew>0 S = Q(a,t); skew<0 S = P(a,t)
    if k > 0:
        f = lambda t: mp.log(mp.gammainc(a, t, mp.inf, regularized=True)) - mp.log(q); t0 = a + 10*abs(b)
    else:
        f = lambda t: mp.log(mp.gammainc(a, 0, t, regularized=True)) - mp.log(q); t0 = a - 8*abs(b)
    tr = mp.findroot(f, t0)
    xr = (tr - a)/b
    print("isf 1e-20", k, stats.pearson3.isf(1e-20, k), mp.nstr(xr, 17))
