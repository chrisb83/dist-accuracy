# F3 check: gammaln(c+1) vs _lgam1p(c) in loggamma._sf's x < _LOGXMIN branch; F1 ppf/isf at subnormal c
import numpy as np, scipy.special as sc, mpmath as mp
from scipy.special import _ufuncs as u
from scipy import stats
mp.mp.dps = 50
for c, x in [(1e-10, -708.3964185322642), (1e-14, -800.0), (1e-6, -800.0)]:
    ref = -mp.expm1(mp.mpf(c) * x - mp.loggamma(1 + mp.mpf(c)))
    print(c, x, -np.expm1(c*x - sc.gammaln(c+1)), -np.expm1(c*x - u._lgam1p(c)), mp.nstr(ref, 17))
with np.errstate(all="ignore"):
    print("ppf(0.99,1e-310)", stats.loggamma.ppf(0.99, 1e-310), "true", np.log(0.99)/1e-310)
    print("isf(0.01,1e-310)", stats.loggamma.isf(0.01, 1e-310))
