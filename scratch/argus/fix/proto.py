import sys, numpy as np, mpmath as mp
from scipy import special as sc
sys.path.insert(0, 'refs'); import argus_ref as R
K = 20
def wts(a):
    # c_k = a**k / (2.5)_k
    c = [np.ones_like(a)]
    for k in range(1, K):
        c.append(c[-1]*a/(k+1.5))
    return c
def sf_mix(x, chi):
    a = chi*chi/2; y = (1-x)*(1+x); c = wts(a)
    return sum(ck*sc.betainc(1.5, k+1, y) for k, ck in enumerate(c))/sum(c)
def cdf_mix(x, chi):
    a = chi*chi/2; w = x*x; c = wts(a)
    num = sum(ck*sc.betainc(k+1, 1.5, w) for k, ck in enumerate(c))
    # 1/M(a) = exp(-a) a**1.5 / (Gamma(2.5) P(a))
    if a <= 1: return num/sum(c)
    return num*np.exp(-a + 1.5*np.log(a) - sc.gammaln(2.5) - np.log(sc.gammainc(1.5, a)))
def cdf_new(x, chi):
    a = chi*chi/2; w = x*x
    if a <= 1 or a*w <= 1: return cdf_mix(x, chi)
    b = a*(1-x)*(1+x)
    return (sc.gammaincc(1.5, b) - sc.gammaincc(1.5, a))/sc.gammainc(1.5, a)
mp.mp.dps = 60
worst = {}
for chi in [1e-300, 1e-200, 1e-103, 1e-50, 1e-3, 0.1, 0.5, 1, 1.4142, 1.5, 2, 3, 10, 100, 1e4]:
    for x in [1e-300, 1e-150, 1e-20, 1e-10, 1e-5, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.45, 0.5, 0.7, 0.9, 0.95, 0.99, 0.999999, 1-1e-12, 1-2**-53]:
        for name, f, rf in [("cdf", cdf_new, R.cdf), ("sf", sf_mix if chi*chi/2 <= 1 else None, R.sf)]:
            if f is None: continue
            ref = rf(mp.mpf(x), mp.mpf(chi))
            if ref == 0 or ref < 1e-300: continue
            got = f(x, chi)
            err = float(abs((mp.mpf(got)-ref)/ref))/2.2e-16 if got != 0 else np.inf
            if err > worst.get((name, chi), (0,))[0]: worst[(name, chi)] = (err, x, got, float(ref))
for k, v in sorted(worst.items()): print(k, "%.1f ulp at x=%r got %r ref %r" % v)
