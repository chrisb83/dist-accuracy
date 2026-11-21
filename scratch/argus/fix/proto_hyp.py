import sys, numpy as np, mpmath as mp
from scipy import special as sc
sys.path.insert(0, 'refs'); import argus_ref as R
mp.mp.dps = 60
def A_new(chi):
    a = chi*chi/2
    return np.log(3) + a - np.log(sc.hyp1f1(1, 2.5, a))
def A_ref(chi):
    chi = mp.mpf(chi); a = chi**2/2
    return mp.log(3) + a - mp.log(mp.hyp1f1(1, 2.5, a))
def sf_hyp(x, chi):
    a = chi*chi/2; y = (1-x)*(1+x); b = a*y
    return y**1.5*np.exp(a*x*x)*sc.hyp1f1(1, 2.5, b)/sc.hyp1f1(1, 2.5, a)
ulp = lambda g, r: float(abs((mp.mpf(g)-r)/r))/2.2e-16
print("A:", max((ulp(A_new(c), A_ref(c)), c) for c in [1e-300, 1e-103, 1e-50, 1e-3, 0.1, 0.5, 1, 1.2, 1.4142]))
w = (0, None)
for chi in [1e-300, 1e-200, 1e-103, 1e-50, 1e-3, 0.1, 0.5, 1, 1.4142]:
    for x in [1e-10, 1e-3, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999999, 1-1e-12, 1-2**-53]:
        e = ulp(sf_hyp(x, chi), R.sf(mp.mpf(x), mp.mpf(chi)))
        if e > w[0]: w = (e, (x, chi))
print("sf:", w)
