import sys, warnings
sys.path.insert(0, sys.argv[1]); sys.path.insert(1, r'C:/Users/Christoph/Documents/repos/dist-accuracy/fixes')
import numpy as np, mpmath as mp
from scipy import stats
from scipy.stats._continuous_distns import _argus_phi
from scipy.stats.tests.test_distributions import TestArgus as T
import argus as A   # check-script closed forms (mp.dps = 100)

def params(f):
    m = [m for m in f.pytestmark if m.name == 'parametrize'][0]
    return m.args[1]

def rel(got, ref):
    return float(abs((mp.mpf(float(got)) - ref)/ref))

X = [0.1, 0.5, 0.9]
for name, meth in [('test_pdf_small_chi', 'pdf'), ('test_sf_small_chi', 'sf'), ('test_cdf_small_chi', 'cdf')]:
    print(name)
    for chi, exp in params(getattr(T, name)):
        refs = [getattr(A, meth)(x, chi) for x in X]
        got = getattr(stats.argus, meth)(X, chi)
        err = max(rel(g, r) for g, r in zip(got, refs))
        stale = max(rel(e, r) for e, r in zip(exp, refs))
        print(f"  chi={chi!r:8}: scipy err {err:.1e}, hard-coded ref err {stale:.1e}")
print('test_sf_near_1')
for x, chi, exp in params(T.test_sf_near_1):
    r = A.sf(x, chi)
    print(f"  {x!r}, {chi!r}: scipy err {rel(stats.argus.sf(x, chi), r):.1e}, ref err {rel(exp, r):.1e}")
print('test_argus_phi_small_chi')
for chi, exp, rtol in params(T.test_argus_phi_small_chi):
    c = mp.mpf(chi); r = mp.gammainc(1.5, 0, c**2/2, regularized=True)/2
    print(f"  chi={chi!r:8}: scipy err {rel(_argus_phi(chi), r):.1e}, ref err {rel(exp, r):.1e}, rtol {rtol}")
