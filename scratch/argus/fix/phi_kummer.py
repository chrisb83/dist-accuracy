import sys, warnings
sys.path.insert(0, sys.argv[1]); sys.path.insert(1, r'C:/Users/Christoph/Documents/repos/dist-accuracy/fixes')
import numpy as np, mpmath as mp
from scipy import special as sc, stats
from scipy.stats._continuous_distns import _argus_phi
import argus as A
mp.mp.dps = 60
def phi_kummer(chi):
    a = 0.5*chi*chi
    return np.where(chi <= np.sqrt(2),
                    chi**3*np.exp(-a)*sc.hyp1f1(1, 2.5, a)/(3*np.sqrt(2*np.pi)),
                    sc.gammainc(1.5, a)/2)
def ref(chi):
    c = mp.mpf(chi); return mp.gammainc(1.5, 0, c**2/2, regularized=True)/2
ulp = lambda g, r: float(abs((mp.mpf(float(g)) - r)/r))/2.22e-16
print("_argus_phi: ulp current vs Kummer")
for chi in [1e-100, 1e-50, 1e-20, 1e-9, 1e-3, 0.1, 0.5, 1.0, 1.414]:
    r = ref(chi)
    print(f"  chi={chi:<7g} {ulp(_argus_phi(chi), r):7.1f} {ulp(phi_kummer(chi), r):7.1f}")
print("sf, chi > sqrt(2), x near 1: ulp current vs ratio of Kummer phi")
for chi in [1.5, 3, 10, 100]:
    for x in [0.9, 0.999999, 1 - 2**-53]:
        r = A.sf(x, chi)
        new = phi_kummer(chi*np.sqrt((1 - x)*(1 + x)))/phi_kummer(chi)
        print(f"  chi={chi:<5g} x={x!r:<20} {ulp(stats.argus.sf(x, chi), r):6.1f} {ulp(new, r):6.1f}")
print("mean (stats) small chi: ulp current vs Kummer phi")
for chi in [1e-9, 1e-3, 0.1, 1.0]:
    c = mp.mpf(chi)
    r = mp.quad(lambda t: t*A.pdf(t, chi), [0, 1])
    m_new = np.sqrt(np.pi/8)*chi*sc.ive(1, chi**2/4)/phi_kummer(chi)
    print(f"  chi={chi:<6g} {ulp(stats.argus.mean(chi), r):6.1f} {ulp(m_new, r):6.1f}")
